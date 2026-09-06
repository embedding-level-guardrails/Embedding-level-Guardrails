"""Цикл дообучения энкодера для RQ3.

Три объектива под одним циклом:
  contrastive    — только contrastive-лосс на триплетах
  classification — только CE поверх линейной головы (это и есть базлайн RQ3)
  joint          — contrastive + CE с весом

Валидация считается тем, что реально важно для guardrail: AUROC и TPR@FPR=1%.
Скор берётся из головы, если она есть, иначе — расстояние до центроидов (это же
прообраз distance-based guardrail из RQ5), так что чисто contrastive-прогон тоже
сравним по тем же числам. Лучший чекпоинт выбирается по TPR@FPR, а не по лоссу:
средний лосс не отражает рабочую точку с низким FPR.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..analysis.probe import binary_metrics
from ..utils import get_logger, l2_normalize
from .data import PairDataset, TextDataset, collate_pairs, collate_texts, texts_and_labels
from .losses import embedding_kl_regularizer, info_nce, supervised_contrastive, triplet_margin
from .model import GuardEncoder

logger = get_logger(__name__)


@dataclass
class TrainConfig:
    objective: str = "contrastive"          # contrastive | classification | joint
    loss: str = "info_nce"                  # info_nce | supervised_contrastive | triplet_margin
    temperature: float = 0.05
    margin: float = 0.2
    epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 128
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1
    ce_weight: float = 1.0                  # вес CE в joint
    contrastive_weight: float = 1.0         # вес contrastive в joint
    kl_weight: float = 0.0                  # >0 включает якорь к базовой модели (RQ4)
    kl_temperature: float = 0.1
    eval_every: int = 200                   # в шагах оптимизатора; 0 = только по эпохам
    log_every: int = 20
    target_fpr: float = 0.01
    pair_types: list[str] = field(default_factory=list)
    seed: int = 42


def _next_cycled(iterator, loader):
    """Следующий батч; исчерпанный загрузчик перезапускается.

    В objective=joint два загрузчика разной длины: пар обычно вдвое больше, чем
    уникальных текстов (16765 против 7904), а число шагов в эпохе считается по
    парам. Без перезапуска короткий загрузчик кончается в середине эпохи и роняет
    обучение StopIteration.
    """
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def build_optimizer(model: GuardEncoder, cfg: TrainConfig, total_steps: int):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if any(k in name for k in ("bias", "LayerNorm.weight", "layer_norm")) else decay).append(param)

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
    )
    warmup = max(int(total_steps * cfg.warmup_ratio), 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _embed_batch(model: GuardEncoder, texts: list[str], device: torch.device) -> torch.Tensor:
    batch = model.tokenize(texts).to(device)
    return model(batch["input_ids"], batch["attention_mask"])


def contrastive_step(model: GuardEncoder, batch: dict, device: torch.device,
                     cfg: TrainConfig) -> tuple[torch.Tensor, dict[str, float]]:
    anchor = _embed_batch(model, batch["anchor"], device)
    positive = _embed_batch(model, batch["positive"], device)
    negative = _embed_batch(model, batch["negative"], device)

    if cfg.loss == "info_nce":
        loss = info_nce(anchor, positive, negative, temperature=cfg.temperature)
    elif cfg.loss == "triplet_margin":
        loss = triplet_margin(anchor, positive, negative, margin=cfg.margin)
    elif cfg.loss == "supervised_contrastive":
        # SupCon смотрит на класс, а не на пару: собираем батч из якорей и негативов.
        embeddings = torch.cat([anchor, negative])
        labels = torch.cat([batch["anchor_label"], batch["negative_label"]]).to(device)
        loss = supervised_contrastive(embeddings, labels, temperature=cfg.temperature)
    else:
        raise ValueError(f"Unknown contrastive loss: {cfg.loss}")

    with torch.no_grad():
        a, p, n = (F.normalize(t, dim=-1) for t in (anchor, positive, negative))
        stats = {
            "cos_anchor_positive": float((a * p).sum(-1).mean()),
            "cos_anchor_negative": float((a * n).sum(-1).mean()),
        }
        stats["cos_margin"] = stats["cos_anchor_positive"] - stats["cos_anchor_negative"]
    return loss, stats


def classification_step(model: GuardEncoder, texts: list[str], labels: torch.Tensor,
                        device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = _embed_batch(model, texts, device)
    logits = model.logits(embeddings)
    return F.cross_entropy(logits, labels.to(device)), embeddings


@torch.no_grad()
def evaluate(model: GuardEncoder, texts: list[str], y: np.ndarray, device: torch.device,
             cfg: TrainConfig, train_embeddings: tuple[np.ndarray, np.ndarray] | None = None
             ) -> dict[str, float]:
    """AUROC / TPR@FPR на валидации.

    Скор — вероятность головы, если она есть; иначе cos(x, mu_harm) - cos(x, mu_safe)
    по центроидам train. Второй вариант нарочно совпадает с distance-based скором
    из RQ5, чтобы contrastive-прогон мерился той же линейкой.
    """
    model.eval()
    embeddings = model.encode_texts(texts, device, batch_size=cfg.eval_batch_size)

    if model.head is not None:
        scores = torch.softmax(model.logits(embeddings.to(device)).float(), dim=-1)[:, 1].cpu().numpy()
        threshold = 0.5
    elif train_embeddings is not None:
        x_train, y_train = train_embeddings
        mu_safe = l2_normalize(x_train[y_train == 0].mean(0)[None])[0]
        mu_harm = l2_normalize(x_train[y_train == 1].mean(0)[None])[0]
        vectors = embeddings.cpu().numpy()
        scores = vectors @ mu_harm - vectors @ mu_safe
        threshold = 0.0
    else:
        raise ValueError("Нечем считать скор: нет ни головы, ни train-эмбеддингов для центроидов")

    model.train()
    return binary_metrics(y, scores, threshold=threshold, target_fpr=cfg.target_fpr)


def train(
    model: GuardEncoder,
    device: torch.device,
    cfg: TrainConfig,
    pairs_train: list[dict],
    texts_train: list[dict],
    val_records: list[dict],
    tracker,
    output_dir: str | Path,
    teacher: GuardEncoder | None = None,
) -> dict[str, Any]:
    """Общий цикл. Возвращает историю и лучшие метрики."""
    torch.manual_seed(cfg.seed)
    use_contrastive = cfg.objective in ("contrastive", "joint")
    use_classification = cfg.objective in ("classification", "joint")

    if use_classification and model.head is None:
        raise ValueError(f"objective={cfg.objective} требует модель с головой (with_head=True)")

    if use_contrastive:
        pair_loader = DataLoader(
            PairDataset(pairs_train, cfg.pair_types or None), batch_size=cfg.batch_size,
            shuffle=True, collate_fn=collate_pairs, drop_last=True,
        )
    else:
        pair_loader = None

    if use_classification:
        text_loader = DataLoader(
            TextDataset(texts_train), batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collate_texts, drop_last=True,
        )
    else:
        text_loader = None

    steps_per_epoch = len(pair_loader) if pair_loader is not None else len(text_loader)
    if pair_loader is not None and text_loader is not None and len(pair_loader) != len(text_loader):
        logger.info("joint: %d батчей пар против %d батчей текстов — короткий загрузчик "
                    "будет циклиться", len(pair_loader), len(text_loader))
    total_steps = max(steps_per_epoch * cfg.epochs // max(cfg.grad_accum_steps, 1), 1)
    optimizer, scheduler = build_optimizer(model, cfg, total_steps)

    val_texts, val_y = texts_and_labels(val_records)
    train_texts_flat, train_y_flat = texts_and_labels(texts_train)

    history: list[dict[str, Any]] = []
    best = {"tpr_at_fpr": -1.0, "step": -1}
    checkpoint_dir = Path(output_dir) / "checkpoint"
    tpr_key = f"tpr_at_fpr_{cfg.target_fpr:g}"
    global_step = 0
    started = time.perf_counter()

    def run_eval(step: int) -> dict[str, float]:
        centroids = None
        if model.head is None:
            # Центроиды считаем на подвыборке train: полный проход после каждого
            # eval стоит дороже самой валидации и ничего не добавляет.
            take = min(len(train_texts_flat), 1024)
            idx = np.random.default_rng(cfg.seed).choice(len(train_texts_flat), take, replace=False)
            subset = [train_texts_flat[i] for i in idx]
            centroids = (model.encode_texts(subset, device, cfg.eval_batch_size).cpu().numpy(),
                         train_y_flat[idx])
        metrics = evaluate(model, val_texts, val_y, device, cfg, train_embeddings=centroids)
        tracker.log_metrics({f"val_{k}": v for k, v in metrics.items()}, step=step)
        logger.info("шаг %d | val AUROC=%.4f TPR@FPR=%.4f FPR=%.4f FNR=%.4f",
                    step, metrics["auroc"], metrics[tpr_key], metrics["fpr"], metrics["fnr"])
        return metrics

    model.train()
    for epoch in range(cfg.epochs):
        pair_iter = iter(pair_loader) if pair_loader is not None else None
        text_iter = iter(text_loader) if text_loader is not None else None

        for micro_step in range(steps_per_epoch):
            loss = torch.zeros((), device=device)
            stats: dict[str, float] = {}

            if pair_iter is not None:
                batch, pair_iter = _next_cycled(pair_iter, pair_loader)
                contrastive_loss, stats = contrastive_step(model, batch, device, cfg)
                loss = loss + cfg.contrastive_weight * contrastive_loss
                stats["loss_contrastive"] = float(contrastive_loss.detach())

            if text_iter is not None:
                text_batch, text_iter = _next_cycled(text_iter, text_loader)
                ce_loss, embeddings = classification_step(
                    model, text_batch["text"], text_batch["label"], device
                )
                loss = loss + cfg.ce_weight * ce_loss
                stats["loss_ce"] = float(ce_loss.detach())

                if teacher is not None and cfg.kl_weight > 0:
                    with torch.no_grad():
                        reference = _embed_batch(teacher, text_batch["text"], device)
                    kl = embedding_kl_regularizer(embeddings, reference, cfg.kl_temperature)
                    loss = loss + cfg.kl_weight * kl
                    stats["loss_kl"] = float(kl.detach())

            (loss / max(cfg.grad_accum_steps, 1)).backward()

            if (micro_step + 1) % max(cfg.grad_accum_steps, 1) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if cfg.log_every and global_step % cfg.log_every == 0:
                    payload = {"loss": float(loss.detach()), "lr": scheduler.get_last_lr()[0], **stats}
                    tracker.log_metrics({f"train_{k}": v for k, v in payload.items()}, step=global_step)
                    logger.info("эпоха %d шаг %d | loss=%.4f", epoch, global_step, float(loss.detach()))

                if cfg.eval_every and global_step % cfg.eval_every == 0:
                    metrics = run_eval(global_step)
                    history.append({"step": global_step, **metrics})
                    if metrics[tpr_key] > best["tpr_at_fpr"]:
                        best = {"tpr_at_fpr": metrics[tpr_key], "auroc": metrics["auroc"],
                                "step": global_step}
                        model.save(checkpoint_dir)

        metrics = run_eval(global_step)
        history.append({"step": global_step, "epoch": epoch, **metrics})
        if metrics[tpr_key] > best["tpr_at_fpr"]:
            best = {"tpr_at_fpr": metrics[tpr_key], "auroc": metrics["auroc"], "step": global_step}
            model.save(checkpoint_dir)

    # Если ни один eval не улучшил стартовое значение, чекпоинта ещё нет.
    if not checkpoint_dir.exists():
        model.save(checkpoint_dir)

    elapsed = time.perf_counter() - started
    tracker.log_metrics({"best_val_tpr_at_fpr": best["tpr_at_fpr"],
                         "best_val_auroc": best.get("auroc", float("nan")),
                         "train_seconds": elapsed}, step=global_step)
    logger.info("Готово за %.1f c. Лучший шаг %d: TPR@FPR=%.4f",
                elapsed, best["step"], best["tpr_at_fpr"])
    return {"history": history, "best": best, "steps": global_step,
            "seconds": elapsed, "checkpoint": str(checkpoint_dir)}
