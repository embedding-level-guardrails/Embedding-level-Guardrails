"""Encoder fine-tuning in one of three modes; only the loss differs.

| mode       | loss updating the encoder         |
|------------|-----------------------------------|
| ce         | CE(classify(h), y)                |
| supcon     | SupCon(project(h), y)             |
| ce_supcon  | CE + lambda * SupCon, one step    |

Every mode at a given seed gets the same training rows, the same batch
sequence (``BalancedBatchSampler``), the same step budget, optimizer and
schedule. The checkpoint is the epoch with the best mode-agnostic selection
score (quick linear probe on ``h``, ROC-AUC on source validation).
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cross_entropy
from transformers import get_linear_schedule_with_warmup

from e5_guardrails import augment, tracking
from e5_guardrails import splits as splits_mod
from e5_guardrails.config import MODES, paths
from e5_guardrails.losses import supcon_loss
from e5_guardrails.model import (
    USES_CE,
    USES_SUPCON,
    E5Encoder,
    GuardrailModel,
    load_tokenizer,
    save_checkpoint,
    tokenize,
)
from e5_guardrails.probe import quick_probe_auc
from e5_guardrails.runtime import autocast, resolve_amp_dtype, resolve_device, set_seed
from e5_guardrails.sampler import BalancedBatchSampler


def compute_losses(out: dict[str, torch.Tensor], labels: torch.Tensor, mode: str, loss_cfg: dict[str, Any]):
    """Total loss plus separately logged components."""
    logs: dict[str, float] = {}
    total = torch.zeros((), device=labels.device)
    if mode in USES_CE:
        ce = cross_entropy(out["logits"].float(), labels)
        logs["ce_loss"] = float(ce.detach())
        total = total + ce
    if mode in USES_SUPCON:
        sc, stats = supcon_loss(out["z"], labels, loss_cfg["temperature"])
        logs["supcon_loss"] = float(sc.detach())
        logs["anchors_without_positive"] = stats["anchors_without_positive"]
        weight = loss_cfg["supcon_weight"] if mode == "ce_supcon" else 1.0
        total = total + weight * sc
    logs["total_loss"] = float(total.detach())
    return total, logs


def training_set(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Training rows (+ train-only augmentation) for the configured variant; cached."""
    p = paths(cfg)
    spec = {"augmentation": cfg["augmentation"], "pairs": cfg["pairs"],
            "split_seed": cfg["preprocess"]["split_seed"]}
    out = p.training_set(hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12])
    if (out / "spec.json").exists() and json.loads((out / "spec.json").read_text()) == spec:
        return (pd.read_parquet(out / "rows.parquet"), pd.read_parquet(out / "pairs.parquet"),
                json.loads((out / "report.json").read_text()))
    train = splits_mod.load_split(p.splits, "train")
    templates, rejected = (None, {})
    if cfg["augmentation"]["wrapper_types"]:
        templates, rejected = augment.load_templates(cfg["augmentation"]["templates"], p.cache)
    rows, pairs, report = augment.build_training_set(train, cfg, templates)
    report["templates_usable"] = len(templates or [])
    report["templates_rejected"] = rejected
    out.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out / "rows.parquet", index=False)
    pairs.to_parquet(out / "pairs.parquet", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "spec.json").write_text(json.dumps(spec, indent=2))
    return rows, pairs, report


@torch.no_grad()
def _validation_losses(model, val, tokenizer, cfg, device, amp_dtype) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    n_batches = 0
    bs = cfg["training"]["batch_size"]
    for start in range(0, len(val), bs):
        chunk = val.iloc[start:start + bs]
        batch = tokenize(tokenizer, chunk["text"].tolist(), cfg["model"]["prefix"], cfg["model"]["max_length"]).to(device)
        labels = torch.tensor(chunk["label"].to_numpy(), device=device)
        with autocast(device, amp_dtype):
            out = model(batch["input_ids"], batch["attention_mask"])
        _, logs = compute_losses(out, labels, model.mode, cfg["loss"])
        for k, v in logs.items():
            sums[k] = sums.get(k, 0.0) + v
        n_batches += 1
    return {f"val_{k}": v / max(n_batches, 1) for k, v in sums.items()}


def train(cfg: dict[str, Any], mode: str, seed: int) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    name = f"e5-{mode}-seed={seed}-lr={cfg['training']['lr']:g}"
    with tracking.start_run(cfg, name, "train", {**cfg, "mode": mode, "seed": seed}) as run:
        return _train(cfg, mode, seed, run)


def _train(cfg: dict[str, Any], mode: str, seed: int, run: Any) -> dict[str, Any]:
    p = paths(cfg)
    run_dir = p.run(mode, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    tcfg = cfg["training"]
    device = resolve_device(tcfg["device"])
    amp_dtype = resolve_amp_dtype(tcfg["precision"], device)

    rows, pairs, aug_report = training_set(cfg)
    val = splits_mod.load_split(p.splits, "validation")
    split_seed = cfg["preprocess"]["split_seed"]
    sel_train = splits_mod.cap_rows(rows[rows["row_kind"] == "anchor"], cfg["selection"]["probe_train_samples"], split_seed)
    sel_val = splits_mod.cap_rows(val, cfg["selection"]["val_samples"], split_seed)

    tokenizer = load_tokenizer(cfg["model"]["name"], cfg["model"]["revision"])
    encoder = E5Encoder.from_pretrained(cfg["model"]["name"], cfg["model"]["revision"])
    model = GuardrailModel(encoder, mode, cfg["model"]["projection_dim"]).to(device)
    if tcfg["gradient_checkpointing"]:
        model.encoder.backbone.gradient_checkpointing_enable()
    head_params = [prm for name, prm in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": model.encoder.parameters(), "lr": tcfg["lr"]},
         {"params": head_params, "lr": tcfg["head_lr"]}],
        weight_decay=tcfg["weight_decay"],
    )
    total_steps = tcfg["epochs"] * tcfg["steps_per_epoch"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(tcfg["warmup_ratio"] * total_steps), total_steps)
    sampler = BalancedBatchSampler(
        rows["label"].to_numpy(), tcfg["batch_size"], tcfg["steps_per_epoch"], seed=seed,
        partners=augment.partners_from_pairs(pairs) if len(pairs) else None,
        pair_prob=cfg["pairs"]["pair_prob"],
    )
    texts, labels_np = rows["text"].tolist(), rows["label"].to_numpy()
    exposures = np.zeros(len(rows), dtype=np.int64)

    best = {"selection_roc_auc": -np.inf, "epoch": None}
    log_path = run_dir / "train_log.jsonl"
    log_path.unlink(missing_ok=True)
    global_step = 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        sums: dict[str, float] = {}
        started = time.time()
        for idx in sampler:
            exposures[idx] += 1
            batch = tokenize(tokenizer, [texts[i] for i in idx], cfg["model"]["prefix"], cfg["model"]["max_length"]).to(device)
            labels = torch.tensor(labels_np[idx], device=device)
            with autocast(device, amp_dtype):
                out = model(batch["input_ids"], batch["attention_mask"])
            loss, logs = compute_losses(out, labels, mode, cfg["loss"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v
            if run is not None:
                run.log({**{f"train/{k}": v for k, v in logs.items()},
                         "train/lr_encoder": scheduler.get_last_lr()[0]}, step=global_step)
        record = {"epoch": epoch, "global_step": global_step, "seconds": round(time.time() - started, 1),
                  "lr_encoder": scheduler.get_last_lr()[0],
                  **{f"train_{k}": v / tcfg["steps_per_epoch"] for k, v in sums.items()}}
        record.update(_validation_losses(model, sel_val, tokenizer, cfg, device, amp_dtype))
        record["selection_roc_auc"] = quick_probe_auc(model.encoder, tokenizer, cfg, sel_train, sel_val, device, amp_dtype)
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps({"mode": mode, "seed": seed, **record}), flush=True)
        if run is not None:
            run.log({f"epoch/{k}": v for k, v in record.items()}, step=global_step)
        if record["selection_roc_auc"] > best["selection_roc_auc"]:
            best = {"selection_roc_auc": record["selection_roc_auc"], "epoch": epoch}
            save_checkpoint(model, tokenizer, cfg, p.checkpoint(mode, seed),
                            {"seed": seed, "epoch": epoch, "global_step": global_step, **best})

    kinds = rows["row_kind"].to_numpy()
    summary = {
        "mode": mode, "seed": seed, "best": best,
        "contrastive_batch_size": tcfg["batch_size"], "gradient_accumulation": 1,
        "total_steps": total_steps, "presentations": int(exposures.sum()),
        "unique_rows_seen": int((exposures > 0).sum()), "rows": len(rows),
        "presentations_by_kind": {k: int(exposures[kinds == k].sum()) for k in np.unique(kinds)},
        "max_presentations_per_row": int(exposures.max()),
        "anchor_presentations_mean": float(exposures[kinds == "anchor"].mean()),
        "training_set": aug_report,
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if run is not None:
        run.summary.update({k: v for k, v in summary.items() if k != "training_set"})
        tracking.log_files(run, f"train-{cfg['protocol']}-{cfg['variant']}-{mode}-seed{seed}", "train-logs",
                           [log_path, run_dir / "train_summary.json"])
    return summary
