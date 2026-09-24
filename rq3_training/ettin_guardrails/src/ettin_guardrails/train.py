import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import hydra
import polars as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_metric_learning.losses import SupConLoss
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase, set_seed
from transformers.utils import logging as transformers_logging

from ettin_guardrails.checkpoint import save_checkpoint
from ettin_guardrails.data import TokenizedDataset, load_data, split_validation_data
from ettin_guardrails.model import Classifier, Embedder

logger = logging.getLogger(__name__)


@dataclass
class TrainingContext:
    cfg: DictConfig
    device: torch.device
    amp_dtype: torch.dtype
    amp_enabled: bool
    train_loader: DataLoader
    val_loader: DataLoader
    supcon_fn: SupConLoss


@dataclass
class ModelTrainingState:
    full_model_name: str
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    global_step: int = 0
    scheduler: torch.optim.lr_scheduler.LinearLR | None = None


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float


def _configure_device(cfg: DictConfig) -> torch.device:
    device_name = cfg.training.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(device_name)


def _configure_precision(cfg: DictConfig, device: torch.device) -> torch.dtype:
    precision = str(cfg.training.get("precision", "auto")).lower()
    if precision == "auto":
        if device.type == "cuda":
            with torch.cuda.device(device):
                precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            precision = "fp16" if device.type == "mps" else "fp32"
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision not in dtypes:
        raise ValueError("training.precision must be auto, fp32, fp16, or bf16")
    logger.info("Training precision: %s on %s", precision, device)
    return dtypes[precision]


def _build_data_loader(
        data: pl.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        cfg: DictConfig,
        device: torch.device,
        is_val: bool = False,
) -> DataLoader:
    dataset = TokenizedDataset(data, tokenizer=tokenizer, **cfg.get("tokenizer", {}))
    sampler = None
    if not is_val:
        sampler = MPerClassSampler(
            dataset.strata_labels,
            m=cfg.training.sampler.number_of_samples_per_category,
            batch_size=cfg.training.sampler.batch_size,
            length_before_new_iter=cfg.training.sampler.number_of_samples_per_epoch,
        )
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=cfg.training.sampler.batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer),
        num_workers=cfg.training.loader.num_workers,
        pin_memory=cfg.training.loader.pin_memory and (device.type == "cuda"),
        persistent_workers=cfg.training.loader.persistent_workers and (cfg.training.loader.num_workers > 0),
        generator=torch.Generator().manual_seed(cfg.seed),
    )


def _prepare_training(cfg: DictConfig) -> TrainingContext:
    set_seed(cfg.seed)
    device = _configure_device(cfg)
    amp_dtype = _configure_precision(cfg, device)
    data = load_data(**cfg.data)
    train_data, val_data = split_validation_data(data, seed=cfg.seed, **cfg.training.validation)
    tokenizer = AutoTokenizer.from_pretrained(cfg.models.backbone)
    return TrainingContext(
        cfg=cfg,
        device=device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_dtype != torch.float32,
        train_loader=_build_data_loader(train_data, tokenizer, cfg, device),
        val_loader=_build_data_loader(val_data, tokenizer, cfg, device, is_val=True),
        supcon_fn=SupConLoss(cfg.supcon_loss.temperature)
    )


def _initialize_model(model_name: str, context: TrainingContext) -> ModelTrainingState:
    cfg = context.cfg
    classifier_cfg = cfg.models.get("classifier", {})
    embedder_cfg = cfg.models.get("embedder", {})
    match model_name:
        case "ettin-ce":
            model = Classifier(**classifier_cfg, model_name=cfg.models.backbone)
            encoder = model.embedder.encoder
        case "ettin-cl":
            model = Embedder(**embedder_cfg, model_name=cfg.models.backbone)
            encoder = model.encoder
        case _:
            model = Classifier(**classifier_cfg, **embedder_cfg, model_name=cfg.models.backbone)
            encoder = model.embedder.encoder
    model = model.to(context.device)
    if cfg.training.gradient_checkpointing:
        encoder.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
        fused=(context.device.type == "cuda"),
    )

    parts = [model_name]
    hypers = {**classifier_cfg, **embedder_cfg}
    parts.extend(f"{key}={value}" for key, value in hypers.items())
    return ModelTrainingState(
        full_model_name="-".join(parts),
        model=model,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler(context.device.type, enabled=context.amp_dtype == torch.float16),
        scheduler=(
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=cfg.scheduler.start_factor,
                end_factor=cfg.scheduler.end_factor,
                total_iters=cfg.scheduler.total_iters,
            )
            if cfg.get("scheduler", {}).get("enabled", False)
            else None
        ),
    )


def _backward_step(loss: torch.Tensor, context: TrainingContext, state: ModelTrainingState) -> None:
    state.scaler.scale(loss).backward()
    if context.cfg.training.max_grad_norm is not None:
        state.scaler.unscale_(state.optimizer)
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), context.cfg.training.max_grad_norm)
    state.scaler.step(state.optimizer)
    state.scaler.update()


def train_classifier_epoch(context: TrainingContext, state: ModelTrainingState) -> EpochMetrics:
    train_loss = 0.0
    train_samples = 0
    val_loss = 0.0
    val_samples = 0
    state.model.train()
    for batch in tqdm(context.train_loader, desc="Training classifier", unit="batch", leave=False):
        input_ids = batch["input_ids"].to(context.device)
        attn_mask = batch["attention_mask"].to(context.device)
        labels = batch["labels"].to(context.device)
        with torch.autocast(context.device.type, dtype=context.amp_dtype, enabled=context.amp_enabled):
            logits, _ = state.model(input_ids, attn_mask)
        loss = cross_entropy(logits.float(), labels.long())
        train_loss += loss.cpu().item() * input_ids.shape[0]
        train_samples += input_ids.shape[0]
        _backward_step(loss, context, state)
        state.global_step += 1
        state.optimizer.zero_grad()

    state.model.eval()
    with torch.inference_mode():
        for batch in tqdm(context.val_loader, desc="Validating classifier", unit="batch", leave=False):
            input_ids = batch["input_ids"].to(context.device)
            attn_mask = batch["attention_mask"].to(context.device)
            labels = batch["labels"].to(context.device)
            with torch.autocast(context.device.type, dtype=context.amp_dtype, enabled=context.amp_enabled):
                logits, _ = state.model(input_ids, attn_mask)
            loss = cross_entropy(logits.float(), labels.long())
            val_loss += loss.cpu().item() * input_ids.shape[0]
            val_samples += input_ids.shape[0]


    return EpochMetrics(train_loss / train_samples, val_loss / val_samples)


def train_embedder_epoch(context: TrainingContext, state: ModelTrainingState) -> EpochMetrics:
    train_loss = 0.0
    train_samples = 0
    val_loss = 0.0
    val_samples = 0

    state.model.train()
    for batch in tqdm(context.train_loader, desc="Training embedder", unit="batch", leave=False):
        input_ids = batch["input_ids"].to(context.device)
        attn_mask = batch["attention_mask"].to(context.device)
        cat_ids = batch["category_ids"].to(context.device)
        with torch.autocast(context.device.type, dtype=context.amp_dtype, enabled=context.amp_enabled):
            embeddings = state.model(input_ids, attn_mask)
        loss = context.supcon_fn(embeddings.float(), cat_ids)
        train_loss += loss.cpu().item() * input_ids.shape[0]
        train_samples += input_ids.shape[0]
        state.global_step += 1
        _backward_step(loss, context, state)
        state.optimizer.zero_grad()

    state.model.eval()
    with torch.inference_mode():
        for batch in tqdm(context.val_loader, desc="Validating embedder", unit="batch", leave=False):
            state.optimizer.zero_grad()
            input_ids = batch["input_ids"].to(context.device)
            attn_mask = batch["attention_mask"].to(context.device)
            cat_ids = batch["category_ids"].to(context.device)
            with torch.autocast(context.device.type, dtype=context.amp_dtype, enabled=context.amp_enabled):
                embeddings = state.model(input_ids, attn_mask)
            loss = context.supcon_fn(embeddings.float(), cat_ids)
            val_loss += loss.cpu().item() * input_ids.shape[0]
            val_samples += input_ids.shape[0]

    return EpochMetrics(train_loss / train_samples, val_loss / val_samples)


def train_joint_epoch(context: TrainingContext, state: ModelTrainingState) -> EpochMetrics:
    train_loss = 0.0
    train_samples = 0
    val_loss = 0.0
    val_samples = 0

    state.model.train()
    for batch in tqdm(context.train_loader, desc="Training joint model", unit="batch", leave=False):
        input_ids = batch["input_ids"].to(context.device)
        attn_mask = batch["attention_mask"].to(context.device)
        cat_ids = batch["category_ids"].to(context.device)
        labels = batch["labels"].to(context.device)
        with torch.autocast(context.device.type, dtype=context.amp_dtype, enabled=context.amp_enabled):
            logits, embeddings = state.model(input_ids, attn_mask)
        loss = context.supcon_fn(embeddings.float(), cat_ids) + cross_entropy(logits.float(), labels.long())
        train_loss += loss.cpu().item() * input_ids.shape[0]
        train_samples += input_ids.shape[0]
        state.global_step += 1
        _backward_step(loss, context, state)
        state.optimizer.zero_grad()

    state.model.eval()
    with torch.inference_mode():
        for batch in tqdm(context.val_loader, desc="Validating joint model", unit="batch", leave=False):
            state.optimizer.zero_grad()
            input_ids = batch["input_ids"].to(context.device)
            attn_mask = batch["attention_mask"].to(context.device)
            cat_ids = batch["category_ids"].to(context.device)
            labels = batch["labels"].to(context.device)
            with torch.autocast(context.device.type, dtype=context.amp_dtype, enabled=context.amp_enabled):
                logits, embeddings = state.model(input_ids, attn_mask)
            loss = context.supcon_fn(embeddings.float(), cat_ids) + cross_entropy(logits.float(), labels.long())
            val_loss += loss.cpu().item() * input_ids.shape[0]
            val_samples += input_ids.shape[0]

    return EpochMetrics(train_loss / train_samples, val_loss / val_samples)


def _wandb_run_name(full_model_name: str, cfg: DictConfig) -> str:
    return full_model_name + f"-lr={cfg.optimizer.lr:g}"


def _wandb_config(model_name: str, cfg: DictConfig) -> dict:
    config = OmegaConf.to_container(
        OmegaConf.masked_copy(cfg, ["seed", "models", "optimizer", "scheduler", "supcon_loss", "tokenizer", "training"]),
        resolve=True,
    )
    config["model_name"] = model_name
    return config


def _train_model(
        model_name: str,
        context: TrainingContext,
        epoch_loop: Callable[[TrainingContext, ModelTrainingState], EpochMetrics],
) -> None:
    cfg = context.cfg
    state = _initialize_model(model_name, context)
    output_dir = Path(cfg.training.output_dir) / model_name
    with wandb.init(
        **cfg.get("wandb", {}),
        name=_wandb_run_name(state.full_model_name, cfg),
        config=_wandb_config(state.full_model_name, cfg),
        reinit="create_new",
    ) as run:
        for epoch in tqdm(range(cfg.training.epochs), desc=model_name, unit="epoch"):
            metrics = epoch_loop(context, state)
            run.log({
                "epoch": epoch + 1,
                "global_step": state.global_step,
                "train/loss": metrics.train_loss,
                "train/lr": state.optimizer.param_groups[0]["lr"],
                "val/loss": metrics.val_loss,
            }, step=state.global_step, commit=True)
            if state.scheduler is not None:
                state.scheduler.step()
            if (epoch + 1) % cfg.training.save_every_epochs == 0 or (epoch + 1) == cfg.training.epochs:
                save_checkpoint(
                    {
                        "model": state.model.state_dict(),
                        "optimizer": state.optimizer.state_dict(),
                        "scaler": state.scaler.state_dict(),
                        "scheduler": state.scheduler.state_dict() if state.scheduler is not None else None,
                        "model_name": model_name,
                        "epoch": epoch + 1,
                        "global_step": state.global_step,
                        "validation_loss": metrics.val_loss,
                        "config": OmegaConf.to_container(cfg, resolve=True),
                    },
                    output_dir / f"epoch_{epoch + 1}",
                )


def train(cfg: DictConfig) -> None:
    epoch_loops = {
        "ettin-ce": train_classifier_epoch,
        "ettin-cl": train_embedder_epoch,
        "ettin-joint": train_joint_epoch,
    }
    selection = cfg.training.models
    if selection == "all":
        selected_models = tuple(epoch_loops)
    elif selection in epoch_loops:
        selected_models = (selection,)
    else:
        raise ValueError(f"Invalid training model: {selection}")

    context = _prepare_training(cfg)
    for model_name in selected_models:
        _train_model(model_name, context, epoch_loops[model_name])


@hydra.main(
    version_base="1.3",
    config_path="../../conf",
    config_name="default.yaml",
)
def main(cfg: DictConfig) -> None:
    transformers_logging.set_verbosity_error()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    train(cfg)


if __name__ == "__main__":
    main()
