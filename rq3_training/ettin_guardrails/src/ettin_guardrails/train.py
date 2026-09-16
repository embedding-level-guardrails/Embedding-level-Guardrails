import logging
from pathlib import Path

import hydra
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, set_seed

from ettin_guardrails.checkpoint import save_checkpoint
from ettin_guardrails.data import TokenizedDataset, StratifiedBatchSampler, load_data, split_validation_data
from ettin_guardrails.loss import info_nce_loss
from ettin_guardrails.model import Embedder

logger = logging.getLogger(__name__)


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
        tokenizer: AutoTokenizer,
        cfg: DictConfig,
        device: torch.device,
        is_val: bool = False
) -> DataLoader:
    dataset = TokenizedDataset(data, tokenizer=tokenizer, max_length=cfg.tokenizer.max_length)
    sampler = None
    if not is_val:
        sampler = StratifiedBatchSampler(dataset.category_ids, seed=cfg.seed, **cfg.sampler)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=cfg.tokenizer.pad_to_multiple_of,
        ),
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory and (device.type == "cuda"),
        persistent_workers=cfg.loader.persistent_workers and (cfg.loader.num_workers > 0),
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    return dataloader


def train(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = _configure_device(cfg)
    amp_dtype = _configure_precision(cfg, device)
    amp_enabled = amp_dtype != torch.float32
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    logger.info("Loading tokenizer %s", cfg.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)

    logger.info("Loading data from %s", cfg.data.path)
    data = load_data(cfg)

    train_data, val_data = split_validation_data(data, cfg.data_split.val_samples_per_cat, cfg.seed)
    logger.info(
        "Held out %d val examples across %d categories",
        len(val_data), val_data["category"].n_unique()
    )
    train_loader = _build_data_loader(train_data, tokenizer, cfg, device)
    val_loader = _build_data_loader(val_data, tokenizer, cfg, device, is_val=True)

    logger.info("Loading model %s on %s", cfg.model.model_name, device)
    model = Embedder(**cfg.model).to(device)
    if cfg.training.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
        fused=(device.type == "cuda"),
    )

    global_step = 0
    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = torch.zeros((), device=device)
        anchor_count = 0
        skipped_batches = 0

        for batch in train_loader:
            categories = batch.pop("category_id")

            # skip the batch when it doesn't contain positive pairs
            _, counts = categories.unique(return_counts=True)
            valid_anchors = int(counts[counts > 1].sum())
            if counts.numel() < 2 or valid_anchors == 0:
                skipped_batches += 1
                continue
            categories = categories.to(device, non_blocking=True)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                embeddings = model(**batch)
            loss = info_nce_loss(
                embeddings.float(), categories, cfg.loss.temperature, cfg.loss.get("eps", 1e-6),
            )
            scaler.scale(loss).backward()

            if cfg.training.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            train_loss += loss.detach() * valid_anchors
            anchor_count += valid_anchors

        model.eval()
        with torch.inference_mode():
            val_embeddings, val_categories = [], []
            for batch in val_loader:
                val_categories.append(batch["category_id"].to(device))
                inputs = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items() if key != "category_id"
                }
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    val_embeddings.append(model(**inputs))
            val_loss = info_nce_loss(
                torch.cat(val_embeddings).float(),
                torch.cat(val_categories),
                cfg.loss.temperature,
                cfg.loss.get("eps", 1e-6),
            )

        logger.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f",
            epoch + 1,
            (train_loss / anchor_count).item() if anchor_count else float("nan"),
            val_loss.item()
        )

        if (epoch + 1) % cfg.training.save_every_epochs == 0 or (epoch + 1) == cfg.training.epochs:
            training_checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "validation_loss": val_loss.item(),
                "config": OmegaConf.to_container(cfg, resolve=True),
            }
            save_checkpoint(training_checkpoint, cfg.training.output_dir)


@hydra.main(
    version_base="1.3",
    config_path="../../conf",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
