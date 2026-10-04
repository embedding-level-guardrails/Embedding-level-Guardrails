"""Shared post-training classification on a frozen encoder.

Main readout (``linear_probe``): a new StandardScaler + logistic regression on
``h`` (before any projection head) fit on source train, with C chosen by
ROC-AUC on source validation (ties -> smaller C). Identical for every encoder,
including the untouched E5 baseline. Additional readouts: ``ce_head`` (the CE
head trained with the encoder, CE modes only) and ``centroid`` (cosine to the
two class means of train ``h``).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from e5_guardrails import splits as splits_mod
from e5_guardrails.config import paths
from e5_guardrails.model import E5Encoder, load_frozen_encoder
from e5_guardrails.runtime import embed, resolve_amp_dtype, resolve_device


def fit_probe(
    x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
    c_grid: list[float], max_iter: int, seed: int,
) -> tuple[Pipeline, dict[str, Any]]:
    best, best_auc, scores = None, -np.inf, {}
    for c in sorted(c_grid):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=max_iter, random_state=seed))
        pipe.fit(x_train, y_train)  # scaler statistics come from train only
        auc = float(roc_auc_score(y_val, pipe.predict_proba(x_val)[:, 1]))
        scores[str(c)] = auc
        if auc > best_auc:
            best, best_auc, best_c = pipe, auc, c
    return best, {"C": best_c, "validation_roc_auc": best_auc, "validation_roc_auc_by_C": scores}


def centroid_scores(h_train: np.ndarray, y_train: np.ndarray, h: np.ndarray) -> np.ndarray:
    mu = [h_train[y_train == c].mean(axis=0) for c in (0, 1)]
    mu = [m / np.linalg.norm(m) for m in mu]
    return h @ mu[1] - h @ mu[0]


def quick_probe_auc(
    encoder: E5Encoder, tokenizer, cfg: dict[str, Any], train: pd.DataFrame, val: pd.DataFrame,
    device: torch.device, amp_dtype: torch.dtype,
) -> float:
    """Checkpoint-selection score: same probe family, fixed small subsets."""
    bs = cfg["probe"]["embed_batch_size"]
    x_tr = embed(encoder, tokenizer, train["text"].tolist(), cfg["model"], bs, device, amp_dtype)
    x_val = embed(encoder, tokenizer, val["text"].tolist(), cfg["model"], bs, device, amp_dtype)
    _, info = fit_probe(x_tr, train["label"].to_numpy(), x_val, val["label"].to_numpy(),
                        cfg["probe"]["c_grid"], cfg["probe"]["max_iter"], seed=0)
    return info["validation_roc_auc"]


def run_probe(cfg: dict[str, Any], encoder_name: str, seed: int) -> pd.DataFrame:
    p = paths(cfg)
    manifest = splits_mod.load_manifest(p.splits)
    device = resolve_device(cfg["training"]["device"])
    amp_dtype = resolve_amp_dtype(cfg["training"]["precision"], device)
    encoder, model, tokenizer = load_frozen_encoder(encoder_name, cfg, p.checkpoint(encoder_name, seed))
    encoder.to(device)

    names = ["train", "validation", "calibration", *manifest["test_sets"]]
    data = {n: splits_mod.load_split(p.splits, n) for n in names}
    data["train"] = splits_mod.cap_rows(data["train"], cfg["probe"]["max_train_samples"], seed)
    bs = cfg["probe"]["embed_batch_size"]
    h = {n: embed(encoder, tokenizer, df["text"].tolist(), cfg["model"], bs, device, amp_dtype) for n, df in data.items()}
    y = {n: df["label"].to_numpy() for n, df in data.items()}

    probe, info = fit_probe(h["train"], y["train"], h["validation"], y["validation"],
                            cfg["probe"]["c_grid"], cfg["probe"]["max_iter"], seed)
    readouts = {
        "linear_probe": lambda x: probe.predict_proba(x)[:, 1],
        "centroid": lambda x: centroid_scores(h["train"], y["train"], x),
    }
    if model is not None and model.classifier is not None:
        head = model.classifier.cpu()
        readouts["ce_head"] = lambda x: torch.softmax(head(torch.from_numpy(x)), dim=-1)[:, 1].numpy()

    frames = []
    for split in names[1:]:  # source train itself is not scored
        df = data[split]
        for readout, score in readouts.items():
            frames.append(pd.DataFrame({
                "protocol": cfg["protocol"], "variant": cfg["variant"], "model": encoder_name,
                "readout": readout, "seed": seed, "split": split, "sample_id": df["sample_id"],
                "label": df["label"], "score": score(h[split]).astype(float),
                "adversarial": df["adversarial"],
            }))
    preds = pd.concat(frames, ignore_index=True)
    out = p.predictions(encoder_name, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)
    (out.parent / "probe.json").write_text(json.dumps(
        {**info, "n_train": len(data["train"]), "readouts": sorted(readouts)}, indent=2))
    return preds
