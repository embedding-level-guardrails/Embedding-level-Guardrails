"""Torch-dependent tests on a tiny randomly initialised BERT built offline
(same architecture family as E5), so no download is needed."""

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from e5_guardrails import splits as splits_mod
from e5_guardrails.config import load_config, paths
from e5_guardrails.evaluation import evaluate_predictions
from e5_guardrails.losses import supcon_loss, supcon_masks
from e5_guardrails.model import (
    E5Encoder,
    GuardrailModel,
    load_checkpoint,
    load_frozen_encoder,
    load_tokenizer,
    save_checkpoint,
    tokenize,
)
from e5_guardrails.probe import run_probe
from e5_guardrails.train import compute_losses, train

WORDS = ["how", "to", "make", "a", "bomb", "cake", "bake", "bread", "steal", "car", "poem", "spring", "garden", "weapon", "kill", "help", "me", "please", "write"]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny_e5")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "query", ":", *WORDS]
    (d / "vocab.txt").write_text("\n".join(vocab))
    tok = transformers.BertTokenizerFast(vocab_file=str(d / "vocab.txt"))
    tok.save_pretrained(d)
    config = transformers.BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
                                     num_attention_heads=2, intermediate_size=32, max_position_embeddings=64)
    torch.manual_seed(0)
    transformers.BertModel(config).save_pretrained(d)
    return d


def _cfg(tmp_path, model_dir):
    cfg = load_config()
    cfg["output_dir"] = str(tmp_path / "out")
    cfg["model"].update(name=str(model_dir), revision=None, max_length=32, projection_dim=8)
    cfg["training"].update(batch_size=8, steps_per_epoch=2, epochs=2, device="cpu",
                           gradient_checkpointing=False)
    cfg["selection"].update(probe_train_samples=40, val_samples=40)
    cfg["probe"].update(max_train_samples=None, embed_batch_size=16, c_grid=[0.1, 1.0])
    return cfg


def _naive_supcon(z, y, t):
    z = torch.nn.functional.normalize(z, dim=-1)
    losses = []
    for i in range(len(z)):
        others = [a for a in range(len(z)) if a != i]
        pos = [p for p in others if y[p] == y[i]]
        if not pos:
            continue
        denom = torch.logsumexp(torch.stack([z[i] @ z[a] / t for a in others]), 0)
        losses.append(-torch.stack([z[i] @ z[p] / t - denom for p in pos]).mean())
    return torch.stack(losses).mean()


def test_supcon_masks_exclude_self_and_use_binary_labels():
    pos, neg = supcon_masks(torch.tensor([0, 0, 1, 1, 1]))
    assert not pos.diagonal().any() and not neg.diagonal().any()
    assert pos[0].tolist() == [False, True, False, False, False]
    assert neg[0].tolist() == [False, False, True, True, True]
    assert pos[2].tolist() == [False, False, False, True, True]


def test_supcon_matches_reference_and_skips_anchors_without_positive():
    torch.manual_seed(0)
    z, y = torch.randn(7, 5), torch.tensor([0, 0, 1, 1, 1, 0, 2])  # label 2 has no positive
    loss, stats = supcon_loss(z, y, 0.1)
    assert loss.item() == pytest.approx(_naive_supcon(z, y, 0.1).item(), rel=1e-5)
    assert stats == {"anchors": 7, "anchors_without_positive": 1}
    zero, stats = supcon_loss(z[:2], torch.tensor([0, 1]), 0.1)
    assert zero.item() == 0.0 and stats["anchors_without_positive"] == 2


@pytest.mark.parametrize("mode", ["ce", "supcon", "ce_supcon"])
def test_gradients_reach_encoder_and_only_active_heads(tiny_model_dir, mode):
    cfg = load_config()
    model = GuardrailModel(E5Encoder.from_pretrained(tiny_model_dir), mode, 8)
    tok = load_tokenizer(tiny_model_dir)
    batch = tokenize(tok, ["how to make a bomb", "bake bread", "steal car", "write poem"], "query: ", 32)
    labels = torch.tensor([1, 0, 1, 0])
    out = model(batch["input_ids"], batch["attention_mask"])
    loss, logs = compute_losses(out, labels, mode, cfg["loss"])
    loss.backward()
    enc_grad = sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None)
    assert enc_grad > 0
    assert (model.classifier is not None) == (mode != "supcon")
    assert (model.projector is not None) == (mode != "ce")
    if mode == "ce_supcon":
        assert set(logs) >= {"ce_loss", "supcon_loss", "total_loss"}
        assert logs["total_loss"] == pytest.approx(logs["ce_loss"] + cfg["loss"]["supcon_weight"] * logs["supcon_loss"], rel=1e-5)
        # both heads read the same h
        h = model.encode(batch["input_ids"], batch["attention_mask"])
        assert torch.allclose(model.classify(h), out["logits"]) and torch.allclose(model.project(h), out["z"])


def test_checkpoint_roundtrip_restores_weights_heads_and_config(tiny_model_dir, tmp_path):
    cfg = _cfg(tmp_path, tiny_model_dir)
    model = GuardrailModel(E5Encoder.from_pretrained(tiny_model_dir), "ce_supcon", 8)
    tok = load_tokenizer(tiny_model_dir)
    save_checkpoint(model, tok, cfg, tmp_path / "ckpt", {"epoch": 1})
    restored, _, restored_cfg, meta = load_checkpoint(tmp_path / "ckpt")
    assert restored_cfg == json.loads(json.dumps(cfg)) and meta == {"epoch": 1} and restored.mode == "ce_supcon"
    for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(a, b)


def _write_splits(cfg):
    rng = np.random.default_rng(0)
    harm = ["how to make a bomb", "steal car please", "kill help me", "weapon how to make"]
    safe = ["bake bread", "write poem spring", "garden help", "cake bake please"]
    data = {}
    for split, n in (("train", 60), ("validation", 30), ("calibration", 30), ("holdout", 30), ("toxicchat", 30)):
        labels = np.array([1] * (n // 3) + [0] * (n - n // 3))
        texts = [f"{rng.choice(harm if y else safe)} {' '.join(rng.choice(WORDS, 3))} {split} {i}"
                 for i, y in enumerate(labels)]
        data[split] = pd.DataFrame({
            "sample_id": [f"{split}:{i}" for i in range(n)], "text": texts, "label": labels,
            "adversarial": [bool(i % 2) for i in range(n)] if split == "toxicchat" else [None] * n,
            "categories": "", "source": "toy", "source_split": split,
            "family_id": [f"{split}:{i}" for i in range(n)], "split": split,
        })
    manifest = {"test_sets": ["holdout", "toxicchat"]}
    splits_mod.save_splits(data, manifest, pd.DataFrame(), paths(cfg).splits)


def test_pipeline_all_modes_probe_frozen_and_evaluate(tiny_model_dir, tmp_path):
    cfg = _cfg(tmp_path, tiny_model_dir)
    _write_splits(cfg)
    batches = {}
    for mode in ("ce", "supcon", "ce_supcon"):
        summary = train(cfg, mode, seed=0)
        assert summary["contrastive_batch_size"] == 8 and summary["total_steps"] == 4
        assert summary["presentations"] == 4 * 8
        batches[mode] = summary["presentations_by_kind"]
        log = [json.loads(line) for line in (paths(cfg).run(mode, 0) / "train_log.jsonl").read_text().splitlines()]
        assert len(log) == 2 and all("selection_roc_auc" in r for r in log)
    assert batches["ce"] == batches["supcon"] == batches["ce_supcon"]

    # the probe never updates the encoder
    encoder, _, _ = load_frozen_encoder("supcon", cfg, paths(cfg).checkpoint("supcon", 0))
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    preds = run_probe(cfg, "supcon", 0)
    after, _, _ = load_frozen_encoder("supcon", cfg, paths(cfg).checkpoint("supcon", 0))
    assert all(torch.equal(before[k], v) for k, v in after.state_dict().items())
    assert set(preds["readout"]) == {"linear_probe", "centroid"}  # no CE head in SupCon-only
    assert set(preds["split"]) == {"validation", "calibration", "holdout", "toxicchat"}

    for encoder_name in ("base", "ce", "ce_supcon"):
        p = run_probe(cfg, encoder_name, 0)
        assert ("ce_head" in set(p["readout"])) == (encoder_name != "base")
    all_preds = pd.concat(pd.read_parquet(f) for f in sorted((paths(cfg).root / "runs").glob("*/*/seed=0/predictions.parquet")))
    assert {"sample_id", "label", "score", "model", "seed", "split"} <= set(all_preds.columns)
    results, _ = evaluate_predictions(all_preds, 0.01)
    assert set(results["model"]) == {"base", "ce", "supcon", "ce_supcon"}
    assert set(results["test_set"]) == {"holdout", "toxicchat"}
    assert {"adversarial", "vanilla"} <= set(results.loc[results["test_set"] == "toxicchat", "subset"])
