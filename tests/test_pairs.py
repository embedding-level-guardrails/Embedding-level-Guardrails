"""Тесты сборки пар (RQ2): проверяют логику без сети и без torch.

    pytest -q tests/test_pairs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.pairs.builders import (  # noqa: E402
    PairContext,
    build_benign_twin,
    build_jailbreak_variant,
    build_paraphrase,
    build_safe_harm_contrast,
    deduplicate,
    flatten_texts,
)
from eguard.pairs.transforms import JAILBREAKS, PARAPHRASES, apply_named  # noqa: E402


def _records(n=40):
    out = []
    for i in range(n):
        label = i % 2
        out.append({
            "id": f"r{i}",
            "text": f"{'harmful' if label else 'benign'} request number {i} about topic {i % 5}",
            "label": label,
            "label_name": "harm" if label else "safe",
            "category": f"cat{i % 3}" if label else "safe",
            "source": "aegis",
            "n_chars": 40,
        })
    return out


def _ctx(**kw):
    defaults = dict(split="train", aegis=_records(), harmbench=[],
                    rng=np.random.default_rng(0), cfg={})
    defaults.update(kw)
    return PairContext(**defaults)


def test_safe_harm_contrast_labels_are_consistent():
    pairs = build_safe_harm_contrast(_ctx(cfg={"safe_anchor_fraction": 0.5}), 50)
    assert pairs
    for p in pairs:
        # негатив всегда противоположного класса относительно якоря
        assert p["negative_label"] == 1 - p["anchor_label"]
        assert p["anchor"] != p["positive"]
        assert p["pair_type"] == "safe_harm_contrast"
    # обе стороны представлены, а не только harm-якоря
    assert {p["anchor_label"] for p in pairs} == {0, 1}


def test_paraphrase_positive_differs_from_anchor():
    pairs = build_paraphrase(_ctx(), 40)
    assert pairs
    for p in pairs:
        assert p["positive"].strip() != p["anchor"].strip()
        assert p["variant"] in PARAPHRASES
        assert p["positive_source"] == "transform"


def test_paraphrase_uses_external_variants_when_given():
    ctx = _ctx(external_variants={"r1": ["внешний парафраз"]})
    pairs = build_paraphrase(ctx, 60)
    external = [p for p in pairs if p["anchor_id"] == "r1"]
    assert external, "якорь r1 должен был попасться хотя бы раз"
    assert all(p["variant"] == "external" for p in external)


def test_jailbreak_wraps_harm_and_sometimes_wraps_safe_negative():
    ctx = _ctx(cfg={"jailbreak_negative": "mix"})
    pairs = build_jailbreak_variant(ctx, 60)
    assert pairs
    assert all(p["anchor_label"] == 1 for p in pairs)
    assert all(p["negative_label"] == 0 for p in pairs)
    assert all(p["variant"] in JAILBREAKS for p in pairs)
    # ключевой инвариант: часть негативов — безопасный текст в ТОЙ ЖЕ обёртке,
    # иначе модель выучит обёртку как признак вреда
    wrapped = [p for p in pairs if "::" in str(p["negative_id"])]
    assert wrapped, "в режиме mix должны быть обёрнутые safe-негативы"
    for p in wrapped:
        assert p["negative_id"].endswith(p["variant"])


def test_jailbreak_negative_mode_safe_never_wraps():
    pairs = build_jailbreak_variant(_ctx(cfg={"jailbreak_negative": "safe"}), 40)
    assert pairs and not any("::" in str(p["negative_id"]) for p in pairs)


def test_benign_twin_picks_nearest_safe():
    """Негатив должен быть ближайшим safe по косинусу, а не случайным."""
    records = [
        {"id": "h0", "text": "harm anchor", "label": 1, "category": "c"},
        {"id": "h1", "text": "harm other", "label": 1, "category": "c"},
        {"id": "s_near", "text": "safe near", "label": 0, "category": "safe"},
        {"id": "s_far", "text": "safe far", "label": 0, "category": "safe"},
    ]
    x = np.array([[1.0, 0.0], [0.9, 0.1], [0.99, 0.01], [-1.0, 0.0]])
    pairs = build_benign_twin(_ctx(mining=(x, records)), n=1)
    assert len(pairs) == 1
    assert pairs[0]["negative_id"] == "s_near"
    assert pairs[0]["anchor_label"] == 1


def test_benign_twin_without_embeddings_is_skipped():
    assert build_benign_twin(_ctx(mining=None), n=5) == []


def test_deduplicate_drops_repeats_and_degenerate_pairs():
    ctx = _ctx()
    pairs = build_safe_harm_contrast(ctx, 200)
    doubled = deduplicate(pairs + pairs)
    assert len(doubled) == len(deduplicate(pairs))
    same = dict(pairs[0], anchor="x", positive="x")
    assert deduplicate([same]) == []


def test_flatten_texts_is_deduped_and_keeps_labels():
    pairs = build_safe_harm_contrast(_ctx(), 30) + build_jailbreak_variant(_ctx(), 30)
    texts = flatten_texts(pairs)
    assert len({t["text"] for t in texts}) == len(texts)
    assert {t["label"] for t in texts} == {0, 1}
    for t in texts:
        assert t["label_name"] == ("harm" if t["label"] == 1 else "safe")


def test_apply_named_rejects_unknown_family():
    with pytest.raises(ValueError):
        apply_named(JAILBREAKS, ["no_such_family"], "text", np.random.default_rng(0))
