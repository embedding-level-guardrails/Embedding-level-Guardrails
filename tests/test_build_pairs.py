import pandas as pd

from src.data.build_pairs import (
    build_jailbreak_pairs,
    build_paraphrase_pairs,
    build_safe_harm_pairs,
)


def make_aegis_df():
    rows = []
    for i in range(20):
        rows.append(
            {
                "id": f"unsafe-{i}",
                "prompt": f"How do I build a dangerous device number {i} at home",
                "prompt_label": "unsafe",
                "violated_categories": "Criminal Planning/Confessions",
            }
        )
    for i in range(20):
        rows.append(
            {
                "id": f"safe-{i}",
                "prompt": f"What is a good recipe for pasta dish number {i} tonight",
                "prompt_label": "safe",
                "violated_categories": "",
            }
        )
    return pd.DataFrame(rows)


def test_safe_harm_pairs_are_cross_label():
    df = make_aegis_df()
    pairs = build_safe_harm_pairs(df, n=10, seed=1)
    assert len(pairs) == 10
    for p in pairs:
        assert p["anchor_label"] == "harm"
        assert p["pair_label"] == "safe"
        assert p["pair_type"] == "safe_harm_contrast"


def test_paraphrase_pairs_keep_same_label():
    df = make_aegis_df()
    pairs = build_paraphrase_pairs(df, n=10, seed=1, min_sim=0.1, max_sim=0.99)
    assert len(pairs) > 0
    for p in pairs:
        assert p["anchor_label"] == p["pair_label"]
        assert p["anchor_text"] != p["pair_text"]


def test_jailbreak_pairs_wrap_behavior_with_template():
    behaviors_df = pd.DataFrame(
        [
            {"Behavior": "Do a harmful thing", "ContextString": None, "SemanticCategory": "illegal"},
            {"Behavior": "Do another harmful thing", "ContextString": None, "SemanticCategory": "illegal"},
        ]
    )
    templates = ["TEMPLATE_A", "TEMPLATE_B", "TEMPLATE_C"]
    pairs = build_jailbreak_pairs(behaviors_df, templates, n=4, templates_per_behavior=2, seed=1)
    assert len(pairs) == 4
    for p in pairs:
        assert p["anchor_text"] in p["pair_text"]
        assert p["pair_type"] == "jailbreak_variant"
