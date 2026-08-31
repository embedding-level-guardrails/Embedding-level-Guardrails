import pandas as pd

from training_pairs.src.data.build_pairs import (
    build_benign_twin_pairs,
    build_code_contrast_pairs,
    build_code_jailbreak_pairs,
    build_code_paraphrase_pairs,
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


def test_benign_twin_pairs_wrap_safe_prompt_and_stay_safe():
    df = make_aegis_df()
    templates = ["TEMPLATE_A", "TEMPLATE_B", "TEMPLATE_C"]
    pairs = build_benign_twin_pairs(df, templates, n=6, templates_per_prompt=2, seed=1)
    assert len(pairs) == 6
    for p in pairs:
        assert p["anchor_text"] in p["pair_text"]
        assert p["anchor_label"] == "safe"
        assert p["pair_label"] == "safe"
        assert p["pair_type"] == "benign_twin"


def make_code_bank():
    return [
        {
            "category": "cat_a",
            "malicious": ["do bad thing 1", "do bad thing 2"],
            "benign_twin": ["do fine thing 1", "do fine thing 2"],
        },
        {
            "category": "cat_b",
            "malicious": ["do bad thing 3", "do bad thing 4"],
            "benign_twin": ["do fine thing 3", "do fine thing 4"],
        },
    ]


def test_code_contrast_pairs_are_cross_label():
    pairs = build_code_contrast_pairs(make_code_bank(), seed=1)
    assert len(pairs) == 2 * 2 * 2  # 2 categories * 2 malicious * 2 benign_twin
    for p in pairs:
        assert p["anchor_label"] == "harm"
        assert p["pair_label"] == "safe"
        assert p["pair_type"] == "code_safe_harm_contrast"


def test_code_paraphrase_pairs_keep_same_label():
    pairs = build_code_paraphrase_pairs(make_code_bank(), seed=1)
    assert len(pairs) > 0
    for p in pairs:
        assert p["anchor_label"] == p["pair_label"]
        assert p["anchor_text"] != p["pair_text"]
        assert p["pair_type"] == "code_paraphrase"


def test_code_jailbreak_pairs_wrap_text_with_template():
    templates = ["TEMPLATE_A", "TEMPLATE_B"]
    pairs = build_code_jailbreak_pairs(templates, make_code_bank(), templates_per_text=1, seed=1, twin=False)
    for p in pairs:
        assert p["anchor_text"] in p["pair_text"]
        assert p["anchor_label"] == "harm"
        assert p["pair_type"] == "code_jailbreak_variant"


def test_code_benign_twin_pairs_wrap_benign_text_and_stay_safe():
    templates = ["TEMPLATE_A", "TEMPLATE_B"]
    pairs = build_code_jailbreak_pairs(templates, make_code_bank(), templates_per_text=1, seed=1, twin=True)
    for p in pairs:
        assert p["anchor_text"] in p["pair_text"]
        assert p["anchor_label"] == "safe"
        assert p["pair_type"] == "code_benign_twin"
