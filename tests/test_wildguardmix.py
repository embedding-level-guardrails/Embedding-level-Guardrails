"""Тесты WildGuardMix: разметка, дедупликация промтов, фильтр adversarial,
новый тип пар adversarial_contrast и выбор затравок jailbreak. Без сети и torch."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.config import DatasetSpec  # noqa: E402
from eguard.data import get_loader  # noqa: E402
from eguard.data.wildguardmix import (  # noqa: E402
    category_of,
    is_adversarial,
    normalize_rows,
    parse_agreement,
    to_binary,
)
from eguard.pairs.builders import (  # noqa: E402
    PairContext,
    build_adversarial_contrast,
    jailbreak_seeds,
)


def _row(prompt="обычный пользовательский запрос", label="unharmful", adversarial=False,
         subcategory="benign", response="ответ"):
    return {"prompt": prompt, "prompt_harm_label": label, "adversarial": adversarial,
            "subcategory": subcategory, "response": response}


def _spec(**kw):
    return DatasetSpec(name="wildguardmix", hf_id="allenai/wildguardmix", **kw)


def test_to_binary():
    assert to_binary("harmful") == 1
    assert to_binary("Unharmful") == 0
    assert to_binary(None) is None and to_binary(float("nan")) is None and to_binary("None") is None
    with pytest.raises(ValueError):
        to_binary("maybe")


def test_adversarial_flag_and_agreement_parsing():
    assert is_adversarial(True) and is_adversarial("true") and not is_adversarial(False)
    assert parse_agreement("2/3") == pytest.approx(2 / 3)
    # реальный формат wildguardtest: число согласных из трёх аннотаторов
    assert parse_agreement(3.0) == 1.0
    assert parse_agreement(2.0) == pytest.approx(2 / 3)
    assert parse_agreement(0.5) == 0.5
    assert parse_agreement(None) == 1.0          # в train согласия нет — LLM-разметка


def test_category():
    assert category_of({"subcategory": "cyberattack"}, 1) == "cyberattack"
    assert category_of({"subcategory": None}, 1) == "others"
    assert category_of({"subcategory": "cyberattack"}, 0) == "safe"


def test_normalize_dedupes_prompts_and_drops_conflicts():
    """Строки train — пары (prompt, response): один промт встречается несколько раз."""
    rows = [
        _row("повторяющийся промт", response="ответ 1"),
        _row("повторяющийся промт", response="ответ 2"),
        _row("спорный промт", label="harmful", subcategory="others"),
        _row("спорный промт", label="unharmful"),           # конфликт меток — выбросить
        _row("промт без метки", label=None),
        _row("x"),                                          # короткий
    ]
    out = normalize_rows(rows, _spec(min_chars=3))
    assert [r["text"] for r in out] == ["повторяющийся промт"]
    record = out[0]
    assert record["label"] == 0 and record["label_name"] == "safe"
    assert record["source"] == "wildguardmix" and record["text_type"] == "user_message"
    assert record["adversarial"] is False


def test_adversarial_filter():
    rows = [_row("ванильный запрос"), _row("адверсариальный запрос", adversarial=True)]
    assert len(normalize_rows(rows, _spec(adversarial_filter="all"))) == 2
    assert [r["text"] for r in normalize_rows(rows, _spec(adversarial_filter="vanilla"))] == ["ванильный запрос"]
    assert [r["text"] for r in normalize_rows(rows, _spec(adversarial_filter="adversarial"))] == \
        ["адверсариальный запрос"]
    with pytest.raises(ValueError):
        normalize_rows(rows, _spec(adversarial_filter="nope"))


def test_max_records_is_stratified():
    rows = [_row(f"безопасный промт {i}") for i in range(80)] + \
           [_row(f"вредный промт {i}", label="harmful", subcategory="others") for i in range(20)]
    out = normalize_rows(rows, _spec(max_records_per_split=50))
    assert len(out) == 50
    assert sum(r["label"] for r in out) == 10                # пропорция 80/20 сохранилась


def test_loader_registered_and_synthetic_covers_edge_cases():
    loader = get_loader("wildguardmix")
    raw = loader.make_synthetic(300, seed=0)
    n_rows = len(raw["train"])
    records = loader.normalize_rows(raw["train"], _spec())
    assert len(records) < n_rows                             # дубли и строки без метки ушли
    assert {r["label"] for r in records} == {0, 1}
    assert {r["adversarial"] for r in records} == {True, False}


# --- пары ------------------------------------------------------------------------

def _wgm_records():
    out = []
    for i in range(40):
        label, adversarial = i % 2, (i // 2) % 2 == 1
        out.append({"id": f"w{i}", "text": f"{'adv ' if adversarial else ''}{'harm' if label else 'safe'} {i}",
                    "label": label, "category": "cyberattack" if label else "safe",
                    "adversarial": adversarial, "source": "wildguardmix"})
    return out


def test_adversarial_contrast_structure():
    ctx = PairContext(split="train", aegis=_wgm_records(), rng=np.random.default_rng(0), cfg={})
    by_id = {r["id"]: r for r in ctx.aegis}
    pairs = build_adversarial_contrast(ctx, 60)
    assert pairs and {p["anchor_label"] for p in pairs} == {0, 1}
    for p in pairs:
        # якорь и негатив — adversarial разных классов, позитив — vanilla того же класса
        assert by_id[p["anchor_id"]]["adversarial"] and by_id[p["negative_id"]]["adversarial"]
        assert not by_id[p["positive_id"]]["adversarial"]
        assert by_id[p["positive_id"]]["label"] == p["anchor_label"]
        assert p["negative_label"] == 1 - p["anchor_label"]


def test_adversarial_contrast_empty_without_flag():
    """У AEGIS нет флага adversarial — тип пар должен быть пустым, а не падать."""
    records = [{"id": f"a{i}", "text": f"t{i}", "label": i % 2, "category": "c"} for i in range(10)]
    ctx = PairContext(split="train", aegis=records, rng=np.random.default_rng(0), cfg={})
    assert build_adversarial_contrast(ctx, 10) == []


def test_jailbreak_seed_modes():
    records = _wgm_records()
    hb = [{"id": "hb0", "text": "behavior", "label": 1, "category": "illegal", "source": "harmbench"}]
    make = lambda mode, harmbench: PairContext(split="train", aegis=records, harmbench=harmbench,
                                               cfg={"jailbreak_seeds": mode})
    vanilla_harm = {r["id"] for r in records if r["label"] == 1 and not r["adversarial"]}

    assert [r["id"] for r in jailbreak_seeds(make("harmbench", hb))] == ["hb0"]
    assert {r["id"] for r in jailbreak_seeds(make("dataset", hb))} == vanilla_harm
    assert {r["id"] for r in jailbreak_seeds(make("both", hb))} == vanilla_harm | {"hb0"}
    # исходное поведение: без HarmBench берётся весь harm-пул
    assert len(jailbreak_seeds(make("harmbench", []))) == sum(r["label"] for r in records)
    with pytest.raises(ValueError):
        jailbreak_seeds(make("nope", hb))
