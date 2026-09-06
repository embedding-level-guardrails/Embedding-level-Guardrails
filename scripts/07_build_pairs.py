"""Шаг 7 (RQ2): сборка обучающих пар из AEGIS + HarmBench.

    python scripts/07_build_pairs.py --config configs/rq2_pairs.yaml
    python scripts/07_build_pairs.py --types safe_harm_contrast jailbreak_variant
    python scripts/07_build_pairs.py --scale 0.05        # быстрый прогон на срезе

На выходе:
    data/processed/pairs/{name}/pairs_{split}.jsonl   — триплеты (anchor, positive, negative)
    data/processed/pairs/{name}/texts_{split}.jsonl   — те же тексты плоско, с бинарной меткой
    data/processed/pairs/{name}/summary.json          — зафиксированные размеры
    results/rq2/PAIRS.md                              — таблица размеров

HarmBench train_split идёт только в train-пары. heldout_split не участвует в
обучении вообще: он сохраняется отдельным файлом для OOD-оценки в RQ3.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.config import load_config  # noqa: E402
from eguard.embeddings import load_xy  # noqa: E402
from eguard.pairs import (  # noqa: E402
    PairContext,
    build_split,
    dedupe_pool,
    exclude_texts,
    flatten_texts,
    load_aegis_records,
    load_external_variants,
    load_harmbench,
    pairs_dir,
    pairs_path,
    split_texts,
)
from eguard.utils import ensure_dir, get_logger, save_json, set_seed, write_jsonl  # noqa: E402

logger = get_logger("build_pairs")


def load_mining(cfg, encoder_key: str | None, split: str, allowed_texts: set[str]):
    """Эмбеддинги фиксированного энкодера для майнинга benign twins.

    Матрица эмбеддингов лежит построчно к ИСХОДНОМУ jsonl сплита, поэтому её надо
    урезать тем же фильтром, что и пулы: иначе benign_twin вытащит дубликат,
    отфильтрованный из пула, и train снова протечёт в val/test.
    """
    if not encoder_key:
        return None
    try:
        x, _, records, _ = load_xy(cfg, encoder_key, split)
    except FileNotFoundError:
        logger.warning("%s/%s: нет эмбеддингов, benign_twin будет пропущен "
                       "(сначала scripts/01_embed.py)", encoder_key, split)
        return None

    keep = [i for i, r in enumerate(records) if r["text"].strip() in allowed_texts]
    if len(keep) < len(records):
        logger.info("%s/%s: майнинг на %d из %d записей (после фильтра пула)",
                    encoder_key, split, len(keep), len(records))
    if not keep:
        return None
    return x[keep], [records[i] for i in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq2_pairs.yaml")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--types", nargs="*", default=None, help="подмножество типов пар")
    ap.add_argument("--scale", type=float, default=1.0, help="множитель размеров из конфига")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    pairs_cfg = cfg.pairs
    name = pairs_cfg.get("name", "pairs_v1")
    hb_cfg = pairs_cfg.get("harmbench", {}) or {}

    external = (load_external_variants(pairs_cfg["external_variants"])
                if pairs_cfg.get("external_variants") else {})
    if external:
        logger.info("Внешние варианты: %d исходных записей", len(external))

    out_dir = ensure_dir(pairs_dir(cfg.paths.processed, name))
    mining_encoder = (pairs_cfg.get("mining") or {}).get("encoder")

    summary = {"name": name, "seed": cfg.seed, "config": pairs_cfg, "splits": []}

    # AEGIS содержит повторяющиеся тексты, в том числе через официальную границу
    # train/test. Поэтому пулы чистятся до сборки, а не после: сначала убираем
    # повторы внутри сплита, затем вычитаем тексты более приоритетных сплитов.
    # Приоритет test > val > train — оценочный сплит не жертвуем ничем.
    raw_records = {
        split: load_aegis_records(cfg.paths.processed, cfg.dataset.name, split)
        for split in args.splits
    }
    priority = [s for s in ("test", "val", "train") if s in raw_records]
    pools: dict[str, list[dict]] = {}
    blocked: set[str] = set()
    for split in priority:
        pool = exclude_texts(dedupe_pool(raw_records[split]), blocked)
        dropped = len(raw_records[split]) - len(pool)
        if dropped:
            logger.info("%s: убрано %d записей (повторы и пересечения со сплитами %s)",
                        split, dropped, priority[: priority.index(split)] or "—")
        pools[split] = pool
        blocked |= split_texts(pool)

    for split in args.splits:
        aegis = pools[split]

        # HarmBench подмешивается только в train: остальное — холд для OOD-оценки.
        harmbench = []
        if split == "train" and hb_cfg.get("train_split"):
            harmbench = load_harmbench(
                hb_cfg["train_split"],
                cache_dir=hb_cfg.get("cache_dir", "data/raw/harmbench"),
                include_context=bool(hb_cfg.get("include_context", False)),
                drop_categories=tuple(hb_cfg.get("drop_categories") or ()),
            )

        sizes = dict((pairs_cfg.get("sizes") or {}).get(split, {}))
        if args.types:
            sizes = {k: v for k, v in sizes.items() if k in args.types}
        if args.scale != 1.0:
            sizes = {k: max(int(round(v * args.scale)), 0) for k, v in sizes.items()}
        if not sizes:
            logger.warning("%s: размеры не заданы, сплит пропущен", split)
            continue

        ctx = PairContext(
            split=split,
            aegis=aegis,
            harmbench=harmbench,
            rng=np.random.default_rng(cfg.seed + hash(split) % 10_000),
            cfg=pairs_cfg,
            external_variants=external,
            mining=load_mining(cfg, mining_encoder, split, split_texts(aegis)),
        )

        pairs = build_split(ctx, sizes)
        texts = flatten_texts(pairs)

        write_jsonl(pairs_path(cfg.paths.processed, name, split, "pairs"), pairs)
        write_jsonl(pairs_path(cfg.paths.processed, name, split, "texts"), texts)

        by_type = Counter(p["pair_type"] for p in pairs)
        by_variant = Counter(f"{p['pair_type']}:{p['variant']}" for p in pairs if p["variant"])
        summary["splits"].append({
            "split": split,
            "n_pairs": len(pairs),
            "n_texts": len(texts),
            "harm_rate_texts": round(float(np.mean([t["label"] for t in texts])), 3) if texts else 0.0,
            "requested": sizes,
            "by_type": dict(by_type),
            "by_variant": dict(by_variant),
            "n_aegis_records": len(aegis),
            "n_harmbench_seeds": len(harmbench),
        })
        logger.info("%s: %d пар, %d уникальных текстов", split, len(pairs), len(texts))

    # HarmBench heldout сохраняем отдельно: в обучении он не участвует.
    if hb_cfg.get("heldout_split"):
        heldout = load_harmbench(
            hb_cfg["heldout_split"],
            cache_dir=hb_cfg.get("cache_dir", "data/raw/harmbench"),
            include_context=bool(hb_cfg.get("include_context", False)),
            drop_categories=tuple(hb_cfg.get("drop_categories") or ()),
        )
        write_jsonl(out_dir / "harmbench_heldout.jsonl", heldout)
        summary["harmbench_heldout"] = {"split": hb_cfg["heldout_split"], "n": len(heldout)}
        logger.info("HarmBench heldout: %d behaviors -> %s", len(heldout), out_dir / "harmbench_heldout.jsonl")

    save_json(out_dir / "summary.json", summary)

    rows = []
    for entry in summary["splits"]:
        row = {"split": entry["split"], "n_pairs": entry["n_pairs"], "n_texts": entry["n_texts"],
               "harm_rate": entry["harm_rate_texts"]}
        row.update(entry["by_type"])
        rows.append(row)
    table = pd.DataFrame(rows).fillna(0)

    report_dir = ensure_dir(Path(cfg.paths.results) / "rq2")
    lines = [
        "# RQ2 — обучающие пары\n",
        f"Набор: `{name}`, seed={cfg.seed}. AEGIS + HarmBench "
        f"(train_split=`{hb_cfg.get('train_split')}`, heldout=`{hb_cfg.get('heldout_split')}`).\n",
        "## Размеры\n", table.to_markdown(index=False) + "\n",
        "\n`n_texts` — число уникальных текстов в парах; это же множество получает "
        "classification-ветка RQ3, чтобы сравнение объективов было при фиксированных данных.\n",
        "\n## Разбивка по вариантам трансформаций\n",
    ]
    for entry in summary["splits"]:
        if entry["by_variant"]:
            variants = pd.DataFrame(
                sorted(entry["by_variant"].items()), columns=["pair_type:variant", "n"]
            )
            lines += [f"\n**{entry['split']}**\n", variants.to_markdown(index=False) + "\n"]
    (report_dir / "PAIRS.md").write_text("".join(lines), encoding="utf-8")

    print("\n=== Размеры пар ===")
    print(table.to_string(index=False))
    logger.info("Готово: %s и %s", out_dir, report_dir / "PAIRS.md")


if __name__ == "__main__":
    main()
