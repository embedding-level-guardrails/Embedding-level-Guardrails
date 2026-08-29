"""Шаг 1: frozen-энкодеры -> эмбеддинги в artifacts/embeddings/.

    python scripts/01_embed.py --config configs/rq1.yaml
    python scripts/01_embed.py --encoders e5-small-v2 mmbert-small --splits test
    python scripts/01_embed.py --encoders dummy   # офлайн, без torch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eguard.config import EncoderSpec, load_config  # noqa: E402
from eguard.data import load_split  # noqa: E402
from eguard.embeddings import compute_and_cache  # noqa: E402
from eguard.encoders import build_encoder  # noqa: E402
from eguard.utils import get_logger, set_seed  # noqa: E402

logger = get_logger("embed")

DUMMY_SPEC = EncoderSpec(key="dummy", hf_id="dummy", pooling="mean")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rq1.yaml")
    ap.add_argument("--encoders", nargs="*", default=None, help="keys from config; 'dummy' for offline-test")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--layer", type=int, default=None, help="redefine embed.layer")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    layer = args.layer if args.layer is not None else cfg.embed.get("layer", -1)
    batch_size = args.batch_size or cfg.embed.get("batch_size", 32)
    device = args.device or cfg.embed.get("device", "auto")

    if args.encoders == ["dummy"]:
        specs = [DUMMY_SPEC]
    else:
        specs = cfg.select_encoders(args.encoders)

    for spec in specs:
        if spec.gated:
            logger.info("%s — gated-repo: need accepted access request and HF_TOKEN in env", spec.key)
        logger.info("=== %s (%s) ===", spec.key, spec.hf_id)
        encoder = build_encoder(spec, device=device, layer=layer, dtype=cfg.embed.get("dtype", "auto"))

        for split in args.splits:
            records = load_split(cfg.paths.processed, cfg.dataset.name, split)
            texts = [r["text"] for r in records]
            compute_and_cache(
                encoder, texts, cfg.paths.embeddings, cfg.dataset.name, split,
                batch_size=batch_size, overwrite=args.overwrite,
            )

        del encoder


if __name__ == "__main__":
    main()
    