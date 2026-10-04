"""CLI: python -m e5_guardrails <command> [--config FILE] [--set key=value ...]"""

from __future__ import annotations

import argparse
import copy
import json
from typing import Any

import numpy as np
import pandas as pd

from e5_guardrails.config import (
    DEFAULT_CONFIG,
    ENCODERS,
    MODES,
    apply_override,
    load_config,
    paths,
)


def prepare_data(cfg: dict[str, Any]) -> None:
    from e5_guardrails import splits
    from e5_guardrails.model import load_tokenizer, token_lengths

    data, manifest, overlap = splits.build_splits(cfg)
    tokenizer = load_tokenizer(cfg["model"]["name"], cfg["model"]["revision"])
    limit = cfg["model"]["max_length"]
    truncation = {}
    for name, df in data.items():
        lengths = token_lengths(tokenizer, df["text"].tolist(), cfg["model"]["prefix"])
        if not len(df):
            continue
        entry = {"n": len(df), "truncated_fraction": float((lengths > limit).mean()),
                 "p95_tokens": float(np.percentile(lengths, 95)), "max_tokens": int(lengths.max())}
        adv = df["adversarial"]
        for value, label in ((True, "adversarial"), (False, "vanilla")):
            mask = adv.map(lambda v, value=value: v is not None and not pd.isna(v) and bool(v) == value).to_numpy()
            if mask.any():
                entry[f"truncated_fraction_{label}"] = float((lengths[mask] > limit).mean())
        truncation[name] = entry
    manifest["truncation"] = {"max_length": limit, "by_split": truncation}
    out = paths(cfg).splits
    splits.save_splits(data, manifest, overlap, out)
    print(json.dumps({"splits_dir": str(out), "counts": manifest["counts"],
                      "eval_overlap": manifest["eval_overlap"], "truncation": manifest["truncation"]}, indent=2))


def with_sweep_best(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    best_path = paths(cfg).sweeps / "best.json"
    if not best_path.exists():
        raise FileNotFoundError(f"{best_path} not found; run sweep first")
    cfg = copy.deepcopy(cfg)
    for key, value in json.loads(best_path.read_text())[mode]["overrides"].items():
        apply_override(cfg, key, value)
    return cfg


def sweep(cfg: dict[str, Any], modes: list[str], seed: int) -> None:
    from e5_guardrails.train import train

    sizes = {m: len(cfg["sweep"][m]) for m in modes}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Unequal hyperparameter budget per mode: {sizes}")
    out = paths(cfg).sweeps
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "best.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else {}
    for mode in modes:
        trials = []
        for i, overrides in enumerate(cfg["sweep"][mode]):
            trial = copy.deepcopy(cfg)
            for key, value in overrides.items():
                apply_override(trial, key, value)
            trial["variant"] = f"{cfg['variant']}__sweep/{mode}-{i}"
            summary = train(trial, mode, seed)
            trials.append({"trial": i, "overrides": overrides, **summary["best"]})
        (out / f"{mode}.json").write_text(json.dumps(trials, indent=2))
        best[mode] = max(trials, key=lambda t: t["selection_roc_auc"])
        best_path.write_text(json.dumps(best, indent=2))
        print(json.dumps({"mode": mode, "best": best[mode]}))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="e5_guardrails", description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-data", help="build and save fixed splits + manifest for the protocol")
    t = sub.add_parser("train", help="fine-tune E5 with one loss")
    t.add_argument("--mode", choices=MODES, required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--use-sweep", action="store_true", help="apply the mode's best sweep overrides")
    pr = sub.add_parser("probe", help="fit the shared linear probe on a frozen encoder and save predictions")
    pr.add_argument("--encoder", choices=ENCODERS, required=True)
    pr.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("evaluate", help="thresholds from calibration, metrics, bootstrap, tables")
    e.add_argument("--no-bootstrap", action="store_true")
    sw = sub.add_parser("sweep", help="equal-budget hyperparameter search on source validation")
    sw.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    sw.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("run-all", help="prepare (optional), train all modes, probe all encoders, evaluate")
    a.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    a.add_argument("--skip-prepare", action="store_true")
    a.add_argument("--use-sweep", action="store_true")
    a.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    if args.command == "prepare-data":
        prepare_data(cfg)
    elif args.command == "train":
        from e5_guardrails.train import train
        print(json.dumps(train(with_sweep_best(cfg, args.mode) if args.use_sweep else cfg, args.mode, args.seed),
                         indent=2, default=str))
    elif args.command == "probe":
        from e5_guardrails.probe import run_probe
        run_probe(cfg, args.encoder, args.seed)
    elif args.command == "evaluate":
        from e5_guardrails.report import run_evaluation
        run_evaluation(cfg, bootstrap=not args.no_bootstrap)
        print((paths(cfg).results / "results.md").read_text())
    elif args.command == "sweep":
        sweep(cfg, args.modes, args.seed)
    elif args.command == "run-all":
        from e5_guardrails.probe import run_probe
        from e5_guardrails.report import run_evaluation
        from e5_guardrails.train import train
        if not args.skip_prepare:
            prepare_data(cfg)
        for seed in args.seeds:
            for mode in args.modes:
                train(with_sweep_best(cfg, mode) if args.use_sweep else cfg, mode, seed)
            for encoder in ("base", *args.modes):
                run_probe(cfg, encoder, seed)
        run_evaluation(cfg, bootstrap=not args.no_bootstrap)
        print((paths(cfg).results / "results.md").read_text())


if __name__ == "__main__":
    main()
