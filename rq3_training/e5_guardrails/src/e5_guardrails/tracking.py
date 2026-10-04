"""Weights & Biases logging, set up like the team's Ettin runs (same project,
default entity, ``online`` mode). Files under ``outputs/`` stay the source of
truth; W&B mirrors them for browsing and comparison.

``wandb.mode: disabled`` turns logging off (tests, offline debugging); wandb is
then never imported.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def group_name(cfg: dict[str, Any]) -> str:
    return f"e5-{cfg['protocol']}-{cfg['variant']}"


@contextmanager
def start_run(cfg: dict[str, Any], name: str, job_type: str, config: dict[str, Any] | None = None) -> Iterator[Any]:
    wcfg = cfg["wandb"]
    if wcfg["mode"] == "disabled":
        yield None
        return
    import wandb

    with wandb.init(
        project=wcfg["project"],
        entity=wcfg["entity"],
        mode=wcfg["mode"],
        name=name,
        group=group_name(cfg),
        job_type=job_type,
        tags=["e5", cfg["protocol"], cfg["variant"]],
        config=config if config is not None else cfg,
        reinit="create_new",
    ) as run:
        yield run


def log_table(run: Any, key: str, df) -> None:
    if run is None:
        return
    import wandb

    run.log({key: wandb.Table(dataframe=df.astype(object).where(df.notna(), None))})


def log_files(run: Any, name: str, artifact_type: str, files: list[Path]) -> None:
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(re.sub(r"[^\w.-]", "-", name), type=artifact_type)
    for f in files:
        if f.exists():
            artifact.add_file(str(f))
    run.log_artifact(artifact)
