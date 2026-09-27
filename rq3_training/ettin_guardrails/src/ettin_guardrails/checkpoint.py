from datetime import datetime
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(training_checkpoint: dict[str, Any], output_path: str | Path):
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(training_checkpoint, output_dir / f"checkpoint_{timestamp}.pt")