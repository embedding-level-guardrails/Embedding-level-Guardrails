from typing import Any

import torch
from omegaconf import DictConfig


def configure_precision(config: DictConfig | dict[str, Any], device: str | torch.device):
    if isinstance(device, str):
        device = torch.device(device)
    precision = str(config.get("training", {}).get("precision", "auto")).lower()
    if precision == "auto":
        if device.type == "cuda":
            with torch.cuda.device(device):
                precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            precision = "fp16" if device.type == "mps" else "fp32"
    dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    if precision not in dtypes:
        raise ValueError("training.precision must be auto, fp32, fp16, or bf16")
    return dtypes[precision]


def configure_device(cfg: DictConfig) -> torch.device:
    device_name = cfg.training.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(device_name)