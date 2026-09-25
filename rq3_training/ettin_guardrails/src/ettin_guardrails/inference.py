from typing import Any, Iterable

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from ettin_guardrails.model import Classifier
from ettin_guardrails.runtime import configure_precision


def predict(
        classifier: Classifier,
        tokenizer: PreTrainedTokenizerBase,
        prompts: str | Iterable[str],
        config: DictConfig | dict[str, Any],
        device: str = "cuda",
        batch_size: int = 32,
) -> torch.Tensor:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    if len(prompts) == 0:
        raise ValueError("prompts must contain data")
    amp_dtype = configure_precision(config, device)
    device = torch.device(device)
    classifier.to(device)
    classifier.eval()
    predictions = []
    start = 0
    with torch.inference_mode():
        while start < len(prompts):
            current_size = min(batch_size, len(prompts) - start)
            try:
                probabilities = _predict_batch(
                    classifier, tokenizer, prompts[start:start + current_size],
                    config, device, amp_dtype,
                )
            except torch.cuda.OutOfMemoryError:
                if device.type != "cuda" or current_size == 1:
                    raise
                batch_size = max(1, current_size // 2)
            else:
                predictions.append(probabilities)
                start += current_size
                continue
            # batch tensors may still be held by the traceback inside the except
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    return torch.cat(predictions, dim=0)


def _predict_batch(classifier, tokenizer, prompts, config, device, amp_dtype):
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=config.get("tokenizer", {})["max_length"],
        return_token_type_ids=False,
        return_tensors="pt",
    )
    with torch.autocast(device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
        logits, _ = classifier(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
        )
    return logits.float().softmax(dim=-1).cpu()
