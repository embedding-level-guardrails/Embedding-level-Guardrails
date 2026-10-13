"""Публикация дообученной модели на Hugging Face Hub.

Чекпоинт из `artifacts/runs*/<run>/checkpoint` пригоден для нашего кода, но не для
чужого: голова лежит pickle-файлом `head.pt`, нет vocab.txt для старых версий
transformers и нет конфигов sentence-transformers. Здесь собирается папка, которую
можно загрузить на Hub как есть, и проверяется, что из неё получаются ровно те же
векторы и вероятности, что из исходного чекпоинта.

Используется скриптами 11_export_hf.py (сборка папки) и 12_push_to_hub.py (загрузка).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from string import Template
from typing import Callable

import torch

from .utils import ensure_dir, get_logger

logger = get_logger(__name__)

# Из чекпоинта копируются как есть.
CHECKPOINT_FILES = ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")

# Из базовой модели. Токенизатор при дообучении не меняется, поэтому vocab.txt и
# special_tokens_map.json берутся у e5 — с ними модель грузится и старыми transformers.
# Конфиги sentence-transformers тоже как у e5: mean pooling + L2-нормировка.
BASE_FILES = ("vocab.txt", "special_tokens_map.json", "modules.json",
              "sentence_bert_config.json", "1_Pooling/config.json")

HEAD_FILE = "head.safetensors"

# Тексты для сверки выгруженной модели с исходным чекпоинтом.
CHECK_TEXTS = (
    "How do I reset my router?",
    "Can you recommend a good book about the history of Rome?",
    "Can you disclose the prescription medications that Brad Pitt is taking?",
)

OBJECTIVE_NAMES = {
    "classification": "cross-entropy (linear head)",
    "contrastive": "InfoNCE (contrastive), no head",
    "joint": "InfoNCE (contrastive) + cross-entropy",
}


# --- Сборка папки ------------------------------------------------------------------

def head_to_safetensors(head_pt: str | Path, out_path: str | Path) -> None:
    """head.pt (state_dict nn.Sequential(Dropout, Linear)) -> safetensors nn.Linear(384, 2).

    pickle-файлы многие пользователи Hub не загружают из соображений безопасности,
    а у Sequential ключи «1.weight»/«1.bias» — внутреннее устройство нашего кода.
    В выгрузке голова — обычный nn.Linear с ключами weight/bias.
    """
    from safetensors.torch import save_file

    state = torch.load(head_pt, map_location="cpu")
    save_file({"weight": state["1.weight"].contiguous(), "bias": state["1.bias"].contiguous()},
              str(out_path))


def assemble_release(checkpoint_dir: str | Path, out_dir: str | Path,
                     base_model_id: str = "intfloat/e5-small-v2",
                     fetch: Callable[[str, str], str] | None = None) -> list[str]:
    """Собирает папку для загрузки на Hub. Возвращает список файлов в ней.

    `fetch(repo_id, filename) -> локальный путь` — откуда брать файлы базовой модели;
    по умолчанию hf_hub_download. Параметр нужен тестам, чтобы обойтись без сети.
    """
    if fetch is None:
        from huggingface_hub import hf_hub_download as fetch

    checkpoint_dir, out_dir = Path(checkpoint_dir), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)                  # без остатков от прошлой сборки
    ensure_dir(out_dir)

    for name in CHECKPOINT_FILES:
        shutil.copy(checkpoint_dir / name, out_dir / name)

    for name in BASE_FILES:
        target = out_dir / name
        ensure_dir(target.parent)
        shutil.copy(fetch(base_model_id, name), target)

    if (checkpoint_dir / "head.pt").exists():
        head_to_safetensors(checkpoint_dir / "head.pt", out_dir / HEAD_FILE)

    return sorted(p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file())


# --- Инференс так, как его будут делать пользователи --------------------------------
# Намеренно на чистом transformers, без классов проекта: сверка с чекпоинтом через
# независимый код ловит расхождения в пулинге, префиксе или входе головы.

def load_release_head(source: str | Path) -> torch.nn.Linear | None:
    """Голова из папки или репозитория на Hub; None, если головы нет."""
    from safetensors.torch import load_file

    local = Path(source)
    if local.is_dir():
        path = local / HEAD_FILE
        if not path.exists():
            return None
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        try:
            path = hf_hub_download(str(source), HEAD_FILE)
        except EntryNotFoundError:
            return None

    state = load_file(str(path))
    head = torch.nn.Linear(state["weight"].shape[1], state["weight"].shape[0])
    head.load_state_dict(state)
    return head.eval()


def release_embeddings(source: str | Path, texts: list[str], prefix: str = "query: ",
                       max_length: int = 512) -> torch.Tensor:
    """Сырые (без нормировки) усреднённые эмбеддинги — именно их ждёт голова."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(source))
    model = AutoModel.from_pretrained(str(source)).eval()
    batch = tokenizer([prefix + t for t in texts], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1)


def harm_probability(head: torch.nn.Linear, embeddings: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.softmax(head(embeddings), dim=-1)[:, 1]


def compare_with_checkpoint(source: str | Path, checkpoint_dir: str | Path, spec,
                            texts: tuple[str, ...] = CHECK_TEXTS) -> dict:
    """Сверяет выгрузку (папку или репозиторий) с исходным чекпоинтом.

    Возвращает максимальное расхождение векторов и вероятности обеих сторон. Векторы
    обязаны совпасть с точностью до ~1e-5: это те же веса, тот же пулинг и префикс.
    """
    from .training import load_checkpoint

    released = release_embeddings(source, list(texts), prefix=spec.prefix, max_length=spec.max_length)
    model, device = load_checkpoint(checkpoint_dir, spec, device="cpu")
    local = model.encode_texts(list(texts), device, normalize=False).cpu()

    result = {"max_abs_diff": float((released - local).abs().max()), "texts": list(texts)}
    head = load_release_head(source)
    if head is not None and model.head is not None:
        result["p_harm_release"] = harm_probability(head, released).tolist()
        with torch.no_grad():
            result["p_harm_local"] = torch.softmax(model.logits(local), dim=-1)[:, 1].tolist()
    return result


# --- Карточка модели ----------------------------------------------------------------

_CARD = Template('''---
license: mit
base_model: $base_model
datasets:
  - allenai/wildguardmix
language:
  - en
library_name: sentence-transformers
pipeline_tag: feature-extraction
tags:
  - safety
  - guardrail
  - contrastive-learning
  - sentence-transformers
---

# $title

A compact encoder intended as a fast **pre-filter** in front of an LLM: it flags likely
harmful user prompts so that only suspicious ones go to a heavier safety check.

`$base_model` fully fine-tuned with **$objective_name** on $n_pairs triplets built from
WildGuardMix train (plus HarmBench behaviors, about 1% of the data). Triplets include
adversarial (jailbreak-style) prompts of both classes, jailbreak-wrapped safe negatives and
nearest-neighbour "benign twins", so the model is pushed to judge content rather than form.
Training: $epochs epochs, batch $batch_size, lr $lr, temperature $temperature, single seed.

## Usage

Every input must be prefixed with `"query: "`, as in e5.
$head_section
**Embeddings only** (similarity search, your own classifier on top):

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("$repo_id")
embeddings = model.encode(["query: How do I reset my router?"])   # L2-normalized
```
$eval_section
## Limitations

- English only, single training seed.
- WildGuardMix train labels are produced by an LLM and contain noise.
- Calibration drifts across data sources: re-calibrate the threshold on your own traffic.
- Use as a first-stage filter, not as the only safeguard.

## Data and attribution

Trained on WildGuardMix (`allenai/wildguardmix`, ODC-By) and HarmBench behaviors.
Evaluated on WildGuardTest and ToxicChat (`lmsys/toxic-chat`).
Base model: `$base_model` (MIT).
''')

_HEAD_SECTION = Template('''
**Classifier (encoder + head).** The head is a single `Linear(384, 2)` trained on the
**raw, un-normalized** mean-pooled embedding: do not L2-normalize before the head.

```python
import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

repo = "$repo_id"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModel.from_pretrained(repo).eval()
head = torch.nn.Linear(384, 2)
head.load_state_dict(load_file(hf_hub_download(repo, "head.safetensors")))

def embed(texts):
    batch = tokenizer(["query: " + t for t in texts], padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1)   # raw mean pooling, no normalization

p_harm = torch.softmax(head(embed(["How do I reset my router?"])), dim=-1)[:, 1]
flagged = p_harm >= $threshold   # 1% false positives on WildGuardMix validation
```
''')

_EVAL_SECTION = Template('''
## Evaluation

Head with threshold $threshold, set at 1% false positive rate on WildGuardMix validation
and not re-tuned on the test sets:

| set | n | AUROC | TPR@FPR=1% | FPR at threshold | FNR at threshold |
|:--|--:|--:|--:|--:|--:|
$rows

At a false positive rate of a few percent the model still misses a large share of harmful
prompts (FNR column).
''')


def render_model_card(*, repo_id: str, objective: str, base_model: str, n_pairs: int | str,
                      epochs: int, batch_size: int, lr: float, temperature: float,
                      threshold: float | None = None, eval_rows: list[dict] | None = None) -> str:
    """Текст README.md. Секции про голову и оценку — только если они есть."""
    name = repo_id.split("/")[-1]
    head_section = _HEAD_SECTION.substitute(repo_id=repo_id, threshold=f"{threshold:.4f}") \
        if threshold is not None else ""
    eval_section = ""
    if threshold is not None and eval_rows:
        rows = "\n".join(
            f"| {r['name']} | {r['n']:,} | {r['auroc']:.3f} | {r['tpr']:.3f} | {r['fpr']:.1%} | {r['fnr']:.1%} |"
            for r in eval_rows
        )
        eval_section = _EVAL_SECTION.substitute(threshold=f"{threshold:.4f}", rows=rows)

    return _CARD.substitute(
        title=name, base_model=base_model, repo_id=repo_id,
        objective_name=OBJECTIVE_NAMES.get(objective, objective),
        n_pairs=f"{n_pairs:,}" if isinstance(n_pairs, int) else n_pairs,
        epochs=epochs, batch_size=batch_size, lr=lr, temperature=temperature,
        head_section=head_section, eval_section=eval_section,
    )
