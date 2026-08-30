# Training pairs: safe/harm contrast, paraphrase, jailbreak variant

Собрано скриптом `src/data/build_pairs.py` из двух источников:

- **AEGIS 2.0** (`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`, train split, 30k
  промптов с бинарной разметкой `safe`/`unsafe` и категориями нарушений).
- **HarmBench** (`centerforaisafety/HarmBench`): 400 канонических harmful
  behaviors + банк из 114 статических jailbreak-шаблонов (`HumanJailbreaks`
  baseline).

## Типы пар

| pair_type            | источник  | anchor              | pair                                   | label anchor/pair |
|-----------------------|-----------|---------------------|------------------------------------------|--------------------|
| `safe_harm_contrast`  | AEGIS     | unsafe-промпт        | safe-промпт                              | harm / safe        |
| `paraphrase`          | AEGIS     | промпт               | ближайший сосед того же класса/категории по TF-IDF cosine similarity (excl. почти-дубликаты) | одинаковый label |
| `jailbreak_variant`   | HarmBench | canonical behavior   | тот же behavior, обёрнутый в jailbreak-шаблон | harm / harm |

Схема записи (JSONL, `data/processed/pairs.jsonl`):
```json
{"id": 0, "pair_type": "...", "anchor_text": "...", "anchor_label": "...",
 "pair_text": "...", "pair_label": "...", "category": "...", "source": "aegis|harmbench"}
```

## Зафиксированные размеры (seed=42, значения по умолчанию)

| pair_type            | n    |
|-----------------------|------|
| safe_harm_contrast    | 1000 |
| paraphrase            | 500  |
| jailbreak_variant     | 500  |
| **итого**             | **2000** |

Полная разбивка по категориям — в `data/processed/manifest.json`, который
генерируется вместе с датасетом при каждом запуске.

Размеры и seed настраиваются через CLI:
```bash
python -m src.data.build_pairs \
  --n-safe-harm 1000 --n-paraphrase 500 --n-jailbreak 500 \
  --templates-per-behavior 5 --seed 42
```

`data/raw/` (кэш скачанных исходников) и `data/processed/` (сгенерированный
датасет) не коммитятся в git (см. `data/.gitignore`) — датасет
детерминированно регенерируется запуском скрипта с тем же seed.

## Известные ограничения

- `paraphrase` пары находятся эвристически (TF-IDF similarity в группе
  `(label, category)`, с сэмплированием до 300 строк на группу) — это лексически
  похожие переформулировки одного и того же запроса, а не paraphrase от LLM.
- `jailbreak_variant` использует только 114 статических человеческих шаблонов
  из HarmBench; сгенерированные оптимизационными методами (GCG/PAIR/AutoDAN и
  т.п.) test cases не включены — полный precomputed-архив HarmBench весит
  ~10 GB (Zenodo, https://zenodo.org/records/10714577) и включает completions
  33 моделей, что избыточно для сбора пар для обучения энкодера.
