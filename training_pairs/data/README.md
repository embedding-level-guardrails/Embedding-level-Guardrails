# Training pairs — safe/harm contrast, paraphrase, jailbreak variant, benign twin

Данные для RQ3 (contrastive-дообучение энкодера).

Собрано скриптом `training_pairs/src/data/build_pairs.py` из двух источников:

- **AEGIS 2.0** (`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`, train split, 30k
  промптов с бинарной разметкой `safe`/`unsafe` и категориями нарушений).
- **HarmBench** (`centerforaisafety/HarmBench`): 400 канонических harmful
  behaviors + банк из 114 статических jailbreak-шаблонов (`HumanJailbreaks`
  baseline).

`sources.py` также содержит `load_wildguardmix()` — необязательный загрузчик
WildGuardMix, gated на HuggingFace (нужен `HF_TOKEN`). Маппинг колонок там
**не проверен** (не было доступа к токену при написании) и в `build_pairs.py`
по умолчанию не подключён — используйте только после ручной проверки.

## Типы пар (основной набор, `data/processed/pairs.jsonl`)

| pair_type            | источник         | anchor              | pair                                   | label anchor/pair |
|-----------------------|------------------|---------------------|------------------------------------------|--------------------|
| `safe_harm_contrast`  | AEGIS            | unsafe-промпт        | safe-промпт                              | harm / safe        |
| `paraphrase`          | AEGIS            | промпт               | ближайший сосед того же класса/категории по TF-IDF cosine similarity (excl. почти-дубликаты) | одинаковый label |
| `jailbreak_variant`   | HarmBench        | canonical behavior   | тот же behavior, обёрнутый в jailbreak-шаблон | harm / harm |
| `benign_twin`         | AEGIS+HarmBench  | safe-промпт          | тот же safe-промпт, обёрнутый в jailbreak-шаблон (TwinBreak-стиль) | safe / safe |

`benign_twin` — твин `jailbreak_variant` для RQ2: та же семья шаблонов-обёрток,
но вокруг безобидного запроса вместо harmful behavior. Сравнение
`jailbreak_variant` vs `benign_twin` изолирует эффект самой обёртки (стиля) от
эффекта вредоносного содержания.

Схема записи (JSONL):
```json
{"id": 0, "pair_type": "...", "anchor_text": "...", "anchor_label": "...",
 "pair_text": "...", "pair_label": "...", "category": "...", "source": "aegis|harmbench|aegis+harmbench"}
```

## Отдельный набор для кода (`data/processed/code/pairs.jsonl`)

Основные источники почти не покрывают code-домен (у AEGIS/HarmBench есть лишь
единичные malware/cybercrime-промпты — недостаточно для сопоставимого по
размеру набора). Поэтому для code-домена — отдельный hand-authored банк,
`training_pairs/src/data/code_bank.py`: 12 категорий (keylogger, ransomware,
ddos_script, sql_injection_tool, phishing_page, credential_stealer,
botnet_client, port_scanner_unauthorized, arp_spoof_mitm, webshell_backdoor,
cryptomining_hijack, license_keygen_crack), по 3 формулировки malicious task
description + 3 benign_twin формулировки на категорию.

Из него строятся те же 4 типа пары (с префиксом `code_`):

| pair_type                  | anchor              | pair                        | label anchor/pair |
|------------------------------|---------------------|------------------------------|--------------------|
| `code_safe_harm_contrast`   | malicious-формулировка | benign_twin той же категории | harm / safe |
| `code_paraphrase`           | формулировка (malicious или benign_twin) | другая формулировка той же категории/label | одинаковый label |
| `code_jailbreak_variant`    | malicious-формулировка | та же формулировка, обёрнутая в jailbreak-шаблон | harm / harm |
| `code_benign_twin`          | benign_twin-формулировка | та же формулировка, обёрнутая в jailbreak-шаблон | safe / safe |

Пишется отдельным манифестом в `data/processed/code/manifest.json`; флаг
`--skip-code` отключает генерацию этого набора.

## Зафиксированные размеры (seed=42, значения по умолчанию)

Основной набор — все 4 типа равны по размеру (500), чтобы сравнение в RQ2 было
честным:

| pair_type            | n    |
|-----------------------|------|
| safe_harm_contrast    | 500  |
| paraphrase            | 500  |
| jailbreak_variant     | 500  |
| benign_twin           | 500  |
| **итого**             | **2000** |

Code-набор ограничен размером банка (12 категорий × 3 формулировки на
сторону), поэтому размеры между собой сопоставимы, но не равны 500 —
`code_safe_harm_contrast`/`code_jailbreak_variant`/`code_benign_twin` — 108,
`code_paraphrase` — 72 (итого 396). Полная разбивка по категориям — в
соответствующих `manifest.json`, которые генерируются вместе с датасетом при
каждом запуске.

Размеры и seed настраиваются через CLI (запускать из корня репозитория):
```bash
python -m training_pairs.src.data.build_pairs \
  --n-safe-harm 500 --n-paraphrase 500 --n-jailbreak 500 --n-benign-twin 500 \
  --templates-per-behavior 5 --code-templates-per-text 3 --seed 42
```

`data/raw/` (кэш скачанных исходников) и `data/processed/` (сгенерированный
датасет) не коммитятся в git (см. `data/.gitignore`) — датасет
детерминированно регенерируется запуском скрипта с тем же seed.

## Известные ограничения

- `paraphrase` пары находятся эвристически (TF-IDF similarity в группе
  `(label, category)`, с сэмплированием до 300 строк на группу) — это лексически
  похожие переформулировки одного и того же запроса, а не paraphrase от LLM.
- `jailbreak_variant`/`benign_twin` использует только 114 статических
  человеческих шаблонов из HarmBench; сгенерированные оптимизационными
  методами (GCG/PAIR/AutoDAN и т.п.) test cases не включены — полный
  precomputed-архив HarmBench весит ~10 GB (Zenodo,
  https://zenodo.org/records/10714577) и включает completions 33 моделей, что
  избыточно для сбора пар для обучения энкодера.
- Code-домен собран вручную (не из скачанного источника), малого размера (36
  malicious + 36 benign_twin формулировок) — это синтетический банк task
  description-уровня, не реальные вредоносные тексты/код.
- `load_wildguardmix()` в `sources.py` не проверен на реальных данных (gated
  датасет, не было HF-токена) и не подключён в `build_pairs.py` по умолчанию.
