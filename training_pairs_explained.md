# Training pairs (RQ3): что сделали, как и почему

Данные для RQ3 (contrastive fine-tuning энкодеров) — `training_pairs/`. Это
набор пар текстов, на которых потом обучается контрастивная модель: пары
должны либо притягиваться (одинаковый label), либо отталкиваться (разный
label) в пространстве эмбеддингов.

Получилось два набора:

- **Основной** (`training_pairs/data/processed/pairs.jsonl`) — 4 типа пар по
  500 штук = 2000 пар, источники AEGIS 2.0 + HarmBench.
- **Отдельный code-domain** (`training_pairs/data/processed/code/pairs.jsonl`)
  — те же 4 типа, но по коду, 396 пар, источник — свой hand-authored банк.

## Как устроен каждый тип (основной набор)

| Тип | Что делает | Зачем |
|---|---|---|
| **safe_harm_contrast** | Случайная пара (unsafe-промпт, safe-промпт) из AEGIS | Базовый контрастивный сигнал: разные классы должны расходиться |
| **paraphrase** | Внутри группы (label, category) ищем близкие по TF-IDF cosine similarity промпты (диапазон 0.35–0.95 — не дубликаты, но и не случайные) | Проверка, что переформулировки одного и того же остаются рядом |
| **jailbreak_variant** | Canonical harmful behavior из HarmBench + тот же behavior, обёрнутый в один из 114 статических jailbreak-шаблонов, оба "harm" | Проверка, не обманывает ли обёртка энкодер (стиль маскирует вредоносность) |
| **benign_twin** | Safe-промпт из AEGIS + тот же safe-промпт, обёрнутый в тот же jailbreak-шаблон, оба "safe" | Твин к jailbreak_variant (TwinBreak-стиль) — сравнение jailbreak_variant vs benign_twin отделяет эффект самой обёртки/стиля от эффекта именно вредоносного содержания |

Источники:
- **AEGIS 2.0** (`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`) — 30k
  промптов с бинарной разметкой `safe`/`unsafe` и категориями нарушений.
- **HarmBench** (`centerforaisafety/HarmBench`) — 400 канонических harmful
  behaviors + банк из 114 статических человеческих jailbreak-шаблонов
  (`HumanJailbreaks` baseline).

## Почему появился code-domain набор отдельно

В AEGIS и HarmBench кода/малвари почти нет — недостаточно, чтобы собрать
сопоставимый по размеру набор именно для code-домена. Поэтому написали
вручную `training_pairs/src/data/code_bank.py`: 12 категорий (keylogger,
ransomware, ddos_script, sql_injection_tool, phishing_page,
credential_stealer, botnet_client, port_scanner_unauthorized, arp_spoof_mitm,
webshell_backdoor, cryptomining_hijack, license_keygen_crack), на каждую — 3
формулировки malicious task description + 3 benign_twin формулировки того же
уровня абстракции (описание задачи, не рабочий эксплойт-код).

Из этого банка строятся те же 4 типа пары с префиксом `code_`:

| Тип | Что делает |
|---|---|
| `code_safe_harm_contrast` | malicious-формулировка × benign_twin-формулировка той же категории (все комбинации) |
| `code_paraphrase` | разные формулировки внутри одного label/категории |
| `code_jailbreak_variant` | malicious-формулировка, обёрнутая в jailbreak-шаблон |
| `code_benign_twin` | benign_twin-формулировка, обёрнутая в тот же jailbreak-шаблон |

## Почему размеры именно такие

Изначально было 1000/500/500 (неровно). Поправили на 500/500/500/500, чтобы
сравнение типов пар в RQ2 было честным (одинаковый вес каждого типа).

Для code-набора равные 500 физически невозможны — банк маленький (36
malicious + 36 benign формулировок), поэтому там 108/72/108/108:
`code_paraphrase` меньше остальных, потому что комбинаций переформулировок в
3 фразах на категорию просто меньше, чем комбинаций contrast/jailbreak-wrap.

## Про источники для RQ1 vs RQ3 (важное решение)

RQ1 (baseline-разделимость без дообучения) теперь считается на
JailbreakBench Behaviors + FORTRESS (598 пар), а не на AEGIS. Решили **не**
подмешивать JailbreakBench/FORTRESS в training-пары для RQ3 — вместо этого
использовать их как held-out eval-набор.

Причина: если RQ3 обучит модель на тех же данных, на которых RQ1 меряет
baseline, сравнение "до/после дообучения" станет нечестным — модель просто
подгонится под эти конкретные 598 пар вместо того, чтобы показать обобщение.
При train (AEGIS+HarmBench) ≠ eval (JailbreakBench+FORTRESS) сравнение "RQ1
baseline на JailbreakBench+FORTRESS" vs "RQ3 fine-tuned на той же
JailbreakBench+FORTRESS" становится честным тестом на generalization, и не
приходится трогать уже зафиксированные размеры (500×4) в `training_pairs`.

## Прочее

- Всё детерминировано (`seed=42`) — одинаковый запуск скрипта всегда даёт
  одинаковый датасет.
- Сами данные (`data/raw/`, `data/processed/`) не коммитятся в git — только
  скрипт, который их регенерирует (`python -m training_pairs.src.data.build_pairs`).
- Известные ограничения (задокументированы в `training_pairs/data/README.md`):
  - `paraphrase` — эвристика (TF-IDF similarity), не LLM-парафраз.
  - `jailbreak_variant`/`benign_twin` — только 114 статических человеческих
    шаблонов из HarmBench, без оптимизационных атак (GCG/PAIR/AutoDAN) —
    полный precomputed-архив HarmBench весит ~10 GB и избыточен для этой
    задачи.
  - code-domain — синтетический и маленький (написан вручную, не выкачан).
  - В `sources.py` есть загрузчик WildGuardMix (`load_wildguardmix()`), но
    он не проверен на реальных данных (gated-датасет, не было HF-токена) и
    не используется по умолчанию.

## Схема одной записи (JSONL)

```json
{
  "id": 0,
  "pair_type": "safe_harm_contrast",
  "anchor_text": "...",
  "anchor_label": "harm",
  "pair_text": "...",
  "pair_label": "safe",
  "category": "...",
  "source": "aegis"
}
```

## Как запустить

```bash
python -m training_pairs.src.data.build_pairs \
  --n-safe-harm 500 --n-paraphrase 500 --n-jailbreak 500 --n-benign-twin 500 \
  --templates-per-behavior 5 --code-templates-per-text 3 --seed 42
```

Флаг `--skip-code` отключает генерацию отдельного code-domain набора.

## Ссылки

- PR: https://github.com/embedding-level-guardrails/Embedding-level-Guardrails/pull/6
- Код: `training_pairs/src/data/build_pairs.py`, `sources.py`, `code_bank.py`
- Документация внутри репозитория: `training_pairs/data/README.md`

