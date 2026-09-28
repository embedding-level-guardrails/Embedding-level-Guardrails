# Training pairs v2 (RQ3): структура данных

Вторая версия пар для контрастивного дообучения энкодера. Скрипт —
`training_pairs/src/data/build_pairs_v2.py`, первая версия (`build_pairs.py`,
описана в `training_pairs_explained.md`) не изменена. Данные не версионируются и
генерируются локально.

## Что изменилось по сравнению с v1 и почему

| проблема v1 | решение v2 |
|---|---|
| В `jailbreak_variant` harm-запросы из HarmBench, в `benign_twin` safe-запросы из AEGIS. Стиль выдавал класс: 76% запросов HarmBench начинались с повелительного глагола, у AEGIS — 2% | Запросы **обоих** классов из AEGIS 2.0 train; из HarmBench берутся только 114 шаблонов-обёрток `HumanJailbreaks` |
| Запрос приклеивался после шаблона, плейсхолдер `{0}` оставался в тексте (он есть в 112 из 114 шаблонов) | Запрос подставляется в `{0}`; если плейсхолдера нет — дописывается в конец |
| Harm и safe обёртки выбирались независимо | Каждому harm-запросу подбирается safe-запрос той же длины и оборачивается в **те же** шаблоны |
| `safe_harm_contrast` — случайные пары unsafe × safe | Safe-промпт подбирается по наибольшему TF-IDF-сходству с unsafe (трудные негативы) |
| В якоря попадали `REDACTED`, промпты, уже похожие на jailbreak (двойная обёртка), противоречивые метки | Фильтры до сборки пар (см. ниже) |
| Категории — строка из нескольких меток, `paraphrase` группировался по строке целиком (1038 разных строк) | `primary_category` — одна категория на промпт; мелкие объединены |
| Три крупнейшие категории — больше половины unsafe | √-стратификация harm-промптов по категориям |

## Источники

- **AEGIS 2.0** (`nvidia/Aegis-AI-Content-Safety-Dataset-2.0`) — train: все запросы и
  метки. Validation и test **не используются** в парах: они нужны для val и hold-out
  в RQ3, их промпты исключены из пула.
- **HarmBench `HumanJailbreaks`** — 114 статических шаблонов-обёрток. Behaviors
  HarmBench в v2 не используются.
- **`code_bank.py`** — ручной банк описаний code-задач (12 подкатегорий, по 3
  malicious и 3 benign формулировки), как в v1.

## Фильтры пула AEGIS train

Применяются по порядку; числа — для запуска с параметрами по умолчанию.

| шаг | удалено строк |
|---|---|
| пустые и `REDACTED` | 912 |
| один и тот же промпт с разными метками | 386 |
| дубликаты промпта | 5 798 |
| промпт есть в AEGIS validation или test | 308 |
| уже похож на jailbreak (DAN, «ignore previous instructions», developer mode и т. п.) | 880 |
| короче 15 или длиннее 2000 символов | 1 124 |

Из 30 007 строк остаётся **10 695 unsafe** и **9 904 safe** промптов. Для обёрток
дополнительно берутся только промпты до 1000 символов, чтобы обёрнутый текст не
выходил далеко за длину контекста при обучении.

Шаблон «похож на jailbreak» — регулярное выражение `JAILBREAK_PATTERN`, оно
сохраняется в `manifest.json`, чтобы val и hold-out размечались тем же правилом.
`DAN` проверяется с учётом регистра, чтобы не отбрасывать имя Dan.

## Категории вреда

- **Бинарная метка:** `harm` (AEGIS `unsafe`) / `safe`. Задача обучения бинарная,
  категории вспомогательные.
- **`primary_category`** — первая категория из `violated_categories` после удаления
  `Needs Caution` (слишком расплывчатая). AEGIS перечисляет категории в порядке
  частоты отметок аннотаторов («in order of frequency of annotations», карточка
  датасета), так что первая — самая согласованная. `Needs Caution` нигде не стоит
  единственной категорией unsafe-промпта, поэтому без категории никто не остаётся.
- **Объединение мелких:** категории, в которых меньше `--min-category-prompts`
  (100) промптов пула, объединяются в `Other`. Исключение — `Malware`: в неё
  попадает code-набор. При параметрах по умолчанию в `Other` ушли Copyright/Trademark/Plagiarism,
  High Risk Gov Decision Making, Illegal Activity, Immoral/Unethical, Manipulation и Threat.
- **Полный список** исходных категорий сохраняется в `*_categories`.
- **У safe-промптов категорий нет** (`*_category = null`), даже если в AEGIS у них
  непустые `violated_categories`: категории, по-видимому, относятся ко всему
  разговору, включая ответ.

## Типы пар

### Основной набор (`processed/v2/pairs.jsonl`, 2000 пар)

| `pair_type` | anchor | pair | метки | n |
|---|---|---|---|---|
| `safe_harm_contrast` | unsafe-промпт (√-стратифицирован по категориям) | самый похожий по TF-IDF неиспользованный safe-промпт, сходство < 0.95 | harm / safe | 500 |
| `paraphrase` | промпт | похожий промпт (TF-IDF-сходство 0.35–0.95) той же категории вреда или, для safe, среди safe | одинаковые | 500 (250 harm, 250 safe) |
| `jailbreak_variant` | unsafe-промпт | он же в шаблоне HarmBench | harm / harm | 500 |
| `benign_twin` | safe-промпт той же длины, что и harm-запрос с тем же номером | он же в тех же шаблонах | safe / safe | 500 |

`jailbreak_variant` и `benign_twin` сопоставлены по порядку: i-я пара обоих списков
использует один и тот же шаблон (`template_id`). 100 harm-запросов × 5 шаблонов.

Harm-категории по типам пар:

| категория | `safe_harm_contrast` | `paraphrase` | `jailbreak_variant` |
|---|---|---|---|
| Criminal Planning/Confessions | 79 | 48 | 80 |
| Violence | 49 | 29 | 50 |
| Hate/Identity Hate | 49 | 36 | 50 |
| Harassment | 38 | 22 | 40 |
| PII/Privacy | 38 | 24 | 40 |
| Sexual | 36 | 17 | 35 |
| Profanity | 32 | 12 | 30 |
| Guns and Illegal Weapons | 27 | 14 | 30 |
| Controlled/Regulated Substances | 27 | 12 | 25 |
| Unauthorized Advice | 24 | 5 | 25 |
| Suicide and Self Harm | 23 | 5 | 20 |
| Other | 20 | 8 | 20 |
| Political/Misinformation/Conspiracy | 17 | 3 | 15 |
| Fraud/Deception | 16 | 4 | 15 |
| Sexual (minor) | 14 | 8 | 15 |
| Malware | 11 | 3 | 10 |

В `paraphrase` малые категории представлены слабее: число пар ограничено тем,
сколько похожих промптов нашлось внутри категории.

Уникальных текстов: 1 515 harm и 1 381 safe.

### Code-набор (`processed/v2/code/pairs.jsonl`, 396 пар)

| `pair_type` | что | n |
|---|---|---|
| `code_safe_harm_contrast` | malicious × benign формулировка той же подкатегории (все сочетания) | 108 |
| `code_paraphrase` | разные формулировки одной подкатегории и метки | 72 |
| `code_jailbreak_variant` | malicious-формулировка в шаблоне | 108 |
| `code_benign_twin` | параллельная benign-формулировка в **тех же** шаблонах | 108 |

Harm-сторона code-пар имеет категорию `Malware`, исходная подкатегория банка
(`keylogger`, `ransomware`, …) — в поле `subcategory`.

## Схема записи (JSONL)

```json
{
  "id": 0,
  "pair_type": "jailbreak_variant",
  "anchor_text": "...",
  "anchor_label": "harm",
  "anchor_category": "Violence",
  "anchor_categories": ["Needs Caution", "Violence"],
  "anchor_source_id": "<AEGIS id>",
  "pair_text": "<шаблон с подставленным запросом>",
  "pair_label": "harm",
  "pair_category": "Violence",
  "pair_categories": ["Needs Caution", "Violence"],
  "pair_source_id": "<AEGIS id>",
  "source": "aegis+harmbench_templates",
  "template_id": 25,
  "match_score": null,
  "subcategory": null
}
```

| поле | значение |
|---|---|
| `*_label` | `harm` / `safe` |
| `*_category` | `primary_category` после объединения; `null` у safe |
| `*_categories` | исходный список категорий AEGIS (с `Needs Caution`); `[]` у safe |
| `*_source_id` | id исходного промпта AEGIS или `code_bank:<подкатегория>:<malicious\|benign_twin>:<номер>`. У запроса и его обёрнутых версий общий — по нему группируются почти-дубликаты |
| `source` | `aegis`, `aegis+harmbench_templates`, `code_bank`, `code_bank+harmbench_templates` |
| `template_id` | номер шаблона HarmBench у обёрнутых пар, иначе `null` |
| `match_score` | TF-IDF-сходство у `safe_harm_contrast` и `paraphrase`, иначе `null` |
| `subcategory` | подкатегория code-банка у code-пар, иначе `null` |

`manifest.json` рядом с парами содержит параметры запуска, статистику фильтров
(`filters`), карту объединения категорий (`category_map`), `dropped_category`,
`other_category`, `jailbreak_pattern`, число шаблонов и распределения пар по типам и
категориям. Блокнот RQ3 читает из него правила разметки категорий и jailbreak-подобных
промптов.

## Проверки на сгенерированных данных

- Подсказки по стилю нет: у harm- и safe-запросов в обёрнутых парах повелительное
  начало 9% и 5%, вопросительный знак в конце 43% и 44%, одинаковое распределение
  длин (медиана 74 символа).
- У каждой пары `jailbreak_variant` / `benign_twin` одинаковый шаблон; `{0}` не
  остался ни в одном тексте.
- 0 текстов пересекается с AEGIS validation и test; 0 jailbreak-подобных промптов без
  обёртки; 0 текстов с противоречивыми метками.
- Медианное TF-IDF-сходство в `safe_harm_contrast` — 0.37 (10-й перцентиль 0.22).

Тесты: `training_pairs/tests/test_build_pairs_v2.py` (без сети, на синтетических данных).

## Известные ограничения

- Метки AEGIS шумные, часть unsafe-промптов пограничная; категории описывают
  разговор, а не только промпт.
- Фильтр jailbreak-подобных промптов — регулярное выражение, он пропускает
  нестандартные формулировки и может задеть безобидные («uncensored version of a movie»).
- Среди самых похожих `safe_harm_contrast` встречаются почти одинаковые
  шаблоны из библиотек промптов («You are a keyword specialist…»), которые AEGIS
  разметил по-разному. Это трудные негативы, но часть из них может быть шумом разметки.
- `paraphrase` — лексическое сходство TF-IDF, а не переформулировки от LLM.
- Обёртки — только 114 статических человеческих шаблонов, без оптимизационных атак
  (GCG, PAIR, AutoDAN).
- Code-набор синтетический и маленький (36 malicious + 36 benign формулировок).
- Файлы `data/raw/code_seeds.parquet` и `data/processed/code/code_benign_twins.parquet`
  (WildJailbreak / WildGuardMix) в v2 не используются.

## Как запустить

Из корня репозитория:

```bash
python -m training_pairs.src.data.build_pairs_v2
```

Параметры (значения по умолчанию):

```bash
python -m training_pairs.src.data.build_pairs_v2 \
  --n-safe-harm 500 --n-paraphrase 500 --n-wrapped 500 \
  --templates-per-prompt 5 --code-templates-per-text 3 \
  --min-category-prompts 100 --max-wrapped-prompt-chars 1000 --seed 42
```

`--n-wrapped` задаёт размер и `jailbreak_variant`, и `benign_twin` — они сопоставлены
попарно. `--skip-code` отключает code-набор. Результат пишется в
`training_pairs/data/processed/v2/`, данные v1 в `processed/` не перезаписываются.
Кэш исходников — `training_pairs/data/raw/` (AEGIS train, validation, test и шаблоны
HarmBench скачиваются при первом запуске).
