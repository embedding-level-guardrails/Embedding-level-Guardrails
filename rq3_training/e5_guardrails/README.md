# RQ3 / E5: CE vs SupCon vs CE + SupCon

Как способ дообучения одного энкодера E5 влияет на бинарную классификацию запросов
(`safe = 0`, `harmful = 1`) на внутреннем hold-out и на OOD (ToxicChat).
Меняется только лосс, обновляющий энкодер. Данные, сплиты, бюджет шагов, порядок
батчей и процедура классификации и оценки одинаковы для всех режимов.

| условие | что обновляет энкодер |
|---|---|
| `base` | ничего (исходный E5) |
| `ce` | CE по бинарным меткам |
| `supcon` | SupCon по тем же бинарным меткам |
| `ce_supcon` | CE + λ·SupCon в одном шаге |

Последовательное SupCon → CE здесь не используется и совместным режимом не называется.

## Протоколы данных

Протокол задаётся через `protocol` в `conf/default.yaml`. Выходы пишутся в
`outputs/<protocol>/`, поэтому протоколы не перезаписывают друг друга.

| | `aegis_to_toxicchat` (основной) | `wildguard_to_toxicchat` |
|---|---|---|
| train | AEGIS 2.0 train минус calibration | WildGuardTrain |
| validation | официальный AEGIS validation | выделен из WildGuardTrain (10%) |
| calibration | выделен из AEGIS train (10%) | выделен из WildGuardTrain (10%) |
| hold-out | официальный AEGIS test | выделен из WildGuardTrain (10%) |
| доп. оценка | — | WildGuardTest |
| OOD | ToxicChat `toxicchat0124` test | то же |

Используются только запрос и его собственная метка: AEGIS `prompt`/`prompt_label`,
WildGuardMix `prompt`/`prompt_harm_label`, ToxicChat `user_input`/`toxicity`.
Ответы моделей, метки отказа и `jailbreaking` целью не являются. `jailbreaking` и
`adversarial` хранятся только как метаданные для разбивки результатов.
Пропущенные и неизвестные метки исключаются с подсчётом в manifest и не
восстанавливаются по категориям. Категории AEGIS сохраняются как метаданные;
«первая категория» вместо бинарной метки не используется.

**Сдвиг политики разметки.** `toxicity` в ToxicChat и harmfulness в AEGIS/WildGuardMix
размечены по разным политикам. OOD-оценка измеряет перенос модели вместе с этим
сдвигом определения, а не только сдвиг распределения текстов.

Ревизии датасетов, модели и шаблонов закреплены по коммитам в конфиге.
WildGuardMix — gated датасет на HF. Протокол `wildguard_to_toxicchat` реализован
и покрыт тестами, но в среде разработки доступа к датасету не было, поэтому на
реальных данных он не запускался.

## Предобработка и утечки (`prepare-data`)

1. Удаляются пустые тексты и заглушки (`REDACTED`, `[REDACTED]`).
2. Дедупликация по нормализованному тексту (регистр и пробелы), включая строки с
   одним промптом и разными ответами. Тексты с конфликтующими метками удаляются
   целиком. При совпадении между официальными train и validation сохраняется
   копия из validation.
3. Near-duplicate (TF-IDF по словам 1–2-граммы, cosine ≥ 0.9) объединяются в
   семейства (union-find). Семейство — единица разбиения: оно целиком попадает в
   один сплит.
4. Пересечения source-данных с каждым тестовым набором (точные и близкие)
   сохраняются в `eval_overlap_report.csv`. Правило `eval_overlap_action: drop`:
   семейства с совпадениями удаляются из train/validation/calibration, тестовые
   наборы не меняются.
5. Недостающие сплиты выделяются из train стратифицированно по
   (label, adversarial), целыми семействами, с `split_seed`.
6. Проверка: ни ID, ни семейство, ни нормализованный текст не встречаются в двух
   source-сплитах или в source-сплите и тестовом наборе.
7. Отчёт о доле запросов длиннее `max_length` токенов, отдельно для adversarial.

`manifest.json` содержит ID всех примеров по сплитам, seed, ревизии, параметры
обработки, отчёты об очистке и числа по классам.

Аугментации и пары (RQ2) строятся только из train после разбиения и
наследуют семейство якоря (см. ниже).

## Модель

`intfloat/e5-base-v2` с закреплённой ревизией: английская E5 (все датасеты
английские), 110M параметров, 768-d, лимит 512 токенов. Как в model card:
префикс `query: `, mean pooling по не-padding токенам, L2-нормализация. Модель,
ревизия, токенизатор, префикс и длина берутся из одной секции конфига.

Интерфейсы (`model.py`):

- `encode(x) -> h` — эмбеддинг E5; его видит linear probe;
- `project(h) -> z` — MLP-проекция, L2-нормированная, только для SupCon;
- `classify(h) -> logits` — линейная голова safe/harm, только для CE.

В `ce_supcon` обе головы получают один и тот же `h` за один проход энкодера.

## Обучение (`train`)

- **SupCon** (Khosla et al., 2020, L_out). Positive — другой пример того же
  safety-класса, negative — пример другого класса. Self-comparisons исключены.
  Сходство — cosine нормированных `z`, температура настраивается. Используются
  бинарные метки, а не категории. Якоря без positive исключаются из среднего и
  логируются.
- **Сэмплер.** В каждом батче `batch_size/2` safe и `batch_size/2` harmful, поэтому
  positive есть у каждого якоря. Последовательность батчей зависит только от
  seed, а не от лосса: все режимы при одном seed видят одни и те же тексты в
  одном порядке.
- **Контрастивный батч** = `training.batch_size`. Gradient accumulation нет: оно
  не расширило бы множество negatives.
- **Логи.** Компоненты совместного лосса (`ce_loss`, `supcon_loss`,
  `total_loss`) логируются отдельно, плюс статистика предъявлений (уникальные
  строки, повторы якорей, предъявления по типу строки).
- **Выбор checkpoint** одинаков для всех режимов: после каждой эпохи быстрый
  linear probe на `h` (фиксированная подвыборка train), ROC-AUC на source
  validation.
- **`sweep`** — подбор гиперпараметров с одинаковым числом проб на режим, по тому
  же критерию на validation; `--use-sweep` применяет лучшие значения.

## Классификация после обучения (`probe`)

Основное сравнение: энкодер заморожен (`eval`, `requires_grad=False`), извлекается
`h` до projection head, обучается новая `StandardScaler + LogisticRegression` на
source train. Scaler обучается только на train, C выбирается по ROC-AUC на
source validation. Процедура одна и та же для `base`, `ce`, `supcon`, `ce_supcon`.
Обучение probe не нарушает название «SupCon-only»: энкодер на этом этапе не меняется.

Дополнительные readout: `ce_head` (голова CE-режимов) и `centroid` (cosine к
средним классов на train). Они сохраняются в predictions, но в основную таблицу
не входят.

Predictions (`predictions.parquet`): `protocol, variant, model, readout, seed,
split, sample_id, label, score, adversarial`.

## Оценка (`evaluate`)

Везде positive = harmful, пример помечается при `score >= threshold`.

**A. Перенесённый рабочий порог.** Для каждой модели, readout и seed порог
выбирается один раз на source calibration: максимальный TPR при эмпирическом
FPR ≤ 1% (при равном TPR — больший порог). Затем он фиксируется и применяется к
hold-out и ToxicChat: FPR, FNR, TPR, TP/FP/TN/FN. На OOD порог заново не
подбирается, поэтому фактический OOD FPR может быть выше 1%.

**B. Тестовая ROC-метрика.** На каждом тестовом наборе отдельно — максимальный TPR
среди достижимых порогов с FPR ≤ 1%. Она использует метки теста, и её порог не
является рабочим. Кандидаты-пороги — уникальные значения score, так что
примеры с одинаковым score всегда помечаются вместе (ties не разделяются).
Вторичные метрики: ROC-AUC и average precision (ступенчатая AP из sklearn, не
трапециевидная PR-AUC).

Разбивка: `all`, а для наборов с метаданными `adversarial`/`vanilla` (у ToxicChat
это флаг `jailbreaking`).

**Неопределённость.**
- *Разброс между запусками* — mean ± std по ≥ 3 seed (`results_summary.csv`).
- *Неопределённость тестовой выборки* — парный bootstrap по примерам: одни и те же
  индексы для всех моделей, 95% интервалы для метрик и для разностей
  (модель − base, режим − режим), порог фиксирован (`bootstrap_ci.csv`). В
  тестовых наборах одно семейство = один пример, поэтому bootstrap по примерам
  совпадает с bootstrap по семействам.

Выходы в `outputs/<protocol>/results/<variant>/`: `results_per_seed.csv`,
`results_summary.csv`, `bootstrap_ci.csv`, `predictions.parquet`, `results.md`.

## Где смотреть результаты

- **Файлы** (основной источник): `outputs/<protocol>/` — сплиты и manifest,
  checkpoint, логи обучения, predictions, таблицы. В Colab ноутбук пишет их на
  Google Drive.
- **W&B**, так же как у запусков Ettin: проект `embedding-level-guardrails`,
  аккаунт по умолчанию (`wandb` в конфиге). Группа `e5-<protocol>-<variant>`,
  теги `e5`, протокол, вариант.
  - `train`-run на каждый (режим, seed): лоссы по шагам (`train/ce_loss`,
    `train/supcon_loss`, `train/total_loss`), метрики по эпохам (`epoch/...`,
    включая `selection_roc_auc`), полный конфиг, итоговая сводка и логи как
    артефакт.
  - `evaluate`-run: таблицы `results_per_seed`, `results_summary`,
    `bootstrap_ci`, сводные метрики `<model>/<test_set>/<metric>` и артефакт с
    CSV, `results.md`, `manifest.json` и отчётом о пересечениях.
  - Отключить: `--set wandb.mode=disabled`; без сети: `--set wandb.mode=offline`.

## RQ2: типы пар (после фиксации RQ3)

| тип | меняет | как |
|---|---|---|
| `jailbreak_variant` | тексты | harmful-якорь в шаблоне HarmBench (positive) |
| `benign_twin` | тексты | safe-якорь в шаблоне (positive, метка унаследована) |
| `paraphrase_candidate` | только сэмплинг | TF-IDF-кандидаты одного класса, не проверенные парафразы |
| `safe_harm_contrast` | только сэмплинг | случайные пары harmful/safe (negative) |

Шаблоны подставляются в слот `{0}` через `str.format`. Сам HarmBench склеивает
`шаблон\n\nзапрос`, оставляя `{0}` и `{{…}}` в тексте. Шаблоны без слота, с
несколькими слотами или с остаточными плейсхолдерами отклоняются явно: из 114
пригодны 110. Метка обёрнутого safe-запроса — допущение (`label_inherited`):
обёртка может добавить самостоятельное вредоносное содержание.

Варианты задаются через `augmentation.wrapper_types`, `pairs.types` и отдельное
имя `variant`. Стандартный бинарный SupCon уже считает все пары одного класса в
батче positive, а пары разных классов — negative. Поэтому удаление метаданных
пары без изменения текстов или батчей ничего не меняет: абляция действует только
через тексты (обёртки) или через совместное попадание пар в батч. Число шагов,
размер батча и баланс классов фиксированы. В `train_summary.json` пишутся число
уникальных якорей, предъявления по типу строки и повторы якорей. Синтетический
кодовый набор — отдельный эксперимент, здесь он не используется. Генератор
`training_pairs/` этими запусками не используется.

## Запуск

Зависимости: torch, transformers, scikit-learn, pandas, pyarrow, pyyaml (в Colab
уже установлены), wandb. В Colab всё запускается из
`notebooks/rq3_e5_colab.ipynb`.

```bash
cd rq3_training/e5_guardrails
export PYTHONPATH=src

python -m e5_guardrails prepare-data                    # сплиты + manifest
python -m e5_guardrails train --mode ce --seed 0        # ce | supcon | ce_supcon
python -m e5_guardrails probe --encoder base --seed 0   # base | ce | supcon | ce_supcon
python -m e5_guardrails evaluate                        # таблицы + bootstrap
python -m e5_guardrails sweep --seed 0                  # опционально, равный бюджет
python -m e5_guardrails run-all --seeds 0 1 2           # всё целиком

# другой протокол
python -m e5_guardrails --set protocol=wildguard_to_toxicchat prepare-data
```

Smoke-тест (маленькая подвыборка, отдельная папка):

```bash
python -m e5_guardrails \
  --set output_dir=outputs_smoke --set preprocess.max_rows_per_split=200 \
  --set training.epochs=1 --set training.steps_per_epoch=3 --set training.batch_size=8 \
  --set model.max_length=128 --set selection.probe_train_samples=100 --set selection.val_samples=100 \
  --set evaluation.bootstrap.n=50 \
  run-all --seeds 0
```

Тесты: `python -m pytest rq3_training/e5_guardrails/tests` (тесты на torch
используют крошечную офлайн-BERT и прогоняют весь pipeline).

## Статус

- `prepare-data` для `aegis_to_toxicchat` выполнен локально: train 19 766,
  validation 1 302, calibration 2 206, hold-out 1 914, ToxicChat 4 951
  (354 harmful).
- Smoke-тест и полные запуски на GPU ещё не выполнялись, результатов нет.
- WildGuardMix: нет доступа в среде разработки.
