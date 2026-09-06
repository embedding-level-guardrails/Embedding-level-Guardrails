# Embedding-level-Guardrails

Embedding-level Guardrails: обучение энкодера для safe / harm separation.

Компактный энкодер как быстрый предварительный фильтр перед LLM: проверяем,
разделяются ли безопасные и вредоносные запросы геометрически уже на уровне
эмбеддингов, и можно ли усилить это разделение contrastive-дообучением.

## Что сделано

| Этап | Скрипты | Результат |
|:--|:--|:--|
| **RQ1** — разделимость на frozen-энкодерах | `00`–`06` | [results/rq1/REPORT.md](results/rq1/REPORT.md) |
| **RQ2** — сборка обучающих пар | `07_build_pairs.py` | [results/rq2/PAIRS.md](results/rq2/PAIRS.md) |
| **RQ3** — три objective + LODO | `08_train_contrastive.py`, `09_eval_lodo.py` | [results/rq3/LODO.md](results/rq3/LODO.md) |

Энкодеры: `jhu-clsp/ettin-encoder-68m`, `jhu-clsp/mmBERT-small`, `intfloat/e5-small-v2`.
Данные: `nvidia/Aegis-AI-Content-Safety-Dataset-1.0` + HarmBench (behaviors).
OOD для LODO: `lmsys/toxic-chat` (реальный трафик чат-бота).

## Быстрый старт

```bash
make setup           # зависимости
make data            # AEGIS -> data/processed/
make ood-data        # ToxicChat -> data/processed/ (OOD для LODO)
make embed           # frozen-эмбеддинги -> artifacts/embeddings/
make rq1             # косинусы, пробы, проекции, error analysis, отчёт
make pairs           # обучающие пары -> data/processed/pairs/
make rq3             # три objective подряд + LODO-оценка
make lodo            # только оценка уже обученных прогонов
make mlflow          # UI трекинга на http://localhost:5000
```

Офлайн-проверка пайплайна без сети и без реальных моделей: `make smoke` (RQ1),
`make smoke-rq3` (пары + пара шагов обучения). Тесты: `make test`.

## Устройство

```
src/eguard/
  data/         загрузка AEGIS, агрегация разметки 5 аннотаторов, caution_policy
  encoders/     frozen HF-энкодеры + dummy для офлайн-прогона
  analysis/     косинусы, геометрия, пробы, контрольные baseline-ы
  pairs/        источники (AEGIS + HarmBench), трансформации, строители пар
  training/     лоссы, цикл дообучения, обёртка MLflow
  viz/          PCA / t-SNE / UMAP
```

## RQ2 — типы пар

Формат — триплет `(anchor, positive, negative)` с меткой `pair_type`, поэтому
абляция «сколько даёт каждый тип» сводится к фильтру:
`08_train_contrastive.py --pair-types safe_harm_contrast jailbreak_variant`.

| Тип | Что учит |
|:--|:--|
| `safe_harm_contrast` | базовый контраст класса; половина пар — от safe-якоря, чтобы моделировалось и безопасное многообразие |
| `paraphrase` | инвариантность к форме запроса |
| `jailbreak_variant` | обёртка не делает вредоносный запрос безопасным |
| `benign_twin` | hard negatives: лексически ближайший safe, намайненный по косинусу |

Ключевая деталь `jailbreak_variant`: половина негативов — **безопасный** текст в
той же самой обёртке. Иначе модель выучит саму обёртку как признак вреда и начнёт
блокировать безобидные запросы, оформленные похоже.

HarmBench разделён по официальным сплитам: `val` (80 behaviors) идёт в обучение,
`test` (320) не участвует нигде и лежит в `harmbench_heldout.jsonl` для OOD-оценки.

## RQ3 — контролируемое сравнение трёх objective

Три варианта поверх одного энкодера и **одних и тех же текстов**
(`texts_*.jsonl` рядом с парами) — иначе сравнение объективов не контролируемое:

```bash
python scripts/08_train_contrastive.py --objective classification   # (а) только CE
python scripts/08_train_contrastive.py --objective contrastive      # (б) только contrastive
python scripts/08_train_contrastive.py --objective joint            # (в) CE + contrastive
python scripts/09_eval_lodo.py                                      # сравнение
```

Лучший чекпоинт выбирается по **TPR@FPR=1%**, а не по лоссу: для guardrail важна
рабочая точка с низким FPR, и средний лосс её не отражает. Чекпоинт сохраняется в
формате `AutoModel`, поэтому его можно дописать в `configs/rq1.yaml` ещё одним
энкодером и прогнать на нём весь RQ1-анализ.

### Как меряется

Две методические вещи, без которых сравнение развалилось бы.

**Единый readout.** Три objective дают три разных пространства, и у чисто
contrastive-варианта своей головы нет вообще. Поэтому основная таблица —
`probe_logreg`: поверх эмбеддингов каждой модели обучается один и тот же линейный
проб на AEGIS train. Различается только пространство. Родная голова и центроидный
скор (прообраз RQ5) идут дополнительными строками.

**Порог с val, а не с теста.** FPR/FNR репортятся при пороге, выставленном на
AEGIS val при FPR=1%, и этот же порог без перекалибровки применяется к hold-out и
к OOD. Подбирать порог на OOD означало бы заглянуть в тестовые данные; деградация
рабочей точки при переносе — это как раз то, что нужно увидеть. Рядом лежат AUROC
и TPR@FPR=1%, посчитанные на каждом наборе отдельно и от порога не зависящие.

**LODO.** Обучение на AEGIS, оценка на ToxicChat: 2853 записи реального трафика с
harm_rate 0.127 против 0.597 у AEGIS. Сдвиг сильный, поэтому AUROC на OOD говорит
мало, а FPR при зафиксированном пороге — много.

Трекинг — MLflow (локально sqlite: `file:./mlruns` в MLflow 3.x отключён).
Метрики LODO дозаписываются в тот же прогон, где училась модель. Выключается
флагом `--no-mlflow`.

## Участники

- Кайгородцева Дарья Андреевна
- Гуминов Дмитрий Андреевич
- Байшев Олег Михайлович
- Никитченко Мария Владиславовна
- Емцова Анна Сергеевна
