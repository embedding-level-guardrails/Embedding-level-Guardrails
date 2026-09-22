# Embedding-level Guardrails — описание проделанной работы

Компактный энкодер как быстрый предварительный фильтр перед LLM. Проверяем,
разделяются ли безопасные и вредоносные запросы геометрически уже на уровне
эмбеддингов, и можно ли усилить это разделение contrastive-дообучением.

**Энкодеры:** `jhu-clsp/ettin-encoder-68m`, `jhu-clsp/mmBERT-small`, `intfloat/e5-small-v2`.
**Данные:** AEGIS (`nvidia/Aegis-AI-Content-Safety-Dataset-1.0`) + HarmBench (behaviors).
**OOD для LODO:** ToxicChat (`lmsys/toxic-chat`, конфиг `toxicchat0124`).

Обозначения ниже: `x` — эмбеддинг текста, `y ∈ {0,1}` — метка (0 = safe, 1 = harm),
`μ_safe`, `μ_harm` — центроиды классов, `α = 0.01` — целевой FPR.

---

## RQ1 — насколько базовые энкодеры разделяют safe/harm без дообучения

**Статус:** выполнено, отчёт в [results/rq1/REPORT.md](../results/rq1/REPORT.md).

### Что сделано

Замеряется разделимость на трёх замороженных энкодерах по четырём осям: косинусная
близость, геометрия пространства, качество пробов и error analysis. Контрольные
baseline-ы (TF-IDF и длина текста) задают нижнюю границу — без них нельзя
утверждать, что разделимость семантическая, а не лексическая.

### Формулы

Косинусная близость и разрыв между внутри- и межклассовой близостью:

```
cos(u, v) = (u · v) / (‖u‖ · ‖v‖)

gap = ½·(mean cos_intra_safe + mean cos_intra_harm) − mean cos_inter

gap_std = gap / sqrt(½·(var_intra_safe + var_intra_harm))
```

`gap_std` нужен потому, что сырой `gap` несопоставим между моделями с разной
анизотропией. Анизотропия меряется средним косинусом случайных пар: у трансформеров
все векторы смещены в общий конус, поэтому любые два текста уже похожи. Вариант
`centered` — те же величины после вычитания глобального среднего.

Единое «harm-направление» (оценивается на train, применяется на eval):

```
d = μ_harm − μ_safe,   d̂ = d / ‖d‖,   score(x) = x · d̂
Cohen's d = (mean(score | y=1) − mean(score | y=0)) / sqrt(½·(var₁ + var₀))
```

Сравнение `AUROC(score)` с AUROC полноразмерного линейного проба показывает,
одномерен ли сигнал безопасности.

Рабочая точка guardrail — главная метрика проекта:

```
TPR@FPR=α  =  max { TPR(τ) : FPR(τ) ≤ α },  α = 0.01
```

Скор distance-based проба (прообраз RQ5):

```
s(x) = cos(x, μ̂_harm) − cos(x, μ̂_safe),  где μ̂ = L2-нормированный центроид
```

Кластерные метрики: silhouette по косинусу, Davies–Bouldin, ARI и NMI между
KMeans-разбиением и меткой безопасности (k ∈ {2, 8}). Доверительные интервалы —
перцентильный стратифицированный bootstrap, 1000 итераций.

### Код

| Файл | Роль |
|:--|:--|
| [src/eguard/data/aegis.py](../src/eguard/data/aegis.py) | загрузка AEGIS, агрегация разметки 5 аннотаторов |
| [src/eguard/encoders/](../src/eguard/encoders) | frozen HF-энкодеры, пулинг, dummy для офлайна |
| [src/eguard/embeddings.py](../src/eguard/embeddings.py) | кеш эмбеддингов на диск |
| [src/eguard/analysis/similarity.py](../src/eguard/analysis/similarity.py) | intra/inter косинусы, поправка на анизотропию |
| [src/eguard/analysis/geometry.py](../src/eguard/analysis/geometry.py) | анизотропия, harm-направление, кластеры |
| [src/eguard/analysis/probe.py](../src/eguard/analysis/probe.py) | logreg / kNN / centroid пробы, метрики, bootstrap |
| [src/eguard/analysis/baselines.py](../src/eguard/analysis/baselines.py) | контроли TF-IDF и длина текста |
| [src/eguard/viz/plots.py](../src/eguard/viz/plots.py) | PCA / t-SNE / UMAP, гистограммы косинусов |
| `scripts/00`…`06` | шаги пайплайна |
| [configs/rq1.yaml](../configs/rq1.yaml) | конфигурация |

### Как запустить

```bash
make data          # AEGIS -> data/processed/
make embed         # эмбеддинги трёх энкодеров -> artifacts/embeddings/
make rq1           # шаги 02..06 + сборка отчёта
```

Или пошагово: `python scripts/02_rq1_similarity.py --config configs/rq1.yaml` и далее
`03_rq1_probe.py`, `04_rq1_viz.py`, `06_error_analysis.py`, `05_report.py`.

### Результаты

Пробы дают AUROC 0.91–0.94, но рабочая точка плохая: **TPR@FPR=1% всего 0.21–0.41**.
Сырой `gap` 0.005–0.01 при среднем косинусе случайных пар 0.78–0.95 — то есть без
поправки на анизотропию косинус почти ничего не показывает; после центрирования
разрыв растёт в 5–15 раз. Безнадзорные кластеры с меткой не совпадают (ARI@k=2
0.09–0.15). Контроли: TF-IDF 0.886, длина 0.767 — эмбеддинги выигрывают, но у
`ettin` и `mmbert` отрыв от чистой лексики невелик.

### Известное ограничение данных

В AEGIS есть повторяющиеся тексты, в том числе через официальную границу train/test:
2318 строк train при 2169 уникальных текстах, 21 текст общий у train и test (метки у
дублей согласованы). Опубликованный `results/rq1/REPORT.md` из-за этого слегка
завышен — около 7% тестовых примеров присутствуют в обучении пробы. В сборке пар
(RQ2) это вычищено, в RQ1 — нет.

---

## RQ2 — какие пары нужны для обучения

**Статус:** выполнено, сводка в [results/rq2/PAIRS.md](../results/rq2/PAIRS.md).

### Что сделано

Собраны обучающие пары из AEGIS + HarmBench в формате триплета
`(anchor, positive, negative)` с тегом `pair_type`. Триплет выбран потому, что он
одинаково ложится и на InfoNCE с in-batch negatives, и на triplet margin, и на SupCon,
а абляция «какой тип пар сколько даёт» сводится к фильтру по полю `pair_type`.

Четыре типа пар:

| Тип | Что учит | Как строится |
|:--|:--|:--|
| `safe_harm_contrast` | базовый контраст класса | positive — другой пример того же класса, по возможности той же категории вреда; negative — случайный пример другого класса |
| `paraphrase` | инвариантность к форме запроса | positive — поверхностный парафраз якоря |
| `jailbreak_variant` | обёртка не делает вредоносный запрос безопасным | positive — behavior HarmBench в jailbreak-обёртке |
| `benign_twin` | hard negatives | negative — лексически ближайший safe, намайненный по косинусу |

Половина пар строится от safe-якоря (`safe_anchor_fraction: 0.5`): если тянуть только
harm, обучение стягивает вредоносный кластер и ничего не говорит о структуре безопасной
области, а именно её размытость даёт ложные пропуски.

Майнинг benign twins формально:

```
negative(a) = argmax_{j : y_j = 0} cos(x_a, x_j)
positive(a) = argmax_{j ≠ a, y_j = 1} cos(x_a, x_j)
```

Эмбеддинги берутся с замороженного `e5-small-v2` из RQ1.

### Два решения, которые определяют качество набора

**Негатив к jailbreak-варианту.** По умолчанию `jailbreak_negative: mix` — половина
негативов это **безопасный** текст в той же самой обёртке. Если негативом всегда брать
обычный safe, модель выучит саму обёртку («ignore previous instructions», ролевую рамку)
как признак вреда и начнёт блокировать безобидные запросы в такой же форме. При `mix`
единственный различающий сигнал остаётся содержательным.

**Плоский набор текстов рядом с триплетами.** Каждый прогон сборки пишет не только
`pairs_{split}.jsonl`, но и `texts_{split}.jsonl` — все тексты из пар с бинарной меткой,
без повторов. Без этого RQ3 не поставить: сравнение contrastive против classification
head требует, чтобы у обеих веток были ровно одни и те же обучающие тексты, включая
парафразы и jailbreak-варианты.

### Отсутствие утечки между сплитами

AEGIS содержит повторяющиеся тексты, пересекающие границу train/test, поэтому пулы
чистятся до сборки: сначала дедупликация внутри сплита, затем вычитание текстов более
приоритетных сплитов. Приоритет `test > val > train` — оценочный сплит не жертвуем ничем.
Отдельно фильтруются записи для майнинга benign twins: матрица эмбеддингов лежит
построчно к исходному jsonl и в обход пулов протаскивала дубликаты. Проверено:
пересечение уникальных текстов между всеми тремя сплитами равно нулю.

### Код

| Файл | Роль |
|:--|:--|
| [src/eguard/pairs/sources.py](../src/eguard/pairs/sources.py) | загрузка HarmBench (CSV с GitHub, без HF-токена) и AEGIS |
| [src/eguard/pairs/transforms.py](../src/eguard/pairs/transforms.py) | парафразы и jailbreak-обёртки |
| [src/eguard/pairs/builders.py](../src/eguard/pairs/builders.py) | четыре строителя пар, дедупликация, `flatten_texts` |
| [src/eguard/pairs/__init__.py](../src/eguard/pairs/__init__.py) | оркестрация, фильтры пулов, пути на диске |
| [scripts/07_build_pairs.py](../scripts/07_build_pairs.py) | CLI, сводка размеров |
| [configs/rq2_pairs.yaml](../configs/rq2_pairs.yaml) | зафиксированные размеры и параметры |

### Как запустить

```bash
make pairs
```

Варианты: `--types safe_harm_contrast jailbreak_variant` (подмножество),
`--scale 0.05` (быстрый прогон на срезе), `--splits train` (один сплит).

### Зафиксированные размеры (набор `aegis_harmbench_v2`)

| split | пар | уник. текстов | harm_rate | contrast | paraphrase | jailbreak | benign_twin |
|:--|--:|--:|--:|--:|--:|--:|--:|
| train | 16 738 | 9 717 | 0.572 | 7998 | 3543 | 3997 | 1200 |
| val | 1 709 | 1 127 | 0.599 | 800 | 363 | 400 | 146 |
| test | 1 712 | 1 162 | 0.585 | 800 | 364 | 400 | 148 |

В обучение идут все **400** behaviors HarmBench (5372 пары с HarmBench-якорем, 3051
уникальный текст). Ранее использовался только официальный `val` (80 behaviors), а `test`
(320) лежал в холде как кандидат в OOD-набор; после переезда LODO на ToxicChat резерв
стал не нужен — на 80 затравках каждая переиспользовалась в ~54 парах. Вернуть холд:
`train_split: val`, `heldout_split: test` в конфиге.

### Ограничение

`paraphrase_*` — **поверхностные** парафразы (обёртки вежливости, смена наклонения,
лёгкий шум ввода), а не семантические перефразировки уровня LLM. Они учат инвариантности
к форме запроса, но не покрывают перефразировку смысла. Крючок для полноценных парафраз
есть: `external_variants` принимает jsonl `{"id": ..., "variants": [...]}`, и строители
берут варианты оттуда. Короткие реплики диалога в якоря парафраз не попадают
(`paraphrase_min_chars: 25`) — иначе шаблоны дают бессмыслицу.

Тип пар `code_safe_harm` из формулировки RQ2 не реализован: под него не выбран датасет.

---

## RQ3 — улучшает ли contrastive objective FPR/FNR по сравнению с classification head

**Статус:** выполнено, отчёт в [results/rq3/LODO.md](../results/rq3/LODO.md). Это ядро проекта.

### Что сделано

Три варианта дообучения `e5-small-v2` поверх одного энкодера и **одних и тех же
обучающих текстов**, различается только objective:

- **(а)** `classification` — только CE поверх линейной головы (базлайн),
- **(б)** `contrastive` — только contrastive-лосс на триплетах,
- **(в)** `joint` — CE + contrastive.

Оценка на hold-out (AEGIS test) и OOD (ToxicChat) по схеме LODO: обучались на AEGIS,
тестируем на реальном трафике чат-бота.

### Формулы

**InfoNCE** с in-batch negatives и явным hard negative. Кандидаты для якоря `i` — все
позитивы батча плюс столбец явных негативов; правильный ответ — позитив с тем же индексом:

```
L_InfoNCE = −(1/N) · Σᵢ log [ exp(âᵢ·p̂ᵢ / τ) / Σⱼ exp(âᵢ·ĉⱼ / τ) ]

где ĉ = [p̂₁ … p̂_N, n̂₁ … n̂_N],  ·̂ = L2-нормировка,  τ = temperature (0.05)
```

**Supervised Contrastive** (Khosla et al.) — позитивы это все примеры того же класса в
батче; стягивает класс целиком, а не пару:

```
L_SupCon = −Σᵢ (1 / |P(i)|) · Σ_{p ∈ P(i)} log [ exp(zᵢ·z_p / τ) / Σ_{k ≠ i} exp(zᵢ·z_k / τ) ]

P(i) = { k ≠ i : y_k = yᵢ }
```

**Triplet margin** (косинусный, нижняя граница):

```
L_triplet = mean max(0, m − cos(a, p) + cos(a, n)),  m = margin (0.2)
```

**KL-якорь к базовой модели** (задел под RQ4): требует, чтобы матрица попарных сходств
внутри батча оставалась близкой к матрице замороженного энкодера, не фиксируя сами векторы:

```
L_KL = KL( softmax(Z_teacher · Z_teacherᵀ / τ) ‖ softmax(Z_student · Z_studentᵀ / τ) )
```

**Итоговый лосс:**

```
L = λ_contrastive · L_contrastive + λ_CE · L_CE + λ_KL · L_KL
```

**Расписание LR** — линейный warmup, затем косинусное затухание:

```
lr(t) = lr₀ · t/W                                    при t < W
lr(t) = lr₀ · ½·(1 + cos(π · (t−W)/(T−W)))           при t ≥ W
```

### Две методические вещи, без которых сравнение развалилось бы

**Единый readout.** Три objective дают три разных пространства, и у чисто
contrastive-варианта своей головы нет вообще — «каждый мерится своей головой» некорректно.
Поэтому основная таблица `probe_logreg`: поверх эмбеддингов каждой модели обучается **один
и тот же** линейный проб на AEGIS train. Различается только пространство. Родная голова и
центроидный скор идут дополнительными строками.

Это не формальность: на валидации во время обучения CE выглядел лучше contrastive
(TPR@FPR 0.736 против 0.642), но там CE мерился своей головой, а contrastive — центроидным
расстоянием. При едином пробе порядок переворачивается.

**Порог с val, а не с теста.** FPR/FNR репортятся при пороге, выставленном на AEGIS val
при FPR=1%, и этот же порог без перекалибровки применяется к hold-out и к OOD:

```
τ* = threshold at FPR = 1% on AEGIS val
FPR/FNR(holdout) и FPR/FNR(ood) считаются при том же τ*
```

Подбирать порог на OOD означало бы заглянуть в тестовые данные. Рядом лежат AUROC и
TPR@FPR=1%, посчитанные на каждом наборе отдельно и от порога не зависящие. Проб для
переноса (`fit_transfer_probe`) обучается **только** на train, в отличие от `logistic_probe`
из RQ1, который в конце дообучается на train+val — иначе val непригоден для калибровки.

### Код

| Файл | Роль |
|:--|:--|
| [src/eguard/training/losses.py](../src/eguard/training/losses.py) | четыре лосса |
| [src/eguard/training/model.py](../src/eguard/training/model.py) | обучаемый энкодер + голова, сохранение и загрузка чекпоинтов |
| [src/eguard/training/data.py](../src/eguard/training/data.py) | датасеты и коллаторы для обеих веток |
| [src/eguard/training/loop.py](../src/eguard/training/loop.py) | цикл обучения, валидация, выбор чекпоинта |
| [src/eguard/training/tracking.py](../src/eguard/training/tracking.py) | обёртка над MLflow |
| [src/eguard/data/toxicchat.py](../src/eguard/data/toxicchat.py) | OOD-набор |
| [src/eguard/analysis/probe.py](../src/eguard/analysis/probe.py) | `fit_transfer_probe` для LODO |
| [scripts/08_train_contrastive.py](../scripts/08_train_contrastive.py) | обучение |
| [scripts/09_eval_lodo.py](../scripts/09_eval_lodo.py) | сравнение hold-out против OOD |
| [configs/rq3_contrastive.yaml](../configs/rq3_contrastive.yaml), [configs/ood_toxicchat.yaml](../configs/ood_toxicchat.yaml) | конфигурации |

Лучший чекпоинт выбирается по **TPR@FPR=1%**, а не по лоссу: средний лосс не отражает
рабочую точку с низким FPR. Чекпоинт сохраняется в формате `AutoModel`, поэтому его можно
дописать в `configs/rq1.yaml` ещё одним энкодером и прогнать на нём весь RQ1-анализ.

### Как запустить

```bash
make ood-data      # ToxicChat -> data/processed/ (один раз)
make rq3           # три objective подряд + LODO-оценка, ~50 минут на MPS
make lodo          # только оценка уже обученных прогонов
make mlflow        # UI трекинга на http://127.0.0.1:5000 (отдельный терминал)
```

Поштучно:

```bash
python scripts/08_train_contrastive.py --config configs/rq3_contrastive.yaml --objective classification --run-name rq3-classification
python scripts/08_train_contrastive.py --config configs/rq3_contrastive.yaml --objective contrastive    --run-name rq3-contrastive
python scripts/08_train_contrastive.py --config configs/rq3_contrastive.yaml --objective joint          --run-name rq3-joint
python scripts/09_eval_lodo.py --config configs/rq3_contrastive.yaml --ood-config configs/ood_toxicchat.yaml
```

Абляция RQ2 (вклад каждого типа пар при прочих равных): `make rq3-ablation` или
`--pair-types safe_harm_contrast jailbreak_variant`.

### Результаты (единый проб, порог с AEGIS val при FPR=1%)

Hold-out: AEGIS test, n=308, harm_rate 0.597. OOD: ToxicChat, n=2853, harm_rate 0.127.

| вариант | набор | AUROC | TPR@FPR=1% | FPR | FNR |
|:--|:--|--:|--:|--:|--:|
| e5 frozen | holdout | 0.936 | 0.364 | 0.040 | 0.370 |
| (а) CE | holdout | 0.954 | 0.495 | 0.048 | 0.245 |
| (б) contrastive | holdout | 0.960 | 0.598 | 0.057 | 0.207 |
| (в) CE+contrastive | holdout | 0.960 | 0.554 | 0.040 | 0.272 |
| e5 frozen | **ood** | 0.872 | 0.191 | 0.063 | 0.525 |
| (а) CE | **ood** | 0.815 | 0.028 | 0.344 | 0.160 |
| (б) contrastive | **ood** | 0.849 | 0.135 | 0.357 | 0.116 |
| (в) CE+contrastive | **ood** | 0.850 | 0.152 | 0.211 | 0.240 |

**На hold-out различий между тремя вариантами нет.** Это главная оговорка, и её нельзя
опускать в отчёте: доверительные интервалы TPR@FPR=1% огромные (CE 0.239–0.745,
contrastive 0.533–0.761, joint 0.489–0.772), потому что в hold-out всего 308 примеров и
метрика при FPR=1% опирается на считанные точки в хвосте. Порядок «contrastive > joint > CE»
держится по точечным оценкам, но статистически не отделим. Нужны несколько seed'ов на
вариант или больший hold-out.

**На OOD различия значимые** (n=2853, интервалы узкие) и картина обратная: CE переносится
хуже всех — TPR@FPR=1% падает до 0.028 [0.011, 0.053] против 0.191 [0.141, 0.251] у
замороженного энкодера, интервалы не пересекаются. Дообучение на AEGIS улучшает in-domain и
портит перенос; contrastive этот ущерб частично компенсирует.

**Порог не переносится.** У всех дообученных вариантов FPR на ToxicChat взлетает до 21–36%
при пороге, откалиброванном на AEGIS. Joint деградирует мягче (0.211). Часть эффекта —
арифметика базовой ставки (harm_rate 0.597 → 0.127), но не вся: у замороженного энкодера при
том же сдвиге FPR остаётся 0.063.

### Важно про соответствие результатов и данных

Числа в `results/rq3/` посчитаны на **предыдущей** версии пар (`aegis_harmbench_v1`, 80
behaviors HarmBench). После этого набор пересобран в `aegis_harmbench_v2` с 400 behaviors,
но переобучение не запускалось. `aegis_harmbench_v1` оставлен на диске, чтобы старые
результаты были воспроизводимы; `configs/rq3_contrastive.yaml` уже указывает на `v2`.

При переобучении на `v2` стоит учесть: `harm_rate` плоского набора текстов вырос с 0.465 до
0.572 (HarmBench состоит только из вредоносных behaviors). На contrastive-ветку это почти не
влияет, а CE-ветка получит заметно более несбалансированные данные — веса классов в CE сейчас
не выставляются.

---

## RQ4 — не ломает ли safety-separation общую семантическую структуру

**Статус:** заготовлено, не прогонялось.

Реализован KL-якорь к замороженной базовой модели (`embedding_kl_regularizer` в
[losses.py](../src/eguard/training/losses.py), параметры `kl_weight` и `kl_temperature` в
конфиге, `frozen_teacher` в [model.py](../src/eguard/training/model.py)). Включается
выставлением `kl_weight > 0`. Прогонов с ним не делалось, downstream-оценка семантической
полезности не реализована.

Часть инструментов для ответа уже есть в RQ1: `anisotropy`, `direction_analysis` и
`cluster_analysis` из [geometry.py](../src/eguard/analysis/geometry.py) применимы к
дообученному чекпоинту — достаточно дописать его в `configs/rq1.yaml` как энкодер и
прогнать `02_rq1_similarity.py`.

---

## RQ5 — расстояние до unsafe-кластеров как confidence score

**Статус:** побочный, но содержательный результат из LODO-оценки.

Скор `s(x) = cos(x, μ̂_harm) − cos(x, μ̂_safe)` считается для всех моделей как отдельный
readout (`centroid_distance` в [09_eval_lodo.py](../scripts/09_eval_lodo.py)).

| пространство | OOD AUROC | OOD FPR при τ* |
|:--|--:|--:|
| e5 frozen | 0.668 | 0.003 |
| после contrastive | **0.892** [0.876, 0.908] | **0.089** |

На замороженном энкодере distance-based скор на OOD почти не работает. После
contrastive-дообучения тот же самый скор даёт лучший OOD-результат среди всех
комбинаций — выше, чем линейный проб на замороженной базе (0.872), и порог при этом
переносится (0.089 против 0.357 у пробы на том же пространстве).

То есть contrastive-обучение не столько улучшает классификацию, сколько перестраивает
геометрию так, что начинает работать distance-based скоринг — ровно гипотеза RQ5.
Вероятностная калибровка скора (чего, по литобзору, не делают) пока не проверялась.

---

## RQ6 — внутренние представления LLM как teacher signal

**Статус:** не начат.

---

## Инфраструктура и качество

**Тесты:** 37 штук, запуск `make test`. Покрывают разметку AEGIS и ToxicChat, логику
сборки пар и отсутствие утечки, лоссы и их градиенты, калибровку порога, обёртку MLflow.
Сети и GPU не требуют (torch нужен только для `test_training.py`).

**Офлайн-проверка пайплайна:** `make smoke` (RQ1 на синтетике и хеш-энкодере, без сети и
torch) и `make smoke-rq3` (маленькие пары + пара шагов обучения + LODO). Синтетика
намеренно не линейно разделима: классы разводятся статистикой словаря, часть словаря общая,
поэтому пробы дают 0.6–0.97, контроль по длине около 0.5, а список ошибок непустой — иначе
смоук проходил бы всегда и не ловил регрессии.

**Трекинг:** MLflow, локально на sqlite (`mlflow.db`). Файловый бэкенд `file:./mlruns` в
MLflow 3.x переведён в maintenance mode и падает без `MLFLOW_ALLOW_FILE_STORE=true`.
Метрики пишутся по ходу обучения (`log_every: 20` шагов, `eval_every: 200`), поэтому
прогресс виден в UI в реальном времени. Метрики LODO дозаписываются в тот же прогон, где
училась модель. Отключается флагом `--no-mlflow`.

---

# Структура проекта

## Директории

| Путь | Назначение |
|:--|:--|
| `src/eguard/` | библиотека: всё переиспользуемое |
| `scripts/` | шаги пайплайна, пронумерованы по порядку выполнения |
| `configs/` | YAML-конфигурации экспериментов |
| `tests/` | pytest, без сети и GPU |
| `docs/` | этот документ |
| `results/` | версионируемые результаты: CSV и markdown-отчёты |
| `data/` | не версионируется: сырые и нормализованные данные, пары |
| `artifacts/` | не версионируется: кеш эмбеддингов, чекпоинты прогонов |
| `mlruns/`, `mlartifacts/`, `mlflow.db` | не версионируется: хранилище MLflow |

Раскладка данных на диске:

```
data/raw/harmbench/                                 кеш CSV с behaviors
data/processed/{dataset}/{split}.jsonl              нормализованные записи
data/processed/{dataset}/summary.json               статистика сплитов
data/processed/pairs/{name}/pairs_{split}.jsonl     триплеты
data/processed/pairs/{name}/texts_{split}.jsonl     те же тексты плоско, с меткой
artifacts/embeddings/{dataset}/{encoder}/{split}.npy
artifacts/runs/{run_name}/checkpoint/               дообученная модель
```

Порядок строк в `{split}.jsonl` — это порядок строк в кеше эмбеддингов; менять его нельзя
без пересчёта (проверка длин стоит в `load_xy`).

## src/eguard — библиотека

### Корень

**[config.py](../src/eguard/config.py)** — загрузка YAML в дата-классы. Используется всеми
скриптами.
- `EncoderSpec`, `DatasetSpec`, `Paths`, `Config` — дата-классы секций конфига.
  `DatasetSpec` несёт и общие поля (`text_types`, `min_chars`), и специфичные для
  отдельных датасетов (`config`, `label_field`, `require_human_annotation` — для ToxicChat).
- `Config.encoder(key)` — спецификация энкодера по ключу, с понятной ошибкой при опечатке.
- `Config.select_encoders(keys)` — подмножество энкодеров или все.
- `load_config(path)` — YAML в `Config`.

**[utils.py](../src/eguard/utils.py)** — мелкие утилиты, общие для всех шагов.
- `get_logger`, `set_seed` (random / numpy / torch), `ensure_dir`.
- `write_jsonl`, `read_jsonl`, `save_json` — ввод-вывод.
- `l2_normalize(x)` — построчная L2-нормировка с защитой от нуля.
- `subsample_indices(n, k, rng)` — k индексов без повторов или все, если `n ≤ k`.
- `batched(seq, size)` — нарезка на батчи.

**[embeddings.py](../src/eguard/embeddings.py)** — кеш эмбеддингов на диске.
- `emb_dir`, `emb_paths` — пути к `.npy`, `.scores.npy`, `.meta.json`.
- `compute_and_cache(...)` — считает и сохраняет, пропускает готовое без `overwrite`.
- `load_embeddings` — чтение матрицы, скоров головы и метаданных.
- `load_xy(cfg, encoder_key, split)` — `(X, y, records, scores)` со сверкой длин: если
  данные пересобрали после эмбеддинга, падает с внятной ошибкой вместо тихого сдвига.
- `available_encoders` — какие энкодеры уже посчитаны.

### data/ — датасеты

**[data/\_\_init\_\_.py](../src/eguard/data/__init__.py)** — реестр загрузчиков.
- `LOADERS = {"aegis": aegis, "toxicchat": toxicchat}`, `get_loader(name)`.
- `split_path`, `load_split` — пути и чтение нормализованных сплитов.

**[data/aegis.py](../src/eguard/data/aegis.py)** — AEGIS: три шага разметки. Используется
в `00_prepare_data.py` и тестах.
- `parse_annotation(value)` — одна аннотация в `(класс, категории)`; классы
  `safe / caution / harm / none`. Обрабатывает `None` и `NaN`.
- `annotation_columns(row)` — столбцы `labels_N` в порядке номера, а не ключей словаря.
- `aggregate_row(row)` — до пяти аннотаций в один класс большинством голосов; **при равенстве
  берётся более консервативный** (`harm > caution > safe`). Отдаёт `agreement` — долю голосов
  за победителя.
- `to_binary(class3, policy)` — бинарная метка по `caution_policy` (`exclude` даёт `None`,
  то есть строка выбрасывается). Разведение `caution` вынесено в конфиг, потому что это ~пятая
  часть разметки и от неё зависит и `harm_rate`, и рабочая точка.
- `normalize_rows(rows, spec)` — фильтры по типу текста, длине и политике; порядок сохраняется.
- `load_raw(spec, splits)` — скачивание с HF Hub, проверка наличия нужных столбцов.
- `make_synthetic(n, seed)` — заглушки в формате сырого AEGIS для `make smoke`.

**[data/toxicchat.py](../src/eguard/data/toxicchat.py)** — ToxicChat, OOD-набор для LODO.
- `to_binary(row, label_field)` — метка из `toxicity` / `jailbreaking` / `any`.
  `any` ближе к задаче guardrail: jailbreak-промт не всегда токсичен по формулировке.
- `category_of(row)` — грубая категория для error analysis, своей таксономии у ToxicChat нет.
- `normalize_rows(rows, spec)` — берёт `user_input` (guardrail видит запрос, а не ответ
  модели — так сопоставимо с AEGIS/`user_message`), фильтрует по `human_annotation`.
- `load_raw`, `make_synthetic` — как у AEGIS; синтетика повторяет дисбаланс оригинала.

### encoders/ — замороженные энкодеры

**[encoders/base.py](../src/eguard/encoders/base.py)** — общий интерфейс.
- `EncodeResult` — матрица `[n, dim]` **без** нормировки, опциональные скоры родной головы,
  метаданные.
- `BaseEncoder.encode(texts, batch_size)` — абстрактный метод.

**[encoders/hf\_encoder.py](../src/eguard/encoders/hf_encoder.py)** — HF-энкодер.
- `resolve_device(name)` — `auto` в cuda / mps / cpu.
- `pool(hidden, mask, mode)` — `cls` / `mean` / `max` пулинг с учётом маски.
- `HFEncoder` — токенизация, forward, пулинг. Замораживает параметры, режет `max_length` по
  пределу модели, сортирует тексты по длине внутри батчей (меньше паддинга, заметно быстрее на
  CPU), умеет снимать вероятность вредоносного класса с родной seqcls-головы.

**[encoders/dummy.py](../src/eguard/encoders/dummy.py)** — `DummyEncoder`: детерминированный
хеш-энкодер символьных 4-грамм. Нужен только чтобы прогнать пайплайн без сети и torch.

**[encoders/\_\_init\_\_.py](../src/eguard/encoders/__init__.py)** — `build_encoder(spec, ...)`,
фабрика; импорт torch спрятан внутрь, чтобы офлайн-прогон его не требовал.

### analysis/ — анализ RQ1 и метрики

**[analysis/similarity.py](../src/eguard/analysis/similarity.py)**
- `cosine_analysis(x, y, ...)` — intra/inter косинусы в двух вариантах (`raw` и `centered`)
  плюс baseline случайной пары.
- `similarity_samples(...)` — наборы значений для гистограмм.
- `flatten_cosine_row(...)` — разворачивание результата в строки CSV.

**[analysis/geometry.py](../src/eguard/analysis/geometry.py)**
- `anisotropy(x)` — средний косинус случайных пар и доля дисперсии в PC1.
- `direction_analysis(...)` — единое harm-направление, его AUROC, Cohen's d, косинус с PC1.
- `cluster_analysis(...)` — silhouette, Davies–Bouldin, ARI и NMI для KMeans.
- `geometry_report(...)` — всё вместе.

**[analysis/probe.py](../src/eguard/analysis/probe.py)** — пробы и метрики. Используется
и в RQ1, и в RQ3.
- `tpr_at_fpr(y, scores, α)` — максимальный TPR при FPR ≤ α и соответствующий порог.
- `binary_metrics(...)` — AUROC, AUPRC, accuracy, F1, FPR, FNR, TPR@FPR при заданном пороге.
- `bootstrap_ci(...)` — перцентильные интервалы стратифицированным bootstrap: после фильтров
  test-сплит AEGIS это несколько сотен примеров, и разница в третьем знаке там ничего не значит.
- `logistic_probe(...)` — L2-логрегрессия, `C` по val, финальное дообучение на train+val.
- `fit_transfer_probe(...)` — то же, но **без** дообучения на val: val остаётся для калибровки
  порога. Используется в LODO.
- `knn_probe(...)` — нелинейная верхняя граница.
- `centroid_distance_probe(...)` — прообраз distance-based guardrail (RQ5).

**[analysis/baselines.py](../src/eguard/analysis/baselines.py)** — контроли, без которых RQ1 не
читается.
- `tfidf_probe(...)` — лексическая нижняя граница (1-2gram + logreg). Если проб на эмбеддингах
  не выигрывает, разделимость лексическая, а не семантическая.
- `length_probe(...)` — скор равен длине текста; AUROC заметно выше 0.5 означает confound.

### pairs/ — сборка обучающих пар (RQ2)

**[pairs/sources.py](../src/eguard/pairs/sources.py)**
- `fetch_harmbench_csv(split, cache_dir)` — скачивает и кеширует CSV с behaviors из
  GitHub-репозитория HarmBench. На HF датасет gated, здесь токен не нужен.
- `load_harmbench(...)` — behaviors в формат записей пайплайна; `include_context` по умолчанию
  выключен (guardrail видит запрос, а не приложенный документ).
- `load_aegis_records(...)` — нормализованный AEGIS с диска с пометкой источника.

**[pairs/transforms.py](../src/eguard/pairs/transforms.py)** — детерминированные при
фиксированном seed трансформации; каждая помечает результат именем семейства, по этой метке
делается абляция.
- `paraphrase_polite / declarative / indirect / noise` — реестр `PARAPHRASES`.
- `jb_role_play / fiction / instruction_override / research_framing / prefix_injection /
  leetspeak / base64` — реестр `JAILBREAKS`. Это намеренно общие, давно опубликованные в
  литературе семейства; они нужны, чтобы разметить их как harm и научить энкодер не считать
  вредоносный запрос безопасным из-за обёртки.
- `apply_named(registry, names, text, rng)` — случайная трансформация из подмножества.
- `load_external_variants(path)` — крючок под сгенерированные снаружи (например LLM) варианты.

**[pairs/builders.py](../src/eguard/pairs/builders.py)**
- `PairContext` — всё, из чего строятся пары для одного сплита; `pool(label)` отдаёт пул класса.
- `make_pair(...)` — один триплет со всей метаинформацией для абляций и error analysis.
- `build_safe_harm_contrast`, `build_paraphrase`, `build_jailbreak_variant`, `build_benign_twin`
  — четыре строителя, реестр `BUILDERS`.
- `deduplicate(pairs)` — убирает повторы и вырожденные пары (`anchor == positive`).
- `flatten_texts(pairs)` — все тексты из пар с бинарной меткой, без повторов; обучающее
  множество classification-ветки RQ3.

**[pairs/\_\_init\_\_.py](../src/eguard/pairs/__init__.py)**
- `pairs_dir`, `pairs_path`, `load_pairs` — пути и чтение.
- `dedupe_pool(records)` — один текст, одна запись.
- `exclude_texts(records, blocked)` — вычитание текстов другого сплита.
- `split_texts(records)` — множество текстов сплита.
- `build_split(ctx, sizes)` — прогон всех запрошенных типов пар для одного сплита.

### training/ — дообучение (RQ3)

**[training/losses.py](../src/eguard/training/losses.py)** — `info_nce`,
`supervised_contrastive`, `triplet_margin`, `embedding_kl_regularizer`, реестр `LOSSES`.
Формулы выше. В `supervised_contrastive` диагональ логитов явно обнуляется после log-softmax:
`-inf`, умноженный на нулевую маску, давал NaN.

**[training/model.py](../src/eguard/training/model.py)**
- `GuardEncoder` — backbone + пулинг (+ линейная голова). Пулинг и префикс те же, что в
  `hf_encoder`, иначе чекпоинт нельзя сравнивать с frozen-базлайном из RQ1.
  Методы: `prepare` (префикс), `tokenize`, `forward`, `logits`, `encode_texts` (инференс для
  валидации), `save` (сохраняет так, чтобы читалось обычным `AutoModel`).
- `build_model(spec, device, with_head)` — сборка и перенос на устройство.
- `load_checkpoint(path, spec, device)` — поднимает сохранённый чекпоинт вместе с головой.
- `frozen_teacher(spec, device)` — замороженная копия базовой модели для KL-якоря.

**[training/data.py](../src/eguard/training/data.py)**
- `PairDataset(pairs, pair_types)` — триплеты; `pair_types` это ручка абляции RQ2.
- `TextDataset(records)` — плоские тексты с меткой.
- `collate_pairs`, `collate_texts` — коллаторы.
- `texts_and_labels(records)` — `(list[str], np.ndarray)`.

**[training/loop.py](../src/eguard/training/loop.py)**
- `TrainConfig` — все гиперпараметры одним дата-классом.
- `build_optimizer(...)` — AdamW с раздельным weight decay плюс warmup и косинусное затухание.
- `contrastive_step(...)` — эмбеддит якорь, позитив, негатив; считает выбранный лосс и
  диагностику (`cos_anchor_positive`, `cos_anchor_negative`, `cos_margin`).
- `classification_step(...)` — CE поверх головы.
- `evaluate(...)` — AUROC и TPR@FPR на валидации. Скор берётся из головы, если она есть, иначе
  из центроидов train, так что чисто contrastive-прогон тоже сравним по тем же числам.
- `_next_cycled(iterator, loader)` — следующий батч с перезапуском исчерпанного загрузчика.
  В `joint` два загрузчика разной длины (пар вдвое больше, чем уникальных текстов), а число
  шагов считается по парам; без перезапуска короткий кончался посреди эпохи и ронял обучение.
- `train(...)` — общий цикл для всех трёх objective, выбор лучшего чекпоинта по TPR@FPR.

**[training/tracking.py](../src/eguard/training/tracking.py)** — обёртка над MLflow. Смысл
обёртки в том, что mlflow не должен быть жёсткой зависимостью: при `enabled: false` или
отсутствии пакета возвращается no-op с тем же интерфейсом.
- `flatten_params(obj)` — вложенный конфиг в плоские ключи вида `training.loss`.
- `NullRun` — заглушка с интерфейсом трекера.
- `MLflowRun` — реальный трекер; режет длинные значения и пишет параметры батчами.
- `start_run(cfg, run_name, tags)` — новый прогон.
- `resume_run(cfg, run_id)` — дозапись в существующий, чтобы метрики LODO лежали в том же
  прогоне, где училась модель.

### viz/ — визуализация

**[viz/plots.py](../src/eguard/viz/plots.py)**
- `project(x, method, seed)` — `[n, d]` в `[n, 2]` через PCA, t-SNE (с PCA-препроцессингом) или
  UMAP. Без установленного `umap-learn` возвращает `None`, и шаг просто пропускает метод.
- `plot_projection(...)` — два подграфика: раскраска по safe/harm и по harm-категории; второе
  нужно, чтобы увидеть, распадается ли harm на подкластеры по темам.
- `plot_similarity_hist(...)` — гистограммы косинусов safe–safe, harm–harm, safe–harm.

## scripts — шаги пайплайна

Все принимают `--config`; нумерация соответствует порядку выполнения.

| Скрипт | Что делает | Ключевые функции |
|:--|:--|:--|
| [00_prepare_data.py](../scripts/00_prepare_data.py) | датасет с HF Hub в нормализованные сплиты; `--synthetic N` для офлайна | `stratified_split` (отрезает val от train со стратификацией, официальный test не трогает), `summarize` |
| [01_embed.py](../scripts/01_embed.py) | frozen-энкодеры в кеш эмбеддингов; `--encoders dummy` для офлайна | — |
| [02_rq1_similarity.py](../scripts/02_rq1_similarity.py) | косинусы и геометрия, гистограммы | — |
| [03_rq1_probe.py](../scripts/03_rq1_probe.py) | logreg / kNN / centroid пробы + контроли, сохраняет скоры теста | — |
| [04_rq1_viz.py](../scripts/04_rq1_viz.py) | PCA / t-SNE / UMAP проекции | — |
| [05_report.py](../scripts/05_report.py) | собирает CSV и картинки в `REPORT.md` | `md_table` |
| [06_error_analysis.py](../scripts/06_error_analysis.py) | hard-пары и ошибки проба в рабочей точке | `hard_pairs` (пары safe/harm с максимальным косинусом — сырьё для hard negatives в RQ2), `probe_errors`, `snippet` |
| [07_build_pairs.py](../scripts/07_build_pairs.py) | сборка обучающих пар | `load_mining` (эмбеддинги для benign twins, урезанные тем же фильтром, что и пулы) |
| [08_train_contrastive.py](../scripts/08_train_contrastive.py) | дообучение с трекингом | `build_train_config` (склейка конфига с CLI-переопределениями) |
| [09_eval_lodo.py](../scripts/09_eval_lodo.py) | сравнение hold-out против OOD | `discover_runs`, `embed_all`, `centroid_scores`, `rows_for_readout` |
| [run_rq1.sh](../scripts/run_rq1.sh) | весь RQ1 одной командой | — |

Файлы `results/rq1/errors/` содержат фрагменты исходных текстов AEGIS, в том числе
вредоносных: это рабочие артефакты анализа, они исключены из версионирования.

## configs

| Файл | Для чего |
|:--|:--|
| [rq1.yaml](../configs/rq1.yaml) | RQ1: три энкодера, AEGIS с `caution_policy: exclude`, параметры анализа и визуализации |
| [rq2_pairs.yaml](../configs/rq2_pairs.yaml) | RQ2: источники, параметры трансформаций, зафиксированные размеры пар |
| [rq3_contrastive.yaml](../configs/rq3_contrastive.yaml) | RQ3: objective, лосс, гиперпараметры, MLflow |
| [ood_toxicchat.yaml](../configs/ood_toxicchat.yaml) | OOD-набор; `val_fraction: 0` — на ToxicChat ничего не подбирается |

## tests

| Файл | Что покрывает |
|:--|:--|
| [test_smoke.py](../tests/test_smoke.py) | разметка AEGIS (мажоритарность, тай-брейк, `caution_policy`, фильтры), косинусный анализ, harm-направление, пробы, контроли, bootstrap |
| [test_pairs.py](../tests/test_pairs.py) | согласованность меток в парах, парафразы и внешние варианты, обёрнутые safe-негативы в jailbreak, майнинг ближайшего safe, дедупликация, `flatten_texts` |
| [test_training.py](../tests/test_training.py) | лоссы и конечность градиентов, регресс на NaN в SupCon, KL против самого себя, `flatten_params`, no-op трекер, регресс на `StopIteration` в joint |
| [test_toxicchat.py](../tests/test_toxicchat.py) | метки и фильтры ToxicChat, обучение transfer-проба только на train, перенос порога с val на сдвинутый набор |

## Makefile

```
setup          зависимости
data           AEGIS -> data/processed/
ood-data       ToxicChat -> data/processed/
embed          frozen-эмбеддинги
rq1            весь RQ1
pairs          обучающие пары
rq3            три objective подряд + LODO
lodo           только оценка обученных прогонов
rq3-ablation   абляция по типам пар
mlflow         UI трекинга
test           pytest
smoke          офлайн-проверка RQ1
smoke-rq3      офлайн-проверка RQ2/RQ3
clean          удалить данные, артефакты, результаты и хранилище MLflow
```

---

## Что осталось сделать

По убыванию пользы для отчёта:

1. **Несколько seed'ов на вариант в RQ3.** Без этого сравнение на hold-out остаётся
   статистически неразличимым, а это ядро проекта.
2. **Переобучение на `aegis_harmbench_v2`** (400 behaviors вместо 80) и пересчёт LODO;
   при этом стоит добавить веса классов в CE из-за сдвига баланса.
3. **Строка с перекалибровкой порога** на небольшой размеченной выборке ToxicChat —
   отделит деградацию ранжирования от сдвига базовой ставки.
4. **RQ4:** прогоны с `kl_weight > 0` и замер геометрии дообученного пространства.
5. **Дедупликация AEGIS** в `00_prepare_data.py` и пересчёт RQ1.
6. **RQ2:** семантические парафразы через `external_variants`; тип пар `code_safe_harm`.
7. **RQ5:** проверка вероятностной калибровки distance-скора.
