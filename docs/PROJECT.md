# Embedding-level Guardrails — описание проделанной работы

Компактный энкодер как быстрый предварительный фильтр перед LLM. Проверяем,
разделяются ли безопасные и вредоносные запросы геометрически уже на уровне
эмбеддингов, и можно ли усилить это разделение contrastive-дообучением.

**Энкодеры:** `jhu-clsp/ettin-encoder-68m`, `jhu-clsp/mmBERT-small`, `intfloat/e5-small-v2`.
**Данные для обучения:** WildGuardMix (`allenai/wildguardmix`) — основной набор;
AEGIS (`nvidia/Aegis-AI-Content-Safety-Dataset-1.0`) — первый эксперимент и RQ1;
HarmBench (behaviors) — источник затравок для jailbreak-вариантов.
**OOD для LODO:** ToxicChat (`lmsys/toxic-chat`, конфиг `toxicchat0124`).

**Главный результат на сейчас (RQ3):** в своём домене CE, contrastive и CE + contrastive
неразличимы, но при переносе на чужой трафик модель, обученная только с CE, теряет рабочую
точку с низким FPR. Эффект воспроизвёлся на двух обучающих наборах. Переход с AEGIS на
WildGuardMix улучшил перенос сильнее, чем любая смена objective. Подробно —
[results/rq3/REPORT.md](../results/rq3/REPORT.md).

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

**Статус:** выполнено для двух наборов — сводки в [results/rq2/PAIRS.md](../results/rq2/PAIRS.md)
(AEGIS) и [results/wildguardmix/rq2/PAIRS.md](../results/wildguardmix/rq2/PAIRS.md) (WildGuardMix).
Абляция по типам пар (сам вопрос RQ2) ещё не прогонялась.

### Что сделано

Собраны обучающие пары в формате триплета `(anchor, positive, negative)` с тегом
`pair_type`: сначала из AEGIS + HarmBench, затем из WildGuardMix + HarmBench. Триплет выбран
потому, что он одинаково ложится и на InfoNCE с in-batch negatives, и на triplet margin, и на
SupCon, а абляция «какой тип пар сколько даёт» сводится к фильтру по полю `pair_type`.

Пять типов пар:

| Тип | Что учит | Как строится |
|:--|:--|:--|
| `safe_harm_contrast` | базовый контраст класса | positive — другой пример того же класса, по возможности той же категории вреда; negative — случайный пример другого класса |
| `paraphrase` | инвариантность к форме запроса | positive — поверхностный парафраз якоря |
| `jailbreak_variant` | обёртка не делает вредоносный запрос безопасным | positive — вредоносная затравка в шаблонной jailbreak-обёртке |
| `benign_twin` | hard negatives | negative — лексически ближайший safe, намайненный по косинусу |
| `adversarial_contrast` | то же, что jailbreak_variant, но на реальных jailbreak-промтах (только WildGuardMix) | anchor — adversarial-промт; positive — vanilla-промт того же класса; negative — adversarial-промт другого класса |

`adversarial_contrast` использует главное свойство WildGuardMix: adversarial-промты там есть
в обоих классах, в том числе безопасные — с jailbreak-обёрткой, но без вредного содержания.
Якорь с позитивом совпадают по содержанию, якорь с негативом — по форме, так что лосс тянет
к содержанию и отталкивает от формы. Строится в обе стороны:

```
harm-якорь: adversarial harmful   → positive: vanilla harmful (той же категории) → negative: adversarial unharmful
safe-якорь: adversarial unharmful → positive: vanilla unharmful                 → negative: adversarial harmful
```

У AEGIS флага `adversarial` нет, поэтому для него этот тип пустой.

Затравки для шаблонных jailbreak-обёрток задаются параметром `jailbreak_seeds`: `harmbench`
(behaviors HarmBench, так собраны пары AEGIS), `dataset` (vanilla-вредоносные промты датасета)
или `both`. Adversarial-промты повторно не оборачиваются.

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
построчно к исходному jsonl и в обход пулов протаскивала дубликаты. Behaviors HarmBench,
дословно совпадающие с промтами из val/test основного датасета, из train тоже убираются.
Проверено на обоих наборах: пересечение уникальных текстов между всеми тремя сплитами равно
нулю.

### Воспроизводимость

До исправления сид сплита считался как `cfg.seed + hash(split)`, а `hash()` строк в Python
меняется в каждом процессе, поэтому каждый запуск давал другие пары. Теперь используется
`zlib.crc32`, и повторный запуск даёт идентичный результат. Следствие: наборы
`aegis_harmbench_v1` и `v2` собраны со случайными сидами и в точности не пересобираются.
Они корректны и без утечек, но при пересборке пары AEGIS будут другими. На CUDA эмбеддинги
для майнинга считаются в fp16, поэтому `benign_twin` на разных машинах может немного
отличаться.

### Код

| Файл | Роль |
|:--|:--|
| [src/eguard/data/wildguardmix.py](../src/eguard/data/wildguardmix.py) | загрузка и нормализация WildGuardMix |
| [src/eguard/pairs/sources.py](../src/eguard/pairs/sources.py) | загрузка HarmBench (CSV с GitHub, без HF-токена) и нормализованных записей датасета |
| [src/eguard/pairs/transforms.py](../src/eguard/pairs/transforms.py) | парафразы и jailbreak-обёртки |
| [src/eguard/pairs/builders.py](../src/eguard/pairs/builders.py) | пять строителей пар, выбор jailbreak-затравок, дедупликация, `flatten_texts` |
| [src/eguard/pairs/__init__.py](../src/eguard/pairs/__init__.py) | оркестрация, фильтры пулов, пути на диске |
| [scripts/07_build_pairs.py](../scripts/07_build_pairs.py) | CLI, сводка размеров |
| [configs/rq2_pairs.yaml](../configs/rq2_pairs.yaml) | AEGIS: зафиксированные размеры и параметры |
| [configs/rq2_pairs_wildguardmix.yaml](../configs/rq2_pairs_wildguardmix.yaml) | WildGuardMix: подготовка данных, эмбеддинги для майнинга и пары одним конфигом |

### Как запустить

```bash
make pairs         # AEGIS
make wgm-pairs     # WildGuardMix: данные -> эмбеддинги e5 -> пары (нужен HF-токен, датасет gated)
```

Варианты: `--types safe_harm_contrast jailbreak_variant` (подмножество),
`--scale 0.05` (быстрый прогон на срезе), `--splits train` (один сплит).

### WildGuardMix: данные

Два конфига на HF: `wildguardtrain` (86 759 строк, разметка LLM) и `wildguardtest`
(1 725 строк, разметка людьми, 3 аннотатора). Датасет gated: нужно принять условия на странице
датасета и войти через `hf auth login`.

Строки train — пары «промт + ответ», один промт встречается с разными ответами. Guardrail
видит только промт, поэтому строки дедуплицируются по тексту промта; промты с разными
метками в разных строках выбрасываются. Метка — `prompt_harm_label`, категория —
`subcategory`, флаг `adversarial` сохраняется в записи.

| сплит | уникальных промтов | harm_rate | доля adversarial | медиана длины, символов |
|:--|--:|--:|--:|--:|
| train | 43 025 | 0.518 | 0.479 | 298 |
| val | 4 781 | 0.518 | 0.478 | 284 |
| test (WildGuardTest) | 1 699 | 0.444 | 0.469 | 250 |

86 759 строк → 47 806 уникальных промтов; 15 выброшены из-за конфликта меток, 26 строк test —
без метки промта. Val отрезан от train. Промты примерно в 6 раз длиннее, чем в AEGIS (медиана
52) и ToxicChat (63). С ToxicChat пересекаются 8 из 2 790 промтов.

### Зафиксированные размеры: WildGuardMix (набор `wildguardmix_harmbench_v1`)

| split | пар | уник. текстов | contrast | paraphrase | jailbreak | adversarial | benign_twin |
|:--|--:|--:|--:|--:|--:|--:|--:|
| train | 32 538 | 48 985 | 12000 | 4539 | 4999 | 8000 | 3000 |
| val | 3 256 | 5 166 | 1200 | 456 | 500 | 800 | 300 |
| test | 3 233 | 2 802 | 1200 | 434 | 500 | 799 | 300 |

`jailbreak_seeds: both`. HarmBench включён, но теряется среди ~22 тысяч вредоносных промтов
WildGuardMix: на него приходится 0.9% пар и 1.0% текстов train. Фактически это обучение на
WildGuardMix.

### Зафиксированные размеры: AEGIS (набор `aegis_harmbench_v2`)

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

**Статус:** выполнено на двух обучающих наборах. Сводный отчёт с выводами —
[results/rq3/REPORT.md](../results/rq3/REPORT.md); таблицы по экспериментам —
[results/wildguardmix/rq3/LODO.md](../results/wildguardmix/rq3/LODO.md) (основной) и
[results/rq3/LODO.md](../results/rq3/LODO.md) (AEGIS). Это ядро проекта.

### Что сделано

Три варианта дообучения `e5-small-v2` поверх одного энкодера и **одних и тех же
обучающих текстов**, различается только objective:

- **(а)** `classification` — только CE поверх линейной головы (базлайн),
- **(б)** `contrastive` — только contrastive-лосс на триплетах,
- **(в)** `joint` — CE + contrastive.

Эксперимент проведён дважды. Оценка — на hold-out своего набора и на OOD (ToxicChat) по
схеме LODO: обучались на одном наборе, тестируем на реальном трафике чат-бота.

| | эксперимент 1 | эксперимент 2 (основной) |
|:--|:--|:--|
| пары | `aegis_harmbench_v1` | `wildguardmix_harmbench_v1` |
| триплетов / уникальных текстов в train | 16 765 / 7 904 | 32 538 / 48 985 |
| доля HarmBench в train: пары / тексты | 25.8% / 10.8% | 0.9% / 1.0% |
| шагов CE / contrastive / joint | 494 / 1046 / 1046 | 3060 / 2032 / 2032 |
| валидация | AEGIS val, 257 | WildGuardMix val, 4 781 |
| hold-out | AEGIS test, 308, harm 0.597 | WildGuardTest, 1 699, harm 0.444 |
| OOD | ToxicChat test, 2 853, harm 0.127 | то же |
| железо, время | Apple M-series (MPS), ~40 мин | RTX 5070 Ti, ~2–2.5 ч с оценкой |

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
и тот же** линейный проб на train своего набора. Различается только пространство. Родная
голова и центроидный скор идут дополнительными строками.

Это не формальность: на AEGIS во время обучения CE выглядел лучше contrastive (TPR@FPR 0.736
против 0.642), но там CE мерился своей головой, а contrastive — центроидным расстоянием. При
едином пробе порядок перевернулся. И обратная проверка: на WildGuardMix у CE-модели единый
проб и родная голова дают почти одинаковые числа (AUROC 0.928 и 0.930), то есть единый
readout CE-вариант не занижает.

**Порог с val, а не с теста.** FPR/FNR репортятся при пороге, выставленном на val своего
набора при FPR=1%, и этот же порог без перекалибровки применяется к hold-out и к OOD:

```
τ* = threshold at FPR = 1% on val
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
| [configs/rq3_wildguardmix.yaml](../configs/rq3_wildguardmix.yaml) | основной эксперимент (WildGuardMix) |
| [configs/rq3_contrastive.yaml](../configs/rq3_contrastive.yaml) | эксперимент на AEGIS |
| [configs/ood_toxicchat.yaml](../configs/ood_toxicchat.yaml) | OOD-набор |

На WildGuardMix включён **gradient checkpointing** (`gradient_checkpointing: true`): промты
длинные, и батч из 96 текстов по 512 токенов без него не влезал в 20 ГБ памяти MPS. Активации
пересчитываются при backward вместо хранения; математика обучения не меняется, время растёт
примерно на 30%. Прогоны и результаты WildGuardMix пишутся отдельно — в
`artifacts/runs_wildguardmix/`, `results/wildguardmix/` и эксперимент MLflow
`eguard-rq3-wildguardmix`, — чтобы не перезаписать AEGIS.

Лучший чекпоинт выбирается по **TPR@FPR=1%**, а не по лоссу: средний лосс не отражает
рабочую точку с низким FPR. Чекпоинт сохраняется в формате `AutoModel`, поэтому его можно
дописать в `configs/rq1.yaml` ещё одним энкодером и прогнать на нём весь RQ1-анализ.

### Как запустить

```bash
make ood-data      # ToxicChat -> data/processed/ (один раз)
make rq3-wgm       # WildGuardMix: три objective подряд + LODO, ~2–2.5 ч на RTX 5070 Ti
make lodo-wgm      # только оценка уже обученных прогонов WildGuardMix
make rq3           # то же на AEGIS, ~50 минут на MPS
make lodo          # только оценка прогонов AEGIS
make mlflow        # UI трекинга на http://127.0.0.1:5000 (отдельный терминал)
```

Поштучно (для AEGIS — тот же набор команд с `configs/rq3_contrastive.yaml`):

```bash
python scripts/08_train_contrastive.py --config configs/rq3_wildguardmix.yaml --objective classification --run-name wgm-classification
python scripts/08_train_contrastive.py --config configs/rq3_wildguardmix.yaml --objective contrastive    --run-name wgm-contrastive
python scripts/08_train_contrastive.py --config configs/rq3_wildguardmix.yaml --objective joint          --run-name wgm-joint
python scripts/09_eval_lodo.py --config configs/rq3_wildguardmix.yaml --ood-config configs/ood_toxicchat.yaml
```

Для настоящих прогонов не используйте `--limit-pairs`: он режет пары и тексты по-разному, и
CE-ветка перестаёт видеть ровно те же тексты, что contrastive. Объём уменьшается через
`--scale` при сборке пар или `epochs` в конфиге.

Абляция RQ2 (вклад каждого типа пар при прочих равных): `make rq3-ablation` или
`--pair-types safe_harm_contrast jailbreak_variant`.

### Результаты: WildGuardMix (основной эксперимент)

Единый проб, порог с val при FPR=1%. Hold-out: WildGuardTest, n=1699, harm_rate 0.444. OOD:
ToxicChat, n=2853, harm_rate 0.127. В скобках — 95% bootstrap-интервалы.

| вариант | набор | AUROC | TPR@FPR=1% | FPR | FNR |
|:--|:--|--:|--:|--:|--:|
| e5 frozen | holdout | 0.868 [0.852, 0.885] | 0.313 [0.253, 0.398] | 0.016 | 0.649 |
| (а) CE | holdout | 0.928 [0.916, 0.940] | 0.363 [0.239, 0.531] | 0.024 | 0.395 |
| (б) contrastive | holdout | 0.919 [0.906, 0.931] | 0.406 [0.241, 0.529] | 0.020 | 0.458 |
| (в) CE+contrastive | holdout | 0.927 [0.915, 0.939] | 0.439 [0.276, 0.535] | 0.030 | 0.406 |
| e5 frozen | **ood** | 0.883 [0.864, 0.899] | 0.160 [0.116, 0.235] | 0.042 | 0.550 |
| (а) CE | **ood** | 0.917 [0.902, 0.929] | **0.069** [0.041, 0.113] | 0.054 | 0.417 |
| (б) contrastive | **ood** | 0.913 [0.897, 0.927] | 0.238 [0.193, 0.321] | 0.037 | 0.481 |
| (в) CE+contrastive | **ood** | 0.923 [0.907, 0.935] | 0.251 [0.163, 0.340] | 0.034 | 0.442 |

### Выводы

1. **CE теряет рабочую точку с низким FPR при переносе — воспроизведено на двух наборах.**
   OOD TPR@FPR=1% у CE: 0.069 [0.041, 0.113] после WildGuardMix и 0.028 [0.011, 0.053] после
   AEGIS. Варианты с contrastive-компонентой — 0.14–0.25, интервалы с CE не пересекаются. По
   AUROC на OOD CE при этом почти не отстаёт (0.917 против 0.913 и 0.923): проблема не в общем
   ранжировании, а в хвосте — у CE-модели небольшая группа безопасных промтов ToxicChat
   получает очень высокие скоры и выставляет порог. AUROC этого не видит.
2. **В своём домене objective не имеет значения.** На hold-out интервалы трёх вариантов
   пересекаются на обоих наборах.
3. **Данные важнее objective.** После AEGIS дообучение на OOD вредило (AUROC 0.815–0.850 против
   0.872 у frozen), а FPR при перенесённом пороге вырастал до 21–36%. После WildGuardMix все три
   варианта лучше frozen на OOD (0.913–0.923 против 0.883), а FPR на ToxicChat — 3–5%. Вклад
   содержания и объёма данных здесь не разделить.
4. **Рекомендуемый вариант — joint.** На WildGuardMix у него лучшие точечные оценки на OOD по
   всем метрикам и самый низкий FPR при перенесённом пороге; на hold-out он не хуже CE.
5. **Практическая рабочая точка.** Joint при пороге с val: FPR 3.0% и FNR 40.6% на WildGuardTest,
   FPR 3.4% и FNR 44.2% на ToxicChat. Как единственный фильтр мало, как дешёвая первая ступень
   перед дорогой проверкой — подходит.
6. **Порог плывёт уже на своём тесте.** На val WildGuardMix порог давал FPR 1%, на WildGuardTest —
   1.6–3.0%. Val взят из train с LLM-разметкой, тест размечен людьми.

Главное ограничение: каждый прогон — один seed, bootstrap-интервалы учитывают только разброс
тестовой выборки. Все выводы и ограничения подробно — в
[results/rq3/REPORT.md](../results/rq3/REPORT.md).

### Результаты: AEGIS (эксперимент 1)

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

На hold-out интервалы TPR@FPR=1% огромные (CE 0.239–0.745, contrastive 0.533–0.761, joint
0.489–0.772): в тесте 308 примеров, и порог при FPR=1% держится на одном-двух безопасных
примерах. На OOD CE переносится хуже всех, а порог, откалиброванный на AEGIS, на ToxicChat
даёт FPR 21–36% у всех дообученных вариантов (у frozen — 6.3%).

Числа посчитаны на парах `aegis_harmbench_v1` (80 behaviors HarmBench). Позже набор пересобран
в `aegis_harmbench_v2` с 400 behaviors, но на нём AEGIS не переобучался;
`configs/rq3_contrastive.yaml` указывает уже на `v2`.

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

**Статус:** побочные измерения из LODO-оценки; первоначальная гипотеза не подтвердилась.

Скор `s(x) = cos(x, μ̂_harm) − cos(x, μ̂_safe)` считается для всех моделей как отдельный
readout (`centroid_distance` в [09_eval_lodo.py](../scripts/09_eval_lodo.py)).

| модель | OOD AUROC, AEGIS | OOD AUROC, WildGuardMix |
|:--|--:|--:|
| e5 frozen, центроиды | 0.668 | 0.789 |
| contrastive, центроиды | **0.892** | 0.897 |
| contrastive, единый проб | 0.849 | **0.913** |
| CE, центроиды | 0.819 | 0.916 |

На AEGIS скор по центроидам после contrastive-обучения дал лучший OOD-результат среди всех
комбинаций, и это выглядело как подтверждение RQ5: contrastive перестраивает геометрию так,
что начинает работать distance-based скоринг. На WildGuardMix это не воспроизвелось:
у contrastive центроиды уступают единому пробу, а у CE центроиды работают не хуже проба.
Сам результат contrastive + центроиды почти не изменился (0.892 → 0.897) — выросли остальные
readout'ы, потому что обучение на WildGuardMix улучшило пространство в целом.

Корректная формулировка: после contrastive-обучения distance-скор даёт ~0.89 AUROC на OOD
независимо от обучающего набора, но лучшим readout'ом не является. Вероятностная калибровка
скора (сам вопрос RQ5) пока не проверялась.

---

## RQ6 — внутренние представления LLM как teacher signal

**Статус:** не начат.

---

## Инфраструктура и качество

**Тесты:** 47 штук, запуск `make test`. Покрывают разметку AEGIS, ToxicChat и WildGuardMix,
логику сборки пар и отсутствие утечки, лоссы и их градиенты, калибровку порога, обёртку MLflow.
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
училась модель. Отключается флагом `--no-mlflow`. Эксперименты: `eguard-rq3` (AEGIS) и
`eguard-rq3-wildguardmix`. Лог обучения на WildGuardMix — `rq3_wgm.log` в корне репозитория.

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
| `results/` | версионируемые результаты: CSV и markdown-отчёты; `results/rq*/` — AEGIS и сводный отчёт RQ3, `results/wildguardmix/` — эксперимент на WildGuardMix |
| `data/` | не версионируется: сырые и нормализованные данные, пары |
| `artifacts/` | не версионируется: кеш эмбеддингов, чекпоинты прогонов (`runs/` — AEGIS, `runs_wildguardmix/`) |
| `mlruns/`, `mlartifacts/`, `mlflow.db` | не версионируется: хранилище MLflow |

Раскладка данных на диске:

```
data/raw/harmbench/                                 кеш CSV с behaviors
data/processed/{dataset}/{split}.jsonl              нормализованные записи
data/processed/{dataset}/summary.json               статистика сплитов
data/processed/pairs/{name}/pairs_{split}.jsonl     триплеты
data/processed/pairs/{name}/texts_{split}.jsonl     те же тексты плоско, с меткой
artifacts/embeddings/{dataset}/{encoder}/{split}.npy
artifacts/runs/{run_name}/checkpoint/               дообученная модель (AEGIS)
artifacts/runs_wildguardmix/{run_name}/checkpoint/  дообученная модель (WildGuardMix)
```

Порядок строк в `{split}.jsonl` — это порядок строк в кеше эмбеддингов; менять его нельзя
без пересчёта (проверка длин стоит в `load_xy`).

## src/eguard — библиотека

### Корень

**[config.py](../src/eguard/config.py)** — загрузка YAML в дата-классы. Используется всеми
скриптами.
- `EncoderSpec`, `DatasetSpec`, `Paths`, `Config` — дата-классы секций конфига.
  `DatasetSpec` несёт и общие поля (`text_types`, `min_chars`), и специфичные для
  отдельных датасетов (`config`, `label_field`, `require_human_annotation` — для ToxicChat;
  `adversarial_filter` — для WildGuardMix; `max_records_per_split` — стратифицированный
  потолок записей на сплит для больших наборов).
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
- `LOADERS = {"aegis": aegis, "toxicchat": toxicchat, "wildguardmix": wildguardmix}`,
  `get_loader(name)`. Все загрузчики с одинаковым интерфейсом (`load_raw`, `normalize_rows`,
  `make_synthetic`), поэтому `00_prepare_data.py` работает с любым из них.
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

**[data/wildguardmix.py](../src/eguard/data/wildguardmix.py)** — WildGuardMix, основной
обучающий набор.
- `load_raw(spec)` — скачивает конфиги `wildguardtrain` и `wildguardtest`. Без доступа к
  gated-репозиторию падает с подсказкой, как его получить; проверяет наличие нужных столбцов.
- `normalize_rows(rows, spec)` — промт как текст, дедупликация по тексту промта (строки train —
  пары «промт + ответ»), выброс промтов с противоречивыми метками и строк без метки, фильтр
  `adversarial_filter`, опциональный стратифицированный срез.
- `to_binary(label)` — `prompt_harm_label` в 1 / 0 / None.
- `is_adversarial(value)` — флаг `adversarial`, понимает и bool, и строки.
- `parse_agreement(value)` — согласие аннотаторов в [0, 1]. В wildguardtest хранится **число**
  согласных из трёх (2.0 или 3.0), а не доля; в train согласия нет, там 1.0.
- `category_of(row, label)` — `subcategory` для вредоносных, `safe` для безопасных.
- `make_synthetic(n, seed)` — заглушки со всеми особыми случаями: повторы промтов, строки
  без метки, оба класса в обоих вариантах (vanilla / adversarial).

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
- `load_processed_records(root, dataset, split)` — нормализованные записи любого датасета с
  диска; источник по умолчанию — имя датасета. `load_aegis_records` — старое имя, оставлено
  алиасом.

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
  Поле с записями основного датасета исторически называется `aegis`, но в него попадает любой
  датасет из `dataset.name`.
- `make_pair(...)` — один триплет со всей метаинформацией для абляций и error analysis.
- `build_safe_harm_contrast`, `build_paraphrase`, `build_jailbreak_variant`, `build_benign_twin`,
  `build_adversarial_contrast` — пять строителей, реестр `BUILDERS`.
- `jailbreak_seeds(ctx)` — затравки для шаблонных jailbreak-обёрток по режиму
  `harmbench` / `dataset` / `both`; из датасета берутся только не-adversarial промты.
- `build_benign_twin` майнит блоками по 512 якорей: на WildGuardMix две плотные матрицы
  косинусов заняли бы около 1 ГБ. Точность float64 сохранена — во float32 почти равные
  косинусы меняют порядок соседей.
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
- `TrainConfig` — все гиперпараметры одним дата-классом, включая `gradient_checkpointing`
  (включается в `08_train_contrastive.py` через `backbone.gradient_checkpointing_enable()`).
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
| [00_prepare_data.py](../scripts/00_prepare_data.py) | датасет с HF Hub в нормализованные сплиты; `--synthetic N` для офлайна | `stratified_split` (отрезает val от train со стратификацией, официальный test не трогает), `summarize` (для WildGuardMix добавляет долю adversarial по классам) |
| [01_embed.py](../scripts/01_embed.py) | frozen-энкодеры в кеш эмбеддингов; `--encoders dummy` для офлайна | — |
| [02_rq1_similarity.py](../scripts/02_rq1_similarity.py) | косинусы и геометрия, гистограммы | — |
| [03_rq1_probe.py](../scripts/03_rq1_probe.py) | logreg / kNN / centroid пробы + контроли, сохраняет скоры теста | — |
| [04_rq1_viz.py](../scripts/04_rq1_viz.py) | PCA / t-SNE / UMAP проекции | — |
| [05_report.py](../scripts/05_report.py) | собирает CSV и картинки в `REPORT.md` | `md_table` |
| [06_error_analysis.py](../scripts/06_error_analysis.py) | hard-пары и ошибки проба в рабочей точке | `hard_pairs` (пары safe/harm с максимальным косинусом — сырьё для hard negatives в RQ2), `probe_errors`, `snippet` |
| [07_build_pairs.py](../scripts/07_build_pairs.py) | сборка обучающих пар; сид сплита через `zlib.crc32`, behaviors HarmBench, совпадающие с val/test, убираются | `load_mining` (эмбеддинги для benign twins, урезанные тем же фильтром, что и пулы) |
| [08_train_contrastive.py](../scripts/08_train_contrastive.py) | дообучение с трекингом | `build_train_config` (склейка конфига с CLI-переопределениями) |
| [09_eval_lodo.py](../scripts/09_eval_lodo.py) | сравнение hold-out против OOD | `discover_runs`, `embed_all`, `centroid_scores`, `rows_for_readout` |
| [run_rq1.sh](../scripts/run_rq1.sh) | весь RQ1 одной командой | — |

Файлы `results/rq1/errors/` содержат фрагменты исходных текстов AEGIS, в том числе
вредоносных: это рабочие артефакты анализа, они исключены из версионирования.

## configs

| Файл | Для чего |
|:--|:--|
| [rq1.yaml](../configs/rq1.yaml) | RQ1: три энкодера, AEGIS с `caution_policy: exclude`, параметры анализа и визуализации |
| [rq2_pairs.yaml](../configs/rq2_pairs.yaml) | RQ2 на AEGIS: источники, параметры трансформаций, зафиксированные размеры пар |
| [rq2_pairs_wildguardmix.yaml](../configs/rq2_pairs_wildguardmix.yaml) | RQ2 на WildGuardMix: датасет, энкодер для майнинга, параметры и размеры пар |
| [rq3_contrastive.yaml](../configs/rq3_contrastive.yaml) | RQ3 на AEGIS: objective, лосс, гиперпараметры, MLflow |
| [rq3_wildguardmix.yaml](../configs/rq3_wildguardmix.yaml) | RQ3 на WildGuardMix: то же + gradient checkpointing, отдельные папки результатов и эксперимент MLflow |
| [ood_toxicchat.yaml](../configs/ood_toxicchat.yaml) | OOD-набор; `val_fraction: 0` — на ToxicChat ничего не подбирается |

## tests

| Файл | Что покрывает |
|:--|:--|
| [test_smoke.py](../tests/test_smoke.py) | разметка AEGIS (мажоритарность, тай-брейк, `caution_policy`, фильтры), косинусный анализ, harm-направление, пробы, контроли, bootstrap |
| [test_pairs.py](../tests/test_pairs.py) | согласованность меток в парах, парафразы и внешние варианты, обёрнутые safe-негативы в jailbreak, майнинг ближайшего safe, дедупликация, `flatten_texts` |
| [test_training.py](../tests/test_training.py) | лоссы и конечность градиентов, регресс на NaN в SupCon, KL против самого себя, `flatten_params`, no-op трекер, регресс на `StopIteration` в joint |
| [test_toxicchat.py](../tests/test_toxicchat.py) | метки и фильтры ToxicChat, обучение transfer-проба только на train, перенос порога с val на сдвинутый набор |
| [test_wildguardmix.py](../tests/test_wildguardmix.py) | метки, дедупликация промтов и конфликты, фильтр adversarial, разбор согласия, стратифицированный срез, структура `adversarial_contrast`, режимы `jailbreak_seeds` |

## Makefile

```
setup          зависимости
data           AEGIS -> data/processed/
ood-data       ToxicChat -> data/processed/
embed          frozen-эмбеддинги
rq1            весь RQ1
pairs          обучающие пары (AEGIS)
wgm-pairs      WildGuardMix: данные -> эмбеддинги для майнинга -> пары
rq3            три objective подряд + LODO (AEGIS)
lodo           только оценка обученных прогонов (AEGIS)
rq3-wgm        три objective подряд + LODO (WildGuardMix)
lodo-wgm       только оценка обученных прогонов (WildGuardMix)
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

1. **3–5 seed'ов на вариант в RQ3 на WildGuardMix** — главное подтверждение вывода про
   потерю хвоста у CE. На RTX 5070 Ti это ~2 часа на seed.
2. **Разбор хвоста:** сохранять скоры в `09_eval_lodo.py` и посмотреть, какие безопасные
   промты ToxicChat CE-модель ставит выше всего.
3. **Абляция типов пар на WildGuardMix** (`make rq3-ablation RQ3_CONFIG=configs/rq3_wildguardmix.yaml`)
   — что именно даёт устойчивость хвоста. Это и есть ответ на RQ2.
4. **Выравнивание по шагам:** CE с тем же числом шагов, что у contrastive (сейчас 3060 против
   2032), чтобы исключить объяснение «CE просто дольше учился».
5. **Строка с перекалибровкой порога** на небольшой размеченной выборке ToxicChat —
   отделит деградацию ранжирования от сдвига базовой ставки.
6. **RQ4:** прогоны с `kl_weight > 0` и замер геометрии дообученного пространства.
7. **Повтор RQ3 на ettin / mmBERT:** предобучение e5 было контрастивным, выводы могут
   не переноситься на MLM-энкодеры.
8. **Дедупликация AEGIS** в `00_prepare_data.py` и пересчёт RQ1.
9. **RQ2:** семантические парафразы через `external_variants`; тип пар `code_safe_harm`.
10. **RQ5:** проверка вероятностной калибровки distance-скора.
