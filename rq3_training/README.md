# RQ3: дообучение энкодера

**Вопрос:** улучшает ли contrastive-дообучение на парах safe/harm
разделимость по сравнению с frozen-энкодерами (RQ1)?

**Статус:** in progress
**Ответственный:** —

## Зависимости

Использует пары для обучения из [`training_pairs`](../training_pairs) —
не дублировать сбор данных здесь.

## Структура (заполнять по мере старта)

```
rq3_training/
  README.md
  notebooks/   <- экспортированные .ipynb
  src/         <- training loop, loss (contrastive/triplet), конфиги
  configs/     <- гиперпараметры экспериментов
  checkpoints/ <- веса моделей (в .gitignore, хранить вне git)
  results/     <- метрики до/после дообучения
```

## Ноутбуки

[`notebooks/RQ3_mmbert_small_training.ipynb`](notebooks/RQ3_mmbert_small_training.ipynb) —
бинарное (harm/safe) дообучение `jhu-clsp/mmBERT-small` в пяти вариантах loss с
одной инициализацией, одними данными и одним сэмплером батчей: `ce`, `supcon`,
`ce+supcon`, `hier-supcon` (взвешенный SupCon: пара → категория вреда → класс),
`ce+hier-supcon`.

- train — пары v2 из `training_pairs` (генерируются локально
  `build_pairs_v2`, не версионируются, структура — в
  [`training_pairs_v2.md`](../training_pairs_v2.md)); val — AEGIS 2.0
  `validation`; hold-out — AEGIS 2.0 `test`; OOD (LODO) — сэмпл ToxicChat;
- батчи: несколько категорий вреда по нескольку harm-пар + safe-пары;
- протокол: короткий перебор гиперпараметров (по 3 конфигурации на вариант) →
  5 вариантов × 3 seed → одинаковая оценка всех моделей (probe с C по val, kNN,
  центроиды; голова — отдельно);
- классификация: AUROC и TPR@FPR=1%, FPR/FNR при пороге с val; среднее ± std
  по seed и парный бутстрэп разницы между вариантами;
- геометрия до/после и по ходу обучения: isotropy, intrinsic dimensionality,
  kNN-purity, alignment/uniformity; шаг и loss, на которых геометрия
  сломалась, пишутся в `results/`;
- диагностика категорий вреда на hold-out: FNR по категориям, kNN purity по
  категории внутри harm, ARI kmeans;
- опционально — выгрузка метрик в Weights & Biases (`WANDB_PROJECT`): читает
  готовые файлы из `results/`, поэтому запускается и в новой сессии, без
  переобучения.

То же самое без блокнота — [`src/log_results_to_wandb.py`](src/log_results_to_wandb.py):
скачивает `results/` из приватного репозитория Hugging Face и логирует в W&B.

```bash
python -m rq3_training.src.log_results_to_wandb \
  --hf-repo <user>/rq3-mmbert-small --project rq3-mmbert-small --entity <team>
```
