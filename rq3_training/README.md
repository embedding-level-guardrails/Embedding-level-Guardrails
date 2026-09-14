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
дообучение `jhu-clsp/mmBERT-small` в трёх вариантах с одной инициализацией и
одними данными: только CE, только contrastive (SupCon), CE + contrastive.

- train — пары из `training_pairs` (основной + code-набор; генерируются
  локально `build_pairs` и не версионируются); val — AEGIS 2.0 `validation`; hold-out —
  AEGIS 2.0 `test`; OOD (LODO) — сэмпл ToxicChat;
- классификация: AUROC, FPR/FNR при пороге с val, TPR@FPR=1%;
- геометрия до/после и по ходу обучения: isotropy, intrinsic dimensionality,
  kNN-purity, alignment/uniformity; шаг и loss, на которых геометрия
  сломалась, пишутся в `results/`;
- в конце — метрики расстояний из RQ1 и t-SNE по датасетам RQ1 для
  сравнения с frozen-бейзлайнами.
