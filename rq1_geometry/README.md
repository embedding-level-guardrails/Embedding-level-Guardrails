# RQ1: baseline разделимость без дообучения

Отчет: https://docs.google.com/document/d/1C7yne8fP11DkqNnEQTwv8ZzR-l4glkCi24rabQEuxf0/edit?usp=sharing

## Что входит

- расстояние внутри классов vs между классами
- PCA анализ
- logistic probe над frozen-эмбеддингами
- Fisher Discriminant Ratio


## Структура (заполнять по мере переноса из Colab)

```
rq1_geometry/
  README.md
  notebooks/   <- экспортированные .ipynb (не только ссылки на Colab)
  src/         <- переиспользуемый код, если понадобится
  results/     <- сохранённые графики/таблицы с выводами
```

## Ноутбуки (Colab, будут перенесены сюда)

- t-SNE/UMAP + kmeans/hdbscan по датасетам и моделям: https://colab.research.google.com/drive/1JCKwxV4ougLRQdIvwQKjkV-yGg96vd2k
- Прогон по WildGuardMix: https://colab.research.google.com/drive/1CMMueCQOI3igLGTnPSqexzq2zKu3bwIQ
