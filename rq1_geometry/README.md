# RQ1: геометрия frozen-энкодеров

**Вопрос:** разделимы ли safe/harm концепты в пространстве эмбеддингов
существующих encoder-моделей без дообучения?

**Статус:** in progress
**Ответственный:** Дарья

## Что входит

- cosine similarity внутри классов vs между классами
- logistic probe над frozen-эмбеддингами
- UMAP/t-SNE визуализация
- кластеризация (kmeans/hdbscan)

## Структура (заполнять по мере переноса из Colab)

```
rq1_geometry/
  README.md
  notebooks/   <- экспортированные .ipynb (не только ссылки на Colab)
  src/         <- переиспользуемый код, если понадобится
  results/     <- сохранённые графики/таблицы с выводами
```

## Ноутбуки (Colab, будут перенесены сюда)

- GLiNER + комментарии: https://colab.research.google.com/drive/1IEiUNCBUg0to-Xb7e3nvZ6kn3fe69gcW
- t-SNE/UMAP + kmeans/hdbscan по датасетам и моделям: https://colab.research.google.com/drive/1JCKwxV4ougLRQdIvwQKjkV-yGg96vd2k
- Прогон по WildGuardMix: https://colab.research.google.com/drive/1CMMueCQOI3igLGTnPSqexzq2zKu3bwIQ
- Обновлённый ноутбук: https://colab.research.google.com/drive/1qx9U5nk5ojcYjXGeV31uKxLXGBDT7Xht

## Модели

- ettin-encoder-68m
- mmBERT-small
- e5-small-v2
- Llama-Prompt-Guard-2-86M (референс-классификатор)
