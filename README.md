# Embedding-level-Guardrails
Embedding-level Guardrails: обучение энкодера для  safe / harm separation.



***Участники:***
- Кайгородцева Дарья Андреевна
- Гуминов Дмитрий Андреевич
- Байшев Олег Михайлович
- Никитченко Мария Владиславовна
- Емцова Анна Сергеевна

## Структура репозитория

Один каталог верхнего уровня на направление/RQ — своя зона ответственности,
внутри — README со статусом. Общий код, нужный нескольким направлениям —
в `common/`.

| Каталог | RQ | Что | Ответственный | Статус |
|---|---|---|---|---|
| [`rq1_geometry`](rq1_geometry) | RQ1 | Геометрия frozen-энкодеров: cosine sim, logistic probe, UMAP/t-SNE, кластеризация | Дарья | in progress |
| [`rq3_training`](rq3_training) | RQ3 | Contrastive-дообучение энкодера | — | not started |
| [`training_pairs`](training_pairs) | подготовка данных для RQ3 | Сбор пар для обучения (safe/harm contrast, paraphrase, jailbreak variant) из AEGIS + HarmBench | — | done |
| [`common`](common) | — | Переиспользуемый код между направлениями | — | пусто |

Ветки/PR называть по той же схеме: `rq1/...`, `rq3/...`.
