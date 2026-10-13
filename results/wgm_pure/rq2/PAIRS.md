# RQ2 — обучающие пары
Набор: `wildguardmix_pure_v1`, seed=42, режим `coverage`. Датасет `wildguardmix`; HarmBench: train_split=`None`, heldout=`None`.
## Размеры
| split   |   n_pairs |   n_texts |   harm_rate |   safe_harm_contrast |   adversarial_contrast |   benign_twin |
|:--------|----------:|----------:|------------:|---------------------:|-----------------------:|--------------:|
| train   |     43025 |     43025 |       0.518 |                18809 |                  20609 |          3607 |
| val     |      4781 |      4781 |       0.518 |                 2090 |                   2287 |           404 |
| test    |      1699 |      1699 |       0.444 |                  779 |                    796 |           124 |

`n_texts` — число уникальных текстов в парах; это же множество получает classification-ветка RQ3, чтобы сравнение объективов было при фиксированных данных.

## Разбивка по вариантам трансформаций

**train**
| pair_type:variant                     |     n |
|:--------------------------------------|------:|
| adversarial_contrast:fallback         |    13 |
| adversarial_contrast:nearest_negative | 20596 |
| benign_twin:nearest                   |  3607 |
| safe_harm_contrast:random             | 18809 |

**val**
| pair_type:variant                     |    n |
|:--------------------------------------|-----:|
| adversarial_contrast:nearest_negative | 2287 |
| benign_twin:nearest                   |  404 |
| safe_harm_contrast:random             | 2090 |

**test**
| pair_type:variant                     |   n |
|:--------------------------------------|----:|
| adversarial_contrast:nearest_negative | 796 |
| benign_twin:nearest                   | 124 |
| safe_harm_contrast:random             | 779 |

## Режим coverage
| split   |   якорей |   триплетов |   подозрений на ошибку разметки |   запасных выборов |   макс. повторов негатива |   у исходного e5 негатив ближе позитива | тексты = весь пул   |
|:--------|---------:|------------:|--------------------------------:|-------------------:|--------------------------:|----------------------------------------:|:--------------------|
| train   |    43025 |       43025 |                             246 |                 13 |                         3 |                                   0.86  | True                |
| val     |     4781 |        4781 |                               1 |                  0 |                         3 |                                   0.863 | True                |
| test    |     1699 |        1699 |                               0 |                  0 |                         3 |                                   0.893 | True                |

Параметры: {'benign_twin_fraction': 0.3, 'top_k': 5, 'search_depth': 100, 'max_negative_uses': 3, 'max_positive_uses': 3, 'noise_threshold': 0.9, 'noise_folds': 5, 'noise_c': 1.0, 'block_size': 512}.

«Подозрения» — тексты, которым out-of-fold логистическая регрессия на эмбеддингах исходного e5 даёт вероятность противоположного класса выше порога. Они остаются якорями с исходной меткой, но не используются как позитивы и негативы. Примеры — в `data/processed/pairs/wildguardmix_pure_v1/summary.json` (тексты могут быть вредоносными, в репозиторий не попадают).
