# RQ2 — обучающие пары
Набор: `aegis_harmbench_v1`, seed=42. AEGIS + HarmBench (train_split=`val`, heldout=`test`).
## Размеры
| split   |   n_pairs |   n_texts |   harm_rate |   safe_harm_contrast |   paraphrase |   jailbreak_variant |   benign_twin |
|:--------|----------:|----------:|------------:|---------------------:|-------------:|--------------------:|--------------:|
| train   |     16765 |      7904 |       0.465 |                 8000 |         3575 |                3990 |          1200 |
| val     |      1702 |      1110 |       0.594 |                  800 |          357 |                 399 |           146 |
| test    |      1699 |      1172 |       0.6   |                  800 |          351 |                 400 |           148 |

`n_texts` — число уникальных текстов в парах; это же множество получает classification-ветка RQ3, чтобы сравнение объективов было при фиксированных данных.

## Разбивка по вариантам трансформаций

**train**
| pair_type:variant                      |    n |
|:---------------------------------------|-----:|
| benign_twin:cosine_mined               | 1200 |
| jailbreak_variant:base64               |  578 |
| jailbreak_variant:fiction              |  603 |
| jailbreak_variant:instruction_override |  533 |
| jailbreak_variant:leetspeak            |  600 |
| jailbreak_variant:prefix_injection     |  604 |
| jailbreak_variant:research_framing     |  569 |
| jailbreak_variant:role_play            |  503 |
| paraphrase:declarative                 | 1037 |
| paraphrase:indirect                    |  956 |
| paraphrase:noise                       |  552 |
| paraphrase:polite                      | 1030 |

**val**
| pair_type:variant                      |   n |
|:---------------------------------------|----:|
| benign_twin:cosine_mined               | 146 |
| jailbreak_variant:base64               |  49 |
| jailbreak_variant:fiction              |  74 |
| jailbreak_variant:instruction_override |  50 |
| jailbreak_variant:leetspeak            |  55 |
| jailbreak_variant:prefix_injection     |  55 |
| jailbreak_variant:research_framing     |  55 |
| jailbreak_variant:role_play            |  61 |
| paraphrase:declarative                 |  80 |
| paraphrase:indirect                    | 108 |
| paraphrase:noise                       |  60 |
| paraphrase:polite                      | 109 |

**test**
| pair_type:variant                      |   n |
|:---------------------------------------|----:|
| benign_twin:cosine_mined               | 148 |
| jailbreak_variant:base64               |  58 |
| jailbreak_variant:fiction              |  64 |
| jailbreak_variant:instruction_override |  57 |
| jailbreak_variant:leetspeak            |  44 |
| jailbreak_variant:prefix_injection     |  66 |
| jailbreak_variant:research_framing     |  51 |
| jailbreak_variant:role_play            |  60 |
| paraphrase:declarative                 |  99 |
| paraphrase:indirect                    | 103 |
| paraphrase:noise                       |  60 |
| paraphrase:polite                      |  89 |
