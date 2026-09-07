# RQ2 — обучающие пары
Набор: `aegis_harmbench_v2`, seed=42. AEGIS + HarmBench (train_split=`all`, heldout=`None`).
## Размеры
| split   |   n_pairs |   n_texts |   harm_rate |   safe_harm_contrast |   paraphrase |   jailbreak_variant |   benign_twin |
|:--------|----------:|----------:|------------:|---------------------:|-------------:|--------------------:|--------------:|
| train   |     16738 |      9717 |       0.572 |                 7998 |         3543 |                3997 |          1200 |
| val     |      1709 |      1127 |       0.599 |                  800 |          363 |                 400 |           146 |
| test    |      1712 |      1162 |       0.585 |                  800 |          364 |                 400 |           148 |

`n_texts` — число уникальных текстов в парах; это же множество получает classification-ветка RQ3, чтобы сравнение объективов было при фиксированных данных.

## Разбивка по вариантам трансформаций

**train**
| pair_type:variant                      |    n |
|:---------------------------------------|-----:|
| benign_twin:cosine_mined               | 1200 |
| jailbreak_variant:base64               |  592 |
| jailbreak_variant:fiction              |  536 |
| jailbreak_variant:instruction_override |  562 |
| jailbreak_variant:leetspeak            |  557 |
| jailbreak_variant:prefix_injection     |  611 |
| jailbreak_variant:research_framing     |  579 |
| jailbreak_variant:role_play            |  560 |
| paraphrase:declarative                 |  996 |
| paraphrase:indirect                    |  946 |
| paraphrase:noise                       |  583 |
| paraphrase:polite                      | 1018 |

**val**
| pair_type:variant                      |   n |
|:---------------------------------------|----:|
| benign_twin:cosine_mined               | 146 |
| jailbreak_variant:base64               |  61 |
| jailbreak_variant:fiction              |  58 |
| jailbreak_variant:instruction_override |  54 |
| jailbreak_variant:leetspeak            |  60 |
| jailbreak_variant:prefix_injection     |  57 |
| jailbreak_variant:research_framing     |  50 |
| jailbreak_variant:role_play            |  60 |
| paraphrase:declarative                 | 104 |
| paraphrase:indirect                    |  93 |
| paraphrase:noise                       |  56 |
| paraphrase:polite                      | 110 |

**test**
| pair_type:variant                      |   n |
|:---------------------------------------|----:|
| benign_twin:cosine_mined               | 148 |
| jailbreak_variant:base64               |  54 |
| jailbreak_variant:fiction              |  56 |
| jailbreak_variant:instruction_override |  51 |
| jailbreak_variant:leetspeak            |  70 |
| jailbreak_variant:prefix_injection     |  67 |
| jailbreak_variant:research_framing     |  60 |
| jailbreak_variant:role_play            |  42 |
| paraphrase:declarative                 | 116 |
| paraphrase:indirect                    | 101 |
| paraphrase:noise                       |  52 |
| paraphrase:polite                      |  95 |
