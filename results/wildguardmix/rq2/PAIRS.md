# RQ2 — обучающие пары
Набор: `wildguardmix_harmbench_v1`, seed=42. AEGIS + HarmBench (train_split=`all`, heldout=`None`).
## Размеры
| split   |   n_pairs |   n_texts |   harm_rate |   safe_harm_contrast |   paraphrase |   jailbreak_variant |   adversarial_contrast |   benign_twin |
|:--------|----------:|----------:|------------:|---------------------:|-------------:|--------------------:|-----------------------:|--------------:|
| train   |     32538 |     48984 |       0.545 |                12000 |         4539 |                4999 |                   8000 |          3000 |
| val     |      3256 |      5165 |       0.544 |                 1200 |          456 |                 500 |                    800 |           300 |
| test    |      3233 |      2802 |       0.503 |                 1200 |          434 |                 500 |                    799 |           300 |

`n_texts` — число уникальных текстов в парах; это же множество получает classification-ветка RQ3, чтобы сравнение объективов было при фиксированных данных.

## Разбивка по вариантам трансформаций

**train**
| pair_type:variant                      |    n |
|:---------------------------------------|-----:|
| adversarial_contrast:adversarial       | 8000 |
| benign_twin:cosine_mined               | 3000 |
| jailbreak_variant:base64               |  728 |
| jailbreak_variant:fiction              |  704 |
| jailbreak_variant:instruction_override |  726 |
| jailbreak_variant:leetspeak            |  690 |
| jailbreak_variant:prefix_injection     |  693 |
| jailbreak_variant:research_framing     |  777 |
| jailbreak_variant:role_play            |  681 |
| paraphrase:declarative                 | 1231 |
| paraphrase:indirect                    | 1264 |
| paraphrase:noise                       |  798 |
| paraphrase:polite                      | 1246 |

**val**
| pair_type:variant                      |   n |
|:---------------------------------------|----:|
| adversarial_contrast:adversarial       | 800 |
| benign_twin:cosine_mined               | 300 |
| jailbreak_variant:base64               |  72 |
| jailbreak_variant:fiction              |  66 |
| jailbreak_variant:instruction_override |  57 |
| jailbreak_variant:leetspeak            |  75 |
| jailbreak_variant:prefix_injection     |  86 |
| jailbreak_variant:research_framing     |  71 |
| jailbreak_variant:role_play            |  73 |
| paraphrase:declarative                 | 117 |
| paraphrase:indirect                    | 127 |
| paraphrase:noise                       |  79 |
| paraphrase:polite                      | 133 |

**test**
| pair_type:variant                      |   n |
|:---------------------------------------|----:|
| adversarial_contrast:adversarial       | 799 |
| benign_twin:cosine_mined               | 300 |
| jailbreak_variant:base64               |  75 |
| jailbreak_variant:fiction              |  65 |
| jailbreak_variant:instruction_override |  70 |
| jailbreak_variant:leetspeak            |  68 |
| jailbreak_variant:prefix_injection     |  77 |
| jailbreak_variant:research_framing     |  81 |
| jailbreak_variant:role_play            |  64 |
| paraphrase:declarative                 | 124 |
| paraphrase:indirect                    | 102 |
| paraphrase:noise                       |  75 |
| paraphrase:polite                      | 133 |
