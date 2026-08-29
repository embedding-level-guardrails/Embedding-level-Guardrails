# RQ1 — baseline-разделимость safe/harm на frozen-энкодерах

Датасет: `nvidia/Aegis-AI-Content-Safety-Dataset-1.0`, text_types=['user_message'], caution_policy=`exclude`, seed=42

## Данные

| split   |    n |   harm_rate |   median_chars |
|:--------|-----:|------------:|---------------:|
| train   | 2318 |       0.576 |             49 |
| test    |  308 |       0.597 |             52 |
| val     |  257 |       0.576 |             46 |

## Косинусная близость intra vs inter

| model        | split   | variant   |   cos_intra_safe |   cos_intra_harm |   cos_inter |   cos_random_pair |    gap |   gap_std_units |   centroid_cos |
|:-------------|:--------|:----------|-----------------:|-----------------:|------------:|------------------:|-------:|----------------:|---------------:|
| e5-small-v2  | test    | raw       |           0.772  |           0.787  |      0.7699 |            0.7763 | 0.0096 |          0.2696 |         0.9858 |
| e5-small-v2  | test    | centered  |           0.027  |           0.0122 |     -0.0246 |           -0.0032 | 0.0442 |          0.3774 |        -0.9928 |
| ettin-68m    | test    | raw       |           0.8642 |           0.9586 |      0.902  |            0.9161 | 0.0093 |          0.0887 |         0.9904 |
| ettin-68m    | test    | centered  |           0.031  |           0.1206 |     -0.0658 |            0.0162 | 0.1417 |          0.3899 |        -0.9432 |
| mmbert-small | test    | raw       |           0.9245 |           0.9656 |      0.9397 |            0.9465 | 0.0054 |          0.106  |         0.9941 |
| mmbert-small | test    | centered  |           0.0256 |           0.0535 |     -0.0396 |            0.0041 | 0.0791 |          0.3618 |        -0.8939 |


Читать так: `gap` — разрыв между средним внутриклассовым и межклассовым косинусом, `gap_std_units` — тот же разрыв в единицах разброса, `cos_random_pair` — уровень анизотропии (на сколько уже смещены косинусы случайных пар). Строки `variant=centered` — после вычитания глобального среднего.


## Геометрия

| model        | split   |   mean_random_cosine |   pc1_explained_var |   top10_explained_var |   direction_auc_roc |   direction_cohens_d |   direction_norm |   direction_vs_pc1_cos |   silhouette_label_cosine |   davies_bouldin_label |   ari_k2 |   nmi_k2 |   ari_k8 |   nmi_k8 |
|:-------------|:--------|---------------------:|--------------------:|----------------------:|--------------------:|---------------------:|-----------------:|-----------------------:|--------------------------:|-----------------------:|---------:|---------:|---------:|---------:|
| e5-small-v2  | test    |               0.7763 |              0.057  |                0.3015 |              0.9065 |               1.8654 |           1.1474 |                 0.8367 |                    0.048  |                 6.253  |   0.1292 |   0.0924 |   0.1049 |   0.1562 |
| ettin-68m    | test    |               0.9161 |              0.5569 |                0.7307 |              0.7649 |               0.9938 |          21.3804 |                 0.9923 |                    0.1867 |                 2.5898 |   0.1469 |   0.1215 |   0.0891 |   0.1161 |
| mmbert-small | test    |               0.9465 |              0.3081 |                0.5946 |              0.7809 |               0.9824 |           6.8647 |                 0.9439 |                    0.1441 |                 3.7394 |   0.0927 |   0.1543 |   0.0348 |   0.0798 |


`direction_auc_roc` — AUC ROC проекции на разницу центроидов (одномерный сигнал), `ari_k2` — совпадает ли неразмеченная кластеризация с safety-меткой, `silhouette_label_cosine` — насколько метки образуют компактные кластеры.


## Пробы

| model        | probe             | auc_roc | auc_prc |     f1 |    fpr |    fnr |   tpr_at_fpr_0.01 |
|:-------------|:------------------|--------:|--------:|-------:|-------:|-------:|------------------:|
| e5-small-v2  | logreg            |  0.9353 |  0.9535 | 0.8815 | 0.1532 | 0.1304 |            0.4076 |
| e5-small-v2  | centroid_distance |  0.9069 |  0.9284 | 0.8696 | 0.2984 | 0.0761 |            0.288  |
| e5-small-v2  | knn_k15           |  0.9341 |  0.9524 | 0.8713 | 0.3548 | 0.0435 |            0.4348 |
| ettin-68m    | logreg            |  0.9106 |  0.9394 | 0.873  | 0.2339 | 0.1033 |            0.3804 |
| ettin-68m    | centroid_distance |  0.7521 |  0.7934 | 0.7838 | 0.5806 | 0.1033 |            0.0435 |
| ettin-68m    | knn_k15           |  0.8628 |  0.9001 | 0.8295 | 0.5645 | 0.0217 |            0.3261 |
| mmbert-small | logreg            |  0.9097 |  0.9275 | 0.8587 | 0.2419 | 0.125  |            0.212  |
| mmbert-small | centroid_distance |  0.7986 |  0.8268 | 0.801  | 0.5081 | 0.1033 |            0.0707 |
| mmbert-small | knn_k15           |  0.8935 |  0.9268 | 0.8364 | 0.5242 | 0.0272 |            0.3804 |
| —            | control_tfidf     |  0.8861 |  0.9227 | 0.8384 | 0.2258 | 0.1685 |            0.2609 |
| —            | control_length    |  0.7665 |  0.7918 | 0.716  | 0.2661 | 0.3424 |            0.0489 |


Разрыв между `logreg` и `knn` показывает долю нелинейного сигнала; `centroid_distance` — прообраз distance-based guardrail из RQ5; `native_head` — собственная голова guardrail-модели на тех же данных. Строки `control_tfidf` и `control_length` — нижняя граница: если проб на эмбеддингах не выигрывает у TF-IDF, разделимость лексическая, а не семантическая. `auc_roc_lo`/`auc_roc_hi` — 95% перцентильный bootstrap-CI.


## Error analysis

| model        |   max_cross_class_cosine |   mean_top_pair_cosine | top_confused_category         |   n_false_positives_listed |   n_false_negatives_listed |
|:-------------|-------------------------:|-----------------------:|:------------------------------|---------------------------:|---------------------------:|
| e5-small-v2  |                   0.9298 |                 0.8758 | Criminal Planning/Confessions |                          1 |                         30 |
| ettin-68m    |                   0.9908 |                 0.9876 | Criminal Planning/Confessions |                          1 |                         30 |
| mmbert-small |                   0.9888 |                 0.9863 | Criminal Planning/Confessions |                          1 |                         30 |


`max_cross_class_cosine` — насколько близко энкодер кладёт safe и harm друг к другу; подробные таблицы hard-пар и ошибок проба лежат в `results/rq1/errors/` (содержат фрагменты вредоносных текстов, не публиковать).


## Визуализации

![cos_hist_e5-small-v2](../figures/cos_hist_e5-small-v2.png)

![cos_hist_ettin-68m](../figures/cos_hist_ettin-68m.png)

![cos_hist_mmbert-small](../figures/cos_hist_mmbert-small.png)

![pca_e5-small-v2_test](../figures/pca_e5-small-v2_test.png)

![pca_ettin-68m_test](../figures/pca_ettin-68m_test.png)

![pca_mmbert-small_test](../figures/pca_mmbert-small_test.png)

![tsne_e5-small-v2_test](../figures/tsne_e5-small-v2_test.png)

![tsne_ettin-68m_test](../figures/tsne_ettin-68m_test.png)

![tsne_mmbert-small_test](../figures/tsne_mmbert-small_test.png)

![umap_e5-small-v2_test](../figures/umap_e5-small-v2_test.png)

![umap_ettin-68m_test](../figures/umap_ettin-68m_test.png)

![umap_mmbert-small_test](../figures/umap_mmbert-small_test.png)
