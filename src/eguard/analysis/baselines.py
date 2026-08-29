"""Контрольные baseline-ы — нижняя граница, без которой RQ1 не читается.

Если TF-IDF + логрегрессия даёт AUROC, сопоставимый с пробом на эмбеддингах, то
«разделимость safe/harm в пространстве энкодера» объясняется лексикой, а не
семантикой, и вывод про геометрию пространства неверен. Аналогично с длиной:
в AEGIS harm-примеры систематически длиннее/короче safe, и проб может ловить
именно это. Оба контроля считаются на тех же сплитах, что и основные пробы.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .probe import binary_metrics


def tfidf_probe(
    train_texts: list[str], y_train: np.ndarray,
    test_texts: list[str], y_test: np.ndarray,
    c: float = 1.0, class_weight: str | None = "balanced",
    target_fpr: float = 0.01, seed: int = 42,
) -> dict[str, float]:
    """Чисто лексическая нижняя граница: слова 1-2gram + logreg."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True)
    xtr = vec.fit_transform(train_texts)
    xte = vec.transform(test_texts)
    clf = LogisticRegression(C=c, max_iter=3000, class_weight=class_weight, random_state=seed)
    clf.fit(xtr, y_train)
    return binary_metrics(y_test, clf.predict_proba(xte)[:, 1], target_fpr=target_fpr)


def length_probe(test_texts: list[str], y_test: np.ndarray, target_fpr: float = 0.01) -> dict[str, float]:
    """Скор = длина текста. AUROC заметно выше 0.5 означает confound по длине."""
    lengths = np.array([len(t) for t in test_texts], dtype=float)
    metrics = binary_metrics(y_test, lengths, threshold=float(np.median(lengths)), target_fpr=target_fpr)
    # AUROC < 0.5 означает «короткие = harm»; для интерпретации важен модуль отклонения от 0.5.
    metrics["auroc_abs_deviation"] = float(abs(metrics["auroc"] - 0.5))
    return metrics
