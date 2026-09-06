"""Функции потерь для RQ3.

RQ3 спрашивает, улучшает ли contrastive objective соотношение FPR/FNR по сравнению
с обычной классификационной головой. Чтобы ответ был про objective, а не про
что-то ещё, все три варианта здесь работают поверх одного и того же энкодера,
одной и той же головы и одних и тех же текстов; различается ровно лосс.

  info_nce               — триплеты (anchor, positive, negative) + in-batch negatives
  supervised_contrastive — SupCon: позитивы = все примеры того же класса в батче
  triplet_margin         — классический margin-лосс, как нижняя граница
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor | None = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """InfoNCE с in-batch negatives и явным hard negative (MultipleNegativesRanking).

    Кандидаты для якоря i: все позитивы батча плюс, если передан, столбец явных
    негативов. Правильный ответ — позитив с тем же индексом.
    """
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    candidates = positive
    if negative is not None:
        candidates = torch.cat([positive, F.normalize(negative, dim=-1)], dim=0)

    logits = (anchor @ candidates.T) / temperature
    target = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, target)


def supervised_contrastive(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """SupCon (Khosla et al.): позитивы — все примеры того же класса в батче.

    В отличие от InfoNCE на триплетах, стягивает класс целиком, а не пару. Для
    guardrail это ближе к цели «безопасное многообразие», но чувствительнее к
    дисбалансу классов в батче.
    """
    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.T / temperature

    n = embeddings.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    same = labels.view(-1, 1).eq(labels.view(1, -1)) & ~eye

    # log-softmax по всем, кроме самого себя
    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    # диагональ уже исключена маской `same`; обнуляем -inf явно, иначе -inf*0 = nan
    log_prob = log_prob.masked_fill(eye, 0.0)

    n_positive = same.sum(1)
    valid = n_positive > 0
    if not valid.any():
        return embeddings.sum() * 0.0

    mean_log_prob = (log_prob * same).sum(1)[valid] / n_positive[valid]
    return -mean_log_prob.mean()


def triplet_margin(
    anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float = 0.2
) -> torch.Tensor:
    """Косинусный triplet margin: max(0, margin - cos(a,p) + cos(a,n))."""
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)
    loss = margin - (anchor * positive).sum(-1) + (anchor * negative).sum(-1)
    return F.relu(loss).mean()


def embedding_kl_regularizer(student: torch.Tensor, teacher: torch.Tensor,
                             temperature: float = 0.1) -> torch.Tensor:
    """Якорь к базовой модели: KL между распределениями попарных сходств (RQ4).

    Contrastive-обучение ломает общую семантическую структуру пространства (см. RQ4).
    Регуляризатор требует, чтобы матрица попарных сходств внутри батча оставалась
    близкой к матрице замороженного энкодера, не фиксируя сами векторы.
    """
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher, dim=-1)
    student_log = F.log_softmax(student @ student.T / temperature, dim=-1)
    teacher_prob = F.softmax(teacher @ teacher.T / temperature, dim=-1)
    return F.kl_div(student_log, teacher_prob, reduction="batchmean")


LOSSES = {
    "info_nce": info_nce,
    "supervised_contrastive": supervised_contrastive,
    "triplet_margin": triplet_margin,
}
