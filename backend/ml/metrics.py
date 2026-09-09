"""Evaluation metrics.

Two families, measuring different things:

* **Rating accuracy** (RMSE, MAE) - how close the predicted star rating is. Easy
  to report, and largely beside the point: a customer never sees a predicted
  number, they see an ordered list.
* **Ranking quality** (Precision@K, Recall@K, NDCG@K) - whether the items the
  user actually went on to like appear near the top. This is what the product
  is judged on.

Plus two health checks that a good accuracy score can hide entirely: catalogue
coverage, and popularity bias. A recommender that shows everyone the same
twenty best-selling dishes can score respectably on ranking metrics while being
useless as a recommender.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ml.split import RatingEvent

#: A held-out rating at or above this counts as an item the user liked.
DEFAULT_RELEVANCE_THRESHOLD = 4.0


@dataclass
class RankingReport:
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    users_evaluated: int
    catalogue_coverage: float
    mean_popularity_percentile: float
    top_decile_share: float
    gini: float
    k: int = 10
    per_user: dict[int, float] = field(default_factory=dict)


def rmse(predictions: np.ndarray, truth: np.ndarray) -> float:
    """Root mean squared error over paired arrays."""
    if predictions.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((predictions - truth) ** 2)))


def mae(predictions: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute error over paired arrays."""
    if predictions.size == 0:
        return float("nan")
    return float(np.mean(np.abs(predictions - truth)))


def rating_accuracy(
    scores: np.ndarray,
    events: list[RatingEvent],
    item_pos: dict[int, int],
    user_pos: dict[int, int],
    *,
    clip: tuple[float, float] | None = (1.0, 5.0),
) -> dict[str, float]:
    """RMSE and MAE of ``scores`` against held-out ratings.

    Predictions are clipped to the rating scale by default. A model is free to
    predict 6.3 stars internally, but reporting the error of a value that could
    never be displayed overstates how wrong it is.
    """
    if not events:
        return {"rmse": float("nan"), "mae": float("nan"), "n": 0}

    predicted = np.array([scores[item_pos[e.item_id], user_pos[e.user_id]] for e in events])
    truth = np.array([e.rating for e in events])
    if clip is not None:
        predicted = np.clip(predicted, *clip)

    return {"rmse": rmse(predicted, truth), "mae": mae(predicted, truth), "n": len(events)}


def _ranked_items(scores_for_user: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Item indices ordered best first, with deterministic tie-breaking.

    The tiebreak key matters more than it looks. A global-mean baseline assigns
    every item the same score, so without a tiebreak the ranking would come out
    in item-id order - which is arbitrary, but not *randomly* arbitrary, and
    could flatter or penalise the baseline depending on how ids happen to
    correlate with quality. A fixed random permutation makes the degenerate case
    an honest random ranking, and keeps it reproducible.
    """
    return np.lexsort((tiebreak, -scores_for_user))


def _dcg(relevances: np.ndarray) -> float:
    positions = np.arange(1, len(relevances) + 1)
    return float(np.sum(relevances / np.log2(positions + 1)))


def gini_coefficient(values: np.ndarray) -> float:
    """Inequality of a distribution: 0 = perfectly even, 1 = one item takes all.

    Applied to how often each dish is recommended.
    """
    if values.size == 0 or values.sum() == 0:
        return 0.0
    ordered = np.sort(values.astype(np.float64))
    n = ordered.size
    index = np.arange(1, n + 1)
    return float(((2 * index - n - 1) @ ordered) / (n * ordered.sum()))


def ranking_quality(
    scores: np.ndarray,
    train_R: np.ndarray,
    test_events: list[RatingEvent],
    item_pos: dict[int, int],
    user_pos: dict[int, int],
    *,
    k: int = 10,
    relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    seed: int = 0,
) -> RankingReport:
    """Precision@K, Recall@K, NDCG@K plus coverage and popularity bias.

    Args:
        scores: (n_m, n_u) predicted score for every item/user pair.
        train_R: (n_m, n_u) mask of what was seen during training. Items a user
            already rated in train are excluded from their candidate list -
            recommending a dish someone has already reviewed is not a
            recommendation.
        test_events: held-out ratings.
        item_pos / user_pos: id to matrix index maps.
        k: cut-off.
        relevance_threshold: held-out rating at or above which an item counts as
            one the user liked.
        seed: controls tie-break ordering only.

    Only users with at least one *relevant* held-out item are scored. Including
    users whose entire held-out block is dishes they disliked would divide by
    zero on recall and drag precision toward zero for reasons unrelated to
    ranking quality.

    Unrated items are treated as not relevant, the standard convention. It
    understates precision - some of those dishes the user would have loved,
    nobody asked them - so the absolute numbers read low across every model in
    the table. The comparison between rows is still fair.
    """
    n_m, _ = scores.shape
    rng = np.random.default_rng(seed)
    tiebreak = rng.permutation(n_m)

    relevant_by_user: dict[int, set[int]] = {}
    for event in test_events:
        if event.rating >= relevance_threshold:
            relevant_by_user.setdefault(event.user_id, set()).add(event.item_id)

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    per_user: dict[int, float] = {}
    recommended_counts = np.zeros(n_m, dtype=np.int64)

    for user_id, relevant_items in relevant_by_user.items():
        j = user_pos[user_id]
        user_scores = scores[:, j].astype(np.float64).copy()
        # Exclude anything already rated in training from the candidate list.
        user_scores[train_R[:, j] == 1] = -np.inf

        order = _ranked_items(user_scores, tiebreak)
        top_k = [idx for idx in order[:k] if np.isfinite(user_scores[idx])]
        if not top_k:
            continue

        recommended_counts[top_k] += 1
        relevant_positions = {item_pos[item_id] for item_id in relevant_items}
        gains = np.array([1.0 if idx in relevant_positions else 0.0 for idx in top_k])

        hits = float(gains.sum())
        precisions.append(hits / k)
        recalls.append(hits / len(relevant_items))

        ideal = np.ones(min(k, len(relevant_items)))
        ndcg = _dcg(gains) / _dcg(ideal) if ideal.size else 0.0
        ndcgs.append(ndcg)
        per_user[user_id] = ndcg

    # Popularity bias, measured against how often each item was rated in train.
    train_popularity = train_R.sum(axis=1)
    percentiles = _to_percentiles(train_popularity)
    recommended_mask = recommended_counts > 0

    total_slots = int(recommended_counts.sum())
    if total_slots:
        weighted_percentile = float(np.sum(percentiles * recommended_counts) / total_slots)
        top_decile = percentiles >= 0.9
        top_decile_share = float(recommended_counts[top_decile].sum() / total_slots)
    else:
        weighted_percentile = float("nan")
        top_decile_share = float("nan")

    return RankingReport(
        precision_at_k=float(np.mean(precisions)) if precisions else float("nan"),
        recall_at_k=float(np.mean(recalls)) if recalls else float("nan"),
        ndcg_at_k=float(np.mean(ndcgs)) if ndcgs else float("nan"),
        users_evaluated=len(precisions),
        catalogue_coverage=float(recommended_mask.sum() / n_m),
        mean_popularity_percentile=weighted_percentile,
        top_decile_share=top_decile_share,
        gini=gini_coefficient(recommended_counts),
        k=k,
        per_user=per_user,
    )


def _to_percentiles(values: np.ndarray) -> np.ndarray:
    """Map values to their percentile rank in [0, 1], ties sharing a rank."""
    if values.size == 0:
        return values
    order = values.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    denominator = max(values.size - 1, 1)
    return ranks / denominator
