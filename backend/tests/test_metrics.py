"""Evaluation metrics, pinned to hand-computed values.

NDCG in particular is easy to implement plausibly and wrongly - an off-by-one in
the log base or the position index changes the number without making anything
crash. The cases below are worked out by hand from the definition.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from ml.baselines import global_mean_model, item_mean_model, most_popular_model
from ml.cofi import RatingMode
from ml.data import RatingMatrices
from ml.metrics import gini_coefficient, mae, ranking_quality, rating_accuracy, rmse
from ml.split import RatingEvent

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def test_rmse_and_mae_on_known_values() -> None:
    predictions = np.array([3.0, 4.0, 5.0])
    truth = np.array([2.0, 4.0, 3.0])
    # errors 1, 0, -2 -> mean square 5/3, mean abs 1
    assert rmse(predictions, truth) == pytest.approx(np.sqrt(5 / 3))
    assert mae(predictions, truth) == pytest.approx(1.0)


def test_rating_accuracy_clips_to_the_rating_scale() -> None:
    """A model may predict 9 stars internally, but nobody could ever be shown
    that, so scoring it as a 4-star error overstates the mistake."""
    scores = np.array([[9.0]])
    events = [RatingEvent(1, 1, 5.0, BASE)]
    clipped = rating_accuracy(scores, events, {1: 0}, {1: 0})
    unclipped = rating_accuracy(scores, events, {1: 0}, {1: 0}, clip=None)

    assert clipped["rmse"] == pytest.approx(0.0)
    assert unclipped["rmse"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _single_user_case(scores_column: list[float], relevant_item: int, k: int = 3):
    """One user, five items, none rated in train, one relevant held-out item."""
    n_m = len(scores_column)
    scores = np.array(scores_column, dtype=np.float64).reshape(n_m, 1)
    train_R = np.zeros((n_m, 1))
    events = [RatingEvent(1, relevant_item, 5.0, BASE)]
    item_pos = {i + 1: i for i in range(n_m)}
    return ranking_quality(scores, train_R, events, item_pos, {1: 0}, k=k)


def test_ndcg_is_one_when_the_relevant_item_ranks_first() -> None:
    report = _single_user_case([9.0, 1.0, 1.0, 1.0, 1.0], relevant_item=1)
    assert report.ndcg_at_k == pytest.approx(1.0)
    assert report.precision_at_k == pytest.approx(1 / 3)
    assert report.recall_at_k == pytest.approx(1.0)


def test_ndcg_discounts_the_second_position() -> None:
    """DCG = 1/log2(3) with a single relevant item in position 2."""
    report = _single_user_case([5.0, 9.0, 1.0, 1.0, 1.0], relevant_item=1)
    assert report.ndcg_at_k == pytest.approx(1 / np.log2(3))


def test_ndcg_discounts_the_third_position() -> None:
    report = _single_user_case([3.0, 9.0, 5.0, 1.0, 1.0], relevant_item=1)
    assert report.ndcg_at_k == pytest.approx(1 / np.log2(4))


def test_ndcg_is_zero_when_the_relevant_item_misses_the_cutoff() -> None:
    report = _single_user_case([0.0, 9.0, 8.0, 7.0, 6.0], relevant_item=1, k=3)
    assert report.ndcg_at_k == pytest.approx(0.0)
    assert report.recall_at_k == pytest.approx(0.0)


def test_items_already_rated_in_training_are_never_recommended() -> None:
    """Recommending a dish someone has already reviewed is not a recommendation,
    and letting it occupy a slot would inflate every ranking metric."""
    scores = np.array([[9.0], [8.0], [7.0]])
    train_R = np.array([[1.0], [0.0], [0.0]])  # item 1 already rated
    events = [RatingEvent(1, 2, 5.0, BASE)]

    report = ranking_quality(scores, train_R, events, {1: 0, 2: 1, 3: 2}, {1: 0}, k=2)
    # Item 2 is now top of the candidate list, so it ranks first.
    assert report.ndcg_at_k == pytest.approx(1.0)


def test_users_with_no_relevant_held_out_items_are_skipped() -> None:
    """Otherwise recall divides by zero and precision is dragged down for
    reasons that have nothing to do with ranking quality."""
    scores = np.array([[1.0], [2.0], [3.0]])
    train_R = np.zeros((3, 1))
    events = [RatingEvent(1, 1, 2.0, BASE)]  # disliked, below the threshold

    report = ranking_quality(scores, train_R, events, {1: 0, 2: 1, 3: 2}, {1: 0}, k=2)
    assert report.users_evaluated == 0
    assert np.isnan(report.precision_at_k)


def test_coverage_counts_distinct_recommended_items() -> None:
    """Two users given identical scores must reach the same few items, which is
    exactly the failure coverage is meant to expose."""
    scores = np.tile(np.array([[5.0], [4.0], [3.0], [2.0]]), (1, 2))
    train_R = np.zeros((4, 2))
    events = [RatingEvent(1, 1, 5.0, BASE), RatingEvent(2, 2, 5.0, BASE)]

    report = ranking_quality(scores, train_R, events, {1: 0, 2: 1, 3: 2, 4: 3}, {1: 0, 2: 1}, k=2)
    assert report.catalogue_coverage == pytest.approx(0.5)  # items 1 and 2 only


# ---------------------------------------------------------------------------
# Popularity bias
# ---------------------------------------------------------------------------


def test_gini_is_zero_for_an_even_distribution() -> None:
    assert gini_coefficient(np.array([5, 5, 5, 5])) == pytest.approx(0.0)


def test_gini_approaches_one_when_a_single_item_takes_everything() -> None:
    assert gini_coefficient(np.array([0, 0, 0, 100])) > 0.7


def test_gini_handles_an_all_zero_distribution() -> None:
    assert gini_coefficient(np.array([0, 0, 0])) == 0.0


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _train_matrices() -> RatingMatrices:
    Y = np.array([[5.0, 3.0, 0.0], [4.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    R = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    return RatingMatrices(
        Y=Y, R=R, item_ids=[1, 2, 3], user_ids=[1, 2, 3], mode=RatingMode.EXPLICIT
    )


def test_global_mean_predicts_one_constant() -> None:
    model = global_mean_model(_train_matrices())
    assert np.allclose(model.scores, np.mean([5.0, 3.0, 4.0, 2.0]))
    assert model.predicts_ratings


def test_item_mean_predicts_per_item_averages() -> None:
    model = item_mean_model(_train_matrices())
    assert model.scores[0, 0] == pytest.approx(4.0)  # (5 + 3) / 2
    assert model.scores[1, 0] == pytest.approx(3.0)  # (4 + 2) / 2
    # Every user gets the same column - no personalisation whatsoever.
    assert np.allclose(model.scores[:, 0], model.scores[:, 1])


def test_item_mean_falls_back_to_the_global_mean_for_unrated_items() -> None:
    model = item_mean_model(_train_matrices())
    assert model.scores[2, 0] == pytest.approx(3.5)


def test_most_popular_is_not_treated_as_a_rating_predictor() -> None:
    """Its scores are counts, not stars. Reporting an RMSE for them would be
    reporting a number on the wrong scale."""
    model = most_popular_model(_train_matrices())
    assert not model.predicts_ratings
    assert model.scores[0, 0] == 2.0
    assert model.scores[2, 0] == 0.0
