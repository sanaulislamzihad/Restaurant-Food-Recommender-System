"""Correctness of the hand-rolled clustering metrics.

These are implemented in numpy rather than pulled from scikit-learn, so they
need their own tests. The Adjusted Rand Index case below is scikit-learn's own
documented example, which pins the implementation to a known-correct value
rather than to whatever it happens to return.
"""

import numpy as np
import pytest

from scripts.verify_clusters import adjusted_rand_index, cosine_similarity, kmeans, purity


def test_ari_is_one_for_identical_partitions() -> None:
    predicted = np.array([0, 0, 1, 1, 2, 2])
    truth = ["a", "a", "b", "b", "c", "c"]
    assert adjusted_rand_index(predicted, truth) == pytest.approx(1.0)


def test_ari_matches_the_reference_worked_example() -> None:
    """scikit-learn's documented example: ARI([0,0,0,1,1,1], [0,0,1,1,2,2]) = 0.2424."""
    predicted = np.array([0, 0, 1, 1, 2, 2])
    truth = ["x", "x", "x", "y", "y", "y"]
    assert adjusted_rand_index(predicted, truth) == pytest.approx(0.24242424, abs=1e-6)


def test_ari_goes_negative_when_the_partition_is_worse_than_chance() -> None:
    predicted = np.array([0, 1, 0, 1])
    truth = ["a", "a", "b", "b"]
    assert adjusted_rand_index(predicted, truth) == pytest.approx(-0.5)


def test_ari_is_near_zero_for_an_unrelated_partition() -> None:
    rng = np.random.default_rng(0)
    truth = ["a"] * 150 + ["b"] * 150
    predicted = rng.integers(0, 2, size=300)
    assert abs(adjusted_rand_index(predicted, truth)) < 0.05


def test_purity_is_one_when_clusters_are_pure() -> None:
    predicted = np.array([0, 0, 1, 1])
    truth = ["a", "a", "b", "b"]
    assert purity(predicted, truth) == pytest.approx(1.0)


def test_purity_reflects_a_fully_mixed_partition() -> None:
    predicted = np.array([0, 1, 0, 1])
    truth = ["a", "a", "b", "b"]
    assert purity(predicted, truth) == pytest.approx(0.5)


def test_cosine_similarity_is_one_on_the_diagonal() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    similarity = cosine_similarity(vectors)
    assert np.allclose(np.diag(similarity), 1.0)
    # Orthogonal rows.
    assert similarity[0, 1] == pytest.approx(0.0)


def test_cosine_similarity_tolerates_all_zero_rows() -> None:
    """A user who has rated nothing produces a zero vector; dividing by its norm
    would emit NaN and silently poison every downstream mean."""
    vectors = np.array([[0.0, 0.0], [1.0, 1.0]])
    similarity = cosine_similarity(vectors)
    assert not np.isnan(similarity).any()


def test_kmeans_separates_well_separated_blobs() -> None:
    rng = np.random.default_rng(1)
    blob_a = rng.normal(loc=[-6.0, -6.0], scale=0.4, size=(60, 2))
    blob_b = rng.normal(loc=[6.0, 6.0], scale=0.4, size=(60, 2))
    vectors = np.vstack([blob_a, blob_b])
    truth = ["a"] * 60 + ["b"] * 60

    labels = kmeans(vectors, k=2, seed=0)
    assert purity(labels, truth) == pytest.approx(1.0)
    assert adjusted_rand_index(labels, truth) == pytest.approx(1.0)


def test_kmeans_is_deterministic_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(80, 5))
    assert np.array_equal(kmeans(vectors, k=3, seed=42), kmeans(vectors, k=3, seed=42))
