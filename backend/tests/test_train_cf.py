"""Training loop and artifact behaviour.

These run on small synthetic matrices rather than the seeded database so the
suite stays fast. The properties under test do not depend on scale.
"""

from pathlib import Path

import numpy as np
import pytest

from ml.artifacts import (
    CFArtifacts,
    latest_version,
    list_versions,
    load_cf_artifacts,
    next_version_dir,
    save_cf_artifacts,
)
from ml.cofi import RatingMode
from ml.data import RatingMatrices
from ml.train_cf import TrainConfig, train


def _matrices(
    n_m: int = 12, n_u: int = 8, density: float = 0.5, seed: int = 0, cold_users: int = 1
) -> RatingMatrices:
    """Synthetic explicit-feedback matrices with a guaranteed cold user column."""
    rng = np.random.default_rng(seed)
    Y = rng.integers(1, 6, size=(n_m, n_u)).astype(np.float64)
    R = (rng.uniform(size=(n_m, n_u)) < density).astype(np.float64)
    # Guarantee at least one rated entry per item so mu is well defined.
    for i in range(n_m):
        if R[i].sum() == 0:
            R[i, 0] = 1.0
    # And guarantee the cold users really are cold.
    for j in range(cold_users):
        R[:, -(j + 1)] = 0.0
    Y = Y * R
    return RatingMatrices(
        Y=Y,
        R=R,
        item_ids=list(range(1, n_m + 1)),
        user_ids=list(range(1, n_u + 1)),
        mode=RatingMode.EXPLICIT,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def test_training_reduces_the_cost() -> None:
    result = train(_matrices(), TrainConfig(iterations=60))
    assert result.history[-1] < result.history[0]
    assert len(result.history) == 60


def test_training_is_deterministic_for_a_fixed_seed() -> None:
    """Without this, no reported metric is reproducible and no two model cards
    can be meaningfully compared."""
    matrices = _matrices()
    first = train(matrices, TrainConfig(iterations=40, seed=123))
    second = train(matrices, TrainConfig(iterations=40, seed=123))

    assert np.allclose(first.X, second.X)
    assert np.allclose(first.W, second.W)
    assert np.allclose(first.b, second.b)
    assert first.history == second.history


def test_different_seeds_give_different_parameters() -> None:
    matrices = _matrices()
    first = train(matrices, TrainConfig(iterations=40, seed=1))
    second = train(matrices, TrainConfig(iterations=40, seed=2))
    assert not np.allclose(first.X, second.X)


def test_a_cold_user_is_predicted_the_item_means_after_real_training() -> None:
    """End-to-end version of the mean-normalisation guarantee.

    The unit test in test_cofi.py checks the arithmetic; this one checks that a
    full optimisation run actually lands there, with the cold column driven to
    zero by the regulariser rather than by construction.
    """
    matrices = _matrices(cold_users=1)
    cold_column = matrices.n_u - 1
    assert matrices.R[:, cold_column].sum() == 0

    result = train(matrices, TrainConfig(iterations=250, lambda_=1.0))

    predictions = result.X @ result.W[cold_column] + result.b[0, cold_column] + result.mu
    assert predictions == pytest.approx(result.mu, abs=1e-6)
    assert np.all(predictions >= 1.0), "cold user must not receive sub-scale predictions"


def test_a_warm_user_is_predicted_something_other_than_the_item_means() -> None:
    """The complement of the test above: if every user got the item means, the
    system would not be personalising anything."""
    matrices = _matrices(cold_users=1)
    result = train(matrices, TrainConfig(iterations=250, lambda_=0.1))

    deviations = [
        np.abs(result.X @ result.W[j] + result.b[0, j]).max() for j in range(matrices.n_u - 1)
    ]
    assert max(deviations) > 0.1


def test_implicit_mode_does_not_mean_normalise() -> None:
    """Subtracting an average from a 0/1 label and sigmoid-ing the result is not
    a defined operation, so mu must stay zero on this path."""
    matrices = _matrices()
    binary = RatingMatrices(
        Y=(matrices.Y >= 4).astype(np.float64) * matrices.R,
        R=matrices.R,
        item_ids=matrices.item_ids,
        user_ids=matrices.user_ids,
        mode=RatingMode.IMPLICIT,
    )
    result = train(binary, TrainConfig(mode=RatingMode.IMPLICIT, iterations=30))

    assert np.all(result.mu == 0.0)
    assert "train_log_loss" in result.metrics
    assert "train_rmse" not in result.metrics


def test_no_normalize_flag_disables_mean_subtraction() -> None:
    result = train(_matrices(), TrainConfig(iterations=20, normalize=False))
    assert np.all(result.mu == 0.0)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def _save_dummy(root: Path, n_m: int = 5, n_u: int = 4, n: int = 3) -> Path:
    rng = np.random.default_rng(0)
    directory = next_version_dir(root)
    save_cf_artifacts(
        directory,
        X=rng.normal(size=(n_m, n)),
        W=rng.normal(size=(n_u, n)),
        b=rng.normal(size=(1, n_u)),
        mu=rng.uniform(1, 5, size=n_m),
        item_ids=list(range(1, n_m + 1)),
        user_ids=list(range(1, n_u + 1)),
        metadata={"model": "test"},
    )
    return directory


def test_artifacts_round_trip(tmp_path: Path) -> None:
    directory = _save_dummy(tmp_path)
    with np.load(directory / "cf_model.npz") as data:
        expected_X = data["X"]

    loaded = load_cf_artifacts(tmp_path)
    assert np.allclose(loaded.X, expected_X)
    assert loaded.version == "v1"
    assert loaded.metadata["model"] == "test"
    assert loaded.item_ids == [1, 2, 3, 4, 5]


def test_versions_increment_and_do_not_overwrite(tmp_path: Path) -> None:
    """Each run must leave the previous model intact so a bad deploy can be
    rolled back by pointing MODEL_VERSION at the last good directory."""
    _save_dummy(tmp_path)
    _save_dummy(tmp_path)
    _save_dummy(tmp_path)

    assert list_versions(tmp_path) == ["v1", "v2", "v3"]
    assert latest_version(tmp_path) == "v3"
    assert (tmp_path / "v1" / "cf_model.npz").exists()


def test_an_explicit_version_can_be_loaded(tmp_path: Path) -> None:
    _save_dummy(tmp_path)
    _save_dummy(tmp_path)
    assert load_cf_artifacts(tmp_path, version="v1").version == "v1"


def test_missing_model_raises_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ml.train_cf"):
        load_cf_artifacts(tmp_path)


def test_version_listing_ignores_unrelated_directories(tmp_path: Path) -> None:
    _save_dummy(tmp_path)
    (tmp_path / "scratch").mkdir()
    (tmp_path / "v_bad").mkdir()
    assert list_versions(tmp_path) == ["v1"]


def test_version_ordering_is_numeric_not_lexicographic(tmp_path: Path) -> None:
    """v10 must sort after v9, which a plain string sort would get wrong and
    would silently serve a stale model."""
    for _ in range(11):
        _save_dummy(tmp_path)
    assert latest_version(tmp_path) == "v11"


def test_prediction_helpers_agree(tmp_path: Path) -> None:
    _save_dummy(tmp_path)
    artifacts: CFArtifacts = load_cf_artifacts(tmp_path)

    full = artifacts.predict_all()
    for j in range(len(artifacts.user_ids)):
        assert artifacts.predict_for_user_index(j) == pytest.approx(full[:, j])
