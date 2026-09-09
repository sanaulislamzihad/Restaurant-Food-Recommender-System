"""Item nearest-neighbour precomputation."""

from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import FoodItem, ItemNeighbor, Restaurant
from ml.artifacts import (
    latest_version_for_mode,
    load_cf_artifacts,
    next_version_dir,
    save_cf_artifacts,
)
from ml.neighbors import rebuild_item_neighbors, squared_distances, top_neighbors


def test_squared_distances_on_known_points() -> None:
    X = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    distances = squared_distances(X)

    assert distances[0, 1] == pytest.approx(25.0)  # 3-4-5 triangle
    assert distances[0, 2] == pytest.approx(1.0)
    assert distances[1, 2] == pytest.approx(9.0 + 9.0)


def test_distances_are_symmetric_with_a_zero_diagonal() -> None:
    X = np.random.default_rng(0).normal(size=(8, 4))
    distances = squared_distances(X)

    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0.0, atol=1e-9)
    assert np.all(distances >= 0.0), "floating point must not produce negative distances"


def test_an_item_is_never_its_own_neighbour() -> None:
    X = np.random.default_rng(1).normal(size=(10, 3))
    for i, neighbours in enumerate(top_neighbors(X, top_n=5)):
        assert i not in [j for j, _ in neighbours]


def test_neighbours_come_back_nearest_first() -> None:
    X = np.array([[0.0], [1.0], [5.0], [20.0]])
    neighbours = top_neighbors(X, top_n=3)

    assert [j for j, _ in neighbours[0]] == [1, 2, 3]
    distances = [d for _, d in neighbours[0]]
    assert distances == sorted(distances)


def test_top_n_is_capped_by_the_catalogue_size() -> None:
    X = np.random.default_rng(2).normal(size=(3, 2))
    assert all(len(row) == 2 for row in top_neighbors(X, top_n=10))


def test_a_single_item_catalogue_has_no_neighbours() -> None:
    assert top_neighbors(np.array([[1.0, 2.0]]), top_n=5) == [[]]


# ---------------------------------------------------------------------------
# Database rebuild
# ---------------------------------------------------------------------------


def _catalogue(session: Session, count: int) -> list[int]:
    restaurant = Restaurant(name="Test", area="Dhanmondi", cuisine_tags=["bengali"])
    session.add(restaurant)
    session.flush()
    ids = []
    for i in range(count):
        item = FoodItem(
            restaurant_id=restaurant.id,
            name=f"Dish {i}",
            cuisine="bengali",
            price=Decimal("100.00"),
            ingredient_tags=[],
        )
        session.add(item)
        session.flush()
        ids.append(item.id)
    session.commit()
    return ids


def test_rebuild_writes_ranked_rows(db_session: Session) -> None:
    item_ids = _catalogue(db_session, 6)
    X = np.random.default_rng(3).normal(size=(6, 3))

    report = rebuild_item_neighbors(db_session, X, item_ids, top_n=3)

    assert report.items_processed == 6
    assert report.rows_written == 18
    rows = db_session.execute(
        select(ItemNeighbor.rank)
        .where(ItemNeighbor.food_item_id == item_ids[0])
        .order_by(ItemNeighbor.rank)
    ).all()
    assert [r[0] for r in rows] == [1, 2, 3]


def test_rebuild_replaces_rather_than_appends(db_session: Session) -> None:
    """Running it twice must not double the table or leave stale pairs behind."""
    item_ids = _catalogue(db_session, 5)
    X = np.random.default_rng(4).normal(size=(5, 3))

    first = rebuild_item_neighbors(db_session, X, item_ids, top_n=2)
    second = rebuild_item_neighbors(db_session, X, item_ids, top_n=2)

    total = db_session.scalar(select(func.count()).select_from(ItemNeighbor))
    assert first.rows_written == second.rows_written == total


def test_items_missing_from_the_database_are_skipped(db_session: Session) -> None:
    """The catalogue can change between training and this run; a stale id must
    not be written as a dangling foreign key."""
    item_ids = _catalogue(db_session, 4)
    stale = [*item_ids, 999_999]
    X = np.random.default_rng(5).normal(size=(5, 3))

    report = rebuild_item_neighbors(db_session, X, stale, top_n=3)

    assert report.items_processed == 4
    written = {row[0] for row in db_session.execute(select(ItemNeighbor.neighbor_id)).all()}
    assert 999_999 not in written


def test_degenerate_feature_rows_are_counted(db_session: Session) -> None:
    """Items nobody rated collapse to the origin, so their neighbours are an
    artefact of regularisation. The count is surfaced rather than hidden."""
    item_ids = _catalogue(db_session, 4)
    X = np.random.default_rng(6).normal(size=(4, 3))
    X[2] = 0.0

    report = rebuild_item_neighbors(db_session, X, item_ids, top_n=2)
    assert report.degenerate_items == 1


# ---------------------------------------------------------------------------
# Mode-aware artifact selection
# ---------------------------------------------------------------------------


def _save(root: Path, mode: str) -> str:
    rng = np.random.default_rng(0)
    directory = next_version_dir(root)
    save_cf_artifacts(
        directory,
        X=rng.normal(size=(4, 2)),
        W=rng.normal(size=(3, 2)),
        b=rng.normal(size=(1, 3)),
        mu=rng.uniform(1, 5, size=4),
        item_ids=[1, 2, 3, 4],
        user_ids=[1, 2, 3],
        metadata={"hyperparameters": {"mode": mode}},
    )
    return directory.name


def test_the_newest_model_of_a_given_mode_is_selected(tmp_path: Path) -> None:
    """Selecting purely by 'newest' silently hands back whichever run finished
    last, which is how the neighbour table first got built from the implicit
    model - measurably worse features for this purpose."""
    _save(tmp_path, "explicit")
    _save(tmp_path, "implicit")

    assert latest_version_for_mode(tmp_path, "explicit") == "v1"
    assert latest_version_for_mode(tmp_path, "implicit") == "v2"
    assert load_cf_artifacts(tmp_path, mode="explicit").version == "v1"
    # Unfiltered still means newest.
    assert load_cf_artifacts(tmp_path).version == "v2"


def test_requesting_an_absent_mode_names_the_command_that_fixes_it(tmp_path: Path) -> None:
    _save(tmp_path, "explicit")
    with pytest.raises(FileNotFoundError, match="--mode implicit"):
        load_cf_artifacts(tmp_path, mode="implicit")


def test_an_explicit_version_overrides_the_mode_filter(tmp_path: Path) -> None:
    _save(tmp_path, "explicit")
    _save(tmp_path, "implicit")
    assert load_cf_artifacts(tmp_path, "v2", mode="explicit").version == "v2"
