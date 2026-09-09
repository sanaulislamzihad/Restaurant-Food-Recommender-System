"""Precompute nearest neighbours on the learned item features.

    python -m ml.build_neighbors

For every dish, find the closest others by squared distance on the learned
feature vectors:

    distance(k, i) = || X[k] - X[i] ||^2

and store the top N in ``item_neighbors``. This is what powers "you liked
Kacchi Biryani — try these". It is precomputed rather than derived per request
because scoring one item against the whole catalogue inside a request handler
would put an O(n_m) numpy pass on the hot path for a result that only changes
when the model is retrained.

A caveat that matters for interpreting the output: a dish nobody has rated has
its feature row pulled to the origin by the regulariser, so it sits near every
other unrated dish and its "neighbours" are an artefact of that, not a real
similarity. Those items are counted and reported rather than silently written
out as though they meant something; the content model in M6 is what actually
answers similarity for cold items.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from app.db.models import FoodItem, ItemNeighbor

#: Below this L2 norm a feature row is treated as uninformative - the
#: regulariser has collapsed it rather than the data having shaped it.
DEGENERATE_NORM = 1e-3


@dataclass
class NeighborReport:
    items_processed: int
    rows_written: int
    degenerate_items: int
    top_n: int


def squared_distances(X: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances between the rows of X.

    Args:
        X: (n_m, n) learned item feature matrix.

    Returns:
        (n_m, n_m) matrix where entry (k, i) is ``|| X[k] - X[i] ||^2``.

    Expanded as ``|x_k|^2 + |x_i|^2 - 2 x_k . x_i`` so the whole thing is one
    matrix multiply instead of a double loop. Floating point can push exact
    zeros very slightly negative on the diagonal, so the result is clamped.
    """
    squared_norms = np.sum(X**2, axis=1)
    distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (X @ X.T)
    clamped: np.ndarray = np.maximum(distances, 0.0)
    return clamped


def top_neighbors(X: np.ndarray, top_n: int = 10) -> list[list[tuple[int, float]]]:
    """For each item row, the ``top_n`` closest other rows.

    Args:
        X: (n_m, n) item feature matrix.
        top_n: neighbours to keep per item.

    Returns:
        A list of length n_m; entry i holds ``(neighbour_index, distance)``
        pairs ordered nearest first, never including i itself.
    """
    distances = squared_distances(X)
    # An item is always its own nearest neighbour at distance 0, and
    # recommending a dish as similar to itself would waste a carousel slot.
    np.fill_diagonal(distances, np.inf)

    n_m = X.shape[0]
    keep = min(top_n, max(n_m - 1, 0))
    if keep == 0:
        return [[] for _ in range(n_m)]

    # argpartition finds the k smallest without sorting all n_m, then only that
    # slice is sorted.
    partitioned = np.argpartition(distances, kth=keep - 1, axis=1)[:, :keep]
    results: list[list[tuple[int, float]]] = []
    for i in range(n_m):
        candidates = partitioned[i]
        ordered = candidates[np.argsort(distances[i, candidates])]
        results.append([(int(j), float(distances[i, j])) for j in ordered])
    return results


def rebuild_item_neighbors(
    session: Session, X: np.ndarray, item_ids: list[int], *, top_n: int = 10
) -> NeighborReport:
    """Recompute and replace the whole ``item_neighbors`` table.

    Args:
        session: open database session; this function commits.
        X: (n_m, n) item features from the trained model.
        item_ids: database ids in matrix row order.
        top_n: neighbours stored per item.

    Items present in the model but no longer in the database are skipped rather
    than written as dangling foreign keys — the catalogue can change between a
    training run and this one.
    """
    live_ids = {row[0] for row in session.execute(select(FoodItem.id)).all()}
    norms = np.linalg.norm(X, axis=1)
    degenerate = int((norms < DEGENERATE_NORM).sum())

    neighbours = top_neighbors(X, top_n=top_n)

    rows: list[dict[str, object]] = []
    processed = 0
    for i, item_id in enumerate(item_ids):
        if item_id not in live_ids:
            continue
        processed += 1
        for rank, (j, distance) in enumerate(neighbours[i], start=1):
            neighbour_id = item_ids[j]
            if neighbour_id not in live_ids:
                continue
            rows.append(
                {
                    "food_item_id": item_id,
                    "neighbor_id": neighbour_id,
                    "distance": round(distance, 6),
                    "rank": rank,
                }
            )

    session.execute(delete(ItemNeighbor))
    for start in range(0, len(rows), 5_000):
        session.execute(insert(ItemNeighbor), rows[start : start + 5_000])
    session.commit()

    return NeighborReport(
        items_processed=processed,
        rows_written=len(rows),
        degenerate_items=degenerate,
        top_n=top_n,
    )
