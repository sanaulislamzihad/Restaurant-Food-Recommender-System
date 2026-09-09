"""Verify that the seeded rating matrix contains discoverable taste clusters.

M1 is not finished just because the tables have rows in them. If the generator
did not actually put structure into the matrix, collaborative filtering has
nothing to find and every later milestone would be measuring noise. This script
is the gate.

Run with::

    python -m scripts.verify_clusters

Three checks, in increasing order of strength:

1. **Cluster x cuisine means** - do the clusters rate their preferred cuisines
   higher? A sanity check that the generator did what it claimed.
2. **Within- vs across-cluster user similarity** - are members of a cluster
   measurably more similar to each other than to outsiders? This is the property
   collaborative filtering actually exploits.
3. **Blind recovery** - can k-means, given no labels at all, rediscover the
   clusters from the rating matrix alone? Reported as purity and Adjusted Rand
   Index. This is the strongest evidence, because it does not get to peek.

Exits non-zero if any check fails, so CI can depend on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select

from app.core.config import BACKEND_DIR
from app.db.models import FoodItem, Rating, User
from app.db.session import SessionLocal

# Thresholds the seeded data must clear. These are deliberately not tight: the
# point is to catch a generator that produces no structure at all, not to pin
# the numbers to whatever today's seed happens to give.
# Separation between within- and across-cluster similarity, measured as a
# standardised effect size (Cohen's d) rather than a raw difference of means.
# The raw gap is not usable as a gate here: at ~10% density the median pair of
# users co-rates only about three items, which compresses every cosine
# similarity toward zero and shrinks the raw gap regardless of how strong the
# underlying structure is. d divides that gap by the spread, so it measures
# separation rather than sparsity. 0.8 is the conventional "large effect" line.
MIN_SIMILARITY_EFFECT_SIZE = 0.50
MIN_PURITY = 0.70
MIN_ARI = 0.35
MIN_DENSITY = 0.02
MAX_DENSITY = 0.20
#: A user needs at least this many ratings before their taste vector means much.
MIN_RATINGS_FOR_CLUSTERING = 5


@dataclass
class Matrix:
    """The rating matrix in the orientation the ML module uses.

    Y: (n_m, n_u) ratings, zero where unrated
    R: (n_m, n_u) 1 where rated
    """

    Y: np.ndarray
    R: np.ndarray
    user_ids: list[int]
    item_ids: list[int]
    cuisines: list[str]


def load_matrix() -> Matrix:
    with SessionLocal() as session:
        users = [row[0] for row in session.execute(select(User.id).order_by(User.id)).all()]
        items = session.execute(select(FoodItem.id, FoodItem.cuisine).order_by(FoodItem.id)).all()
        ratings = session.execute(select(Rating.user_id, Rating.food_item_id, Rating.rating)).all()

    item_ids = [row[0] for row in items]
    cuisines = [row[1] for row in items]
    user_pos = {uid: idx for idx, uid in enumerate(users)}
    item_pos = {iid: idx for idx, iid in enumerate(item_ids)}

    Y = np.zeros((len(item_ids), len(users)), dtype=np.float64)
    R = np.zeros_like(Y)
    for user_id, item_id, value in ratings:
        i, j = item_pos[item_id], user_pos[user_id]
        Y[i, j] = float(value)
        R[i, j] = 1.0

    return Matrix(Y=Y, R=R, user_ids=users, item_ids=item_ids, cuisines=cuisines)


def load_true_labels() -> dict[int, str]:
    """Read the generator's ground-truth cluster assignment from the seed sidecar."""
    path = BACKEND_DIR / "data" / "seed_meta.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run `python -m scripts.seed --reset` first - the cluster "
            "labels live in that sidecar, deliberately not in the database."
        )
    meta = json.loads(path.read_text(encoding="utf-8"))
    return {int(uid): cluster for uid, cluster in meta["user_clusters"].items()}


def centered_user_vectors(matrix: Matrix) -> np.ndarray:
    """Users as mean-centered taste vectors, shape (n_u, n_m).

    Each rating has its item's mean subtracted, so what remains is "how this
    person feels about the dish relative to everyone else" rather than how good
    the dish is. Unrated entries stay at zero, which after centering reads as
    "no opinion" and contributes nothing to a dot product.
    """
    rated_counts = matrix.R.sum(axis=1)
    safe_counts = np.where(rated_counts == 0, 1.0, rated_counts)
    item_means = (matrix.Y * matrix.R).sum(axis=1) / safe_counts

    centered = (matrix.Y - item_means[:, None]) * matrix.R
    return centered.T


def cosine_similarity(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    unit = vectors / norms
    return unit @ unit.T


def kmeans(vectors: np.ndarray, k: int, seed: int = 0, restarts: int = 10) -> np.ndarray:
    """Plain k-means with random restarts.

    Hand-rolled rather than pulled from scikit-learn: this is the only place the
    project would need it, and a 30-line implementation is cheaper than another
    dependency in the serving image.
    """
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    best_labels = np.zeros(n, dtype=int)
    best_inertia = np.inf

    for _ in range(restarts):
        centroids = vectors[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(100):
            distances = ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster in range(k):
                members = vectors[labels == cluster]
                if len(members):
                    centroids[cluster] = members.mean(axis=0)

        inertia = float(((vectors - centroids[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels

    return best_labels


def purity(predicted: np.ndarray, truth: list[str]) -> float:
    """Fraction of users landing in a cluster dominated by their true label."""
    total = 0
    for cluster in np.unique(predicted):
        members = [truth[i] for i in range(len(truth)) if predicted[i] == cluster]
        if members:
            total += max(members.count(label) for label in set(members))
    return total / len(truth)


def adjusted_rand_index(predicted: np.ndarray, truth: list[str]) -> float:
    """ARI: agreement between two partitions, corrected for chance.

    0 means "no better than random", 1 means identical partitions. Stricter than
    purity, which can be inflated by simply producing many small clusters.
    """
    label_ids = {label: idx for idx, label in enumerate(sorted(set(truth)))}
    truth_ids = np.array([label_ids[t] for t in truth])

    contingency = np.zeros((predicted.max() + 1, truth_ids.max() + 1), dtype=np.int64)
    for p, t in zip(predicted, truth_ids, strict=True):
        contingency[p, t] += 1

    def comb2(x: np.ndarray | np.int64) -> np.ndarray | np.int64:
        return x * (x - 1) // 2

    sum_ij = comb2(contingency).sum()
    a = comb2(contingency.sum(axis=1)).sum()
    b = comb2(contingency.sum(axis=0)).sum()
    n_pairs = comb2(np.int64(len(truth)))

    expected = a * b / n_pairs
    maximum = (a + b) / 2
    if maximum == expected:
        return 0.0
    return float((sum_ij - expected) / (maximum - expected))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify seeded taste-cluster structure.")
    parser.add_argument("--seed", type=int, default=0, help="k-means restart seed")
    args = parser.parse_args()

    matrix = load_matrix()
    labels_by_user = load_true_labels()
    n_m, n_u = matrix.Y.shape
    n_ratings = int(matrix.R.sum())
    density = n_ratings / (n_m * n_u)

    failures: list[str] = []

    print("=" * 74)
    print("RATING MATRIX")
    print("=" * 74)
    print(f"  shape (n_m x n_u)   : {n_m} items x {n_u} users")
    print(f"  observed ratings    : {n_ratings:,}")
    print(f"  density             : {density:.2%}")
    print(f"  mean rating         : {(matrix.Y * matrix.R).sum() / n_ratings:.3f}")
    if not MIN_DENSITY <= density <= MAX_DENSITY:
        failures.append(f"density {density:.2%} outside [{MIN_DENSITY:.0%}, {MAX_DENSITY:.0%}]")

    # ---- Check 1: cluster x cuisine means --------------------------------
    print()
    print("=" * 74)
    print("CHECK 1  mean rating by taste cluster x cuisine")
    print("=" * 74)
    cuisine_list = sorted(set(matrix.cuisines))
    cluster_list = sorted(set(labels_by_user.values()))
    cuisine_of_item = np.array(matrix.cuisines)

    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for j, user_id in enumerate(matrix.user_ids):
        cluster = labels_by_user.get(user_id)
        if cluster is None:
            continue
        rated = matrix.R[:, j] == 1
        for cuisine in cuisine_list:
            mask = rated & (cuisine_of_item == cuisine)
            if mask.any():
                totals[(cluster, cuisine)].extend(matrix.Y[mask, j].tolist())

    header = f"  {'cuisine':<12}" + "".join(f"{c:>16}" for c in cluster_list)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cuisine in cuisine_list:
        row = f"  {cuisine:<12}"
        for cluster in cluster_list:
            values = totals[(cluster, cuisine)]
            row += f"{np.mean(values):>16.2f}" if values else f"{'-':>16}"
        print(row)

    # Each cluster's own top cuisine should be one it was designed to love.
    for cluster in cluster_list:
        means = {c: np.mean(totals[(cluster, c)]) for c in cuisine_list if totals[(cluster, c)]}
        best = max(means, key=lambda c: means[c])
        worst = min(means, key=lambda c: means[c])
        spread = means[best] - means[worst]
        print(f"  {cluster:<16} favourite={best:<12} spread={spread:.2f}")
        if spread < 0.8:
            failures.append(f"cluster {cluster} has a flat cuisine profile (spread {spread:.2f})")

    # ---- Check 2: within- vs across-cluster similarity --------------------
    print()
    print("=" * 74)
    print("CHECK 2  user-user similarity, within cluster vs across")
    print("=" * 74)
    vectors = centered_user_vectors(matrix)
    counts = matrix.R.sum(axis=0)
    eligible = counts >= MIN_RATINGS_FOR_CLUSTERING
    print(f"  users with >= {MIN_RATINGS_FOR_CLUSTERING} ratings : {int(eligible.sum())} of {n_u}")
    print(f"  cold-start users excluded  : {int((~eligible).sum())}")

    sub_vectors = vectors[eligible]
    sub_labels = [
        labels_by_user[uid] for uid, keep in zip(matrix.user_ids, eligible, strict=True) if keep
    ]

    similarity = cosine_similarity(sub_vectors)
    np.fill_diagonal(similarity, np.nan)
    label_array = np.array(sub_labels)
    same = label_array[:, None] == label_array[None, :]

    within_values = similarity[same & ~np.isnan(similarity)]
    across_values = similarity[~same & ~np.isnan(similarity)]
    within, across = float(within_values.mean()), float(across_values.mean())
    gap = within - across
    pooled_sd = float(np.sqrt((within_values.var() + across_values.var()) / 2))
    effect_size = gap / pooled_sd if pooled_sd else 0.0

    # How much evidence each pairwise similarity is actually based on.
    sub_R = matrix.R[:, eligible]
    overlap = sub_R.T @ sub_R
    np.fill_diagonal(overlap, 0)
    well_observed = overlap >= 8
    within_well = similarity[same & well_observed & ~np.isnan(similarity)]
    across_well = similarity[~same & well_observed & ~np.isnan(similarity)]

    print(f"  mean similarity within cluster : {within:+.4f}  (sd {within_values.std():.4f})")
    print(f"  mean similarity across clusters: {across:+.4f}  (sd {across_values.std():.4f})")
    print(f"  raw gap                        : {gap:+.4f}  (reported, not gated)")
    print(
        f"  effect size (Cohen's d)        : {effect_size:.4f}  "
        f"(need >= {MIN_SIMILARITY_EFFECT_SIZE})"
    )
    print(f"  median co-rated items per pair : {int(np.median(overlap))}")
    print(
        f"  raw gap among pairs co-rating >= 8 items: "
        f"{float(within_well.mean()) - float(across_well.mean()):+.4f}"
    )
    if effect_size < MIN_SIMILARITY_EFFECT_SIZE:
        failures.append(
            f"similarity effect size {effect_size:.4f} below {MIN_SIMILARITY_EFFECT_SIZE}"
        )

    # ---- Check 3: blind recovery -----------------------------------------
    print()
    print("=" * 74)
    print("CHECK 3  blind k-means recovery of the clusters (no labels given)")
    print("=" * 74)
    predicted = kmeans(sub_vectors, k=len(cluster_list), seed=args.seed)
    pur = purity(predicted, sub_labels)
    ari = adjusted_rand_index(predicted, sub_labels)
    majority = max(sub_labels.count(c) for c in cluster_list) / len(sub_labels)

    print(f"  k                    : {len(cluster_list)}")
    print(
        f"  purity               : {pur:.3f}  (need >= {MIN_PURITY}; "
        f"majority-class baseline {majority:.3f})"
    )
    print(f"  adjusted Rand index  : {ari:.3f}  (need >= {MIN_ARI}; 0 = chance)")
    print("\n  discovered cluster vs true label:")
    for cluster in np.unique(predicted):
        members = [sub_labels[i] for i in range(len(sub_labels)) if predicted[i] == cluster]
        breakdown = ", ".join(
            f"{label}={members.count(label)}" for label in cluster_list if members.count(label)
        )
        print(f"    cluster {cluster}: n={len(members):3d}  {breakdown}")

    if pur < MIN_PURITY:
        failures.append(f"k-means purity {pur:.3f} below {MIN_PURITY}")
    if ari < MIN_ARI:
        failures.append(f"adjusted Rand index {ari:.3f} below {MIN_ARI}")

    # ---- Verdict ----------------------------------------------------------
    print()
    print("=" * 74)
    if failures:
        print("RESULT: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        print("=" * 74)
        sys.exit(1)
    print("RESULT: PASS - the rating matrix carries recoverable taste structure.")
    print("=" * 74)


if __name__ == "__main__":
    main()
