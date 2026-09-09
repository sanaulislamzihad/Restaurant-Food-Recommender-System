"""Baselines the learned model has to beat to justify existing.

Every one of these is a few lines long and needs no training. If collaborative
filtering cannot beat them, the honest conclusion is that the extra machinery is
not earning its place - not that the baselines were unfair.

All four models expose the same shape, a full (n_m, n_u) score matrix, so the
evaluation harness treats them uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.data import RatingMatrices


@dataclass
class ScoredModel:
    """A model reduced to what evaluation needs: a score for every pair.

    ``predicts_ratings`` distinguishes models whose scores live on the 1-5 star
    scale from ranking-only models. Reporting an RMSE for a popularity count
    would be meaningless, so the harness prints a dash instead of inventing one.
    """

    name: str
    scores: np.ndarray
    predicts_ratings: bool
    detail: str = ""


def global_mean_model(train: RatingMatrices) -> ScoredModel:
    """Predict the same number - the overall training mean - for everything.

    The floor. Any model that cannot beat this has learned nothing whatsoever.
    Its ranking is arbitrary by construction, since every item ties.
    """
    observed = train.R.sum()
    mean = float((train.Y * train.R).sum() / observed) if observed else 0.0
    return ScoredModel(
        name="global mean",
        scores=np.full_like(train.Y, mean),
        predicts_ratings=True,
        detail=f"constant {mean:.3f}",
    )


def item_mean_model(train: RatingMatrices) -> ScoredModel:
    """Predict each dish's average rating, the same for every user.

    The baseline that actually matters. It captures "some dishes are better than
    others", which is most of the variance in rating data and none of the
    personalisation. Beating it is the whole claim of a recommender system.
    """
    counts = train.R.sum(axis=1)
    observed = train.R.sum()
    global_mean = float((train.Y * train.R).sum() / observed) if observed else 0.0

    safe = np.where(counts == 0, 1.0, counts)
    means = (train.Y * train.R).sum(axis=1) / safe
    # A dish nobody rated in train has no mean of its own.
    means = np.where(counts == 0, global_mean, means)

    return ScoredModel(
        name="item mean",
        scores=np.tile(means[:, None], (1, train.n_u)),
        predicts_ratings=True,
        detail="per-item average, identical for all users",
    )


def most_popular_model(train: RatingMatrices) -> ScoredModel:
    """Rank by how many people rated the dish. Same list for everybody.

    Scores are rating counts, not stars, so rating-accuracy metrics do not apply
    and the harness reports them as not applicable rather than fabricating a
    number on the wrong scale.
    """
    popularity = train.R.sum(axis=1)
    return ScoredModel(
        name="most popular",
        scores=np.tile(popularity[:, None], (1, train.n_u)),
        predicts_ratings=False,
        detail="ranked by training rating count",
    )


def collaborative_model(
    X: np.ndarray, W: np.ndarray, b: np.ndarray, mu: np.ndarray, *, name: str, detail: str = ""
) -> ScoredModel:
    """Wrap trained collaborative filtering parameters as a scored model.

    X:  (n_m, n)   W: (n_u, n)   b: (1, n_u)   mu: (n_m,)
    """
    return ScoredModel(
        name=name,
        scores=X @ W.T + b + mu[:, None],
        predicts_ratings=True,
        detail=detail,
    )


def build_baselines(train: RatingMatrices) -> list[ScoredModel]:
    return [global_mean_model(train), item_mean_model(train), most_popular_model(train)]
