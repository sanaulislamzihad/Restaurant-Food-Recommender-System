"""Evaluation harness: chronological split, metrics, baseline comparison.

    python -m ml.evaluate
    python -m ml.evaluate --k 10 --lambdas 0.5 1 2 5 10 20

Protocol, in the order it runs:

1. Split every user's ratings by time into train / validation / test.
2. Fit collaborative filtering on **train** at several regularisation strengths
   and score each on **validation**. Pick one.
3. Refit every model - baselines included, so nobody gets more data than anyone
   else - on **train + validation**.
4. Report on **test**, which nothing above has touched.

The separation in steps 2 and 4 is the point. Choosing a hyperparameter on the
same data used to report the result is how a model comes to look better than it
is, and it is the specific mistake this file exists to avoid.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import BACKEND_DIR
from app.db.session import SessionLocal
from ml.baselines import ScoredModel, build_baselines, collaborative_model
from ml.data import RatingMatrices
from ml.metrics import DEFAULT_RELEVANCE_THRESHOLD, RankingReport, ranking_quality, rating_accuracy
from ml.split import RatingEvent, chronological_split, load_rating_events, matrices_from_events
from ml.train_cf import TrainConfig, train

DEFAULT_LAMBDAS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def content_scores_for_split(
    fitting_events: list[RatingEvent],
    test_events: list[RatingEvent],
    item_ids: list[int],
    user_ids: list[int],
    *,
    epochs: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]] | None:
    """Fit the two-tower model on the fitting split and score every pair.

    Imports TensorFlow lazily so an evaluation run without the content half does
    not pay for loading it.

    Leakage matters more here than it looks. The item and user feature vectors
    contain rating aggregates, so computing them from the whole database would
    feed the test block's ratings into the model's inputs - the model would
    "know" how test items were rated before predicting them. Every aggregate is
    therefore derived from the fitting events alone, and the order-based user
    features are cut off at each user's first test rating.
    """
    try:
        from ml.features import FeatureBuilder, FeatureVocabulary, ItemStats, UserStats
        from ml.train_content import (  # noqa: PLC0415 - deliberately lazy
            ContentTrainConfig,
            load_catalogue,
            train_content_model,
        )
        from ml.two_tower import unscale_ratings
    except ImportError:
        return None

    from sqlalchemy import select as sa_select

    from app.db.models import Order

    with SessionLocal() as session:
        items, users = load_catalogue(session)
        order_rows = session.execute(
            sa_select(Order.user_id, Order.created_at, Order.total_amount)
        ).all()

    # ---- Aggregates from the fitting split only --------------------------
    item_sums: dict[int, float] = {}
    item_counts: dict[int, int] = {}
    user_sums: dict[int, float] = {}
    user_counts: dict[int, int] = {}
    cuisine_sums: dict[tuple[int, str], float] = {}
    cuisine_counts: dict[tuple[int, str], int] = {}
    cuisine_of = {int(item["id"]): str(item["cuisine"]) for item in items}

    for event in fitting_events:
        item_sums[event.item_id] = item_sums.get(event.item_id, 0.0) + event.rating
        item_counts[event.item_id] = item_counts.get(event.item_id, 0) + 1
        user_sums[event.user_id] = user_sums.get(event.user_id, 0.0) + event.rating
        user_counts[event.user_id] = user_counts.get(event.user_id, 0) + 1
        key = (event.user_id, cuisine_of.get(event.item_id, ""))
        cuisine_sums[key] = cuisine_sums.get(key, 0.0) + event.rating
        cuisine_counts[key] = cuisine_counts.get(key, 0) + 1

    global_mean = sum(item_sums.values()) / sum(item_counts.values()) if item_counts else 3.5

    # Orders are cut off at each user's first held-out rating, so no behaviour
    # from the test period reaches the features.
    cutoff: dict[int, Any] = {}
    for event in test_events:
        current = cutoff.get(event.user_id)
        if current is None or event.created_at < current:
            cutoff[event.user_id] = event.created_at

    order_totals: dict[int, float] = {}
    order_counts: dict[int, int] = {}
    for user_id, created_at, total in order_rows:
        limit = cutoff.get(int(user_id))
        if limit is not None and created_at >= limit:
            continue
        order_totals[int(user_id)] = order_totals.get(int(user_id), 0.0) + float(total)
        order_counts[int(user_id)] = order_counts.get(int(user_id), 0) + 1

    item_stats = ItemStats(
        average_rating={i: item_sums[i] / item_counts[i] for i in item_counts},
        rating_count=dict(item_counts),
        global_mean=global_mean,
    )
    user_stats = UserStats(
        total_orders=order_counts,
        average_order_value={u: order_totals[u] / order_counts[u] for u in order_counts},
        average_rating={u: user_sums[u] / user_counts[u] for u in user_counts},
        cuisine_rating={k: cuisine_sums[k] / cuisine_counts[k] for k in cuisine_counts},
        global_mean=global_mean,
    )

    vocabulary = FeatureVocabulary.build(items, users)
    builder = FeatureBuilder(vocabulary)
    item_matrix = builder.item_matrix(items, item_stats)
    user_matrix = builder.user_matrix(users, user_stats)

    item_index = {int(item["id"]): i for i, item in enumerate(items)}
    user_index = {int(user["id"]): i for i, user in enumerate(users)}
    interactions = [(e.user_id, e.item_id, e.rating) for e in fitting_events]

    cfg = ContentTrainConfig(epochs=epochs, seed=seed)
    model, metrics = train_content_model(
        cfg, user_matrix, item_matrix, user_index, item_index, interactions
    )

    # Full (n_items, n_users) score matrix, in star units, to match the others.
    user_tower = model.get_layer("user_NN")
    item_tower = model.get_layer("item_NN")
    vu = user_tower.predict(user_matrix, batch_size=512, verbose=0)
    vm = item_tower.predict(item_matrix, batch_size=512, verbose=0)
    vu /= np.maximum(np.linalg.norm(vu, axis=1, keepdims=True), 1e-12)
    vm /= np.maximum(np.linalg.norm(vm, axis=1, keepdims=True), 1e-12)

    scores = unscale_ratings(vm @ vu.T)
    ordered = np.zeros((len(item_ids), len(user_ids)), dtype=np.float64)
    for position, item_id in enumerate(item_ids):
        row = item_index.get(item_id)
        if row is None:
            continue
        for column, user_id in enumerate(user_ids):
            col = user_index.get(user_id)
            if col is not None:
                ordered[position, column] = scores[row, col]
    return ordered, metrics


@dataclass
class ModelReport:
    name: str
    detail: str
    rmse: float | None
    mae: float | None
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    catalogue_coverage: float
    mean_popularity_percentile: float
    top_decile_share: float
    gini: float


def evaluate_model(
    model: ScoredModel,
    train_R: np.ndarray,
    test_events: list[RatingEvent],
    item_pos: dict[int, int],
    user_pos: dict[int, int],
    *,
    k: int,
    relevance_threshold: float,
) -> ModelReport:
    accuracy = (
        rating_accuracy(model.scores, test_events, item_pos, user_pos)
        if model.predicts_ratings
        else {"rmse": None, "mae": None}
    )
    ranking: RankingReport = ranking_quality(
        model.scores,
        train_R,
        test_events,
        item_pos,
        user_pos,
        k=k,
        relevance_threshold=relevance_threshold,
    )
    return ModelReport(
        name=model.name,
        detail=model.detail,
        rmse=accuracy["rmse"],
        mae=accuracy["mae"],
        precision_at_k=ranking.precision_at_k,
        recall_at_k=ranking.recall_at_k,
        ndcg_at_k=ranking.ndcg_at_k,
        catalogue_coverage=ranking.catalogue_coverage,
        mean_popularity_percentile=ranking.mean_popularity_percentile,
        top_decile_share=ranking.top_decile_share,
        gini=ranking.gini,
    )


def select_lambda(
    train_matrices: RatingMatrices,
    validation_events: list[RatingEvent],
    item_pos: dict[int, int],
    user_pos: dict[int, int],
    *,
    lambdas: tuple[float, ...],
    iterations: int,
    k: int,
    relevance_threshold: float,
) -> tuple[float, list[dict[str, float]]]:
    """Fit on train, score on validation, return the best lambda and the sweep.

    Selection is on validation NDCG@K rather than RMSE. The product ranks dishes;
    it never shows anyone a predicted number, so the ordering metric is the one
    worth optimising. Validation RMSE is recorded alongside it, and the two do
    not always agree - which is itself worth seeing in the report.
    """
    sweep: list[dict[str, float]] = []

    for lambda_ in lambdas:
        result = train(
            train_matrices, TrainConfig(lambda_=lambda_, iterations=iterations, verbose=False)
        )
        model = collaborative_model(
            result.X, result.W, result.b, result.mu, name=f"cf (lambda={lambda_})"
        )
        accuracy = rating_accuracy(model.scores, validation_events, item_pos, user_pos)
        ranking = ranking_quality(
            model.scores,
            train_matrices.R,
            validation_events,
            item_pos,
            user_pos,
            k=k,
            relevance_threshold=relevance_threshold,
        )
        sweep.append(
            {
                "lambda": lambda_,
                "train_rmse": result.metrics.get("train_rmse", float("nan")),
                "validation_rmse": accuracy["rmse"],
                "validation_ndcg": ranking.ndcg_at_k,
            }
        )
        print(
            f"  lambda {lambda_:>6.1f}   train RMSE {sweep[-1]['train_rmse']:.4f}"
            f"   val RMSE {accuracy['rmse']:.4f}   val NDCG@{k} {ranking.ndcg_at_k:.4f}"
        )

    best = max(sweep, key=lambda row: row["validation_ndcg"])
    return float(best["lambda"]), sweep


def select_alpha(
    collaborative: np.ndarray,
    content: np.ndarray,
    train_R: np.ndarray,
    validation_events: list[RatingEvent],
    item_pos: dict[int, int],
    user_pos: dict[int, int],
    *,
    k: int,
    relevance_threshold: float,
    candidates: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> tuple[float, list[dict[str, float]]]:
    """Choose the blend weight on validation, exactly as lambda was chosen.

    alpha=0.6 was a configuration default, not a measured one. Reporting a
    hybrid built on an unexamined constant would be presenting an arbitrary
    number as a result, so it is selected here on data the test set never sees.

    The per-item and per-user switching rules are applied at every alpha, so
    what is being tuned is the blend for the cases where both models are
    actually usable - which is what the serving code does too.
    """
    item_counts = train_R.sum(axis=1)
    user_counts = train_R.sum(axis=0)

    sweep: list[dict[str, float]] = []
    for alpha in candidates:
        weights = np.full_like(collaborative, alpha)
        weights[item_counts < 3, :] = 0.0
        weights[:, user_counts < 5] = 0.0
        blended = weights * collaborative + (1.0 - weights) * content

        accuracy = rating_accuracy(blended, validation_events, item_pos, user_pos)
        ranking = ranking_quality(
            blended,
            train_R,
            validation_events,
            item_pos,
            user_pos,
            k=k,
            relevance_threshold=relevance_threshold,
        )
        sweep.append(
            {
                "alpha": alpha,
                "validation_rmse": accuracy["rmse"],
                "validation_ndcg": ranking.ndcg_at_k,
            }
        )
        print(
            f"  alpha {alpha:>4.1f}   val RMSE {accuracy['rmse']:.4f}"
            f"   val NDCG@{k} {ranking.ndcg_at_k:.4f}"
        )

    best = max(sweep, key=lambda row: row["validation_ndcg"])
    return float(best["alpha"]), sweep


def _format_table(reports: list[ModelReport], k: int) -> str:
    header = (
        f"{'model':<26}{'RMSE':>8}{'MAE':>8}{'P@' + str(k):>8}"
        f"{'R@' + str(k):>8}{'NDCG@' + str(k):>9}{'cover':>8}{'pop%ile':>9}{'gini':>7}"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        rmse_text = f"{report.rmse:>8.4f}" if report.rmse is not None else f"{'-':>8}"
        mae_text = f"{report.mae:>8.4f}" if report.mae is not None else f"{'-':>8}"
        lines.append(
            f"{report.name:<26}{rmse_text}{mae_text}"
            f"{report.precision_at_k:>8.4f}{report.recall_at_k:>8.4f}{report.ndcg_at_k:>9.4f}"
            f"{report.catalogue_coverage:>8.1%}{report.mean_popularity_percentile:>9.3f}"
            f"{report.gini:>7.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the recommender against baselines.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--relevance-threshold", type=float, default=DEFAULT_RELEVANCE_THRESHOLD)
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="skip the content and hybrid rows (they require TensorFlow)",
    )
    parser.add_argument("--content-epochs", type=int, default=30)
    parser.add_argument("--output", type=Path, default=BACKEND_DIR / "data" / "evaluation.json")
    args = parser.parse_args()

    with SessionLocal() as session:
        events, item_ids, user_ids = load_rating_events(session)

    split = chronological_split(
        events,
        item_ids,
        user_ids,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
    )
    item_pos = {item_id: idx for idx, item_id in enumerate(item_ids)}
    user_pos = {user_id: idx for idx, user_id in enumerate(user_ids)}

    print("=" * 78)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 78)
    for key, value in split.summary().items():
        print(f"  {key:24s} {value:>8,}")

    train_only = matrices_from_events(split.train, item_ids, user_ids)

    print()
    print("=" * 78)
    print(
        f"REGULARISATION SWEEP  (fit on train, scored on validation, {len(split.validation):,} "
        "held-out ratings)"
    )
    print("=" * 78)
    best_lambda, sweep = select_lambda(
        train_only,
        split.validation,
        item_pos,
        user_pos,
        lambdas=tuple(args.lambdas),
        iterations=args.iterations,
        k=args.k,
        relevance_threshold=args.relevance_threshold,
    )
    print(f"\n  selected lambda = {best_lambda} (highest validation NDCG@{args.k})")

    # Refit everything on train + validation. The baselines get the same data as
    # the learned model; otherwise the comparison would not be like for like.
    fitting_events = split.train + split.validation
    fitting = matrices_from_events(fitting_events, item_ids, user_ids)

    print()
    print("=" * 78)
    print(
        f"FINAL FIT on train + validation ({len(fitting_events):,} ratings), "
        f"reported on test ({len(split.test):,})"
    )
    print("=" * 78)

    final = train(
        fitting, TrainConfig(lambda_=best_lambda, iterations=args.iterations, verbose=False)
    )
    models = build_baselines(fitting)
    models.append(
        collaborative_model(
            final.X,
            final.W,
            final.b,
            final.mu,
            name="collaborative filtering",
            detail=f"n=10 latent features, lambda={best_lambda}",
        )
    )

    selected_alpha = 0.6
    alpha_sweep: list[dict[str, float]] = []

    if not args.no_content:
        print("")
        print("=" * 78)
        print("BLEND SWEEP  (both models fit on train, scored on validation)")
        print("=" * 78)
        train_only_cf = train(
            train_only, TrainConfig(lambda_=best_lambda, iterations=args.iterations, verbose=False)
        )
        train_only_content = content_scores_for_split(
            split.train,
            split.validation,
            item_ids,
            user_ids,
            epochs=args.content_epochs,
            seed=42,
        )
        if train_only_content is not None:
            selected_alpha, alpha_sweep = select_alpha(
                train_only_cf.X @ train_only_cf.W.T + train_only_cf.b + train_only_cf.mu[:, None],
                train_only_content[0],
                train_only.R,
                split.validation,
                item_pos,
                user_pos,
                k=args.k,
                relevance_threshold=args.relevance_threshold,
            )
            print("")
            print(f"  selected alpha = {selected_alpha} " f"(highest validation NDCG@{args.k})")

        print("")
        print("Fitting the content model on the full fitting split ...")
        content = content_scores_for_split(
            fitting_events,
            split.test,
            item_ids,
            user_ids,
            epochs=args.content_epochs,
            seed=42,
        )
        if content is not None:
            content_matrix, content_metrics = content
            print(
                f"  content model train RMSE {content_metrics.get('train_rmse', float('nan')):.4f}"
            )
            models.append(
                ScoredModel(
                    name="content (two-tower)",
                    scores=content_matrix,
                    predicts_ratings=True,
                    detail="32-dim towers, features only, no ids",
                )
            )

            # The hybrid, using exactly the switching rules the API applies.
            collaborative_matrix = final.X @ final.W.T + final.b + final.mu[:, None]
            counts = fitting.R.sum(axis=1)
            user_counts = fitting.R.sum(axis=0)

            alpha_matrix = np.full_like(collaborative_matrix, selected_alpha)
            # alpha -> 0 wherever a switching rule fires, per user and per item.
            alpha_matrix[counts < 3, :] = 0.0
            alpha_matrix[:, user_counts < 5] = 0.0
            hybrid_matrix = (
                alpha_matrix * collaborative_matrix + (1.0 - alpha_matrix) * content_matrix
            )
            models.append(
                ScoredModel(
                    name="hybrid",
                    scores=hybrid_matrix,
                    predicts_ratings=True,
                    detail=f"alpha={selected_alpha}, selected on validation",
                )
            )

    reports = [
        evaluate_model(
            model,
            fitting.R,
            split.test,
            item_pos,
            user_pos,
            k=args.k,
            relevance_threshold=args.relevance_threshold,
        )
        for model in models
    ]

    print()
    print(_format_table(reports, args.k))
    print()
    _print_verdict(reports)

    payload: dict[str, Any] = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "split": "per-user chronological",
            "test_fraction": args.test_fraction,
            "validation_fraction": args.validation_fraction,
            "k": args.k,
            "relevance_threshold": args.relevance_threshold,
            "selected_lambda": best_lambda,
            "selected_alpha": selected_alpha,
            "selection_metric": f"validation NDCG@{args.k}",
            "iterations": args.iterations,
        },
        "split": split.summary(),
        "lambda_sweep": sweep,
        "alpha_sweep": alpha_sweep,
        "results": [asdict(report) for report in reports],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


def _print_verdict(reports: list[ModelReport]) -> None:
    """State plainly whether the model beat the baseline that matters."""
    by_name = {report.name: report for report in reports}
    model = by_name.get("collaborative filtering")
    item_mean = by_name.get("item mean")
    if model is None or item_mean is None:
        return

    print("VERDICT")
    print("-" * 78)
    if model.rmse is not None and item_mean.rmse is not None:
        delta = (item_mean.rmse - model.rmse) / item_mean.rmse * 100
        verb = "beats" if model.rmse < item_mean.rmse else "LOSES TO"
        print(
            f"  RMSE     : CF {model.rmse:.4f} vs item-mean {item_mean.rmse:.4f}  "
            f"-> CF {verb} item-mean by {abs(delta):.1f}%"
        )
    if item_mean.ndcg_at_k > 0:
        delta = (model.ndcg_at_k - item_mean.ndcg_at_k) / item_mean.ndcg_at_k * 100
        verb = "beats" if model.ndcg_at_k > item_mean.ndcg_at_k else "LOSES TO"
        print(
            f"  NDCG     : CF {model.ndcg_at_k:.4f} vs item-mean {item_mean.ndcg_at_k:.4f}  "
            f"-> CF {verb} item-mean by {abs(delta):.1f}%"
        )
    print(
        f"  coverage : CF reaches {model.catalogue_coverage:.1%} of the catalogue, "
        f"item-mean {item_mean.catalogue_coverage:.1%}"
    )


if __name__ == "__main__":
    main()
