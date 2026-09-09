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
            "selection_metric": f"validation NDCG@{args.k}",
            "iterations": args.iterations,
        },
        "split": split.summary(),
        "lambda_sweep": sweep,
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
