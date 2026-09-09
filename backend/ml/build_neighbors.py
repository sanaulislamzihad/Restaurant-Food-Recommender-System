"""CLI to rebuild the item-neighbour table from a trained model.

    python -m ml.build_neighbors
    python -m ml.build_neighbors --version v1 --top-n 10

Run after training. The API reads this table; it never computes distances.
"""

from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.db.session import SessionLocal
from ml.artifacts import load_cf_artifacts
from ml.neighbors import rebuild_item_neighbors


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute item nearest neighbours.")
    parser.add_argument("--version", default=None, help="model version, defaults to the newest")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--mode",
        default="explicit",
        choices=["explicit", "implicit"],
        help=(
            "which trained model to take features from. Defaults to explicit: "
            "item similarity here means 'people rate these alike', and the "
            "explicit features carry that far better (47.8%% of top-10 "
            "neighbours share a cuisine, against 32.4%% for implicit and 15.7%% "
            "at random)"
        ),
    )
    args = parser.parse_args()

    settings = get_settings()
    artifacts = load_cf_artifacts(settings.model_root, args.version, mode=args.mode)
    trained_mode = artifacts.metadata.get("hyperparameters", {}).get("mode", "unknown")
    print(
        f"Loaded model {artifacts.version} [{trained_mode}] "
        f"({artifacts.X.shape[0]} items, {artifacts.X.shape[1]} latent features)"
    )

    with SessionLocal() as session:
        report = rebuild_item_neighbors(session, artifacts.X, artifacts.item_ids, top_n=args.top_n)

    print(f"  items processed  : {report.items_processed:,}")
    print(f"  neighbour rows   : {report.rows_written:,}")
    print(f"  top-n per item   : {report.top_n}")
    if report.degenerate_items:
        print(
            f"  WARNING: {report.degenerate_items} item(s) have near-zero feature vectors. "
            "Those are dishes with too few ratings for the model to place, so their "
            "neighbours are an artefact of regularisation rather than real similarity."
        )


if __name__ == "__main__":
    main()
