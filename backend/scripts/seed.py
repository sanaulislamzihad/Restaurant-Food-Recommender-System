"""Seed the database with realistic, deliberately clustered synthetic data.

Run with::

    python -m scripts.seed --reset

The point of this script is not just to fill tables. The ratings are generated
from three latent taste clusters (spicy-desi, continental, dessert-lover) so the
collaborative filtering has genuine structure to discover. ``verify_clusters.py``
checks that the structure actually survived generation.

Design notes worth knowing before changing anything here:

* Ratings are drawn from a cluster affinity plus a per-item quality term and a
  per-user generosity term. Without the item-quality term the item-mean baseline
  would be meaningless, and beating it would prove nothing.
* A user only rates an item that already existed, so recently-added items stay
  cold. This is what exercises the item-side cold-start path.
* Roughly 5% of users are left with fewer than five ratings so the user-side
  cold-start path has real subjects to be tested against.
* Every seeded user shares one bcrypt digest. Hashing 500 passwords separately
  would take about a minute and prove nothing.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import bcrypt
import numpy as np
from sqlalchemy import delete, func, insert, select, text
from sqlalchemy.orm import Session

from app.core.config import BACKEND_DIR
from app.db.models import (
    FoodItem,
    Gender,
    Impression,
    ItemNeighbor,
    Order,
    OrderItem,
    OrderStatus,
    Rating,
    Restaurant,
    User,
)
from app.db.session import SessionLocal, engine
from scripts.menu_data import (
    DHAKA_AREAS,
    DISHES,
    RESTAURANTS,
    SIDE_CUISINES,
    DishSpec,
)

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------

SEED_PASSWORD = "foodrec123"

#: History window. Ratings and orders are spread across this many days so the
#: evaluation harness has a meaningful chronological axis to split on.
HISTORY_DAYS = 240

#: Items created inside this many days of "now" are treated as new arrivals and
#: attract few or no ratings.
NEW_ITEM_WINDOW_DAYS = 14
NEW_ITEM_COUNT = 16

#: Fraction of users deliberately left below the cold-start threshold.
COLD_START_USER_FRACTION = 0.05

#: Dishes flagged as promoted, so the ranking stage's boost rule has subjects.
PROMOTED_ITEM_COUNT = 18
#: A fixed, separate seed: promotions must not perturb the rating stream.
PROMOTION_SEED_OFFSET = 20_260_910

TASTE_CLUSTERS: tuple[str, ...] = ("spicy_desi", "continental", "dessert_lover")
CLUSTER_WEIGHTS: tuple[float, ...] = (0.42, 0.34, 0.24)

#: Expected rating a member of each cluster gives to each cuisine, before the
#: spice penalty, item quality, user generosity and noise terms are applied.
CLUSTER_AFFINITY: dict[str, dict[str, float]] = {
    "spicy_desi": {
        "bengali": 4.5,
        "mughlai": 4.6,
        "thai": 4.0,
        "chinese": 3.4,
        "fast_food": 3.0,
        "italian": 2.4,
        "continental": 2.3,
        "dessert": 3.2,
        "beverage": 3.3,
    },
    "continental": {
        "italian": 4.5,
        "continental": 4.6,
        "fast_food": 4.1,
        "chinese": 3.8,
        "thai": 3.3,
        "bengali": 2.5,
        "mughlai": 2.6,
        "dessert": 3.5,
        "beverage": 3.6,
    },
    "dessert_lover": {
        "dessert": 4.7,
        "beverage": 4.5,
        "italian": 3.6,
        "fast_food": 3.5,
        "continental": 3.4,
        "chinese": 3.1,
        "bengali": 2.9,
        "mughlai": 3.0,
        "thai": 2.6,
    },
}

#: Spice tolerance is correlated with cluster: the desi crowd asks for heat.
CLUSTER_SPICE_TOLERANCE: dict[str, tuple[int, int]] = {
    "spicy_desi": (3, 5),
    "continental": (0, 3),
    "dessert_lover": (1, 3),
}

FIRST_NAMES = (
    "Rakib",
    "Tanvir",
    "Sadia",
    "Nusrat",
    "Arif",
    "Mehedi",
    "Farhana",
    "Sabbir",
    "Ishrat",
    "Jubair",
    "Tasnim",
    "Rifat",
    "Sumaiya",
    "Nayeem",
    "Anika",
    "Shakib",
    "Mahmuda",
    "Imran",
    "Rubaiya",
    "Fahim",
    "Naznin",
    "Asif",
    "Sharmin",
    "Tousif",
    "Lamia",
    "Rasel",
    "Jarin",
    "Sourav",
    "Mitu",
    "Zahid",
    "Priya",
    "Nabil",
    "Sanjida",
    "Riyad",
    "Tahmina",
    "Mizan",
    "Oishi",
    "Shuvo",
    "Marjan",
    "Adnan",
)
LAST_NAMES = (
    "Hasan",
    "Islam",
    "Ahmed",
    "Rahman",
    "Chowdhury",
    "Karim",
    "Akter",
    "Haque",
    "Siddique",
    "Mahmud",
    "Sarkar",
    "Bhuiyan",
    "Alam",
    "Khan",
    "Talukder",
    "Mollah",
)


@dataclass
class SeedConfig:
    users: int = 500
    ratings: int = 15_000
    negatives_per_positive: int = 3
    random_seed: int = 42
    reset: bool = False
    #: Where to write the cluster-label sidecar. Overridable so tests do not
    #: clobber the development seed's metadata.
    meta_path: Path | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC)


def _cuisine_sampling_weights(cluster: str, cuisines: list[str]) -> np.ndarray:
    """Turn cluster affinities into a distribution over which cuisines a user browses.

    People do not sample the menu uniformly: they mostly order what they like and
    occasionally explore. Exponentiating the affinity concentrates mass on the
    preferred cuisines while leaving every cuisine reachable, which is what puts
    cluster structure into the *sparsity pattern* R and not only the values Y.
    """
    affinity = np.array([CLUSTER_AFFINITY[cluster][c] for c in cuisines], dtype=np.float64)
    weights = np.exp(1.15 * (affinity - affinity.mean()))
    normalised: np.ndarray = weights / weights.sum()
    return normalised


def _rating_counts(rng: np.random.Generator, cfg: SeedConfig) -> np.ndarray:
    """Ratings per user: a long-tailed distribution plus deliberate cold-start users."""
    n_cold = max(1, int(cfg.users * COLD_START_USER_FRACTION))
    n_regular = cfg.users - n_cold

    # A handful of true zero-history users, the rest just below the threshold.
    cold = rng.integers(0, 5, size=n_cold)

    raw = rng.lognormal(mean=3.05, sigma=0.62, size=n_regular)
    raw = np.clip(raw, 5, 95)
    # Rescale so the totals land on the requested rating budget.
    budget = max(cfg.ratings - int(cold.sum()), n_regular * 5)
    raw = raw * (budget / raw.sum())
    regular = np.clip(np.round(raw), 5, 120).astype(int)

    counts = np.concatenate([cold, regular])
    rng.shuffle(counts)
    return counts


def _truncate(session: Session) -> None:
    """Delete seeded rows in FK-safe order."""
    for model in (ItemNeighbor, Impression, OrderItem, Order, Rating, FoodItem, User, Restaurant):
        session.execute(delete(model))
    session.commit()


def _resync_sequences(session: Session) -> None:
    """Advance Postgres identity sequences past the explicitly-assigned ids.

    The seeder assigns primary keys itself so it can build foreign-key references
    without round-tripping to the database. Postgres does not notice, so the next
    API insert would collide on a duplicate id. SQLite derives the next rowid from
    max(id) and needs no fixing.
    """
    if engine.dialect.name != "postgresql":
        return
    for table in (
        "restaurants",
        "users",
        "food_items",
        "ratings",
        "orders",
        "order_items",
        "impressions",
    ):
        session.execute(
            text(
                "SELECT setval(pg_get_serial_sequence(:t, 'id'), "
                "COALESCE((SELECT MAX(id) FROM " + table + "), 1))"
            ),
            {"t": table},
        )
    session.commit()


# ---------------------------------------------------------------------------
# Generation stages
# ---------------------------------------------------------------------------


def _build_restaurants() -> list[dict[str, Any]]:
    return [
        {
            "id": idx + 1,
            "name": spec.name,
            "area": spec.area,
            "cuisine_tags": list(spec.cuisine_tags),
            "is_active": True,
        }
        for idx, spec in enumerate(RESTAURANTS)
    ]


def _build_food_items(rng: np.random.Generator, now: datetime) -> list[dict[str, Any]]:
    """Draw each restaurant's menu from its cuisine pools plus a dessert/drink quota."""
    by_cuisine: dict[str, list[DishSpec]] = defaultdict(list)
    for dish in DISHES:
        by_cuisine[dish.cuisine].append(dish)

    rows: list[dict[str, Any]] = []
    item_id = 1

    for restaurant_index, spec in enumerate(RESTAURANTS, start=1):
        core_pool = [d for c in spec.cuisine_tags for d in by_cuisine[c]]
        side_pool = [d for c in SIDE_CUISINES for d in by_cuisine[c]]

        core_need = spec.menu_size - spec.side_quota
        chosen = list(rng.choice(np.array(core_pool, dtype=object), size=core_need, replace=False))
        if spec.side_quota:
            chosen += list(
                rng.choice(np.array(side_pool, dtype=object), size=spec.side_quota, replace=False)
            )

        for dish in chosen:
            price = round(dish.base_price * spec.price_multiplier / 5) * 5
            rows.append(
                {
                    "id": item_id,
                    "restaurant_id": restaurant_index,
                    "name": dish.name,
                    "description": _describe(dish),
                    "cuisine": dish.cuisine,
                    "spice_level": dish.spice_level,
                    "is_veg": dish.is_veg,
                    "is_rice_based": dish.is_rice_based,
                    "price": Decimal(str(price)),
                    "prep_time_min": dish.prep_time_min,
                    # No image assets ship with the repo, and inventing a path
                    # to a file that does not exist is data claiming something
                    # untrue: the browser fetches every one, gets a 404, and the
                    # console fills with errors that mask real ones. NULL lets
                    # the frontend render its designed placeholder immediately.
                    # A real deployment populates this from object storage.
                    "image_url": None,
                    "ingredient_tags": list(dish.ingredient_tags),
                    "is_available": True,
                    "is_promoted": False,
                    # Overwritten below for the designated new arrivals.
                    "created_at": now - timedelta(days=HISTORY_DAYS + 5),
                }
            )
            item_id += 1

    # Mark a slice of the catalogue as new arrivals so item-side cold start is
    # exercised: nobody can have rated these before they existed.
    new_item_ids = rng.choice(len(rows), size=NEW_ITEM_COUNT, replace=False)
    for idx in new_item_ids:
        age_days = float(rng.uniform(0, NEW_ITEM_WINDOW_DAYS))
        rows[idx]["created_at"] = now - timedelta(days=age_days)

    # A few items are temporarily off the menu; the ranking stage must drop them.
    for idx in rng.choice(len(rows), size=12, replace=False):
        rows[idx]["is_available"] = False

    # Promotions are drawn from their own generator rather than `rng`. Taking
    # them from the shared stream would shift every subsequent draw and change
    # the ratings, invalidating the trained model and every published metric for
    # the sake of a display flag.
    promo_rng = np.random.default_rng(PROMOTION_SEED_OFFSET)
    for idx in promo_rng.choice(len(rows), size=PROMOTED_ITEM_COUNT, replace=False):
        if rows[idx]["is_available"]:
            rows[idx]["is_promoted"] = True

    return rows


def _describe(dish: DishSpec) -> str:
    heat = ("no chilli", "mild", "lightly spiced", "medium hot", "hot", "fiery")[dish.spice_level]
    diet = "Vegetarian" if dish.is_veg else "Non-vegetarian"
    tags = ", ".join(dish.ingredient_tags)
    return f"{diet}, {heat}. Made with {tags}. Ready in about {dish.prep_time_min} minutes."


def _build_users(
    rng: np.random.Generator, cfg: SeedConfig, now: datetime
) -> tuple[list[dict[str, Any]], list[str]]:
    """Create users and assign each a hidden taste cluster.

    The cluster label is returned separately rather than stored on the row: it is
    a property of the data generator, not of the product, and keeping it out of
    the schema means no model can accidentally train on the answer.
    """
    password_hash = bcrypt.hashpw(SEED_PASSWORD.encode(), bcrypt.gensalt()).decode()
    clusters = list(
        rng.choice(np.array(TASTE_CLUSTERS), size=cfg.users, p=np.array(CLUSTER_WEIGHTS))
    )

    rows: list[dict[str, Any]] = []
    for i in range(cfg.users):
        cluster = clusters[i]
        low, high = CLUSTER_SPICE_TOLERANCE[cluster]
        first = FIRST_NAMES[int(rng.integers(len(FIRST_NAMES)))]
        last = LAST_NAMES[int(rng.integers(len(LAST_NAMES)))]
        joined_days_ago = float(rng.uniform(HISTORY_DAYS * 0.25, HISTORY_DAYS))
        rows.append(
            {
                "id": i + 1,
                "name": f"{first} {last}",
                "email": f"{first.lower()}.{last.lower()}{i + 1}@example.com",
                "password_hash": password_hash,
                "age": int(rng.integers(18, 55)),
                "gender": Gender(
                    rng.choice(
                        np.array(["male", "female", "other"]), p=np.array([0.52, 0.46, 0.02])
                    )
                ),
                "area": DHAKA_AREAS[int(rng.integers(len(DHAKA_AREAS)))],
                "spice_tolerance": int(rng.integers(low, high + 1)),
                "created_at": now - timedelta(days=joined_days_ago),
            }
        )
    return rows, clusters


def _build_ratings(
    rng: np.random.Generator,
    cfg: SeedConfig,
    users: list[dict[str, Any]],
    clusters: list[str],
    items: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    """Generate the rating matrix from the latent cluster model."""
    counts = _rating_counts(rng, cfg)

    cuisines = sorted({item["cuisine"] for item in items})
    items_by_cuisine: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        items_by_cuisine[item["cuisine"]].append(idx)

    # Some dishes are simply better cooked than others. This is the signal the
    # item-mean baseline picks up, and what the model has to beat.
    item_quality = rng.normal(0.0, 0.35, size=len(items))
    # Some people rate everything generously, some are hard to please.
    user_generosity = rng.normal(0.0, 0.30, size=len(users))

    # Fraction of the history window each item has been orderable for, used above
    # to thin the ratings of recent arrivals.
    window_seconds = HISTORY_DAYS * 86_400
    item_exposure = np.array(
        [
            min(1.0, max(0.0, (now - item["created_at"]).total_seconds() / window_seconds))
            for item in items
        ]
    )

    rows: list[dict[str, Any]] = []
    rating_id = 1

    for user_index, user in enumerate(users):
        n_ratings = int(counts[user_index])
        if n_ratings == 0:
            continue

        cluster = clusters[user_index]
        weights = _cuisine_sampling_weights(cluster, cuisines)
        joined_at = user["created_at"]

        # Draw a candidate pool by first choosing cuisines, then items within them.
        picked_cuisines = rng.choice(np.array(cuisines), size=n_ratings * 5, p=weights)
        candidates: list[int] = []
        seen: set[int] = set()
        for cuisine in picked_cuisines:
            pool = items_by_cuisine[str(cuisine)]
            idx = int(pool[int(rng.integers(len(pool)))])
            if idx in seen:
                continue
            # An item that has only been on the menu a few days cannot have
            # collected a full history. Accepting a candidate in proportion to
            # how long it has existed is what leaves the new arrivals genuinely
            # cold rather than merely recent, which is what the item-side
            # cold-start path needs in order to be testable at all.
            if rng.random() > item_exposure[idx]:
                continue
            seen.add(idx)
            candidates.append(idx)
            if len(candidates) == n_ratings:
                break

        for item_index in candidates:
            item = items[item_index]
            base = CLUSTER_AFFINITY[cluster][item["cuisine"]]
            # Ordering something hotter than you can handle costs you.
            spice_penalty = -0.28 * max(0, item["spice_level"] - user["spice_tolerance"])
            score = (
                base
                + spice_penalty
                + item_quality[item_index]
                + user_generosity[user_index]
                + rng.normal(0.0, 0.45)
            )
            value = int(np.clip(round(score), 1, 5))

            earliest = max(joined_at, item["created_at"])
            span = (now - earliest).total_seconds()
            created_at = earliest + timedelta(seconds=float(rng.uniform(0, max(span, 1))))

            rows.append(
                {
                    "id": rating_id,
                    "user_id": user["id"],
                    "food_item_id": item["id"],
                    "rating": Decimal(value),
                    "created_at": created_at,
                }
            )
            rating_id += 1

    rows.sort(key=lambda r: (r["user_id"], r["created_at"]))
    for new_id, row in enumerate(rows, start=1):
        row["id"] = new_id
    return rows


def _build_orders(
    rng: np.random.Generator,
    ratings: list[dict[str, Any]],
    items_by_id: dict[int, dict[str, Any]],
    now: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn rating events into order baskets.

    A rating implies the person actually ate the dish, so orders are derived from
    ratings rather than invented independently. Consecutive ratings by the same
    user are grouped into baskets of one to three items.
    """
    by_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in ratings:
        by_user[row["user_id"]].append(row)

    orders: list[dict[str, Any]] = []
    order_items: list[dict[str, Any]] = []
    order_id = 1
    order_item_id = 1

    for user_id, user_ratings in by_user.items():
        position = 0
        while position < len(user_ratings):
            basket_size = int(rng.integers(1, 4))
            basket = user_ratings[position : position + basket_size]
            position += basket_size

            placed_at = basket[0]["created_at"]
            total = Decimal("0.00")
            pending: list[dict[str, Any]] = []
            for row in basket:
                item = items_by_id[row["food_item_id"]]
                quantity = int(rng.integers(1, 3))
                unit_price = Decimal(item["price"])
                total += unit_price * quantity
                pending.append(
                    {
                        "id": order_item_id,
                        "order_id": order_id,
                        "food_item_id": item["id"],
                        "quantity": quantity,
                        "unit_price": unit_price,
                    }
                )
                order_item_id += 1

            # Recent orders may still be in flight; older ones are settled.
            age_hours = (now - placed_at).total_seconds() / 3600
            if age_hours < 2:
                status = OrderStatus.PREPARING
            elif age_hours < 6 and rng.random() < 0.3:
                status = OrderStatus.CONFIRMED
            elif rng.random() < 0.04:
                status = OrderStatus.CANCELLED
            else:
                status = OrderStatus.DELIVERED

            orders.append(
                {
                    "id": order_id,
                    "user_id": user_id,
                    "total_amount": total,
                    "status": status,
                    "created_at": placed_at,
                }
            )
            order_items.extend(pending)
            order_id += 1

    return orders, order_items


def _build_impressions(
    rng: np.random.Generator,
    cfg: SeedConfig,
    orders: list[dict[str, Any]],
    order_items: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Log what was shown, converted or not.

    Only the positives are known from the order log; the negatives are the whole
    point of this table. They are drawn from the same restaurant the user ordered
    from, because that is what the customer would actually have been looking at.
    """
    orders_by_id = {order["id"]: order for order in orders}
    items_by_restaurant: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        items_by_restaurant[item["restaurant_id"]].append(item)
    items_by_id = {item["id"]: item for item in items}

    rows: list[dict[str, Any]] = []
    impression_id = 1

    for order_item in order_items:
        order = orders_by_id[order_item["order_id"]]
        item = items_by_id[order_item["food_item_id"]]
        shown_at = order["created_at"]

        rows.append(
            {
                "id": impression_id,
                "user_id": order["user_id"],
                "food_item_id": item["id"],
                "was_ordered": True,
                "shown_at": shown_at,
            }
        )
        impression_id += 1

        siblings = items_by_restaurant[item["restaurant_id"]]
        for _ in range(cfg.negatives_per_positive):
            candidate = siblings[int(rng.integers(len(siblings)))]
            if candidate["id"] == item["id"] or candidate["created_at"] > shown_at:
                continue
            rows.append(
                {
                    "id": impression_id,
                    "user_id": order["user_id"],
                    "food_item_id": candidate["id"],
                    "was_ordered": False,
                    # Shown moments before the order was placed.
                    "shown_at": shown_at - timedelta(seconds=float(rng.uniform(5, 180))),
                }
            )
            impression_id += 1

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _bulk_insert(
    session: Session, model: Any, rows: list[dict[str, Any]], chunk: int = 5_000
) -> None:
    """Core-level bulk insert. The ORM unit of work is far too slow at this size."""
    for start in range(0, len(rows), chunk):
        session.execute(insert(model), rows[start : start + chunk])
    session.commit()


def seed(cfg: SeedConfig) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.random_seed)
    now = _now()

    with SessionLocal() as session:
        existing = session.scalar(select(func.count()).select_from(FoodItem)) or 0
        if existing and not cfg.reset:
            raise SystemExit(
                f"Database already holds {existing} food items. "
                "Re-run with --reset to wipe and reseed."
            )
        if cfg.reset:
            _truncate(session)

        restaurants = _build_restaurants()
        items = _build_food_items(rng, now)
        users, clusters = _build_users(rng, cfg, now)
        ratings = _build_ratings(rng, cfg, users, clusters, items, now)
        items_by_id = {item["id"]: item for item in items}
        orders, order_items = _build_orders(rng, ratings, items_by_id, now)
        impressions = _build_impressions(rng, cfg, orders, order_items, items)

        _bulk_insert(session, Restaurant, restaurants)
        _bulk_insert(session, FoodItem, items)
        _bulk_insert(session, User, users)
        _bulk_insert(session, Rating, ratings)
        _bulk_insert(session, Order, orders)
        _bulk_insert(session, OrderItem, order_items)
        _bulk_insert(session, Impression, impressions)
        _resync_sequences(session)

    rated_item_ids = {row["food_item_id"] for row in ratings}
    summary: dict[str, Any] = {
        "generated_at": now.isoformat(),
        "random_seed": cfg.random_seed,
        "history_days": HISTORY_DAYS,
        "counts": {
            "restaurants": len(restaurants),
            "food_items": len(items),
            "users": len(users),
            "ratings": len(ratings),
            "orders": len(orders),
            "order_items": len(order_items),
            "impressions": len(impressions),
        },
        "matrix": {
            "density": round(len(ratings) / (len(users) * len(items)), 4),
            "items_with_zero_ratings": len(items) - len(rated_item_ids),
        },
        "clusters": {name: clusters.count(name) for name in TASTE_CLUSTERS},
        # Consumed by verify_clusters.py. Deliberately a sidecar file rather than
        # a database column: the label is generator ground truth, not product data.
        "user_clusters": {
            str(user["id"]): cluster for user, cluster in zip(users, clusters, strict=True)
        },
    }

    meta_path = cfg.meta_path or (BACKEND_DIR / "data" / "seed_meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the food recommender database.")
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--ratings", type=int, default=15_000, help="approximate rating budget")
    parser.add_argument("--negatives-per-positive", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42, dest="random_seed")
    parser.add_argument("--reset", action="store_true", help="wipe existing rows first")
    args = parser.parse_args()

    cfg = SeedConfig(
        users=args.users,
        ratings=args.ratings,
        negatives_per_positive=args.negatives_per_positive,
        random_seed=args.random_seed,
        reset=args.reset,
    )
    summary = seed(cfg)

    counts = summary["counts"]
    print("Seed complete")
    for key, value in counts.items():
        print(f"  {key:14s} {value:>7,}")
    print(f"  {'density':14s} {summary['matrix']['density']:>7.2%}")
    print(f"  {'cold items':14s} {summary['matrix']['items_with_zero_ratings']:>7,}")
    print(f"  clusters: {summary['clusters']}")


if __name__ == "__main__":
    main()
