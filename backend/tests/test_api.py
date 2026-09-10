"""API contract tests.

Built on a small hand-made dataset rather than the full seeder, so each
assertion is about a value that can be reasoned about by hand.
"""

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.security import hash_password
from app.db.base import Base
from app.db.models import FoodItem, ItemNeighbor, Rating, Restaurant, User
from app.db.session import SessionLocal, engine
from app.main import app
from ml.artifacts import next_version_dir, save_cf_artifacts

PASSWORD = "testpass123"
#: Hashed once for the whole module. bcrypt is deliberately slow, and hashing
#: three users per test was costing more than the tests themselves.
PASSWORD_HASH = hash_password(PASSWORD)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A client over a freshly built database with a small known catalogue.

    Six dishes, three users:
      * ``warm@example.com``  - 5 ratings, above the cold-start threshold
      * ``cold@example.com``  - 1 rating, below it
      * ``other@example.com`` - used to check ownership boundaries
    """
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    with SessionLocal() as db:
        restaurant = Restaurant(name="Test Kitchen", area="Dhanmondi", cuisine_tags=["bengali"])
        db.add(restaurant)
        db.flush()

        specs = [
            ("Kacchi Biryani", "mughlai", 4, False, Decimal("450.00"), True),
            ("Beef Bhuna", "bengali", 4, False, Decimal("320.00"), True),
            ("Margherita Pizza", "italian", 0, True, Decimal("480.00"), True),
            ("Rasmalai", "dessert", 0, True, Decimal("110.00"), True),
            ("Masala Cha", "beverage", 2, True, Decimal("50.00"), True),
            ("Sold Out Dish", "bengali", 3, False, Decimal("200.00"), False),
            # The warm user rates the first five, so these three exist purely to
            # leave the recommender something to actually recommend. They are
            # chosen to fall outside every filter assertion below: non-veg,
            # spice >= 2, priced between 110 and 400, in cuisines already present.
            ("Morog Polao", "mughlai", 2, False, Decimal("300.00"), True),
            ("Chicken Rezala", "bengali", 3, False, Decimal("280.00"), True),
            ("Shami Kabab", "mughlai", 3, False, Decimal("180.00"), True),
        ]
        items = []
        for name, cuisine, spice, veg, price, available in specs:
            item = FoodItem(
                restaurant_id=restaurant.id,
                name=name,
                description=f"{name} description",
                cuisine=cuisine,
                spice_level=spice,
                is_veg=veg,
                price=price,
                is_available=available,
                ingredient_tags=["test"],
            )
            db.add(item)
            db.flush()
            items.append(item)

        users = {}
        for email, count in (("warm", 5), ("cold", 1), ("other", 0)):
            user = User(
                name=email.title(),
                email=f"{email}@example.com",
                password_hash=PASSWORD_HASH,
                spice_tolerance=3,
            )
            db.add(user)
            db.flush()
            users[email] = user
            for i in range(count):
                db.add(
                    Rating(
                        user_id=user.id,
                        food_item_id=items[i].id,
                        rating=Decimal(5 - (i % 3)),
                        created_at=datetime.now(UTC) - timedelta(days=count - i),
                    )
                )

        # A neighbour link so reason attribution has something real to cite.
        db.add(
            ItemNeighbor(
                food_item_id=items[0].id,
                neighbor_id=items[1].id,
                distance=Decimal("0.5"),
                rank=1,
            )
        )
        db.commit()

    yield TestClient(app)


def _token(client: TestClient, email: str) -> dict[str, str]:
    response = client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture
def trained_model(client: TestClient, model_root: Path) -> TestClient:
    """Save a model whose ids match this fixture's catalogue, then load it.

    Built so ``warm@example.com`` most prefers item 3 (Margherita Pizza), giving
    the personalisation assertions something deterministic to check.
    """
    from app.services.recommender import reload_model

    with SessionLocal() as db:
        item_ids = [
            row[0]
            for row in db.execute(
                FoodItem.__table__.select().with_only_columns(FoodItem.id).order_by(FoodItem.id)
            )
        ]
        user_ids = [
            row[0]
            for row in db.execute(
                User.__table__.select().with_only_columns(User.id).order_by(User.id)
            )
        ]

    n_m, n_u, n = len(item_ids), len(user_ids), 3
    X = np.zeros((n_m, n))
    W = np.zeros((n_u, n))
    X[:, 0] = np.linspace(0.1, 1.0, n_m)
    W[:, 0] = 1.0
    save_cf_artifacts(
        next_version_dir(model_root),
        X=X,
        W=W,
        b=np.zeros((1, n_u)),
        mu=np.full(n_m, 3.0),
        item_ids=item_ids,
        user_ids=user_ids,
        metadata={"model": "test", "hyperparameters": {"mode": "explicit"}},
    )
    reload_model()
    return client


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_register_returns_a_usable_token(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register",
        json={"name": "New", "email": "new@example.com", "password": PASSWORD},
    )
    assert response.status_code == 201
    token = response.json()["access_token"]

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "new@example.com"


def test_register_rejects_a_duplicate_email(client: TestClient) -> None:
    payload = {"name": "Dup", "email": "warm@example.com", "password": PASSWORD}
    assert client.post("/api/auth/register", json=payload).status_code == 409


def test_register_rejects_a_short_password(client: TestClient) -> None:
    response = client.post(
        "/api/auth/register", json={"name": "X", "email": "x@example.com", "password": "short"}
    )
    assert response.status_code == 422


def test_login_rejects_a_wrong_password(client: TestClient) -> None:
    response = client.post(
        "/api/auth/login", json={"email": "warm@example.com", "password": "wrong"}
    )
    assert response.status_code == 401


def test_an_unknown_email_is_indistinguishable_from_a_wrong_password(client: TestClient) -> None:
    """Same status and same body, so login cannot be used to enumerate accounts."""
    unknown = client.post(
        "/api/auth/login", json={"email": "nobody@example.com", "password": PASSWORD}
    )
    wrong = client.post("/api/auth/login", json={"email": "warm@example.com", "password": "bad"})

    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json()


@pytest.mark.parametrize("header", [None, "Bearer nonsense", "Bearer "])
def test_protected_endpoints_reject_bad_credentials(client: TestClient, header: str | None) -> None:
    headers = {"Authorization": header} if header else {}
    assert client.get("/api/auth/me", headers=headers).status_code in (401, 403)


def test_taste_profile_reports_cold_start(client: TestClient) -> None:
    profile = client.get("/api/auth/me/taste-profile", headers=_token(client, "cold@example.com"))
    assert profile.status_code == 200
    assert profile.json()["is_cold_start"] is True

    warm = client.get("/api/auth/me/taste-profile", headers=_token(client, "warm@example.com"))
    assert warm.json()["is_cold_start"] is False
    assert warm.json()["total_ratings"] == 5


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------


def test_menu_hides_unavailable_items_by_default(client: TestClient) -> None:
    body = client.get("/api/menu").json()
    names = {item["name"] for item in body["items"]}
    assert "Sold Out Dish" not in names
    assert body["total"] == 8

    including = client.get("/api/menu?available_only=false").json()
    assert including["total"] == 9


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("cuisine=dessert", {"Rasmalai"}),
        ("cuisine=dessert&cuisine=beverage", {"Rasmalai", "Masala Cha"}),
        ("is_veg=true", {"Margherita Pizza", "Rasmalai", "Masala Cha"}),
        ("max_spice=0", {"Margherita Pizza", "Rasmalai"}),
        ("min_price=400", {"Kacchi Biryani", "Margherita Pizza"}),
        ("max_price=110", {"Rasmalai", "Masala Cha"}),
        ("search=biryani", {"Kacchi Biryani"}),
    ],
)
def test_menu_filters(client: TestClient, query: str, expected: set[str]) -> None:
    body = client.get(f"/api/menu?{query}").json()
    assert {item["name"] for item in body["items"]} == expected


def test_menu_sorts_by_price(client: TestClient) -> None:
    ascending = client.get("/api/menu?sort=price_asc").json()["items"]
    prices = [float(item["price"]) for item in ascending]
    assert prices == sorted(prices)

    descending = client.get("/api/menu?sort=price_desc").json()["items"]
    assert [float(i["price"]) for i in descending] == sorted(prices, reverse=True)


def test_menu_paginates_without_dropping_or_repeating_items(client: TestClient) -> None:
    first = client.get("/api/menu?sort=name&limit=2&offset=0").json()
    second = client.get("/api/menu?sort=name&limit=2&offset=2").json()

    assert first["total"] == second["total"] == 8
    assert len(first["items"]) == len(second["items"]) == 2
    assert not {i["id"] for i in first["items"]} & {i["id"] for i in second["items"]}


def test_menu_rejects_an_oversized_page(client: TestClient) -> None:
    assert client.get("/api/menu?limit=1000").status_code == 422


def test_menu_detail_includes_rating_aggregates(client: TestClient) -> None:
    item_id = client.get("/api/menu?search=biryani").json()["items"][0]["id"]
    detail = client.get(f"/api/menu/{item_id}").json()

    assert detail["name"] == "Kacchi Biryani"
    assert detail["rating_count"] == 2  # warm and cold both rated the first dish
    assert detail["average_rating"] is not None
    assert detail["restaurant"]["name"] == "Test Kitchen"


def test_menu_detail_404s_for_an_unknown_item(client: TestClient) -> None:
    assert client.get("/api/menu/999999").status_code == 404


def test_cuisines_endpoint_lists_only_live_cuisines(client: TestClient) -> None:
    assert client.get("/api/menu/cuisines").json() == [
        "bengali",
        "beverage",
        "dessert",
        "italian",
        "mughlai",
    ]


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------


def test_rating_requires_authentication(client: TestClient) -> None:
    assert client.post("/api/ratings", json={"food_item_id": 1, "rating": 5}).status_code in (
        401,
        403,
    )


def test_rating_the_same_item_twice_updates_rather_than_conflicts(client: TestClient) -> None:
    headers = _token(client, "other@example.com")
    item_id = client.get("/api/menu?search=Rasmalai").json()["items"][0]["id"]

    assert (
        client.post(
            "/api/ratings", json={"food_item_id": item_id, "rating": 5}, headers=headers
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/api/ratings", json={"food_item_id": item_id, "rating": 2}, headers=headers
        ).status_code
        == 201
    )

    listing = client.get("/api/ratings/me", headers=headers).json()
    assert listing["total"] == 1
    assert float(listing["items"][0]["rating"]) == 2.0


@pytest.mark.parametrize("value", [0, 6, -1])
def test_ratings_outside_the_scale_are_rejected(client: TestClient, value: int) -> None:
    headers = _token(client, "other@example.com")
    response = client.post(
        "/api/ratings", json={"food_item_id": 1, "rating": value}, headers=headers
    )
    assert response.status_code == 422


def test_rating_an_unknown_item_404s(client: TestClient) -> None:
    headers = _token(client, "other@example.com")
    response = client.post(
        "/api/ratings", json={"food_item_id": 999999, "rating": 5}, headers=headers
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def test_order_total_is_computed_from_database_prices(client: TestClient) -> None:
    """A client must not be able to influence what it is charged."""
    headers = _token(client, "other@example.com")
    item = client.get("/api/menu?search=Rasmalai").json()["items"][0]

    response = client.post(
        "/api/orders",
        # unit_price is deliberately not part of the request schema; sending it
        # should change nothing.
        json={"items": [{"food_item_id": item["id"], "quantity": 3, "unit_price": "0.01"}]},
        headers=headers,
    )
    assert response.status_code == 201
    body = response.json()
    assert float(body["total_amount"]) == pytest.approx(330.0)  # 110 * 3
    assert float(body["items"][0]["unit_price"]) == pytest.approx(110.0)


def test_ordering_an_unavailable_item_is_rejected(client: TestClient) -> None:
    headers = _token(client, "other@example.com")
    sold_out = client.get("/api/menu?available_only=false&search=Sold").json()["items"][0]

    response = client.post(
        "/api/orders",
        json={"items": [{"food_item_id": sold_out["id"], "quantity": 1}]},
        headers=headers,
    )
    assert response.status_code == 409


def test_ordering_an_unknown_item_404s(client: TestClient) -> None:
    headers = _token(client, "other@example.com")
    response = client.post(
        "/api/orders", json={"items": [{"food_item_id": 999999, "quantity": 1}]}, headers=headers
    )
    assert response.status_code == 404


def test_an_empty_order_is_rejected(client: TestClient) -> None:
    headers = _token(client, "other@example.com")
    assert client.post("/api/orders", json={"items": []}, headers=headers).status_code == 422


def test_another_users_order_is_not_readable(client: TestClient) -> None:
    """404 rather than 403 - confirming the order exists is itself a leak."""
    owner = _token(client, "warm@example.com")
    created = client.post(
        "/api/orders", json={"items": [{"food_item_id": 1, "quantity": 1}]}, headers=owner
    )
    order_id = created.json()["id"]

    intruder = _token(client, "other@example.com")
    assert client.get(f"/api/orders/{order_id}", headers=intruder).status_code == 404
    assert client.get(f"/api/orders/{order_id}", headers=owner).status_code == 200


# ---------------------------------------------------------------------------
# Impressions
# ---------------------------------------------------------------------------


def test_impression_batch_records_known_items_and_skips_the_rest(client: TestClient) -> None:
    headers = _token(client, "other@example.com")
    response = client.post(
        "/api/impressions/batch",
        json={
            "impressions": [
                {"food_item_id": 1, "was_ordered": True},
                {"food_item_id": 2, "was_ordered": False},
                {"food_item_id": 999999, "was_ordered": False},
            ]
        },
        headers=headers,
    )
    assert response.status_code == 202
    assert response.json() == {"recorded": 2, "skipped": 1}


def test_an_empty_impression_batch_is_rejected(client: TestClient) -> None:
    headers = _token(client, "other@example.com")
    assert (
        client.post("/api/impressions/batch", json={"impressions": []}, headers=headers).status_code
        == 422
    )


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def test_popular_is_non_empty_without_any_model(client: TestClient) -> None:
    """The API must serve recommendations before anything has been trained."""
    body = client.get("/api/recommendations/popular").json()
    assert len(body["items"]) > 0
    assert body["model_version"] is None
    assert all(item["reason"] for item in body["items"])


def test_a_brand_new_user_still_receives_recommendations(client: TestClient) -> None:
    """The cold-start requirement: retrieval must never return an empty feed."""
    registered = client.post(
        "/api/auth/register",
        json={"name": "Brand New", "email": "brandnew@example.com", "password": PASSWORD},
    )
    headers = {"Authorization": f"Bearer {registered.json()['access_token']}"}

    body = client.get("/api/recommendations/for-me", headers=headers).json()
    assert len(body["items"]) > 0
    assert body["is_cold_start"] is True
    assert all(item["reason"] for item in body["items"])


def test_a_user_below_the_rating_threshold_is_treated_as_cold(
    trained_model: TestClient,
) -> None:
    body = trained_model.get(
        "/api/recommendations/for-me", headers=_token(trained_model, "cold@example.com")
    ).json()
    assert body["is_cold_start"] is True


def test_a_warm_user_gets_model_backed_recommendations(trained_model: TestClient) -> None:
    body = trained_model.get(
        "/api/recommendations/for-me", headers=_token(trained_model, "warm@example.com")
    ).json()

    assert body["is_cold_start"] is False
    assert body["model_version"] == "v1"
    assert body["candidates_considered"] > 0
    assert all(item["reason"] for item in body["items"])


def test_recommendations_never_repeat_something_already_rated(
    trained_model: TestClient,
) -> None:
    """Recommending a dish the customer has already reviewed is not a
    recommendation, and it wastes a slot in a short list."""
    headers = _token(trained_model, "warm@example.com")
    rated = {
        item["food_item"]["id"]
        for item in trained_model.get("/api/ratings/me", headers=headers).json()["items"]
    }
    recommended = {
        item["item"]["id"]
        for item in trained_model.get("/api/recommendations/for-me", headers=headers).json()[
            "items"
        ]
    }
    assert not rated & recommended


def test_recommendations_exclude_unavailable_items(trained_model: TestClient) -> None:
    headers = _token(trained_model, "warm@example.com")
    body = trained_model.get("/api/recommendations/for-me", headers=headers).json()
    assert all(item["item"]["is_available"] for item in body["items"])


def test_the_second_identical_request_is_served_from_cache(trained_model: TestClient) -> None:
    headers = _token(trained_model, "warm@example.com")
    first = trained_model.get("/api/recommendations/for-me", headers=headers).json()
    second = trained_model.get("/api/recommendations/for-me", headers=headers).json()

    assert first["cached"] is False
    assert second["cached"] is True
    assert [i["item"]["id"] for i in first["items"]] == [i["item"]["id"] for i in second["items"]]


def test_rating_something_invalidates_the_cached_feed(trained_model: TestClient) -> None:
    """Otherwise a customer rates a dish, reloads, and sees an identical list -
    which reads as the rating having been ignored."""
    headers = _token(trained_model, "warm@example.com")
    trained_model.get("/api/recommendations/for-me?limit=20", headers=headers)
    assert trained_model.get("/api/recommendations/for-me?limit=20", headers=headers).json()[
        "cached"
    ]

    trained_model.post("/api/ratings", json={"food_item_id": 5, "rating": 5}, headers=headers)
    assert not trained_model.get("/api/recommendations/for-me?limit=20", headers=headers).json()[
        "cached"
    ]


def test_similar_items_cites_the_source_dish(client: TestClient) -> None:
    source_id = client.get("/api/menu?search=biryani").json()["items"][0]["id"]
    body = client.get(f"/api/recommendations/similar/{source_id}").json()

    assert len(body["items"]) == 1
    assert body["items"][0]["item"]["name"] == "Beef Bhuna"
    assert body["items"][0]["reason"] == "Similar to Kacchi Biryani"


def test_similar_items_is_empty_rather_than_an_error_for_an_unknown_item(
    client: TestClient,
) -> None:
    body = client.get("/api/recommendations/similar/999999").json()
    assert body["items"] == []


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


def test_admin_metrics_requires_authentication(client: TestClient) -> None:
    assert client.get("/api/admin/model/metrics").status_code in (401, 403)


def test_admin_metrics_reports_no_model_before_training(client: TestClient) -> None:
    body = client.get("/api/admin/model/metrics", headers=_token(client, "warm@example.com")).json()
    assert body["loaded"] is None
    assert body["available_versions"] == []


def test_admin_metrics_reports_the_loaded_model(trained_model: TestClient) -> None:
    body = trained_model.get(
        "/api/admin/model/metrics", headers=_token(trained_model, "warm@example.com")
    ).json()
    assert body["loaded"]["version"] == "v1"
    assert body["available_versions"] == ["v1"]
    assert body["neighbours_indexed"] == 1


def test_retrain_reports_the_command_instead_of_training_in_process(client: TestClient) -> None:
    """Training inside a request handler would block a worker for minutes."""
    response = client.post(
        "/api/admin/model/retrain",
        json={"mode": "explicit", "iterations": 200, "lambda": 2.0},
        headers=_token(client, "warm@example.com"),
    )
    assert response.status_code == 202
    body = response.json()
    assert "python -m ml.train_cf" in body["command"]
    assert "--lambda 2.0" in body["command"]
    assert "build_neighbors" in body["command"]


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "origin", ["http://localhost:3000", "http://127.0.0.1:3000"]
)
def test_both_development_origins_are_allowed(client: TestClient, origin: str) -> None:
    """A browser treats localhost and 127.0.0.1 as different origins.

    Allowing only one produced an app that rendered its shell perfectly and then
    fetched nothing at all, with the cause visible only in the browser console.
    Found by driving the real UI, not by any request-level test.
    """
    response = client.options(
        "/api/menu",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin
