# Restaurant Food Recommender System

A production-shaped food recommendation engine for a restaurant ordering app. Two
recommendation models are implemented from scratch and blended:

- **Collaborative filtering** (Andrew Ng's formulation) — learns latent item features `X`,
  user preferences `W` and per-user biases `b` directly from the rating matrix, with a
  vectorized cost function and a custom `tf.GradientTape` training loop.
- **Content-based two-tower network** — separate user and item networks producing
  L2-normalized 32-dimensional embeddings, combined with a dot product. This is what makes
  brand-new users and brand-new menu items work.

Every recommendation carries a human-readable `reason` ("because you liked Kacchi Biryani").

## Status

Built milestone by milestone. Current progress:

- [x] **M1** — schema, migrations, clustered seed data
- [x] **M2** — collaborative filtering training pipeline
- [x] **M3** — evaluation harness and baseline comparison
- [x] **M4** — FastAPI backend
- [x] **M5** — Next.js frontend
- [ ] **M6** — two-tower content model, hybrid scoring, retrieval + ranking
- [ ] **M7** — docs, admin dashboard, CI

## Stack

| Layer | Choice |
| --- | --- |
| API | FastAPI, Pydantic v2 |
| ML | TensorFlow 2.x (both models hand-written, no recommender library) |
| Data | SQLAlchemy 2.0 + Alembic; Postgres in Docker/CI, SQLite for local dev |
| Cache | Redis, with an in-process fallback so the app runs without it |
| Web | Next.js 14 (App Router), TypeScript, Tailwind, shadcn/ui |

The database URL drives the dialect, so the whole system runs locally with no services
at all. `docker-compose` brings up the Postgres + Redis configuration.

## Quick start

```bash
cp .env.example .env
cd backend
py -3.11 -m venv .venv && source .venv/Scripts/activate   # Windows (Git Bash)
pip install -e ".[dev]"
alembic upgrade head
python -m scripts.seed
python -m scripts.verify_clusters   # proves the seeded matrix has real structure
```

That runs with no Docker, Postgres or Redis: the SQLite default in
`.env.example` is enough. `docker compose up` brings up the Postgres + Redis
configuration instead.

### What the seed produces

| | |
| --- | --- |
| Restaurants / menu items | 8 / 302 |
| Users / ratings | 500 / 14,998 |
| Orders / impressions | 7,645 / 55,813 |
| Matrix density | 9.93% |
| Cold-start users (< 5 ratings) | 25, of which 6 have none |
| Cold-start items (< 3 ratings) | 9, of which 3 have none |

Ratings are generated from three latent taste clusters. `verify_clusters`
confirms k-means can rediscover them from the matrix alone, with no labels:
purity 0.937, Adjusted Rand Index 0.815 against a 0.417 majority-class
baseline.

## Training

```bash
cd backend
python -m ml.train_cf                     # explicit ratings, squared error
python -m ml.train_cf --mode implicit     # impression log, cross-entropy
python -m ml.train_cf --lambda 5 --iterations 600
```

Each run writes a new `models/v{n}/` containing the weights, the id maps, a
loss curve and a `metadata.json` model card. Training is always an offline job;
the API only ever loads artifacts read-only.

The vectorized cost function is checked against a loop transcription of the
same formula on every test run — they agree to within 1e-11, against the 1e-4
the spec requires.

**On mean normalization.** A user who has rated nothing ends with `W[j] = 0`
and `b[0,j] = 0`, so their raw prediction for every dish is zero — below the
bottom of a 1–5 scale, ranking the entire menu identically badly. Training on
mean-centred data and adding `mu[i]` back means that user instead sees each
dish's average rating. Verified against a trained model: the cold user's
predictions match the item means to 3.4e-10.

## Running the app

```bash
# terminal 1 - API
cd backend && uvicorn app.main:app --reload

# terminal 2 - web
cd web && npm install && npm run dev
```

Then open http://localhost:3000. Sign in as any seeded address (they all use
the password `foodrec123`) to see a personalised feed, or register a fresh
account to see the cold-start path.

## API

20 endpoints under `/api`, with interactive docs at `/docs` once the server is
running:

```bash
cd backend
uvicorn app.main:app --reload
```

| | |
| --- | --- |
| auth | `POST /auth/register`, `POST /auth/login`, `GET /auth/me`, `GET /auth/me/taste-profile` |
| menu | `GET /menu` (filter, sort, paginate), `GET /menu/{id}`, `GET /menu/cuisines` |
| interactions | `POST /ratings`, `GET /ratings/me`, `POST /orders`, `GET /orders/history`, `GET /orders/{id}`, `POST /impressions/batch` |
| recommendations | `GET /recommendations/for-me`, `GET /recommendations/similar/{id}`, `GET /recommendations/popular` |
| admin | `GET /admin/model/metrics`, `POST /admin/model/retrain` |

Every recommended item carries a `reason` derived from what actually surfaced
it — "Because you liked Mutton Kacchi Biryani", not a generic string. A user
with too little history falls back to popularity with `is_cold_start: true` and
`model_version: null`, so the frontend can label that row honestly rather than
calling a popularity list "picked for you".

Recommendations are cached with the model version in the key, so retraining
invalidates them implicitly. Measured 30.7 ms uncached against 0.6 ms cached.
Redis is used when `REDIS_URL` is set and an in-process TTL cache when it is
not, so the API runs with no external services at all.

After training, rebuild the "similar items" index:

```bash
python -m ml.build_neighbors
```

## Results

Held-out test set, per-user chronological split. Full detail and the
limitations that qualify these numbers are in
[`docs/EVALUATION.md`](docs/EVALUATION.md).

| model | RMSE ↓ | NDCG@10 ↑ | coverage ↑ |
| --- | --- | --- | --- |
| global mean | 0.9689 | 0.0257 | 6.3% |
| item mean | 0.8870 | 0.0331 | 5.0% |
| most popular | — | 0.0681 | 14.6% |
| **collaborative filtering** | **0.7268** | **0.0908** | **74.2%** |

Collaborative filtering beats item-mean by 18.1% on RMSE and 174% on NDCG@10,
while reaching 74.2% of the menu instead of 5%. It is also close to
popularity-neutral: 10.3% of its recommendations land in the top popularity
decile, against 99.5% for the most-popular baseline.

## Documentation

- [`docs/ALGORITHM.md`](docs/ALGORITHM.md) — the math, and why collaborative filtering
  alone is not enough
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — metrics against baselines, honest limitations
