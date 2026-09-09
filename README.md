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
- [ ] **M2** — collaborative filtering training pipeline
- [ ] **M3** — evaluation harness and baseline comparison
- [ ] **M4** — FastAPI backend
- [ ] **M5** — Next.js frontend
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
```

## Documentation

- [`docs/ALGORITHM.md`](docs/ALGORITHM.md) — the math, and why collaborative filtering
  alone is not enough
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — metrics against baselines, honest limitations
