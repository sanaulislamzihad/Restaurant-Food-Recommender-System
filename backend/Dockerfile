# syntax=docker/dockerfile:1

# Serving image. TensorFlow is deliberately absent: collaborative-filtering
# inference is a dot product against saved numpy arrays, so the API does not
# need a ~600MB ML framework in it. Training runs as a separate offline job.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv/app

# libpq5 is the psycopg runtime; curl is used by the compose healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

# The whole source is copied before installing because the project is installed
# in editable mode against these packages. It costs a dependency reinstall on
# every source change; a separate requirements lock would cache better, and is
# worth doing once dependencies stop moving.
COPY pyproject.toml ./
COPY app ./app
COPY ml ./ml
COPY scripts ./scripts
COPY alembic ./alembic
COPY alembic.ini ./

RUN pip install --no-cache-dir -e .

# Run unprivileged.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /srv/app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Migrations run on start so a fresh volume comes up ready. That is fine for a
# single-replica demo; a multi-replica deployment would run them as a job so
# concurrent instances do not race for the Alembic lock.
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
