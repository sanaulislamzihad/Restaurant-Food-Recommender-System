"""Versioned model artifacts on disk.

Layout::

    models/
      v1/
        cf_model.npz     X, W, b, mu
        id_maps.json     matrix row/column order -> database ids
        metadata.json    the model card: when, on what, with which knobs, how well
        loss_curve.png

Each training run writes a new version rather than overwriting the last, so a
regression can be diagnosed by comparing model cards, and a bad model can be
rolled back by pointing MODEL_VERSION at the previous directory.

The id maps are saved because matrix positions are meaningless on their own:
row 7 is only "Kacchi Biryani" relative to the exact catalogue that was loaded
at training time. Serving a model against a shifted catalogue would silently
recommend the wrong dishes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

VERSION_PATTERN = re.compile(r"^v(\d+)$")

MODEL_FILE = "cf_model.npz"
ID_MAP_FILE = "id_maps.json"
METADATA_FILE = "metadata.json"
LOSS_CURVE_FILE = "loss_curve.png"


@dataclass
class CFArtifacts:
    """A trained collaborative filtering model, loaded from disk.

    X:  (n_m, n)   item feature matrix
    W:  (n_u, n)   user parameter matrix
    b:  (1, n_u)   per-user bias
    mu: (n_m,)     per-item mean rating, added back at prediction time
    """

    X: np.ndarray
    W: np.ndarray
    b: np.ndarray
    mu: np.ndarray
    item_ids: list[int]
    user_ids: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = ""

    def predict_all(self) -> np.ndarray:
        """Full (n_m, n_u) prediction matrix with the item means added back."""
        predictions: np.ndarray = self.X @ self.W.T + self.b + self.mu[:, None]
        return predictions

    def predict_for_user_index(self, user_index: int) -> np.ndarray:
        """Predicted rating of every item for one user column, shape (n_m,)."""
        predictions: np.ndarray = self.X @ self.W[user_index] + self.b[0, user_index] + self.mu
        return predictions

    def item_index(self) -> dict[int, int]:
        return {item_id: idx for idx, item_id in enumerate(self.item_ids)}

    def user_index(self) -> dict[int, int]:
        return {user_id: idx for idx, user_id in enumerate(self.user_ids)}


def list_versions(root: Path) -> list[str]:
    """Existing version directory names, ordered oldest to newest."""
    if not root.exists():
        return []
    versions = [
        (int(match.group(1)), path.name)
        for path in root.iterdir()
        if path.is_dir() and (match := VERSION_PATTERN.match(path.name))
    ]
    return [name for _, name in sorted(versions)]


def latest_version(root: Path) -> str | None:
    versions = list_versions(root)
    return versions[-1] if versions else None


def read_metadata(root: Path, version: str) -> dict[str, Any]:
    """Model card for one version, or an empty dict if it has none."""
    path = root / version / METADATA_FILE
    if not path.exists():
        return {}
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def latest_version_for_mode(root: Path, mode: str) -> str | None:
    """Newest version trained in a given feedback mode.

    Selecting purely by "newest" is a trap once both explicit and implicit
    models are on disk: the two learn different feature spaces, and a caller
    that wants rating-based item similarity would silently get whichever run
    happened to finish last.
    """
    for version in reversed(list_versions(root)):
        metadata = read_metadata(root, version)
        if metadata.get("hyperparameters", {}).get("mode") == mode:
            return version
    return None


def latest_version_with(root: Path, filename: str) -> str | None:
    """Newest version directory containing a given artifact file.

    The collaborative and content models are trained by separate jobs and land
    in separate version directories. Serving looks each half up independently,
    so training one does not invalidate the other and either can be absent.
    """
    for version in reversed(list_versions(root)):
        if (root / version / filename).exists():
            return version
    return None


def next_version_dir(root: Path) -> Path:
    """Allocate the next unused ``models/v{n}`` directory."""
    existing = list_versions(root)
    next_number = 1 + (int(VERSION_PATTERN.match(existing[-1]).group(1)) if existing else 0)  # type: ignore[union-attr]
    path = root / f"v{next_number}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_cf_artifacts(
    directory: Path,
    *,
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    item_ids: list[int],
    user_ids: list[int],
    metadata: dict[str, Any],
) -> Path:
    """Write one trained model into ``directory``. Returns the directory."""
    directory.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(directory / MODEL_FILE, X=X, W=W, b=b, mu=mu)
    (directory / ID_MAP_FILE).write_text(
        json.dumps({"item_ids": item_ids, "user_ids": user_ids}), encoding="utf-8"
    )

    card = {"saved_at": datetime.now(UTC).isoformat(), **metadata}
    (directory / METADATA_FILE).write_text(json.dumps(card, indent=2), encoding="utf-8")
    return directory


def load_cf_artifacts(
    root: Path, version: str | None = None, *, mode: str | None = None
) -> CFArtifacts:
    """Load a trained model.

    Args:
        root: the ``models/`` directory.
        version: an explicit version name. Wins over ``mode``.
        mode: restrict to the newest model trained in this feedback mode.

    Raises:
        FileNotFoundError: if no matching model exists, naming the command that
            would produce one - a missing model is a setup step not yet run,
            not a mystery to debug.
    """
    if version is not None:
        resolved: str | None = version
    elif mode is not None:
        resolved = latest_version_for_mode(root, mode)
        if resolved is None:
            raise FileNotFoundError(
                f"No {mode} model found under {root}. "
                f"Run `python -m ml.train_cf --mode {mode}` first."
            )
    else:
        resolved = latest_version(root)

    if resolved is None:
        raise FileNotFoundError(
            f"No trained model found under {root}. Run `python -m ml.train_cf` first."
        )

    directory = root / resolved
    if not (directory / MODEL_FILE).exists():
        raise FileNotFoundError(f"{directory / MODEL_FILE} is missing; the version is incomplete.")

    with np.load(directory / MODEL_FILE) as data:
        X, W, b, mu = data["X"], data["W"], data["b"], data["mu"]

    id_maps = json.loads((directory / ID_MAP_FILE).read_text(encoding="utf-8"))
    metadata_path = directory / METADATA_FILE
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    )

    return CFArtifacts(
        X=X,
        W=W,
        b=b,
        mu=mu,
        item_ids=id_maps["item_ids"],
        user_ids=id_maps["user_ids"],
        metadata=metadata,
        version=resolved,
    )
