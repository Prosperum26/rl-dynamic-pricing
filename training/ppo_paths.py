"""
Paths and helpers for per-category PPO checkpoints.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config.constants import DEFAULT_PRODUCT_CATEGORY, MODELS_DIR, PRODUCT_CATEGORIES

PPO_BEST_DIR = MODELS_DIR / "best_model"
LEGACY_BEST_PATH = PPO_BEST_DIR / "best_model.zip"
MULTI_CATEGORY_KEY = "all"


def category_slug(category: str) -> str:
    """Filesystem-safe id, e.g. 'Home & Kitchen' -> 'home_and_kitchen'."""
    s = category.strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "unknown"


def ppo_model_path(category: str) -> Path:
    """Path to the best PPO zip for a category (or 'all' for multi-category policy)."""
    slug = category_slug(category) if category != MULTI_CATEGORY_KEY else MULTI_CATEGORY_KEY
    return PPO_BEST_DIR / f"best_model_{slug}.zip"


def ppo_metadata_path(category: str) -> Path:
    slug = category_slug(category) if category != MULTI_CATEGORY_KEY else MULTI_CATEGORY_KEY
    return PPO_BEST_DIR / f"best_model_{slug}.json"


def normalize_category_arg(category: str) -> str:
    """Resolve CLI category string to a known category or 'all'."""
    raw = category.strip()
    if raw.lower() in (MULTI_CATEGORY_KEY, "multi", "*"):
        return MULTI_CATEGORY_KEY
    for cat in PRODUCT_CATEGORIES:
        if raw.lower() == cat.lower():
            return cat
    return raw


def archive_best_model(
    category: str,
    source: Path | None = None,
    *,
    timesteps: int | None = None,
    run_name: str | None = None,
) -> Path:
    """
    Copy EvalCallback best checkpoint to best_model_<category>.zip (+ metadata json).

    Also updates legacy best_model.zip for backward compatibility.
    """
    src = source or LEGACY_BEST_PATH
    if not src.exists():
        raise FileNotFoundError(f"No PPO checkpoint to archive: {src}")

    PPO_BEST_DIR.mkdir(parents=True, exist_ok=True)
    dest = ppo_model_path(category)
    src_resolved = src.resolve()
    dest_resolved = dest.resolve()
    legacy_resolved = LEGACY_BEST_PATH.resolve()

    if src_resolved != dest_resolved:
        shutil.copy2(src, dest)
    # EvalCallback already writes best_model.zip; do not copy onto itself
    if legacy_resolved not in (src_resolved, dest_resolved):
        shutil.copy2(src, LEGACY_BEST_PATH)

    meta = {
        "category": category,
        "category_slug": category_slug(category) if category != MULTI_CATEGORY_KEY else MULTI_CATEGORY_KEY,
        "model_path": str(dest),
        "archived_at": datetime.now().isoformat(timespec="seconds"),
        "timesteps": timesteps,
        "run_name": run_name,
    }
    meta_path = ppo_metadata_path(category)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return dest


def resolve_ppo_path_for_category(category: str) -> Path:
    """
    Pick the PPO file to load for a dashboard category.

    Order: per-category zip -> multi-category (all) -> legacy best_model.zip.
    """
    per_cat = ppo_model_path(category)
    if per_cat.exists():
        return per_cat
    multi = ppo_model_path(MULTI_CATEGORY_KEY)
    if multi.exists():
        return multi
    if LEGACY_BEST_PATH.exists():
        return LEGACY_BEST_PATH
    return per_cat


def list_available_ppo_models() -> Dict[str, Path]:
    """Map category label -> path for existing per-category (and all) checkpoints."""
    found: Dict[str, Path] = {}
    if not PPO_BEST_DIR.exists():
        return found
    for path in sorted(PPO_BEST_DIR.glob("best_model_*.zip")):
        stem = path.stem.replace("best_model_", "", 1)
        if stem == MULTI_CATEGORY_KEY:
            found[MULTI_CATEGORY_KEY] = path
            continue
        for cat in PRODUCT_CATEGORIES:
            if category_slug(cat) == stem:
                found[cat] = path
                break
    if LEGACY_BEST_PATH.exists() and not found:
        found["legacy"] = LEGACY_BEST_PATH
    return found


def categories_for_training(category_arg: str) -> List[str]:
    """Expand --category / --train-all-categories into a list of training jobs."""
    norm = normalize_category_arg(category_arg)
    if norm == MULTI_CATEGORY_KEY:
        return [MULTI_CATEGORY_KEY]
    return [norm]
