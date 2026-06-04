"""Application helpers for Didact AI Streamlit integration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.model_utils import load_assets, resolve_dataset_path
from src.train_models import train_all

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"
REQUIRED_FILES = [
    MODELS_DIR / "structured_difficulty_model.joblib",
    MODELS_DIR / "unstructured_domain_model.joblib",
    MODELS_DIR / "evaluation_report.json",
    resolve_dataset_path(),
]


def asset_status() -> Dict[str, object]:
    missing = [str(path) for path in REQUIRED_FILES if not path.exists()]
    return {
        "dataset_path": str(resolve_dataset_path()),
        "missing": missing,
        "assets_available": len(missing) == 0,
    }


def load_app_assets() -> tuple[object, object, object, Dict]:
    """Load the trained ML assets and data for the app.

    The Streamlit app should not rely on implicit background training of large
    ML services. If assets are missing, it fails fast and asks the user to run
    the training pipeline explicitly.
    """
    status = asset_status()
    if not status["assets_available"]:
        raise FileNotFoundError(
            "Required trained assets are missing. Please run `python -m src.train_models` "
            "and make sure models/ and data/processed/ are populated. "
            f"Missing: {', '.join(status['missing'])}"
        )

    structured, unstructured, data, report = load_assets()
    if not isinstance(report, dict) or "dataset" not in report:
        raise ValueError("Loaded evaluation report does not contain expected metadata.")
    return structured, unstructured, data, report


def ensure_assets_or_train() -> tuple[object, object, object, Dict]:
    """Attempt to load assets and train if they are not present.

    This helper exists for CLI workflows where explicit reproducibility is
    expected, not for normal app startup.
    """
    status = asset_status()
    if not status["assets_available"]:
        train_all()
    return load_app_assets()


def initialize_session_state(session_state) -> None:
    defaults = {
        "mastery": 0.55,
        "hints_used": 0,
        "attempts": 1,
        "diagnostic_started": False,
        "diagnostic_results": None,
        "diagnostic_seed": 0,
        "interaction_log": [],
        "current_exercise_start_time": None,
        "current_exercise_attempt_count": 0,
        "current_exercise_hint_count": 0,
        "current_exercise_mistake_count": 0,
        "current_exercise_consecutive_errors": 0,
        "neural_available": False,
    }
    for key, value in defaults.items():
        if key not in session_state:
            session_state[key] = value


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
