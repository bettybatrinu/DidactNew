"""Model loading and inference helpers for Didact AI."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from .data_prep import engineer_features
    from .train_models import STRUCTURED_FEATURES_CAT, STRUCTURED_FEATURES_NUM
except ImportError:
    from data_prep import engineer_features
    from train_models import STRUCTURED_FEATURES_CAT, STRUCTURED_FEATURES_NUM

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PROCESSED_PATH = ROOT / "data" / "processed" / "exercises_augmented.csv"
FALLBACK_PROCESSED_PATH = ROOT / "data" / "processed" / "exercises_processed.csv"


def resolve_dataset_path() -> Path:
    if PROCESSED_PATH.exists():
        return PROCESSED_PATH
    return FALLBACK_PROCESSED_PATH


def load_assets():
    structured = joblib.load(MODELS_DIR / "structured_difficulty_model.joblib")
    unstructured = joblib.load(MODELS_DIR / "unstructured_domain_model.joblib")
    data = pd.read_csv(resolve_dataset_path())
    with open(MODELS_DIR / "evaluation_report.json", "r", encoding="utf-8") as f:
        report = json.load(f)
    return structured, unstructured, data, report


def prepare_single_problem(problem: str, tema_norm: str, domeniu: str, item: int = 1, sursa_type: str = "manual") -> pd.DataFrame:
    row = pd.DataFrame(
        [{
            "Sursa": sursa_type,
            "Itemul": item,
            "Problema": problem,
            "Pasii de rezolvare": "",
            "Raspunsul": "",
            "Dificultate": np.nan,
            "Tema": tema_norm,
        }]
    )
    features = engineer_features(row)
    # Preserve user-selected normalized labels instead of re-normalizing unknown manual label.
    features["Tema_norm"] = tema_norm
    features["Domeniu"] = domeniu
    features["Sursa_type"] = sursa_type
    return features[STRUCTURED_FEATURES_NUM + STRUCTURED_FEATURES_CAT]


def predict_structured_difficulty(model, feature_frame: pd.DataFrame) -> Dict:
    pred = model.predict(feature_frame)[0]
    proba = None
    classes = getattr(model.named_steps.get("clf"), "classes_", None) if hasattr(model, "named_steps") else None
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(feature_frame)[0]
            proba = {str(c): float(v) for c, v in zip(classes, p)}
        except Exception:
            proba = None
    return {"prediction": str(pred), "probabilities": proba}


def predict_domain_from_text(model, text: str) -> Dict:
    pred = model.predict([text])[0]
    result = {"prediction": str(pred), "probabilities": None}
    if hasattr(model, "predict_proba"):
        p = model.predict_proba([text])[0]
        classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["clf"].classes_
        result["probabilities"] = {str(c): float(v) for c, v in zip(classes, p)}
    return result
