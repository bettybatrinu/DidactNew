from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping
except ModuleNotFoundError:  # pragma: no cover - runtime fallback for deployment images
    tf = None
    layers = None
    models = None
    EarlyStopping = None

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"

FEATURE_COLUMNS = [
    "time_spent_seconds",
    "hint_count",
    "attempt_count",
    "is_correct",
    "mistake_count",
    "exercise_difficulty_encoded",
    "previous_mastery",
    "consecutive_errors",
    "help_level_requested",
]
LABEL_COLUMN = "learner_state"


def _ensure_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def generate_student_interactions_csv(path: Path = PROCESSED_DIR / "student_interactions.csv") -> pd.DataFrame:
    """Generate a realistic synthetic dataset when no real interaction logs exist yet."""
    _ensure_seed(42)
    n_rows = 2200
    rng = np.random.default_rng(42)

    difficulty = rng.integers(1, 5, size=n_rows)
    mastery = np.clip(rng.normal(0.55, 0.22, size=n_rows), 0.05, 0.98)
    consecutive_errors = rng.integers(0, 6, size=n_rows)
    hint_count = np.clip(rng.poisson(1.1, size=n_rows) + (difficulty > 3).astype(int) * 1, 0, 6)
    attempt_count = np.clip(rng.poisson(1.8, size=n_rows) + (hint_count > 2).astype(int) * 1, 1, 6)
    mistake_count = np.clip((consecutive_errors * 0.7 + rng.poisson(0.8, size=n_rows) + (difficulty > 2).astype(int) * 1), 0, 8)
    time_spent_seconds = np.clip(25 + difficulty * 18 + attempt_count * 12 + mistake_count * 10 + rng.normal(0, 12, size=n_rows), 10, 240).astype(int)
    help_level_requested = np.clip((hint_count + (mistake_count > 3).astype(int) + (time_spent_seconds > 110).astype(int)), 0, 3).astype(int)

    states = []
    for i in range(n_rows):
        d = int(difficulty[i])
        m = float(mastery[i])
        e = int(consecutive_errors[i])
        h = int(hint_count[i])
        a = int(attempt_count[i])
        mc = int(mistake_count[i])
        hl = int(help_level_requested[i])

        if m < 0.35 and (e >= 3 or h >= 3 or mc >= 4):
            state = "blocaj"
        elif m < 0.60 and (a >= 3 or hl >= 2):
            state = "progres"
        elif m >= 0.75 and (h <= 1 and a <= 2 and mc <= 2):
            state = "autonomie_buna"
        else:
            state = "supraincarcare" if (h >= 2 or hl >= 2 or time_spent_seconds[i] > 160) else "progres"

        if d >= 4 and m > 0.72 and e <= 1:
            state = "autonomie_buna"
        if d <= 2 and (h >= 4 or mc >= 5):
            state = "blocaj"

        states.append(state)

    df = pd.DataFrame(
        {
            "time_spent_seconds": time_spent_seconds,
            "hint_count": hint_count,
            "attempt_count": attempt_count,
            "is_correct": np.where(np.array(states) == "autonomie_buna", 1, np.where(np.array(states) == "blocaj", 0, rng.integers(0, 2, size=n_rows))).astype(int),
            "mistake_count": mistake_count,
            "exercise_difficulty_encoded": difficulty,
            "previous_mastery": np.clip(mastery + (np.array(states) == "autonomie_buna") * 0.12 + (np.array(states) == "blocaj") * -0.10 + rng.normal(0, 0.04, size=n_rows), 0.05, 0.98),
            "consecutive_errors": consecutive_errors,
            "help_level_requested": help_level_requested,
            LABEL_COLUMN: states,
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_student_interactions(path: Path = PROCESSED_DIR / "student_interactions.csv") -> pd.DataFrame:
    if not path.exists():
        return generate_student_interactions_csv(path)
    return pd.read_csv(path)


def train_neural_student_state_model(path: Path = PROCESSED_DIR / "student_interactions.csv") -> Dict:
    """Train the neural network and save model artifacts to models/."""
    if tf is None or models is None or layers is None or EarlyStopping is None:
        raise ModuleNotFoundError("TensorFlow is required to train the neural student-state model.")

    _ensure_seed(42)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_student_interactions(path).copy()
    df = df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN])

    X = df[FEATURE_COLUMNS].astype(float)
    y = df[LABEL_COLUMN].astype(str)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    num_classes = len(label_encoder.classes_)
    model = models.Sequential(
        [
            layers.Input(shape=(X_train_scaled.shape[1],)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.15,
        epochs=120,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=0,
    )

    y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    balanced = float(balanced_accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    model.save(MODELS_DIR / "neural_student_state_model.keras")
    joblib.dump(scaler, MODELS_DIR / "neural_student_state_scaler.joblib")
    joblib.dump(label_encoder, MODELS_DIR / "neural_student_state_label_encoder.joblib")

    evaluation_report = {
        "task": "Neural student learning-state prediction",
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "classes": label_encoder.classes_.tolist(),
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced,
        },
        "classification_report": report,
        "confusion_matrix": {
            "labels": label_encoder.classes_.tolist(),
            "matrix": cm.tolist(),
        },
        "training_history": {
            "epochs": int(len(history.history.get("loss", []))),
            "best_val_loss": float(np.min(history.history.get("val_loss", [np.inf]))),
        },
    }
    with open(MODELS_DIR / "neural_student_state_report.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, ensure_ascii=False, indent=2)

    return evaluation_report


def load_neural_model_artifacts() -> Dict[str, object]:
    if tf is None or models is None:
        raise ModuleNotFoundError("TensorFlow is required to load the neural student-state model.")

    model = tf.keras.models.load_model(MODELS_DIR / "neural_student_state_model.keras")
    scaler = joblib.load(MODELS_DIR / "neural_student_state_scaler.joblib")
    label_encoder = joblib.load(MODELS_DIR / "neural_student_state_label_encoder.joblib")
    return {"model": model, "scaler": scaler, "label_encoder": label_encoder}


def predict_neural_student_state(features_dict: Dict[str, float]) -> Dict[str, object]:
    if tf is None or models is None:
        raise ModuleNotFoundError("TensorFlow is required to run the neural student-state model.")

    artifacts = load_neural_model_artifacts()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    label_encoder = artifacts["label_encoder"]

    frame = pd.DataFrame([features_dict])[FEATURE_COLUMNS]
    scaled = scaler.transform(frame.astype(float))
    probabilities = model.predict(scaled, verbose=0)[0]
    pred_index = int(np.argmax(probabilities))
    label = label_encoder.inverse_transform([pred_index])[0]

    action_map = {
        "blocaj": "Exercițiu mai ușor + indiciu concret, cu un pas de rezolvare simplu și o nouă încercare mică.",
        "progres": "Urmărește cu un exercițiu puțin mai dificil, păstrând un singur indiciu de sprijin.",
        "supraincarcare": "Reducerea sarcinii: un task ghidat, mai simplu, cu feedback scurt și mai puține etape.",
        "autonomie_buna": "Provocă elevul cu un exercițiu mai dificil și o problemă de transfer, fără indiciu imediat.",
    }

    return {
        "predicted_state": label,
        "probabilities": {cls: float(probabilities[i]) for i, cls in enumerate(label_encoder.classes_)},
        "recommended_action": action_map.get(label, "Continuă cu exerciții adaptate nivelului actual."),
    }


def ensure_neural_model_exists() -> Dict:
    if tf is None or models is None or layers is None or EarlyStopping is None:
        return {"status": "unavailable", "reason": "TensorFlow runtime is not available in this environment."}
    if not (MODELS_DIR / "neural_student_state_model.keras").exists():
        return train_neural_student_state_model()
    return {"status": "ready", "model": str(MODELS_DIR / "neural_student_state_model.keras")}
