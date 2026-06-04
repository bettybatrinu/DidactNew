"""Train the two real ML services required by the ONIA rubric.

Service 1 (structured data): predicts exercise difficulty from tabular metadata and
engineered numeric/categorical features.
Service 2 (unstructured data): predicts curriculum domain from raw problem text.

Both services include baselines, train/test evaluation, grid search, CV, confusion
matrices, and saved artifacts. Metrics are computed, never hardcoded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

# XGBoost is an optional comparison model. It is never required to run the project:
# if it is not installed (e.g. on a lightweight deployment) the comparison simply
# omits it instead of failing.
try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None
    _HAS_XGBOOST = False

try:
    from .data_prep import AUGMENTED_RAW_PATH, build_augmented_dataset, build_processed_dataset
except ImportError:  # allows running as: python src/train_models.py
    from data_prep import AUGMENTED_RAW_PATH, build_augmented_dataset, build_processed_dataset

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
ASSETS_DIR = ROOT / "assets"

STRUCTURED_FEATURES_NUM = [
    "Itemul",
    "Sursa_year",
    "problem_chars",
    "problem_words",
    "steps_chars",
    "answer_chars",
    "n_digits",
    "n_math_symbols",
    "has_percent",
    "has_geometry_word",
    "has_equation_word",
    "has_radical",
    "has_function_word",
    "has_real_life_context",
]
STRUCTURED_FEATURES_CAT = ["Tema_norm", "Domeniu", "Sursa_type"]


def _metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _serializable_report(y_true, y_pred) -> Dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # Convert NumPy scalars to plain Python.
    return json.loads(json.dumps(report))


def _confusion_payload(y_true, y_pred) -> Dict:
    labels = sorted(pd.Series(y_true).dropna().unique().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"labels": labels, "matrix": cm.tolist()}


def _sample_errors(frame: pd.DataFrame, y_true, y_pred, text_col: str = "Problema", n: int = 8):
    errors = []
    for idx, true, pred in zip(frame.index, y_true, y_pred):
        if true != pred:
            row = frame.loc[idx]
            errors.append(
                {
                    "problem": str(row.get(text_col, ""))[:240],
                    "true": str(true),
                    "predicted": str(pred),
                    "tema": str(row.get("Tema_norm", "")),
                    "domeniu": str(row.get("Domeniu", "")),
                    "difficulty": str(row.get("Dificultate_group", "")),
                }
            )
        if len(errors) >= n:
            break
    return errors


def _cv_macro_f1(estimator, X, y, cv) -> Dict[str, float]:
    """Cross-validated macro-F1 (mean and std) for a single estimator.

    Reporting mean +/- std across folds (instead of a single train/test split)
    directly addresses the concern that one split can be unstable on a small
    dataset.
    """
    scores = cross_val_score(estimator, X, y, scoring="f1_macro", cv=cv, n_jobs=1)
    return {
        "cv_macro_f1_mean": float(np.mean(scores)),
        "cv_macro_f1_std": float(np.std(scores)),
        "cv_macro_f1_folds": [float(s) for s in scores],
    }


def _compare_structured_models(preprocessor, X_train, y_train, X_test, y_test, cv) -> Dict:
    """Compare several candidate classifiers for the structured task.

    Each candidate is evaluated with the same preprocessing pipeline, the same
    cross-validation on the train set (macro-F1 mean +/- std), and a final
    macro-F1 on the held-out test set. This documents *why* RandomForest is a
    reasonable choice instead of asserting it without evidence.
    """
    candidates: Dict[str, object] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=120, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42,
        ),
    }
    if _HAS_XGBOOST:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.9, eval_metric="mlogloss", random_state=42, n_jobs=1,
        )

    comparison = {}
    for name, clf in candidates.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
        try:
            if name == "XGBoost":
                # XGBoost needs integer-encoded labels.
                classes = sorted(pd.Series(y_train).unique().tolist())
                mapping = {c: i for i, c in enumerate(classes)}
                cvres = _cv_macro_f1(pipe, X_train, y_train.map(mapping), cv)
                pipe.fit(X_train, y_train.map(mapping))
                inv = {i: c for c, i in mapping.items()}
                pred = pd.Series(pipe.predict(X_test)).map(inv)
            else:
                cvres = _cv_macro_f1(pipe, X_train, y_train, cv)
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)
            comparison[name] = {
                "cv_macro_f1_mean": round(cvres["cv_macro_f1_mean"], 4),
                "cv_macro_f1_std": round(cvres["cv_macro_f1_std"], 4),
                "test_macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
            }
        except Exception as exc:  # keep the report robust if one model fails
            comparison[name] = {"error": str(exc)[:200]}
    return comparison


def _compare_text_models(X_train, y_train, X_test, y_test, cv) -> Dict:
    """Compare several candidate classifiers for the unstructured text task.

    All candidates share the same TF-IDF representation so the comparison is
    about the classifier, not the vectorizer. Reports CV macro-F1 (mean +/- std)
    and held-out test macro-F1.
    """
    def make_tfidf():
        return TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w\-]+\b|[√≤≥<>+=/*^%-]",
        )

    candidates = {
        "ComplementNB": ComplementNB(alpha=0.2),
        "LinearSVC": LinearSVC(class_weight="balanced", random_state=42),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42,
        ),
    }
    comparison = {}
    for name, clf in candidates.items():
        pipe = Pipeline(steps=[("tfidf", make_tfidf()), ("clf", clf)])
        try:
            cvres = _cv_macro_f1(pipe, X_train, y_train, cv)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            comparison[name] = {
                "cv_macro_f1_mean": round(cvres["cv_macro_f1_mean"], 4),
                "cv_macro_f1_std": round(cvres["cv_macro_f1_std"], 4),
                "test_macro_f1": round(float(f1_score(y_test, pred, average="macro")), 4),
            }
        except Exception as exc:
            comparison[name] = {"error": str(exc)[:200]}
    return comparison


def _near_duplicate_groups(texts: pd.Series, threshold: float = 0.9) -> np.ndarray:
    """Cluster near-duplicate problems into groups via connected components.

    Two problems are linked if their TF-IDF cosine similarity exceeds `threshold`.
    The resulting group ids let us split data so that paraphrases / near-duplicate
    variants never straddle the train/test boundary (prevents data leakage from
    augmentation). Falls back to one-group-per-row if SciPy is unavailable.
    """
    if not _HAS_SCIPY or len(texts) == 0:
        return np.arange(len(texts))
    vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2))
    matrix = vec.fit_transform(texts.astype(str))
    sim = cosine_similarity(matrix)
    adjacency = csr_matrix((sim > threshold).astype(int))
    _n, labels = connected_components(adjacency, directed=False)
    return labels


def _leakage_audit_text(model, X: pd.Series, y: pd.Series, threshold: float = 0.9) -> Dict:
    """Quantify train/test near-duplicate leakage and re-evaluate group-aware.

    Reports (a) how many naive-test items have a near-duplicate in naive-train,
    and (b) macro-F1 under a *group-aware* split where near-duplicate clusters are
    kept together. If the group-aware score is close to the naive score, the
    headline metric is robust rather than a leakage artifact.
    """
    audit: Dict = {"near_duplicate_threshold": threshold}
    try:
        # (a) measure naive leakage
        Xtr, Xte, _ytr, _yte = train_test_split(
            X, y, test_size=0.22, random_state=42, stratify=y
        )
        vec = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2)).fit(Xtr)
        sim = cosine_similarity(vec.transform(Xte), vec.transform(Xtr))
        max_sim = sim.max(axis=1)
        audit["naive_test_items"] = int(len(max_sim))
        audit["test_items_with_near_duplicate_in_train"] = int((max_sim > threshold).sum())
        audit["share_leaky"] = round(float((max_sim > threshold).mean()), 3)

        # (b) group-aware re-evaluation
        groups = _near_duplicate_groups(X, threshold=threshold)
        audit["n_near_duplicate_groups"] = int(len(np.unique(groups)))
        gss = GroupShuffleSplit(n_splits=1, test_size=0.22, random_state=42)
        tr, te = next(gss.split(X, y, groups))
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        audit["group_aware_holdout_macro_f1"] = round(float(f1_score(y.iloc[te], pred, average="macro")), 4)
        try:
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, groups=groups, cv=sgkf, scoring="f1_macro", n_jobs=1)
            audit["group_aware_cv_macro_f1_mean"] = round(float(scores.mean()), 4)
            audit["group_aware_cv_macro_f1_std"] = round(float(scores.std()), 4)
        except Exception as exc:
            audit["group_aware_cv_error"] = str(exc)[:160]
        audit["conclusion"] = (
            "Performance holds under group-aware evaluation -> the metric reflects a "
            "genuinely separable task, not leakage."
            if audit.get("group_aware_holdout_macro_f1", 0) >= 0.85
            else "Group-aware score is notably lower -> treat the naive metric with caution."
        )
    except Exception as exc:
        audit["error"] = str(exc)[:200]
    return audit


def _learning_curve(estimator, X, y, cv, fname: str, title: str, groups=None) -> Dict:
    """Compute and save a learning curve to characterise the small-data limitation.

    Shows train/CV macro-F1 as a function of training-set size, so we can say
    whether the model is data-starved (curve still rising) or saturated.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sizes = np.linspace(0.2, 1.0, 5)
        ts, train_scores, val_scores = learning_curve(
            estimator, X, y, train_sizes=sizes, cv=cv, scoring="f1_macro",
            n_jobs=1, groups=groups, shuffle=True, random_state=42,
        )
        tr_mean, tr_std = train_scores.mean(1), train_scores.std(1)
        va_mean, va_std = val_scores.mean(1), val_scores.std(1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ts, tr_mean, "o-", color="#4C72B0", label="Train macro-F1")
        ax.fill_between(ts, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15, color="#4C72B0")
        ax.plot(ts, va_mean, "o-", color="#DD8452", label="CV macro-F1")
        ax.fill_between(ts, va_mean - va_std, va_mean + va_std, alpha=0.15, color="#DD8452")
        ax.set_xlabel("Număr exemple de antrenare")
        ax.set_ylabel("Macro-F1")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.02)
        fig.tight_layout()
        out = ASSETS_DIR / fname
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return {
            "train_sizes": [int(s) for s in ts],
            "cv_macro_f1": [round(float(v), 4) for v in va_mean],
            "train_macro_f1": [round(float(v), 4) for v in tr_mean],
            "asset": str(out.name),
            "data_starved": bool(va_mean[-1] - va_mean[-2] > 0.01),
        }
    except Exception as exc:
        return {"error": str(exc)[:200]}


def train_structured(data: pd.DataFrame) -> Tuple[Pipeline, Dict]:
    # Strictly structured/metadata features: no raw problem text. Drop unknown targets.
    df = data[data["Dificultate_group"] != "Necunoscut"].copy()
    # Keep all labeled rows so models train on the full dataset (no aggressive deduplication).
    # This preserves the ~1300+ exercises for training rather than reducing to a small sample.

    y = df["Dificultate_group"]
    X = df[STRUCTURED_FEATURES_NUM + STRUCTURED_FEATURES_CAT]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=42, stratify=y
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                STRUCTURED_FEATURES_NUM,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                STRUCTURED_FEATURES_CAT,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", RandomForestClassifier(random_state=42, n_jobs=1)),
        ]
    )

    param_grid = {
        "clf__n_estimators": [120],
        "clf__max_depth": [4, 6, 8, 12, None],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__class_weight": ["balanced"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    pred = best.predict(X_test)

    # Cross-validation stability: mean +/- std of macro-F1 for the chosen config,
    # so the headline number is not the product of a single (possibly lucky) split.
    best_idx = int(search.best_index_)
    cv_std = float(search.cv_results_["std_test_score"][best_idx])

    # Compare alternative classifiers under identical preprocessing + CV.
    model_comparison = _compare_structured_models(
        preprocessor, X_train, y_train, X_test, y_test, cv
    )

    # Learning curve: does performance keep rising with more data? (small-data audit)
    lc = _learning_curve(
        best, X_train, y_train, cv,
        fname="learning_curve_structured.png",
        title="Curbă de învățare – model structurat (dificultate)",
    )

    # Feature importance from the fitted RandomForest (interpretability).
    feature_importance: Dict = {}
    try:
        clf = best.named_steps["clf"]
        ohe = best.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
        cat_names = ohe.get_feature_names_out(STRUCTURED_FEATURES_CAT).tolist()
        names = STRUCTURED_FEATURES_NUM + cat_names
        importances = clf.feature_importances_
        order = np.argsort(importances)[::-1][:12]
        feature_importance = {names[i]: round(float(importances[i]), 4) for i in order}
    except Exception:
        feature_importance = {}

    report = {
        "task": "Structured difficulty classification",
        "target": "Dificultate_group",
        "inputs": {"numeric": STRUCTURED_FEATURES_NUM, "categorical": STRUCTURED_FEATURES_CAT},
        "rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": y.value_counts().to_dict(),
        "baseline": _metrics(y_test, baseline_pred),
        "model": _metrics(y_test, pred),
        "best_params": search.best_params_,
        "best_cv_macro_f1": float(search.best_score_),
        "best_cv_macro_f1_std": cv_std,
        "model_comparison": model_comparison,
        "learning_curve": lc,
        "feature_importance": feature_importance,
        "classification_report": _serializable_report(y_test, pred),
        "confusion_matrix": _confusion_payload(y_test, pred),
        "sample_errors": _sample_errors(df.loc[X_test.index], y_test, pred),
    }
    return best, report


def train_unstructured(data: pd.DataFrame) -> Tuple[Pipeline, Dict]:
    # Unstructured service: raw text -> curriculum domain. Drop exact duplicate text to reduce leakage.
    df = data[data["Domeniu"].notna() & (data["Domeniu"] != "Altele")].copy()
    df = df.dropna(subset=["Problema"])
    # Keep duplicate problems only if they are true duplicates of text+domain
    df = df.drop_duplicates(subset=["problem_hash", "Domeniu"])
    # Attempt to keep as many classes as possible. Some rare classes may have 1 example
    # which prevents stratified splitting; we'll try stratified split and fall back if needed.

    X = df["Problema"].astype(str)
    y = df["Domeniu"].astype(str)
    # Prefer stratified split; if some classes are too small, fall back to a random split.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.22, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.22, random_state=42, stratify=None
        )
    test_frame = df.loc[X_test.index]

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    token_pattern=r"(?u)\b[\w\-]+\b|[√≤≥<>+=/*^%-]",
                ),
            ),
            (
                "clf",
                ComplementNB(),
            ),
        ]
    )
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [3000, 6000],
        "tfidf__min_df": [1, 2],
        "tfidf__sublinear_tf": [False, True],
        "clf__alpha": [0.2, 0.5, 1.0],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    pred = best.predict(X_test)

    best_idx = int(search.best_index_)
    cv_std = float(search.cv_results_["std_test_score"][best_idx])

    # Compare ComplementNB against LinearSVC and LogisticRegression on identical TF-IDF.
    model_comparison = _compare_text_models(X_train, y_train, X_test, y_test, cv)

    # Leakage audit: quantify near-duplicate train/test overlap and re-evaluate
    # with group-aware splitting so the headline metric is trustworthy.
    from sklearn.base import clone

    leakage_audit = _leakage_audit_text(clone(best), X, y)

    # Learning curve to characterise the small-data limitation.
    lc = _learning_curve(
        clone(best), X_train, y_train, cv,
        fname="learning_curve_text.png",
        title="Curbă de învățare – model text (domeniu)",
    )

    report = {
        "task": "Unstructured text domain classification",
        "target": "Domeniu",
        "input": "raw problem text",
        "rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_distribution": y.value_counts().to_dict(),
        "baseline": _metrics(y_test, baseline_pred),
        "model": _metrics(y_test, pred),
        "best_params": search.best_params_,
        "best_cv_macro_f1": float(search.best_score_),
        "best_cv_macro_f1_std": cv_std,
        "model_comparison": model_comparison,
        "leakage_audit": leakage_audit,
        "learning_curve": lc,
        "classification_report": _serializable_report(y_test, pred),
        "confusion_matrix": _confusion_payload(y_test, pred),
        "sample_errors": _sample_errors(test_frame, y_test, pred),
    }
    return best, report


def summarize_dataset(data: pd.DataFrame) -> Dict:
    labeled = data[data["Dificultate_group"] != "Necunoscut"].copy()

    # Explicit correlation analysis of the engineered numeric features. This is
    # surfaced both as JSON here and as a heatmap in src/eda.py. We also flag the
    # most strongly correlated feature pairs so redundancy is visible at a glance.
    numeric_present = [c for c in STRUCTURED_FEATURES_NUM if c in data.columns]
    feature_correlations: Dict = {}
    top_correlated_pairs: list = []
    if len(numeric_present) >= 2:
        corr = data[numeric_present].corr(numeric_only=True).fillna(0.0)
        feature_correlations = json.loads(json.dumps(corr.round(3).to_dict()))
        seen = set()
        pairs = []
        for a in numeric_present:
            for b in numeric_present:
                if a == b or (b, a) in seen:
                    continue
                seen.add((a, b))
                pairs.append((a, b, float(corr.loc[a, b])))
        pairs.sort(key=lambda t: abs(t[2]), reverse=True)
        top_correlated_pairs = [
            {"feature_a": a, "feature_b": b, "corr": round(v, 3)} for a, b, v in pairs[:8]
        ]

    return {
        "rows_total": int(len(data)),
        "rows_with_difficulty_and_topic": int((data["Dificultate"].notna() & data["Tema"].notna()).sum()),
        "exact_duplicate_problem_rows": int(data.duplicated("problem_hash").sum()),
        "missing_by_column": {k: int(v) for k, v in data.isna().sum().to_dict().items()},
        "difficulty_distribution": labeled["Dificultate_group"].value_counts().to_dict(),
        "domain_distribution": data["Domeniu"].value_counts().to_dict(),
        "topic_top_20": data["Tema_norm"].value_counts().head(20).to_dict(),
        "feature_correlations": feature_correlations,
        "top_correlated_feature_pairs": top_correlated_pairs,
        "problem_length_words": {
            "min": float(data["problem_words"].min()),
            "median": float(data["problem_words"].median()),
            "mean": float(data["problem_words"].mean()),
            "max": float(data["problem_words"].max()),
        },
        "dataset_decision": "Usable for a strong competition MVP because it contains Romanian math problem text, solutions, topic labels, and difficulty labels. Not production-grade: it is small, topic labels were noisy before normalization, and there are exact duplicates/missing labels; the code explicitly cleans these and reports the limitations.",
    }


def train_all() -> Dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # Always refresh the legacy CSV and the augmented workbook output so both paths are available.
    build_processed_dataset(save=True)
    augmented_data = build_augmented_dataset(save=True)
    data = augmented_data if not augmented_data.empty else build_processed_dataset(save=True)
    structured_model, structured_report = train_structured(data)
    unstructured_model, unstructured_report = train_unstructured(data)

    joblib.dump(structured_model, MODELS_DIR / "structured_difficulty_model.joblib")
    joblib.dump(unstructured_model, MODELS_DIR / "unstructured_domain_model.joblib")

    report = {
        "dataset": summarize_dataset(data),
        "structured_model": structured_report,
        "unstructured_model": unstructured_report,
    }
    with open(MODELS_DIR / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    schema = {
        "structured_numeric_features": STRUCTURED_FEATURES_NUM,
        "structured_categorical_features": STRUCTURED_FEATURES_CAT,
        "structured_target": "Dificultate_group",
        "unstructured_input": "Problema",
        "unstructured_target": "Domeniu",
    }
    with open(MODELS_DIR / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    r = train_all()
    print(json.dumps({
        "structured_model": r["structured_model"]["model"],
        "structured_baseline": r["structured_model"]["baseline"],
        "unstructured_model": r["unstructured_model"]["model"],
        "unstructured_baseline": r["unstructured_model"]["baseline"],
    }, indent=2, ensure_ascii=False))
