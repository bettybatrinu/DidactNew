"""Exploratory data analysis visuals for Didact AI.

Generates the figures that back the data-understanding criteria of the rubric:

Structured data (criterion 3):
  * a correlation heatmap of the engineered numeric features, so feature
    redundancy (e.g. problem_chars vs problem_words) is explicit instead of
    implied.

Unstructured / text data (criterion 6):
  * word clouds per curriculum domain,
  * top discriminative tokens per domain (bar charts),
  * problem-length distribution per domain (box plot).

All figures are written to /assets and are rendered read-only by the Streamlit
app, so the app itself needs no plotting dependencies at run time. Run with:

    python -m src.eda
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .data_prep import clean_text
    from .train_models import STRUCTURED_FEATURES_NUM
except ImportError:  # allow running as a script
    from data_prep import clean_text
    from train_models import STRUCTURED_FEATURES_NUM

try:
    from wordcloud import WordCloud

    _HAS_WORDCLOUD = True
except Exception:  # pragma: no cover - optional dependency
    WordCloud = None
    _HAS_WORDCLOUD = False

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
PROCESSED_DIR = ROOT / "data" / "processed"

# Romanian stop words that carry no curricular signal; kept small and explicit.
STOPWORDS = {
    "de", "la", "si", "in", "un", "o", "cu", "pe", "se", "este", "sunt", "care",
    "ce", "din", "al", "ale", "a", "ai", "il", "ii", "iar", "sa", "sau", "fie",
    "daca", "ca", "pentru", "lui", "cele", "cel", "cea", "cei", "le", "ne", "ei",
    "este", "are", "intr", "dintre", "prin", "asa", "dupa", "fara", "mai", "doua",
    "sa", "te", "ti", "va", "vor", "fiecare", "astfel", "incat", "unde", "cat",
}


def _load_dataset() -> pd.DataFrame:
    augmented = PROCESSED_DIR / "exercises_augmented.csv"
    processed = PROCESSED_DIR / "exercises_processed.csv"
    path = augmented if augmented.exists() else processed
    return pd.read_csv(path)


def _tokenize(text: str) -> List[str]:
    text = clean_text(text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]


def correlation_heatmap(data: pd.DataFrame) -> Path:
    """Correlation heatmap of engineered numeric features (criterion 3)."""
    cols = [c for c in STRUCTURED_FEATURES_NUM if c in data.columns]
    corr = data[cols].corr(numeric_only=True).fillna(0.0)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    ax.set_title("Corelații între features numerice (date structurate)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = ASSETS_DIR / "feature_correlation_heatmap.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def text_length_by_domain(data: pd.DataFrame) -> Path:
    """Box plot of problem length (words) per curriculum domain (criterion 6)."""
    df = data[data["Domeniu"].notna() & (data["Domeniu"] != "Altele")].copy()
    domains = df["Domeniu"].value_counts().index.tolist()
    groups = [df.loc[df["Domeniu"] == d, "problem_words"].dropna().values for d in domains]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(groups, tick_labels=domains, vert=True, showfliers=False)
    ax.set_ylabel("Lungime enunț (cuvinte)")
    ax.set_title("Distribuția lungimii enunțului pe domeniu curricular")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    out = ASSETS_DIR / "text_length_by_domain.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def top_tokens_by_domain(data: pd.DataFrame, top_domains: int = 4, top_tokens: int = 12) -> Path:
    """Bar charts of the most frequent tokens per domain (criterion 6)."""
    df = data[data["Domeniu"].notna() & (data["Domeniu"] != "Altele")].copy()
    domains = df["Domeniu"].value_counts().index.tolist()[:top_domains]

    n = len(domains)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, domain in zip(axes, domains):
        tokens: Counter = Counter()
        for text in df.loc[df["Domeniu"] == domain, "Problema"].astype(str):
            tokens.update(_tokenize(text))
        common = tokens.most_common(top_tokens)
        if not common:
            ax.axis("off")
            continue
        words, counts = zip(*reversed(common))
        ax.barh(words, counts, color="#4C72B0")
        ax.set_title(f"{domain} – termeni frecvenți", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Termeni cei mai frecvenți pe domeniu (text brut)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ASSETS_DIR / "top_tokens_by_domain.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def wordclouds_by_domain(data: pd.DataFrame, top_domains: int = 4) -> Path | None:
    """Word clouds per domain (criterion 6). Skipped gracefully if wordcloud is absent."""
    if not _HAS_WORDCLOUD:
        return None
    df = data[data["Domeniu"].notna() & (data["Domeniu"] != "Altele")].copy()
    domains = df["Domeniu"].value_counts().index.tolist()[:top_domains]

    n = len(domains)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, domain in zip(axes, domains):
        text = " ".join(
            " ".join(_tokenize(t)) for t in df.loc[df["Domeniu"] == domain, "Problema"].astype(str)
        )
        if not text.strip():
            ax.axis("off")
            continue
        wc = WordCloud(width=600, height=350, background_color="white",
                       colormap="viridis").generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(domain, fontsize=11)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Word clouds pe domeniu curricular", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ASSETS_DIR / "wordclouds_by_domain.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def difficulty_distribution(data: pd.DataFrame) -> Path:
    """Bar chart of the difficulty class distribution (criterion 3 EDA)."""
    labeled = data[data["Dificultate_group"] != "Necunoscut"]
    counts = labeled["Dificultate_group"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(counts.index, counts.values, color="#DD8452")
    ax.set_title("Distribuția dificultății")
    ax.set_ylabel("Număr exerciții")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = ASSETS_DIR / "difficulty_distribution.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def domain_distribution(data: pd.DataFrame) -> Path:
    """Bar chart of the curriculum-domain distribution (criterion 6 EDA)."""
    counts = data["Domeniu"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.set_title("Distribuția domeniilor curriculare")
    ax.set_ylabel("Număr exerciții")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = ASSETS_DIR / "domain_distribution.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def macro_f1_comparison() -> Path | None:
    """Model-vs-baseline macro-F1 comparison from the evaluation report (criteria 5/7)."""
    report_path = ROOT / "models" / "evaluation_report.json"
    if not report_path.exists():
        return None
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    labels = ["Structurat\n(dificultate)", "Nestructurat\n(domeniu)"]
    model_f1 = [
        report["structured_model"]["model"]["macro_f1"],
        report["unstructured_model"]["model"]["macro_f1"],
    ]
    base_f1 = [
        report["structured_model"]["baseline"]["macro_f1"],
        report["unstructured_model"]["baseline"]["macro_f1"],
    ]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, base_f1, width, label="Baseline", color="#999999")
    ax.bar(x + width / 2, model_f1, width, label="Model", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("Macro-F1: model vs baseline")
    ax.legend()
    for i in range(len(labels)):
        ax.text(i - width / 2, base_f1[i], f"{base_f1[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, model_f1[i], f"{model_f1[i]:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = ASSETS_DIR / "macro_f1_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_all() -> dict:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_dataset()
    outputs = {
        "correlation_heatmap": str(correlation_heatmap(data)),
        "difficulty_distribution": str(difficulty_distribution(data)),
        "domain_distribution": str(domain_distribution(data)),
        "text_length_by_domain": str(text_length_by_domain(data)),
        "top_tokens_by_domain": str(top_tokens_by_domain(data)),
    }
    mf1 = macro_f1_comparison()
    outputs["macro_f1_comparison"] = str(mf1) if mf1 else "skipped (no report)"
    wc = wordclouds_by_domain(data)
    outputs["wordclouds_by_domain"] = str(wc) if wc else "skipped (wordcloud not installed)"
    return outputs


if __name__ == "__main__":
    import json

    print(json.dumps(generate_all(), indent=2, ensure_ascii=False))
