"""Educational impact KPIs for Didact AI.

A good macro-F1 does not guarantee pedagogical value, so we define and measure an
explicit educational KPI from the interaction data:

  * mastery_improvement_rate  : mean change in mastery after a *clean* success
                                (correct, no hints) — the signal that the tutor
                                is moving learners forward, not just labelling.
  * hints_per_correct          : mean number of hints used before a correct
                                answer — lower is better (less dependence).
  * clean_success_rate         : share of items solved correctly with 0 hints.

These are computed from data/processed/student_interactions.csv using the same
mastery-update rule the app uses, so the number is reproducible. On synthetic
interaction data these are proxies; on real students they would be validated
longitudinally.

Run:
    python -m src.educational_kpi
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

try:
    from .pedagogical_engine import update_mastery
except ImportError:  # allow running as a script
    from pedagogical_engine import update_mastery

ROOT = Path(__file__).resolve().parents[1]
INTERACTIONS = ROOT / "data" / "processed" / "student_interactions.csv"
OUT = ROOT / "models" / "educational_kpi.json"


def compute_kpi(path: Path = INTERACTIONS) -> Dict:
    df = pd.read_csv(path)

    correct = df["is_correct"].astype(bool)
    hints = df["hint_count"].fillna(0).astype(int)
    attempts = df["attempt_count"].fillna(1).astype(int)
    prev = df["previous_mastery"].astype(float)

    # Apply the app's mastery-update rule to estimate post-interaction mastery,
    # then measure the improvement specifically on clean successes.
    deltas = []
    for c, h, a, p in zip(correct, hints, attempts, prev):
        new = update_mastery(p, bool(c), int(h), int(a))
        deltas.append(new - p)
    df = df.assign(_delta=deltas)

    clean_success = correct & (hints == 0)
    mastery_improvement_rate = float(df.loc[clean_success, "_delta"].mean()) if clean_success.any() else 0.0
    hints_per_correct = float(hints[correct].mean()) if correct.any() else 0.0
    clean_success_rate = float(clean_success.mean())

    return {
        "n_interactions": int(len(df)),
        "mastery_improvement_rate_clean_success": round(mastery_improvement_rate, 4),
        "hints_per_correct": round(hints_per_correct, 3),
        "clean_success_rate": round(clean_success_rate, 3),
        "note": (
            "Proxy KPI on synthetic interaction data. Targets (pre-registered): "
            "positive mastery_improvement_rate on clean successes, and "
            "hints_per_correct kept low. Validate longitudinally on real students."
        ),
    }


if __name__ == "__main__":
    kpi = compute_kpi()
    OUT.write_text(json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(kpi, ensure_ascii=False, indent=2))
