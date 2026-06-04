"""Pedagogical decision layer for the Streamlit MVP.

This is intentionally rule-based around the ML services. The concept document
recommends a hybrid model: ML predictions + controlled pedagogical rules, so the
tutor remains safe, explainable, and not a black-box answer machine.
"""
from __future__ import annotations

import math
import re
import unicodedata
from datetime import date, timedelta
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

DIFF_ORDER = ["1 - bază", "2 - mediu", "3 - consolidare", "4 - avansat"]
METACOGNITIVE_QUESTIONS = [
    "Ce știi deja din problemă și ce trebuie aflat?",
    "De ce ai ales această formulă sau metodă?",
    "Care este primul pas obligatoriu pe care nu trebuie să îl sărim?",
    "Cum poți verifica dacă răspunsul tău are sens?",
    "Poți explica aceeași idee într-o problemă din viața reală?",
]


def normalize_answer(text: str) -> str:
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", "", text)
    return text


def extract_numbers(text: str) -> List[str]:
    return re.findall(r"-?\d+(?:[\.,/]\d+)?", normalize_answer(text))


def evaluate_answer(student_answer: str, expected_answer: str) -> Dict:
    """Lightweight answer check for demo purposes.

    It is deliberately transparent: exact normalized match, containment, or overlap
    of numeric tokens. It does not pretend to be a full symbolic mathematics grader.
    """
    s = normalize_answer(student_answer)
    e = normalize_answer(expected_answer)
    student_nums = set(extract_numbers(student_answer))
    expected_nums = set(extract_numbers(expected_answer))

    if not s:
        status = "no_answer"
        correct = False
        feedback = "Nu ai introdus încă un răspuns. Începe cu ce se cere și cu datele cunoscute."
    elif s == e or (len(e) > 0 and (s in e or e in s)):
        status = "correct"
        correct = True
        feedback = "Răspunsul se potrivește cu cheia. Acum justifică metoda, nu doar rezultatul."
    elif expected_nums and expected_nums.issubset(student_nums):
        status = "probably_correct_format_differs"
        correct = True
        feedback = "Valorile numerice principale coincid. Verifică forma finală și unitățile."
    elif student_nums & expected_nums:
        status = "partial_numeric_overlap"
        correct = False
        feedback = "Ai unele valori corecte, dar rezultatul final nu este complet. Verifică semnele, operațiile și unitățile."
    else:
        status = "incorrect"
        correct = False
        feedback = "Răspunsul nu coincide. Nu îți dau soluția completă: verifică primul pas și alege următorul indiciu."

    return {
        "correct": correct,
        "status": status,
        "feedback": feedback,
        "student_numbers": sorted(student_nums),
        "expected_numbers": sorted(expected_nums),
    }


def split_solution_steps(solution: str) -> List[str]:
    text = "" if pd.isna(solution) else str(solution)
    # Split on semicolons, line breaks, or sentence boundaries while keeping useful chunks.
    parts = re.split(r"(?:\n+|;|\.\s+)", text)
    steps = [p.strip(" .;\n\t") for p in parts if len(p.strip()) >= 4]
    return steps[:8] if steps else [text[:240]]


def choose_hint(problem: str, solution_steps: str, mastery: float, hints_used: int) -> Dict:
    steps = split_solution_steps(solution_steps)
    level = min(max(int(hints_used), 0), 3)
    if mastery >= 0.75 and level == 0:
        hint_type = "abstract"
        hint = "Alege conceptul principal și scrie relația de pornire. Nu calcula încă."
    elif mastery >= 0.55 and level <= 1:
        hint_type = "guided"
        hint = f"Următorul pas util: {steps[0]}" if steps else "Scrie datele cunoscute și necunoscuta."
    else:
        hint_type = "concrete"
        idx = min(level, len(steps) - 1)
        hint = f"Indiciu concret pentru pasul {idx + 1}: {steps[idx]}"
    return {"hint_type": hint_type, "hint": hint, "shown_solution_steps": level >= 3}


def diagnose_learning_state(correct: bool, hints_used: int, attempts: int, time_seconds: int) -> Dict:
    if correct and hints_used == 0 and attempts <= 1:
        state = "progres stabil"
        intervention = "crește dificultatea sau treci la transfer într-un context nou"
    elif correct and hints_used > 0:
        state = "înțelegere în formare"
        intervention = "dă o problemă izomorfă fără indiciu pentru verificare"
    elif not correct and hints_used >= 2:
        state = "blocaj conceptual"
        intervention = "revino la fundament și oferă succes rapid ghidat"
    elif not correct and attempts >= 2:
        state = "eroare procedurală repetată"
        intervention = "exercițiu țintit pe tipul de pas greșit"
    elif time_seconds < 30 and not correct:
        state = "stil impulsiv"
        intervention = "întrebare de conștientizare înainte de calcul"
    else:
        state = "oscilare"
        intervention = "menține dificultatea și cere justificarea strategiei"
    return {"state": state, "intervention": intervention}


def update_mastery(mastery: float, correct: bool, hints_used: int, attempts: int) -> float:
    mastery = float(np.clip(mastery, 0.05, 0.95))
    evidence = 0.16 if correct else -0.14
    evidence -= 0.04 * max(hints_used, 0)
    evidence -= 0.03 * max(attempts - 1, 0)
    return float(np.clip(mastery + evidence, 0.05, 0.97))


def review_interval_days(mastery: float, correct: bool) -> int:
    if not correct or mastery < 0.45:
        return 2
    if mastery < 0.65:
        return 7
    if mastery < 0.85:
        return 30
    return 90


def target_difficulty_from_mastery(mastery: float, correct: bool) -> str:
    if not correct or mastery < 0.35:
        return "1 - bază"
    if mastery < 0.60:
        return "2 - mediu"
    if mastery < 0.80:
        return "3 - consolidare"
    return "4 - avansat"


def generate_diagnostic_bank(data: pd.DataFrame, n_questions: int = 5, random_state: int = 42) -> List[Dict]:
    """Build a varied diagnostic set from sensible, non-empty exercise rows."""
    candidates = data.copy()
    candidates = candidates[candidates["Domeniu"].notna() & (candidates["Domeniu"] != "Altele")]
    candidates = candidates[candidates["Problema"].astype(str).str.strip().ne("")]
    candidates = candidates[candidates["Raspunsul"].astype(str).str.strip().ne("")]
    candidates = candidates.drop_duplicates(subset=["Problema", "Domeniu", "Tema_norm"], keep="first")

    if len(candidates) == 0:
        return []

    bank = []
    for domain, group in candidates.groupby("Domeniu", sort=True):
        if len(group) == 0:
            continue
        seed = (random_state + sum(ord(ch) for ch in str(domain))) % 100000
        sample = group.sample(n=min(2, len(group)), random_state=seed)
        bank.extend(sample.to_dict("records"))
        if len(bank) >= n_questions:
            break

    return bank[:n_questions]


def assess_diagnostic_results(responses: List[Dict], data: pd.DataFrame) -> Dict:
    """Estimate weak domains from a short diagnostic quiz."""
    if not responses:
        return {"weak_domains": [], "domain_scores": {}, "recommended_themes": []}

    scores = {}
    for domain in sorted(data["Domeniu"].dropna().unique().tolist()):
        scores[domain] = {"correct": 0, "total": 0}

    for item in responses:
        domain = item.get("domain") or item.get("Domeniu") or "Alt"
        correct = bool(item.get("correct", False))
        scores.setdefault(domain, {"correct": 0, "total": 0})
        scores[domain]["correct"] += int(correct)
        scores[domain]["total"] += 1

    domain_scores = {}
    for domain, stats in scores.items():
        ratio = stats["correct"] / stats["total"] if stats["total"] else 0.0
        domain_scores[domain] = {
            "accuracy": round(ratio, 2),
            "correct": stats["correct"],
            "total": stats["total"],
            "priority": round(1.0 - ratio, 2),
        }

    weak_domains = [
        domain for domain, stats in sorted(domain_scores.items(), key=lambda kv: (kv[1]["priority"], kv[1]["accuracy"]), reverse=False)
        if stats["total"] >= 1 and stats["accuracy"] < 0.75
    ]

    recommended_themes = []
    for domain in weak_domains[:3]:
        theme_pool = data[(data["Domeniu"] == domain) & (data["Tema_norm"].notna())].copy()
        if len(theme_pool):
            theme_pool = theme_pool.sort_values(["Dificultate", "problem_words"], ascending=[True, True])
            recommended_themes.append({"domain": domain, "themes": theme_pool["Tema_norm"].dropna().unique().tolist()[:5]})

    return {
        "weak_domains": weak_domains,
        "domain_scores": domain_scores,
        "recommended_themes": recommended_themes,
    }


def recommend_next_exercise(
    data: pd.DataFrame,
    domain: str,
    target_difficulty: str,
    exclude_problem: str | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a single-row DataFrame with the recommended exercise, or an empty DataFrame.

    Returning a DataFrame keeps the return type consistent for callers in the Streamlit app.
    """
    candidates = data.copy()
    if domain:
        same = candidates[candidates["Domeniu"] == domain]
        if len(same) >= 3:
            candidates = same
    if exclude_problem:
        candidates = candidates[candidates["Problema"].astype(str) != str(exclude_problem)]
    exact = candidates[candidates["Dificultate_group"] == target_difficulty]
    if len(exact) > 0:
        candidates = exact
    if len(candidates) == 0:
        return candidates.head(0)
    return candidates.sample(1, random_state=random_state)


def next_review_date(mastery: float, correct: bool) -> str:
    days = review_interval_days(mastery, correct)
    return (date.today() + timedelta(days=days)).isoformat()
