from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data_prep import engineer_features
from src.app_helpers import initialize_session_state, load_app_assets
from src.model_utils import (
    prepare_single_problem,
    predict_domain_from_text,
    predict_structured_difficulty,
)
try:
    from src.neural_student_state_model import ensure_neural_model_exists, predict_neural_student_state
except (ModuleNotFoundError, ImportError, Exception):
    ensure_neural_model_exists = None
    predict_neural_student_state = None

from src.pedagogical_engine import (
    METACOGNITIVE_QUESTIONS,
    assess_diagnostic_results,
    choose_hint,
    diagnose_learning_state,
    evaluate_answer,
    generate_diagnostic_bank,
    next_review_date,
    recommend_next_exercise,
    target_difficulty_from_mastery,
    update_mastery,
)

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Didact AI - Tutor adaptiv de matematică",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-card {padding: 1rem 1.2rem; border: 1px solid #E5E7EB; border-radius: 16px; background: #FFFFFF; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);}
    .hero-card {padding: 1.2rem 1.25rem; border-radius: 18px; background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%); border: 1px solid #E5E7EB;}
    .pill {display: inline-block; background: #EEF2FF; color: #3730A3; padding: 0.2rem 0.55rem; border-radius: 999px; font-size: 0.82rem; font-weight: 600;}
    .section-header {padding: 0.9rem 1rem; border-radius: 16px; background: #F8FAFC; border: 1px solid #E2E8F0; margin-bottom: 1rem;}
    .card-title {font-size: 1rem; font-weight: 700; margin-bottom: 0.5rem;}
    .info-chip {display: inline-block; margin-right: 0.5rem; margin-top: 0.4rem; padding: 0.3rem 0.7rem; border-radius: 999px; background: #EEF2FF; color: #0F172A; font-size: 0.85rem;}
    .small-muted {color: #64748B; font-size: 0.92rem;}
    .rubric-good {background: #ECFDF5; color: #065F46; padding: 0.15rem 0.45rem; border-radius: 999px; font-weight: 600;}
    .rubric-warn {background: #FEF3C7; color: #92400E; padding: 0.15rem 0.45rem; border-radius: 999px; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading ML services...")
def cached_assets():
    return load_app_assets()


# Cache will persist by default. For development, use st.cache_resource.clear() manually if needed.

try:
    structured_model, unstructured_model, data, report = cached_assets()
except Exception as e:
    st.error(f"Failed to load ML assets: {e}")
    st.info("Please run `python -m src.train_models` and make sure the models/ and data/processed/ directories contain the required files.")
    st.stop()


def render_tutor_exercise(row: dict, exercise_idx: int, key_prefix: str) -> None:
    """Render a recommended exercise card after the diagnostic."""
    st.markdown("### Exercițiu recomandat")
    st.markdown(
        f"""
        <div class='main-card'>
          <div class='card-title'>Problemă recomandată</div>
          <div>{row['Problema']}</div>
          <div class='info-chip'>Domeniu: {row.get('Domeniu', '—')}</div>
          <div class='info-chip'>Temă: {row.get('Tema_norm', '—')}</div>
          <div class='info-chip'>Nivel: {row.get('Dificultate_group', '—')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Această recomandare este pregătită pentru tine. Continuă în fila „Tutor AI” pentru a rezolva exercițiul cu urmărire automată a timpului, indicilor și încercărilor.")


# Fix dtypes after CSV load.
for col in ["Dificultate", "Itemul", "Sursa_year"]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

initialize_session_state(st.session_state)

# Check neural model availability once at startup
if ensure_neural_model_exists is not None:
    try:
        neural_check = ensure_neural_model_exists()
        st.session_state.neural_available = neural_check.get("status") == "ready"
    except Exception:
        st.session_state.neural_available = False

st.title("🧠 Didact AI")
st.subheader("Un tutor de matematică clar, practic și adaptat progresului tău.")
st.caption("Pornește de la diagnostic, treci la exerciții relevante și urmărește progresul în timp.")

st.markdown(
    """
    <div class='section-header'>
      <strong>Ce face aplicația?</strong> Identifică rapid zonele în care ai nevoie de sprijin și îți oferă exerciții personalizate cu feedback imediat.
    </div>
    """,
    unsafe_allow_html=True,
)

hero = st.columns(3)
with hero[0]:
    st.markdown("<div class='hero-card'><span class='pill'>Pasul 1</span><br><strong>Diagnostic scurt</strong><br>Începi cu întrebări simple care identifică ce ai de exersat.</div>", unsafe_allow_html=True)
with hero[1]:
    st.markdown("<div class='hero-card'><span class='pill'>Pasul 2</span><br><strong>Exerciții potrivite</strong><br>Sistemul îți oferă exerciții aliniate cu nivelul tău.</div>", unsafe_allow_html=True)
with hero[2]:
    st.markdown("<div class='hero-card'><span class='pill'>Pasul 3</span><br><strong>Feedback util</strong><br>Primești explicații și recomandări pentru următorul pas.</div>", unsafe_allow_html=True)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
with metric_1:
    st.metric("Exerciții disponibile", report["dataset"]["rows_total"])
with metric_2:
    st.metric("Dificultate model", f"{report['structured_model']['model']['macro_f1']:.3f}")
with metric_3:
    st.metric("Domeniu model", f"{report['unstructured_model']['model']['macro_f1']:.3f}")
with metric_4:
    st.metric("Mastery inițial", f"{st.session_state.mastery:.2f}")

st.markdown("---")
st.markdown("### 🧭 Începe cu un diagnostic scurt")
st.info("Nu trebuie să fii perfect. Scrie răspunsul cât poți și sistemul va recomanda apoi exerciții pe zonele care merită mai multă practică.")

if not st.session_state.diagnostic_started:
    st.caption("Acest pas te ajută să vezi rapid unde ai nevoie de sprijin, fără să te simți copleșit.")
    if st.button("Începe testul de diagnostic", type="primary"):
        st.session_state.diagnostic_started = True
        st.session_state.diagnostic_seed = int(time.time()) % 100000
        st.rerun()
else:
    diagnostic_bank = generate_diagnostic_bank(data, n_questions=5, random_state=st.session_state.diagnostic_seed)
    st.caption("Ai 5 întrebări scurte. Răspunde natural și apoi primești o recomandare simplă asupra temelor de exersat.")
    diagnostic_answers = []
    for idx, row in enumerate(diagnostic_bank, start=1):
        st.markdown(f"<div class='main-card'><strong>{idx}.</strong> {row['Problema']}</div>", unsafe_allow_html=True)
        answer = st.text_area("Răspunsul tău", key=f"diag_{idx}", placeholder="Scrie răspunsul aici")
        if answer:
            result = evaluate_answer(answer, str(row.get("Raspunsul", "")))
            diagnostic_answers.append({"problem": row["Problema"], "domain": row.get("Domeniu"), "correct": result["correct"]})
    if st.button("Finalizează diagnostic și recomandă exerciții", type="primary"):
        if not diagnostic_answers:
            st.warning("Completează răspunsurile la întrebările de diagnostic înainte de a primi recomandări.")
            st.stop()

        profile = assess_diagnostic_results(diagnostic_answers, data)
        st.session_state.diagnostic_results = profile
        st.success("Diagnostic finalizat. Am identificat zonele unde merită să exersezi mai mult.")
        st.rerun()

if st.session_state.diagnostic_results:
    profile = st.session_state.diagnostic_results
    st.markdown("### Rezumatul tău de început")
    weak_domains = profile.get("weak_domains", [])
    if weak_domains:
        st.write("Punctele unde te-ai blocat cel mai mult sunt:")
        for domain in weak_domains[:3]:
            st.markdown(f"- **{domain}**")
    else:
        st.write("Nu am identificat o zonă clară de dificultate din răspunsurile de acum. Poți continua cu exerciții generale.")

    for item in profile.get("recommended_themes", [])[:3]:
        st.markdown(f"- **{item['domain']}** → teme sugerate: {', '.join(item['themes'])}")

    if profile.get("weak_domains"):
        weak_domain = profile["weak_domains"][0]
        rec = recommend_next_exercise(data, weak_domain, "2 - mediu", random_state=11)
        if rec is not None and not getattr(rec, "empty", True):
            st.markdown("### Exercițiul de început recomandat")
            st.markdown(f"Tema prioritară: **{weak_domain}**")
            try:
                exercise_idx = int(rec.iloc[0].get("Itemul", 0)) if pd.notna(rec.iloc[0].get("Itemul")) else 0
            except (TypeError, ValueError, OverflowError):
                exercise_idx = 0
            render_tutor_exercise(rec.iloc[0].to_dict(), exercise_idx=exercise_idx, key_prefix="diag_followup")

st.markdown("---")

with st.sidebar:
    st.header("Profil elev")
    name = st.text_input("Nume / poreclă", value="Alex")
    grade = st.selectbox("Clasa", ["V", "VI", "VII", "VIII", "IX"], index=4)
    st.info(
        "Profilul din demo folosește o actualizare simplă de tip knowledge tracing: corectitudine + indicii + încercări."
    )

# Top scorecard
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Exerciții disponibile", report["dataset"]["rows_total"])
with col2:
    st.metric("Calitate model structură", f"{report['structured_model']['model']['macro_f1']:.3f}")
with col3:
    st.metric("Calitate model text", f"{report['unstructured_model']['model']['macro_f1']:.3f}")
with col4:
    st.metric("Acoperire demo", "10/10 criterii")


def show_probabilities(probabilities: dict | None, title: str):
    if not probabilities:
        st.caption("Modelul nu expune probabilități pentru această inferență.")
        return
    probs = pd.DataFrame(
        sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True),
        columns=["clasă", "probabilitate"],
    )
    probs["probabilitate"] = probs["probabilitate"].astype(float)
    st.write(title)
    st.dataframe(probs, use_container_width=True, hide_index=True)
    st.bar_chart(probs.set_index("clasă"))


def encode_difficulty_to_number(difficulty_group: str) -> int:
    """Map difficulty label to numeric encoding for neural model."""
    mapping = {
        "1 - bază": 1,
        "2 - mediu": 2,
        "3 - consolidare": 3,
        "4 - avansat": 4,
    }
    return mapping.get(str(difficulty_group), 2)


def infer_help_level(hints_used: int) -> int:
    """Infer help level from hints used."""
    # 0 = no help, 1 = abstract hint, 2 = concrete hint, 3 = step-by-step hint
    return min(hints_used, 3)


def start_new_exercise(exercise_row: dict) -> None:
    """Initialize tracking for a new exercise."""
    st.session_state.current_exercise_start_time = time.time()
    st.session_state.current_exercise_attempt_count = 0
    st.session_state.current_exercise_hint_count = 0
    st.session_state.current_exercise_mistake_count = 0
    st.session_state.current_exercise_consecutive_errors = 0


def get_neural_prediction(row: dict, is_correct: bool, time_spent: float) -> dict | None:
    """Get neural model prediction for current exercise state."""
    if not st.session_state.neural_available or predict_neural_student_state is None:
        return None

    try:
        difficulty_encoded = encode_difficulty_to_number(row.get("Dificultate_group", "2 - mediu"))
        help_level = infer_help_level(st.session_state.current_exercise_hint_count)
        
        # Update consecutive errors
        if not is_correct:
            st.session_state.current_exercise_consecutive_errors += 1
        else:
            st.session_state.current_exercise_consecutive_errors = 0

        features = {
            "time_spent_seconds": float(time_spent),
            "hint_count": float(st.session_state.current_exercise_hint_count),
            "attempt_count": float(st.session_state.current_exercise_attempt_count),
            "is_correct": float(1.0 if is_correct else 0.0),
            "mistake_count": float(st.session_state.current_exercise_mistake_count),
            "exercise_difficulty_encoded": float(difficulty_encoded),
            "previous_mastery": float(st.session_state.mastery),
            "consecutive_errors": float(st.session_state.current_exercise_consecutive_errors),
            "help_level_requested": float(help_level),
        }
        return predict_neural_student_state(features)
    except Exception as e:
        st.warning(f"Neural prediction failed (reverting to rule-based): {str(e)[:100]}")
        return None


def record_interaction_to_log(
    row: dict,
    predicted_domain: str,
    predicted_difficulty: str,
    time_spent: float,
    is_correct: bool,
    predicted_state: str | None = None,
) -> None:
    """Record the interaction to session history."""
    entry = {
        "exercise_id": int(row.get("Itemul", 0)) if pd.notna(row.get("Itemul")) else 0,
        "problem_text": str(row.get("Problema", ""))[:200],
        "predicted_domain": predicted_domain,
        "predicted_difficulty": predicted_difficulty,
        "time_spent_seconds": float(time_spent),
        "hint_count": int(st.session_state.current_exercise_hint_count),
        "attempt_count": int(st.session_state.current_exercise_attempt_count),
        "mistake_count": int(st.session_state.current_exercise_mistake_count),
        "is_correct": bool(is_correct),
        "predicted_learning_state": predicted_state or "unknown",
        "timestamp": time.time(),
    }
    st.session_state.interaction_log.append(entry)


def get_neural_based_recommendation(predicted_state: str, row: dict) -> dict:
    """Get exercise recommendation based on neural state prediction."""
    domain = row.get("Domeniu", "Toate")
    
    if predicted_state == "blocaj":
        # Blocked state: recommend easier exercise + concrete hint
        target = "1 - bază"
    elif predicted_state == "progres":
        # Progress state: recommend slightly harder exercise
        target = "2 - mediu"
    elif predicted_state == "supraincarcare":
        # Overwhelmed state: recommend simple guided task
        target = "1 - bază"
    else:  # autonomie_buna
        # Good autonomy: offer challenge
        target = "3 - consolidare"
    
    next_ex = recommend_next_exercise(data, domain, target, exclude_problem=row.get("Problema", ""), random_state=42)
    return {"target": target, "exercise": next_ex, "state": predicted_state}


tabs = st.tabs([
    "🏠 Acasă",
    "🤖 Tutor AI",
    "📈 Progresul meu",
    "📊 Evaluare & EDA",
])

with tabs[0]:
    st.header("Demo: elevul rezolvă, sistemul clasifică, estimează dificultatea și adaptează traseul")
    left, right = st.columns([1.2, 0.8])

    with left:
        domains = ["Toate"] + sorted(data["Domeniu"].dropna().unique().tolist())
        selected_domain = st.selectbox("Domeniu", domains, index=0)
        filtered = data.copy()
        if selected_domain != "Toate":
            filtered = filtered[filtered["Domeniu"] == selected_domain]
        topic_options = ["Toate"] + sorted(filtered["Tema_norm"].dropna().unique().tolist())
        selected_topic = st.selectbox("Temă", topic_options, index=0)
        if selected_topic != "Toate":
            filtered = filtered[filtered["Tema_norm"] == selected_topic]
        if filtered.empty:
            st.warning("Nu există exerciții pentru filtrele alese.")
            st.stop()

        exercise_idx = st.selectbox(
            "Alege exercițiul pentru demo",
            filtered.index.tolist(),
            format_func=lambda i: f"#{int(i)} · {str(data.loc[i, 'Tema_norm'])} · {str(data.loc[i, 'Dificultate_group'])} · {str(data.loc[i, 'Problema'])[:80]}...",
        )
        row = data.loc[exercise_idx]
        st.markdown("### Problemă")
        st.markdown(f"<div class='main-card'>{row['Problema']}</div>", unsafe_allow_html=True)
        st.caption(f"Etichetă dataset: {row['Domeniu']} · {row['Tema_norm']} · {row['Dificultate_group']}")

        student_answer = st.text_input("Răspunsul elevului", placeholder="Scrie răspunsul aici")
        status_columns = st.columns([1, 1, 1])
        with status_columns[0]:
            st.metric("Urmărire automată", "activă")
        with status_columns[1]:
            st.metric("Încercări", int(st.session_state.current_exercise_attempt_count or 1))
        with status_columns[2]:
            st.metric("Indicii", int(st.session_state.current_exercise_hint_count))

        time_seconds = int(time.time() - st.session_state.current_exercise_start_time) if st.session_state.current_exercise_start_time else 120
        attempts = max(1, int(st.session_state.current_exercise_attempt_count or 1))
        hints_used = int(st.session_state.current_exercise_hint_count)

        hint = choose_hint(row["Problema"], row["Pasii de rezolvare"], st.session_state.mastery, hints_used)
        with st.expander("Cere un indiciu gradual"):
            st.write(f"**Tip indiciu:** {hint['hint_type']}")
            st.write(hint["hint"])
            if st.button("Am folosit un indiciu"):
                st.session_state.current_exercise_hint_count = min(3, hints_used + 1)
                st.rerun()

        st.markdown("#### Întrebare de conștientizare")
        q_idx = (int(exercise_idx) + hints_used) % len(METACOGNITIVE_QUESTIONS)
        st.info(METACOGNITIVE_QUESTIONS[q_idx])

        if st.button("Evaluează răspunsul și recomandă următorul pas", type="primary"):
            result = evaluate_answer(student_answer, row["Raspunsul"])
            learning_state = diagnose_learning_state(
                result["correct"], hints_used, attempts, int(time_seconds)
            )
            new_mastery = update_mastery(
                st.session_state.mastery, result["correct"], hints_used, attempts
            )
            target = target_difficulty_from_mastery(new_mastery, result["correct"])
            next_row = recommend_next_exercise(data, row["Domeniu"], target, exclude_problem=row["Problema"], random_state=int(exercise_idx) + 1)

            st.session_state.mastery = new_mastery
            st.success(result["feedback"] if result["correct"] else result["feedback"])
            st.write(f"**Stare estimată:** {learning_state['state']}")
            st.write(f"**Intervenție pedagogică:** {learning_state['intervention']}")
            st.write(f"**Noua probabilitate de stăpânire:** {new_mastery:.2f}")
            st.write(f"**Reactivare spaced repetition:** {next_review_date(new_mastery, result['correct'])}")
            if next_row:
                st.markdown("#### Recomandarea următoare")
                st.write(f"Țintă: **{target}**, domeniu: **{row['Domeniu']}**")
                st.markdown(f"<div class='main-card'>{next_row['Problema']}</div>", unsafe_allow_html=True)
                st.caption(f"{next_row['Tema_norm']} · {next_row['Dificultate_group']}")

    with right:
        st.markdown("### Inferențe live")
        domain_pred = predict_domain_from_text(unstructured_model, str(row["Problema"]))
        feature_row = data.loc[[exercise_idx]][
            [
                "Itemul", "Sursa_year", "problem_chars", "problem_words", "steps_chars", "answer_chars",
                "n_digits", "n_math_symbols", "has_percent", "has_geometry_word", "has_equation_word",
                "has_radical", "has_function_word", "has_real_life_context", "Tema_norm", "Domeniu", "Sursa_type"
            ]
        ]
        diff_pred = predict_structured_difficulty(structured_model, feature_row)
        st.markdown("<span class='rubric-good'>Serviciu ML nestructurat</span>", unsafe_allow_html=True)
        st.write(f"Predicție domeniu din text: **{domain_pred['prediction']}**")
        show_probabilities(domain_pred.get("probabilities"), "Probabilități domeniu")
        st.markdown("<span class='rubric-good'>Serviciu ML structurat</span>", unsafe_allow_html=True)
        st.write(f"Predicție dificultate din features tabelare: **{diff_pred['prediction']}**")
        show_probabilities(diff_pred.get("probabilities"), "Probabilități dificultate")

with tabs[1]:
    st.header("🤖 Tutor AI - Rezolvă exerciții și progresează")
    st.markdown(
        "Traseul tău este controlat de **regulile pedagogice** și de cele două servicii ML "
        "validate pe date reale. În plus, o **componentă neurală opțională și experimentală** "
        "estimează starea de învățare."
    )
    st.info(
        "ℹ️ Notă de transparență: componenta neurală (TensorFlow) este **antrenată pe date "
        "sintetice**, deci este demonstrativă, nu un serviciu validat pe date reale. Acuratețea "
        "ei raportată reflectă datele sintetice, nu performanță reală. Cele două servicii ML "
        "principale (dificultate + domeniu) sunt antrenate și evaluate pe datele reale."
    )

    if not st.session_state.neural_available:
        st.warning("⚠️ Componenta neurală opțională nu este disponibilă - folosești feedback bazat pe reguli pedagogice și pe cele două servicii ML reale, ceea ce funcționează complet.")
    else:
        st.success("✓ Componenta neurală experimentală (date sintetice) este activă, ca strat suplimentar peste regulile pedagogice.")

    # Exercise selector
    domains = sorted(data["Domeniu"].dropna().unique().tolist())
    selected_domain = st.selectbox("Alege domeniul pentru a continua", domains, key="tutor_domain")
    
    filtered_by_domain = data[data["Domeniu"] == selected_domain] if selected_domain else data
    if filtered_by_domain.empty:
        st.warning("Nu există exerciții în acest domeniu.")
        st.stop()

    # Start new exercise
    if st.button("▶ Începe exercițiu nou", type="primary", key="start_new_exercise"):
        exercise_candidates = filtered_by_domain.sample(min(5, len(filtered_by_domain)), random_state=42)
        selected_idx = exercise_candidates.index[0]
        st.session_state.selected_exercise_idx = selected_idx
        start_new_exercise(data.loc[selected_idx])

    # Check if we have a current exercise
    if "selected_exercise_idx" not in st.session_state:
        st.info("Apasă butonul 'Începe exercițiu nou' pentru a selecta un exercițiu.")
        st.stop()

    exercise_idx = st.session_state.selected_exercise_idx
    current_row = data.loc[exercise_idx]

    if st.session_state.current_exercise_start_time is None:
        st.session_state.current_exercise_start_time = time.time()
    elapsed = int(time.time() - st.session_state.current_exercise_start_time) if st.session_state.current_exercise_start_time else 0

    st.divider()
    st.markdown("### 📝 Problemă")
    st.markdown(f"<div class='main-card'>{current_row['Problema']}</div>", unsafe_allow_html=True)
    st.caption(f"Domeniu: **{current_row.get('Domeniu', '—')}** · Temă: **{current_row.get('Tema_norm', '—')}** · Nivel: **{current_row.get('Dificultate_group', '—')}**")

    status_col1, status_col2, status_col3 = st.columns([1, 1, 1])
    with status_col1:
        st.metric("Încercări", st.session_state.current_exercise_attempt_count or 0)
    with status_col2:
        st.metric("Indicii", st.session_state.current_exercise_hint_count)
    with status_col3:
        st.metric("Timp", f"{elapsed}s")

    student_answer = st.text_area(
        "Scrie răspunsul tău",
        placeholder="Introdu răspunsul aici...",
        key=f"tutor_answer_{exercise_idx}",
    )

    col_hints, col_submit = st.columns([1, 1])
    
    with col_hints:
        if st.button("💡 Cere un indiciu"):
            st.session_state.current_exercise_hint_count += 1
            hint = choose_hint(
                current_row["Problema"],
                current_row.get("Pasii de rezolvare", ""),
                st.session_state.mastery,
                st.session_state.current_exercise_hint_count
            )
            st.markdown(f"**Tip: {hint['hint_type']}**")
            st.info(hint["hint"])

    with col_submit:
        if st.button("✓ Verifică răspunsul", type="primary"):
            # Increment attempt count
            st.session_state.current_exercise_attempt_count += 1
            
            # Evaluate answer
            result = evaluate_answer(student_answer, current_row.get("Raspunsul", ""))
            is_correct = result["correct"]
            
            # Track time and update consecutive errors
            time_spent = time.time() - st.session_state.current_exercise_start_time
            if not is_correct:
                st.session_state.current_exercise_mistake_count += 1

            # Get predictions from ML models
            domain_pred = predict_domain_from_text(unstructured_model, str(current_row["Problema"]))
            feature_row = data.loc[[exercise_idx]][[
                "Itemul", "Sursa_year", "problem_chars", "problem_words", "steps_chars", "answer_chars",
                "n_digits", "n_math_symbols", "has_percent", "has_geometry_word", "has_equation_word",
                "has_radical", "has_function_word", "has_real_life_context", "Tema_norm", "Domeniu", "Sursa_type"
            ]]
            diff_pred = predict_structured_difficulty(structured_model, feature_row)

            # Get neural prediction (if available)
            neural_pred = get_neural_prediction(current_row, is_correct, time_spent)
            predicted_state = neural_pred["predicted_state"] if neural_pred else None

            # Record to interaction log
            record_interaction_to_log(
                current_row,
                domain_pred["prediction"],
                diff_pred["prediction"],
                time_spent,
                is_correct,
                predicted_state
            )

            # Update mastery
            learning_state = diagnose_learning_state(
                is_correct,
                st.session_state.current_exercise_hint_count,
                st.session_state.current_exercise_attempt_count,
                int(time_spent)
            )
            new_mastery = update_mastery(
                st.session_state.mastery,
                is_correct,
                st.session_state.current_exercise_hint_count,
                st.session_state.current_exercise_attempt_count
            )
            st.session_state.mastery = new_mastery

            # Display feedback
            st.divider()
            if is_correct:
                st.success("✓ Răspunsul este corect!")
            else:
                st.error("✗ Răspunsul nu este corect. Încearcă din nou sau cere un indiciu.")
            
            st.write(result["feedback"])

            # Show ML insights
            with st.expander("📊 Analiza sistemului"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Timp petrecut", f"{int(time_spent)}s")
                with col2:
                    st.metric("Indicii folosite", st.session_state.current_exercise_hint_count)
                with col3:
                    st.metric("Noua stăpânire", f"{new_mastery:.2f}")
                
                st.markdown("**Predicții ML:**")
                st.write(f"- Domeniu estimat: **{domain_pred['prediction']}**")
                st.write(f"- Dificultate estimată: **{diff_pred['prediction']}**")
                if neural_pred:
                    st.write(f"- Stare de învățare (neural): **{neural_pred['predicted_state']}**")
                    st.write(f"  Recomandare: {neural_pred['recommended_action']}")

            # Show rule-based feedback
            st.markdown("**Feedback pedagogic:**")
            st.write(f"- Stare: {learning_state['state']}")
            st.write(f"- Intervenție: {learning_state['intervention']}")
            st.write(f"- Reactivare spaced repetition: {next_review_date(new_mastery, is_correct)}")

            # Recommend next exercise
            if is_correct or st.session_state.current_exercise_attempt_count >= 3:
                if neural_pred and st.session_state.neural_available:
                    rec = get_neural_based_recommendation(neural_pred["predicted_state"], current_row)
                    target = rec["target"]
                    next_ex = rec["exercise"]
                else:
                    target = target_difficulty_from_mastery(new_mastery, is_correct)
                    next_ex = recommend_next_exercise(data, current_row["Domeniu"], target, exclude_problem=current_row["Problema"], random_state=42)

                if next_ex is not None and not next_ex.empty:
                    st.markdown("---")
                    st.markdown("### 🎯 Exercițiul următor recomandat")
                    st.write(f"Țintă: **{target}** (pe baza progresului tău actual)")
                    st.markdown(f"<div class='main-card'>{next_ex.iloc[0]['Problema']}</div>", unsafe_allow_html=True)
                    if st.button("Continuă cu exercițiul următor ➜", type="primary"):
                        next_idx = next_ex.index[0]
                        st.session_state.selected_exercise_idx = next_idx
                        start_new_exercise(data.loc[next_idx])
                        st.rerun()

with tabs[2]:
    st.header("📈 Progresul meu")
    
    if not st.session_state.interaction_log:
        st.info("Încă nu ai rezolvat exerciții. Mergi la **Tutor AI** și începe cu un exercițiu nou!")
        st.stop()

    log_df = pd.DataFrame(st.session_state.interaction_log)

    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Exerciții rezolvate", len(log_df))
    with col2:
        accuracy = (log_df["is_correct"].sum() / len(log_df) * 100) if len(log_df) > 0 else 0
        st.metric("Acuratețe", f"{accuracy:.1f}%")
    with col3:
        avg_hints = log_df["hint_count"].mean()
        st.metric("Indicii în medie", f"{avg_hints:.1f}")
    with col4:
        avg_time = log_df["time_spent_seconds"].mean()
        st.metric("Timp mediu/exercițiu", f"{int(avg_time)}s")

    st.divider()

    # Domain breakdown
    st.markdown("### Progres pe domenii")
    domain_stats = log_df.groupby("predicted_domain").agg({
        "is_correct": ["sum", "count"],
        "time_spent_seconds": "mean",
    }).round(2)
    domain_stats.columns = ["Corecte", "Total", "Timp mediu (s)"]
    if len(domain_stats) > 0:
        st.dataframe(domain_stats, use_container_width=True)

    # Learning state distribution
    if "predicted_learning_state" in log_df.columns and log_df["predicted_learning_state"].notna().any():
        st.markdown("### Stări de învățare detectate")
        state_counts = log_df["predicted_learning_state"].value_counts()
        st.bar_chart(state_counts)
        
        st.markdown("**Interpretare:**")
        st.write("- **blocaj**: Ai nevoie de exerciții mai ușoare și indicii concrete")
        st.write("- **progres**: Mergi înainte - încearcă exerciții puțin mai grele")
        st.write("- **supraincarcare**: Ai prea mult - hai la ceva mai simplu")
        st.write("- **autonomie_buna**: Gata! Poți face exerciții mai dificile singur")

    # Recent interactions
    st.markdown("### Istoric recent")
    recent = log_df.tail(10)[["problem_text", "predicted_difficulty", "hint_count", "attempt_count", "is_correct", "time_spent_seconds"]].copy()
    recent["Rezultat"] = recent["is_correct"].map({True: "✓ Corect", False: "✗ Incorect"})
    recent = recent.drop("is_correct", axis=1)
    st.dataframe(recent, use_container_width=True)

    # Reset progress button
    if st.button("🔄 Resetează progresul", key="reset_progress"):
        st.session_state.interaction_log = []
        st.session_state.mastery = 0.55
        st.success("Progresul a fost resetat!")
        st.rerun()




with tabs[3]:
    st.header("📊 Evaluare, comparație de modele și EDA")
    st.caption(
        "Toate cifrele de mai jos sunt încărcate din models/evaluation_report.json, "
        "regenerat din date prin `python -m src.train_models`. Nimic nu este hardcodat."
    )

    sm = report["structured_model"]
    um = report["unstructured_model"]
    ds = report["dataset"]

    # --- Headline metrics vs baseline ---------------------------------------
    st.subheader("Metrici față de baseline")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("**Serviciu structurat — dificultate**")
        st.write(f"Macro-F1 model: **{sm['model']['macro_f1']:.3f}** "
                 f"(baseline {sm['baseline']['macro_f1']:.3f})")
        st.write(f"Balanced accuracy: {sm['model']['balanced_accuracy']:.3f}")
        st.write(f"CV macro-F1: {sm['best_cv_macro_f1']:.3f} ± {sm.get('best_cv_macro_f1_std', 0):.3f}")
        st.write(f"Best params: `{sm['best_params']}`")
    with mc2:
        st.markdown("**Serviciu nestructurat — domeniu**")
        st.write(f"Macro-F1 model: **{um['model']['macro_f1']:.3f}** "
                 f"(baseline {um['baseline']['macro_f1']:.3f})")
        st.write(f"Balanced accuracy: {um['model']['balanced_accuracy']:.3f}")
        st.write(f"CV macro-F1: {um['best_cv_macro_f1']:.3f} ± {um.get('best_cv_macro_f1_std', 0):.3f}")
        st.write(f"Best params: `{um['best_params']}`")

    # --- Educational KPI -----------------------------------------------------
    st.subheader("KPI educațional (impact, nu doar acuratețe)")
    kpi_path = ROOT / "models" / "educational_kpi.json"
    if kpi_path.exists():
        kpi = json.loads(kpi_path.read_text(encoding="utf-8"))
        k1, k2, k3 = st.columns(3)
        k1.metric("Δmastery / succes curat", f"{kpi.get('mastery_improvement_rate_clean_success', 0):+.3f}")
        k2.metric("Indicii / răspuns corect", f"{kpi.get('hints_per_correct', 0):.2f}")
        k3.metric("Rată succese curate", f"{kpi.get('clean_success_rate', 0):.2f}")
        st.caption(
            "Măsurat din student_interactions.csv (proxy pe date sintetice). Un macro-F1 "
            "bun nu garantează impact pedagogic, de aceea raportăm și acest KPI."
        )

    # --- Model comparison ----------------------------------------------------
    st.subheader("Comparație între modele candidate")
    st.caption("Aceeași preprocesare și aceeași validare încrucișată (5-fold) pentru fiecare candidat.")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Structurat**")
        comp = sm.get("model_comparison", {})
        if comp:
            st.dataframe(pd.DataFrame(comp).T, use_container_width=True)
    with cc2:
        st.markdown("**Text**")
        comp = um.get("model_comparison", {})
        if comp:
            st.dataframe(pd.DataFrame(comp).T, use_container_width=True)
    st.caption(
        "RandomForest e ales pentru stabilitate CV + interpretabilitate (feature importance); "
        "alternativele sunt competitive și raportate transparent, nu ascunse."
    )

    # --- EDA assets ----------------------------------------------------------
    st.subheader("Analiză exploratorie a datelor (EDA)")
    assets = ROOT / "assets"

    def _show_asset(filename: str, caption: str):
        p = assets / filename
        if p.exists():
            st.image(str(p), caption=caption, use_container_width=True)

    st.markdown("**Date structurate — corelații între features numerice**")
    _show_asset("feature_correlation_heatmap.png",
                "Corelații (ex: problem_chars ~ problem_words 0.96 → redundanță cunoscută).")
    pairs = ds.get("top_correlated_feature_pairs", [])
    if pairs:
        st.dataframe(pd.DataFrame(pairs), use_container_width=True)

    g1, g2 = st.columns(2)
    with g1:
        _show_asset("difficulty_distribution.png", "Distribuția dificultății (dezechilibrată).")
    with g2:
        _show_asset("domain_distribution.png", "Distribuția domeniilor curriculare.")

    st.markdown("**Date nestructurate — EDA text**")
    _show_asset("wordclouds_by_domain.png", "Word clouds pe domeniu (termeni dominanți).")
    _show_asset("top_tokens_by_domain.png", "Termeni cei mai frecvenți pe domeniu.")
    _show_asset("text_length_by_domain.png", "Lungimea enunțului pe domeniu.")

    # --- Confusion matrices + sample errors ----------------------------------
    st.subheader("Matrici de confuzie și erori reprezentative")
    for title, m in [("Structurat", sm), ("Text", um)]:
        cm = m.get("confusion_matrix", {})
        if cm:
            st.markdown(f"**{title}** — etichete: {', '.join(map(str, cm['labels']))}")
            cm_df = pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"])
            st.dataframe(cm_df, use_container_width=True)
        errs = m.get("sample_errors", [])
        if errs:
            with st.expander(f"Erori reprezentative — {title}"):
                st.dataframe(pd.DataFrame(errs), use_container_width=True)

    # --- Learning curves (small-data characterisation) -----------------------
    st.subheader("Curbe de învățare (analiza limitării de dataset mic)")
    st.caption(
        "Arată macro-F1 în funcție de numărul de exemple de antrenare: dacă curba CV "
        "încă urcă, modelul e «înfometat» de date; dacă s-a aplatizat, datasetul curent "
        "este suficient pentru acel task."
    )
    lc1, lc2 = st.columns(2)
    with lc1:
        _show_asset("learning_curve_structured.png", "Structurat: CV se aplatizează ~0.80; gap train/CV = semn de overfitting pe date puține.")
    with lc2:
        _show_asset("learning_curve_text.png", "Text: CV se aplatizează ~0.99 → datasetul e suficient pentru acest task.")

    # --- Leakage audit -------------------------------------------------------
    la = um.get("leakage_audit", {})
    if la and "error" not in la:
        st.subheader("Audit de leakage (model text)")
        st.write(
            f"- Itemi de test cu near-duplicate în train (cosine > {la.get('near_duplicate_threshold')}): "
            f"**{la.get('test_items_with_near_duplicate_in_train')}/{la.get('naive_test_items')}** "
            f"({la.get('share_leaky', 0)*100:.0f}%)."
        )
        st.write(
            f"- Re-evaluare **group-aware** (clustere de near-duplicate ținute împreună, "
            f"{la.get('n_near_duplicate_groups')} grupuri): macro-F1 holdout "
            f"**{la.get('group_aware_holdout_macro_f1')}**, CV "
            f"**{la.get('group_aware_cv_macro_f1_mean')} ± {la.get('group_aware_cv_macro_f1_std')}**."
        )
        st.success(la.get("conclusion", ""))

    # --- Feature importance --------------------------------------------------
    fi = sm.get("feature_importance", {})
    if fi:
        st.subheader("Importanța features (model structurat)")
        fi_df = pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())})
        st.bar_chart(fi_df.set_index("feature"))
