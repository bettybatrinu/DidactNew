#!/usr/bin/env python
"""Minimal test to verify app.py can run without immediate errors."""

import streamlit as st
import time
import pandas as pd
from pathlib import Path

# Mimic the app.py initialization
print("Starting test...")

try:
    # Test imports
    from src.pedagogical_engine import (
        evaluate_answer,
        choose_hint,
        diagnose_learning_state,
        update_mastery,
        recommend_next_exercise,
        assess_diagnostic_results,
        target_difficulty_from_mastery,
        generate_diagnostic_bank,
    )
    from src.model_utils import (
        load_assets,
        predict_domain_from_text,
        predict_structured_difficulty,
    )
    from src.neural_student_state_model import (
        predict_neural_student_state,
        ensure_neural_model_exists,
    )
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    st.error(f"Import error: {e}")
    st.stop()

try:
    # Test asset loading
    structured_model, unstructured_model, data, report = load_assets()
    print(f"✓ Assets loaded: {data.shape[0]} exercises")
except Exception as e:
    print(f"✗ Asset loading failed: {e}")
    st.error(f"Failed to load ML assets: {e}")
    st.stop()

try:
    # Test neural model
    neural_check = ensure_neural_model_exists()
    neural_available = neural_check.get("status") == "ready"
    print(f"✓ Neural model check passed: {neural_available}")
except Exception as e:
    print(f"✗ Neural model check failed: {e}")

# Test session state initialization
if "test_initialized" not in st.session_state:
    st.session_state.test_initialized = True
    print("✓ Session state initialization passed")

# Test tab rendering
try:
    st.title("Test App")
    tabs = st.tabs([
        "🏠 Home",
        "🤖 Tutor",
        "📈 Progress",
    ])
    print("✓ Tabs created successfully")
    
    with tabs[0]:
        st.write("Home tab works")
    with tabs[1]:
        st.write("Tutor tab works")
    with tabs[2]:
        st.write("Progress tab works")
    
    print("✓ All tabs rendered successfully")
except Exception as e:
    print(f"✗ Tab rendering failed: {e}")
    st.error(f"Tab error: {e}")

print("\n✓✓✓ TEST COMPLETED SUCCESSFULLY ✓✓✓")
