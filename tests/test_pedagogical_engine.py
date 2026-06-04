from src.pedagogical_engine import evaluate_answer, update_mastery, target_difficulty_from_mastery, diagnose_learning_state


def test_evaluate_answer_exact_match():
    result = evaluate_answer("10", "10")
    assert result["correct"] is True
    assert result["status"] == "correct"


def test_evaluate_answer_numeric_overlap():
    result = evaluate_answer("rezultatul este 12", "12")
    assert result["correct"] is True
    assert result["status"] in {"correct", "probably_correct_format_differs"}


def test_update_mastery_increases_for_correct():
    mastery = update_mastery(0.60, True, hints_used=0, attempts=1)
    assert mastery > 0.60


def test_update_mastery_decreases_for_incorrect():
    mastery = update_mastery(0.60, False, hints_used=2, attempts=3)
    assert mastery < 0.60


def test_target_difficulty_from_mastery():
    assert target_difficulty_from_mastery(0.30, False) == "1 - bază"
    assert target_difficulty_from_mastery(0.50, True) == "2 - mediu"
    assert target_difficulty_from_mastery(0.85, True) == "4 - avansat"


def test_diagnose_learning_state_uses_hints_and_attempts():
    state = diagnose_learning_state(False, hints_used=3, attempts=1, time_seconds=120)
    assert state["state"] == "blocaj conceptual"
