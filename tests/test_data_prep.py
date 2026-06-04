from src.data_prep import canonicalize_tema, clean_text, difficulty_group, infer_domeniu


def test_clean_text_removes_diacritics_and_normalizes():
    assert clean_text("ĂȘȚî -  5  ") == "asti - 5"


def test_canonicalize_tema_maps_common_topics():
    assert canonicalize_tema("Ecuație de gradul II") == "Ecuații de gradul II"
    assert canonicalize_tema("procent și raport") == "Procente"
    assert canonicalize_tema("triunghi isoscel") == "Geometrie - Triunghiuri"


def test_infer_domeniu_from_canonical_topic():
    assert infer_domeniu("Funcții de gradul II") == "Funcții"
    assert infer_domeniu("Geometrie - Cerc") == "Geometrie"
    assert infer_domeniu("Inecuații cu parametri") == "Ecuații, inecuații și sisteme"


def test_difficulty_group_rounds_and_labels():
    assert difficulty_group(1.1) == "1 - bază"
    assert difficulty_group(2.0) == "2 - mediu"
    assert difficulty_group(3.9) == "4 - avansat"
    assert difficulty_group(float("nan")) == "Necunoscut"
