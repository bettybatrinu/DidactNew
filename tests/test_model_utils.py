import pandas as pd

from src.model_utils import prepare_single_problem, resolve_dataset_path


def test_prepare_single_problem_returns_feature_frame():
    df = prepare_single_problem(
        problem="Aflați aria unui triunghi cu baza 5 și înălțimea 4.",
        tema_norm="Geometrie",
        domeniu="Geometrie",
        item=3,
        sursa_type="manual",
    )
    assert isinstance(df, pd.DataFrame)
    assert "Tema_norm" in df.columns
    assert "Domeniu" in df.columns
    assert df.iloc[0]["Tema_norm"] == "Geometrie"
    assert df.iloc[0]["Domeniu"] == "Geometrie"


def test_resolve_dataset_path_returns_existing_file():
    path = resolve_dataset_path()
    assert path.exists()
    assert path.suffix == ".csv"
