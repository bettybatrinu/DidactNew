"""REST API for the two Didact AI ML services (FastAPI + Swagger).

This exposes the *same* trained models the Streamlit app uses, so each ML
service can be tested programmatically and independently — not only through the
UI. Interactive documentation (Swagger UI) is served automatically at /docs and
the OpenAPI schema at /openapi.json.

Run:
    pip install -r requirements-api.txt
    uvicorn api:app --reload --port 8000
    # then open http://localhost:8000/docs

Endpoints:
    GET  /health             -> liveness + which models are loaded
    POST /predict/domain     -> unstructured service: raw text -> curriculum domain
    POST /predict/difficulty -> structured service: metadata -> difficulty class
    GET  /schema             -> feature schema used by the structured model
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model_utils import (
    load_assets,
    predict_domain_from_text,
    predict_structured_difficulty,
    prepare_single_problem,
)

# Models are loaded once at startup and reused across requests.
_STRUCTURED = None
_UNSTRUCTURED = None
_REPORT: Dict = {}


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    global _STRUCTURED, _UNSTRUCTURED, _REPORT
    try:
        structured, unstructured, _data, report = load_assets()
        _STRUCTURED, _UNSTRUCTURED, _REPORT = structured, unstructured, report
    except Exception as exc:  # surfaced through /health
        _STRUCTURED, _UNSTRUCTURED, _REPORT = None, None, {"load_error": str(exc)}
    yield


app = FastAPI(
    title="Didact AI – ML Services API",
    description=(
        "Două servicii ML reale pentru tutorul adaptiv de matematică:\n\n"
        "1. **Structurat** – prezice dificultatea exercițiului din metadate tabulare.\n"
        "2. **Nestructurat** – prezice domeniul curricular din textul brut al problemei.\n\n"
        "Aceleași modele folosite de aplicația Streamlit, expuse aici pentru testare "
        "programatică independentă."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class DomainRequest(BaseModel):
    text: str = Field(..., description="Enunțul brut al problemei de matematică.",
                      examples=["Aflați aria unui triunghi cu baza 6 cm și înălțimea 4 cm."])


class DifficultyRequest(BaseModel):
    problem: str = Field(..., description="Enunțul problemei (folosit pentru feature engineering).")
    tema_norm: str = Field("Geometrie", description="Tema normalizată.")
    domeniu: str = Field("Geometrie", description="Domeniul curricular.")
    item: int = Field(1, description="Numărul itemului în sursă.")
    sursa_type: str = Field("manual", description="Tipul sursei.")


class DomainResponse(BaseModel):
    service: str
    prediction: str
    probabilities: Optional[Dict[str, float]] = None


class DifficultyResponse(BaseModel):
    service: str
    prediction: str
    probabilities: Optional[Dict[str, float]] = None


@app.get("/health", summary="Liveness + loaded models")
def health() -> Dict:
    return {
        "status": "ok" if _STRUCTURED is not None and _UNSTRUCTURED is not None else "degraded",
        "structured_model_loaded": _STRUCTURED is not None,
        "unstructured_model_loaded": _UNSTRUCTURED is not None,
        "structured_macro_f1": _REPORT.get("structured_model", {}).get("model", {}).get("macro_f1"),
        "unstructured_macro_f1": _REPORT.get("unstructured_model", {}).get("model", {}).get("macro_f1"),
    }


@app.get("/schema", summary="Feature schema of the structured service")
def schema() -> Dict:
    return {
        "structured_inputs": _REPORT.get("structured_model", {}).get("inputs", {}),
        "structured_target": "Dificultate_group (1-bază / 2-mediu / 3-consolidare / 4-avansat)",
        "unstructured_input": "raw problem text",
        "unstructured_target": "Domeniu (curriculum domain)",
    }


@app.post("/predict/domain", response_model=DomainResponse,
          summary="Unstructured service: text -> curriculum domain")
def predict_domain(req: DomainRequest) -> DomainResponse:
    if _UNSTRUCTURED is None:
        raise HTTPException(status_code=503, detail="Unstructured model not loaded.")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Field 'text' must not be empty.")
    out = predict_domain_from_text(_UNSTRUCTURED, req.text)
    return DomainResponse(service="unstructured_domain", **out)


@app.post("/predict/difficulty", response_model=DifficultyResponse,
          summary="Structured service: metadata -> difficulty class")
def predict_difficulty(req: DifficultyRequest) -> DifficultyResponse:
    if _STRUCTURED is None:
        raise HTTPException(status_code=503, detail="Structured model not loaded.")
    features = prepare_single_problem(
        problem=req.problem, tema_norm=req.tema_norm, domeniu=req.domeniu,
        item=req.item, sursa_type=req.sursa_type,
    )
    out = predict_structured_difficulty(_STRUCTURED, features)
    return DifficultyResponse(service="structured_difficulty", **out)


@app.get("/", summary="Root")
def root() -> Dict:
    return {
        "name": "Didact AI ML Services API",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": ["/health", "/schema", "/predict/domain", "/predict/difficulty"],
    }
