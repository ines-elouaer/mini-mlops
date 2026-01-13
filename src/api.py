from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# -----------------------
# Config
# -----------------------
MODEL_PATH = Path("models")
MODEL_FILE = "model.joblib"
 # ex: "model_xxx.joblib" si tu veux forcer

WEB_DIR = Path("web")  # web/index.html + web/app.js


def _load_model():
    if MODEL_FILE:
        p = MODEL_PATH / MODEL_FILE
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        return joblib.load(p)

    if not MODEL_PATH.exists():
        raise FileNotFoundError("models/ folder not found. Train and save a model first.")

    candidates = sorted(MODEL_PATH.glob("*.joblib"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No .joblib model found in models/. Train and save a model first.")

    return joblib.load(candidates[0])


app = FastAPI(title="Breast Cancer Predictor API", version="1.0")

# Servir /web/*
# Important: mount marche même si web n'existe pas, donc on protège
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


@app.get("/ui")
def ui():
    index = WEB_DIR / "index.html"
    if not index.exists():
        raise HTTPException(
            status_code=404,
            detail="UI not found. Create web/index.html (and web/app.js) at project root.",
        )
    return FileResponse(str(index))


# Charger modèle
model = _load_model()


class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Dictionnaire {nom_feature: valeur}. Doit correspondre aux colonnes d'entraînement.",
        examples=[{"mean radius": 14.0, "mean texture": 20.0}],
    )


class PredictResponse(BaseModel):
    prediction: int
    proba: Optional[float]
    used_features_count: int


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "API is running. Go to /docs for Swagger UI. Go to /ui for simple UI."}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/features")
def get_expected_features() -> Dict[str, Any]:
    if not hasattr(model, "feature_names_in_"):
        return {"error": "Model has no feature_names_in_. Retrain with a DataFrame (pandas) input."}
    return {"expected_features": list(model.feature_names_in_)}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not hasattr(model, "feature_names_in_"):
        raise HTTPException(
            status_code=500,
            detail="Model does not have feature_names_in_. Retrain with pandas DataFrame to keep feature names.",
        )

    expected: List[str] = list(model.feature_names_in_)
    incoming = payload.features

    # Garde uniquement les features attendues (ignore clés inconnues)
    filtered = {k: v for k, v in incoming.items() if k in expected}

    missing = [f for f in expected if f not in filtered]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Missing {len(missing)} feature(s). Example missing: {missing[:5]} ... "
                f"Use GET /features to see the full list."
            ),
        )

    row = {f: filtered[f] for f in expected}
    df = pd.DataFrame([row], columns=expected)

    pred = int(model.predict(df)[0])

    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(df)[0][1])

    return PredictResponse(prediction=pred, proba=proba, used_features_count=len(expected))
