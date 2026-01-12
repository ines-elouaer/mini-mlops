from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -----------------------
# Config
# -----------------------
# Mets ici TON fichier modèle (celui que tu veux servir)
# Exemple: models/model_xxx.joblib
MODEL_PATH = Path("models")  # dossier
MODEL_FILE = None  # si tu veux forcer un fichier précis: "model_XXXX.joblib"


def _load_model():
    # 1) si MODEL_FILE est défini → on charge ce fichier
    if MODEL_FILE:
        p = MODEL_PATH / MODEL_FILE
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        return joblib.load(p)

    # 2) sinon on prend le dernier .joblib du dossier models/
    if not MODEL_PATH.exists():
        raise FileNotFoundError("models/ folder not found. Train and save a model first.")

    candidates = sorted(MODEL_PATH.glob("*.joblib"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No .joblib model found in models/. Train and save a model first.")

    return joblib.load(candidates[0])


# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Breast Cancer Predictor API", version="1.0")

model = _load_model()


class PredictRequest(BaseModel):
    # On accepte un dict "feature -> valeur"
    # Swagger mettra "additionalProp1" par défaut, mais nous on va IGNORER les clés inconnues.
    features: Dict[str, float] = Field(
        ...,
        description="Dictionnaire {nom_feature: valeur}. Les noms doivent correspondre aux colonnes d'entraînement.",
        examples=[
            {
                "mean radius": 14.0,
                "mean texture": 20.0
            }
        ],
    )


class PredictResponse(BaseModel):
    prediction: int
    proba: Optional[float]
    used_features_count: int


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "API is running. Go to /docs for Swagger UI."}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/features")
def get_expected_features() -> Dict[str, Any]:
    """Pratique: permet de copier/coller la liste des features attendues."""
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

    # 1) garder uniquement les features attendues (ignore additionalProp1, etc.)
    filtered = {k: v for k, v in incoming.items() if k in expected}

    # 2) vérifier les manquantes
    missing = [f for f in expected if f not in filtered]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing {len(missing)} feature(s). Example missing: {missing[:5]} ... "
                   f"Use GET /features to see the full list.",
        )

    # 3) construire DataFrame dans le bon ordre
    row = {f: filtered[f] for f in expected}
    df = pd.DataFrame([row], columns=expected)

    # 4) prédiction
    pred = int(model.predict(df)[0])

    # proba si dispo
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(df)[0][1])

    return PredictResponse(prediction=pred, proba=proba, used_features_count=len(expected))
