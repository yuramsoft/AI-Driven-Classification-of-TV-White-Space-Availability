from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from classify import DecisionPolicy, classify_channel

MODEL_PATH = Path(os.getenv("TVWS_MODEL_PATH", "models/tvws_power.joblib"))
app = FastAPI(title="TV White-Space Research Classifier", version="0.1.0")


class PredictionRequest(BaseModel):
    frequency_mhz: float = Field(gt=0)
    tx_distance_km: float = Field(ge=0)
    tx_height_m: float = Field(gt=0)
    rx_height_m: float = Field(gt=0)
    terrain_loss_db: float
    landcover_urban_fraction: float = Field(ge=0, le=1)
    landcover_forest_fraction: float = Field(ge=0, le=1)
    uncertainty_db: float = Field(ge=0, description="Validated model interval half-width")
    regulator_allows: bool = Field(description="Result from authoritative rules/database")


@lru_cache(maxsize=1)
def load_bundle():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        bundle = load_bundle()
        features = bundle["features"]
        row = pd.DataFrame([request.model_dump()])[features]
        power = float(bundle["model"].predict(row)[0])
        decision = classify_channel(
            power,
            request.uncertainty_db,
            DecisionPolicy(),
            request.regulator_allows,
        )
        return {"model_version": "0.1.0", **decision}
    except (RuntimeError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc