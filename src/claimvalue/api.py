"""FastAPI service. Inputs are non-identifying by design (ids are pseudonyms, no free text)."""

from __future__ import annotations

import json
import secrets
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from claimvalue import __version__
from claimvalue.config import get_settings
from claimvalue.features import FEATURES, NUMERIC

app = FastAPI(title="claimvalue", version=__version__)


class CaseFeatures(BaseModel):
    uf: str = Field(min_length=2, max_length=2, examples=["SP"])
    comarca: str = Field(max_length=80)
    vara: str = Field(max_length=80)
    judge_id: str = Field(max_length=40, examples=["JUDGE_0123456789"])
    filing_month: int = Field(ge=1, le=12)
    judge_n_prior: float | None = None
    judge_avg_value_prior: float | None = None
    judge_unfav_rate_prior: float | None = Field(default=None, ge=0, le=1)
    comarca_n_prior: float | None = None
    comarca_avg_value_prior: float | None = None
    comarca_unfav_rate_prior: float | None = Field(default=None, ge=0, le=1)


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().api_key
    if not expected:
        raise HTTPException(503, "API_KEY is not configured on the server")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(401, "invalid or missing API key")


@lru_cache(maxsize=4)
def _load(model_dir: str, task: str):
    path = Path(model_dir) / f"{task}.joblib"
    if not path.exists():
        return None, {}
    meta = json.loads((Path(model_dir) / f"{task}.json").read_text())
    return joblib.load(path), meta  # trusted, self-produced artefact only


def _frame(case: CaseFeatures) -> pd.DataFrame:
    df = pd.DataFrame([case.model_dump()])[FEATURES]
    df[NUMERIC] = df[NUMERIC].astype(float)  # None -> NaN (the models handle missing values)
    return df


def _model(task: str):
    model, meta = _load(str(get_settings().model_dir), task)
    if model is None:
        raise HTTPException(503, f"no trained {task} model available")
    return model, meta


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/predict/claim-value", dependencies=[Depends(require_key)])
def predict_claim_value(case: CaseFeatures) -> dict:
    model, meta = _model("regression")
    return {
        "predicted_claim_value": round(float(model.predict(_frame(case))[0]), 2),
        "model": meta.get("model"),
        "trained_at": meta.get("created_at"),
    }


@app.post("/predict/unfavorable-probability", dependencies=[Depends(require_key)])
def predict_unfavorable(case: CaseFeatures) -> dict:
    model, meta = _model("classification")
    return {
        "probability_unfavorable": round(float(model.predict_proba(_frame(case))[0, 1]), 4),
        "model": meta.get("model"),
        "trained_at": meta.get("created_at"),
    }
