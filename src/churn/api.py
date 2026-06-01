from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
import os 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.churn.config import load_config

CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/debug.yaml")

app = FastAPI(
    title="Customer Churn Prediction API",
    description = "Production-stype API for customer churn prediction using xgboost",
    version="0.1.0"
)

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    
class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int
    threshold: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

def load_fold_models(model_dir: Path):
    model_paths = sorted(model_dir.glob("xgb_fold_*.pkl"))

    if not model_paths:
        raise FileNotFoundError(
            f"no model existed in {model_dir}"
        )
    
    models = []

    for path in model_paths:
        model = joblib.load(path)

        # Server inference on CPU by default
        try:
            model.set_params(device="cpu")
        except Exception:
            pass
        models.append(model)
    return models

def load_artifacts():
    config = load_config(CONFIG_PATH)

    model_dir = Path(config["training"]["model_dir"])
    feature_pipeline_path = model_dir / "feature_pipeline.pkl"

    if not feature_pipeline_path:
        raise FileNotFoundError(
            f"feature pipeline not found: {feature_pipeline_path}"
        )
    
    feature_builder = joblib.load(feature_pipeline_path)
    models = load_fold_models(model_dir)

    return config, feature_builder, models

try:
    CONFIG, FEATURE_BUILDER, MODELS = load_artifacts()
except Exception as e:
    CONFIG, FEATURE_BUILDER, MODELS = None, None, None
    STARTUP_ERROR = str(e)
else:
    STARTUP_ERROR = None 

def payload_to_dict(payload: BaseModel) -> dict:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()

def predict_dataframe(df: pd.DataFrame, threshold: float=0.5) -> np.ndarray:
    if FEATURE_BUILDER is None or MODELS is None:
        raise RuntimeError(f"Artifact were not load correctly: {STARTUP_ERROR}")
    X = FEATURE_BUILDER.transform(df)
    preds = np.zeros(len(df))
    for model in MODELS:
        preds += model.predict_proba(X)[:,1] / len(MODELS)
    return preds

@app.get("/health")
def health_check():
    return {
        "status": "ok" if STARTUP_ERROR is None else "error",
        "startup_error": STARTUP_ERROR,
        "num_models": len(MODELS) if MODELS is not None else 0,
    }

@app.post("/predict")
def predict(customer: CustomerInput, threshold: float= 0.5):
    try:
        customer_predict = payload_to_dict(customer)
        df = pd.DataFrame([customer_predict])
        
        prob = float(predict_dataframe(df, threshold=threshold)[0])
        pred = int(prob > threshold)
        
        return PredictionResponse(
            churn_probability=prob,
            prediction=pred,
            threshold=threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict-batch", response_model=BatchPredictionResponse)
def predict_batch(customers: List[CustomerInput], threshold: float = 0.5):
    try:
        rows = [payload_to_dict(customer) for customer in customers]
        df = pd.DataFrame(rows)

        probs = predict_dataframe(df, threshold=threshold)

        results = [
            PredictionResponse(
                churn_probability=probs,
                prediction=int(prob>threshold),
                threshold=threshold
            )
            for prob in probs
        ]

        return BatchPredictionResponse(predictions=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
