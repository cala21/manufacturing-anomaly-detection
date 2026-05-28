"""FastAPI inference endpoint for real-time anomaly detection."""
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import time
import os
import pickle

app = FastAPI(title="Manufacturing Anomaly Detection API", version="1.0.0")

ANOMALIES = Counter("anomalies_detected_total", "Total anomalies detected", ["line"])
LATENCY = Histogram("inference_latency_seconds", "Inference latency")

model = None
scaler = None

@app.on_event("startup")
async def load_model():
    global model, scaler
    model_path = os.getenv("MODEL_PATH", "models/isolation_forest.pkl")
    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)
        model = artifacts["model"]
        scaler = artifacts["scaler"]

class SensorData(BaseModel):
    line_id: str
    features: dict[str, float]
    timestamp: str

class AnomalyResult(BaseModel):
    line_id: str
    is_anomaly: bool
    anomaly_score: float
    latency_ms: float

@app.post("/detect", response_model=AnomalyResult)
async def detect(data: SensorData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    start = time.time()
    X = np.array(list(data.features.values())).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    score = float(model.score_samples(X_scaled)[0])
    is_anomaly = prediction == -1
    latency = (time.time() - start) * 1000
    LATENCY.observe(latency / 1000)
    if is_anomaly:
        ANOMALIES.labels(line=data.line_id).inc()
    return AnomalyResult(
        line_id=data.line_id,
        is_anomaly=is_anomaly,
        anomaly_score=score,
        latency_ms=round(latency, 2),
    )

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
