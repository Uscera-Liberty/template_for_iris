import os
import time
import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import uvicorn

app = FastAPI(title="Iris Model Server")

# ============================================
# PROMETHEUS МЕТРИКИ
# ============================================

PREDICTION_COUNTER = Counter(
    'iris_predictions_total',
    'Total number of predictions',
    ['predicted_class', 'status']
)

PREDICTION_LATENCY = Histogram(
    'iris_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0]
)

MODEL_ACCURACY = Gauge(
    'iris_model_accuracy',
    'Current model accuracy from MLflow',
    ['model_version', 'run_id']
)

MODEL_VERSION = Gauge(
    'iris_model_version',
    'Currently loaded model version'
)

ERROR_COUNTER = Counter(
    'iris_errors_total',
    'Total errors',
    ['error_type']
)

# ============================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
model = None
current_model_info = {}

def load_production_model():
    global model, current_model_info

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.get_latest_versions("iris-classifier", stages=["Production"])
        if not versions:
            raise Exception("No Production model found")

        latest = versions[0]

        model = mlflow.sklearn.load_model(f"models:/iris-classifier/Production")

        run = client.get_run(latest.run_id)
        accuracy = run.data.metrics.get('accuracy', 0)

        current_model_info = {
            'version': latest.version,
            'run_id': latest.run_id,
            'accuracy': accuracy
        }

        MODEL_ACCURACY.labels(
            model_version=latest.version,
            run_id=latest.run_id
        ).set(accuracy)
        MODEL_VERSION.set(float(latest.version))

    except Exception as e:
        ERROR_COUNTER.labels(error_type='model_load').inc()
        raise

# ============================================
# API
# ============================================

class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictResponse(BaseModel):
    prediction: int
    class_name: str
    confidence: float
    model_version: str

TARGET_NAMES = ['setosa', 'versicolor', 'virginica']

@app.on_event("startup")
async def startup():
    load_production_model()

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_version": current_model_info.get('version')}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None:
        ERROR_COUNTER.labels(error_type='model_not_loaded').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        features = np.array([[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width
        ]])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[prediction])
        class_name = TARGET_NAMES[prediction]

        PREDICTION_LATENCY.observe(time.time() - start_time)
        PREDICTION_COUNTER.labels(predicted_class=class_name, status='success').inc()

        return PredictResponse(
            prediction=int(prediction),
            class_name=class_name,
            confidence=confidence,
            model_version=str(current_model_info.get('version', 'unknown'))
        )

    except Exception as e:
        ERROR_COUNTER.labels(error_type='prediction').inc()
        PREDICTION_COUNTER.labels(predicted_class='unknown', status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-model")
async def reload_model():
    load_production_model()
    return {"status": "reloaded", "model_info": current_model_info}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)