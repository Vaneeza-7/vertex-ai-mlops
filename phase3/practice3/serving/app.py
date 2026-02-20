# serving/app.py
import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.environ.get("AIP_STORAGE_URI", "")  # not always set locally
LOCAL_MODEL_FILE = "/model/model.joblib"

app = FastAPI()
model = None

class PredictRequest(BaseModel):
    instances: list

@app.on_event("startup")
def load_model():
    global model
    # Vertex mounts model artifacts at /model for custom containers
    model = joblib.load(LOCAL_MODEL_FILE)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.instances, dtype=float)
    preds = model.predict(X).tolist()
    return {"predictions": preds}
