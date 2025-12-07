# scripts/app.py
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Classifier (FastAPI wrapper)")

# Config via environment variables (easy to override in docker-compose)
TF_SERVING_HOST = os.getenv("TF_SERVING_HOST", "tf_serving")
TF_SERVING_REST_PORT = os.getenv("TF_SERVING_REST_PORT", "8501")
MODEL_NAME = os.getenv("MODEL_NAME", "image_classifier")
TF_SERVING_URL = f"http://{TF_SERVING_HOST}:{TF_SERVING_REST_PORT}/v1/models/{MODEL_NAME}:predict"

# input size — set to model's expected size; default 224x224 (change if your model expects 32)
INPUT_SIZE = int(os.getenv("INPUT_SIZE", "224"))

REQUEST_TIMEOUT = float(os.getenv("TF_REQUEST_TIMEOUT", "10.0"))

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

@app.get("/health")
def health():
    # quick check: ping TF Serving model status
    try:
        status_url = f"http://{TF_SERVING_HOST}:{TF_SERVING_REST_PORT}/v1/models/{MODEL_NAME}"
        r = requests.get(status_url, timeout=2.0)
        if r.status_code == 200:
            return {"status": "ok"}
    except Exception:
        # if TF Serving not reachable, still return 503 so orchestrator knows
        raise HTTPException(status_code=503, detail="TF Serving not reachable")
    raise HTTPException(status_code=503, detail="model not available")

@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    # read image bytes
    try:
        img = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # preprocess: resize & normalize
    try:
        img = img.resize((INPUT_SIZE, INPUT_SIZE))
        image_array = np.array(img).astype(np.float32) / 255.0
        # ensure shape (1,H,W,3)
        if image_array.ndim == 3:
            image_array = np.expand_dims(image_array, axis=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}")

    payload = {"instances": image_array.tolist()}

    # call TF Serving with timeout and basic error handling
    try:
        resp = requests.post(TF_SERVING_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("TF Serving request failed: %s", e)
        raise HTTPException(status_code=502, detail=f"TF Serving error: {e}")

    result = resp.json()
    if "predictions" not in result:
        logger.error("Unexpected TF Serving response: %s", result)
        raise HTTPException(status_code=502, detail="Invalid response from TF Serving")

    probs = np.array(result["predictions"][0], dtype=float)
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    class_name = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else str(top_idx)

    return JSONResponse({"class": class_name, "confidence": confidence})
