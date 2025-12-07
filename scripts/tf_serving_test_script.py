# improved_predict.py
import requests
import numpy as np
import time
import uuid
from typing import List, Tuple
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scripts.preprocess import preprocess_image  # your existing preprocess function

TF_SERVING_URL = "http://localhost:8501/v1/models/image_classifier:predict"
REQUEST_TIMEOUT = 10  # seconds
RETRY_TOTAL = 3
RETRY_BACKOFF_FACTOR = 0.5

def requests_session_with_retries(total=RETRY_TOTAL, backoff=RETRY_BACKOFF_FACTOR) -> requests.Session:
    session = requests.Session()
    retries = Retry(total=total,
                    backoff_factor=backoff,
                    status_forcelist=[500, 502, 503, 504],
                    allowed_methods=frozenset(["POST", "GET"]))
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def load_labels(label_file_path: str) -> List[str]:
    with open(label_file_path, "r") as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

def top_k_from_probs(probs: List[float], labels: List[str], k: int = 3) -> List[Tuple[str, float]]:
    probs = np.array(probs, dtype=float)
    if probs.ndim != 1:
        probs = probs.flatten()
    topk_idx = probs.argsort()[::-1][:k]
    return [(labels[i], float(probs[i])) for i in topk_idx]

def get_prediction(image_path: str, label_file: str = "labels.txt", top_k: int = 3):
    labels = load_labels(label_file)
    # Preprocess (should return np.array shape (H,W,3) or (1,H,W,3))
    img_arr = preprocess_image(image_path)
    # Make sure we have a batch dimension
    arr = np.array(img_arr)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    if arr.ndim != 4:
        raise ValueError(f"Preprocessed image must be 3D or 4D array, got shape {arr.shape}")

    payload = {"instances": arr.tolist()}
    session = requests_session_with_retries()

    req_id = str(uuid.uuid4())
    start = time.time()
    try:
        resp = session.post(TF_SERVING_URL, json=payload, timeout=REQUEST_TIMEOUT)
        latency = time.time() - start
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Request {req_id} failed: {e}")

    result = resp.json()
    # Commonly the key is 'predictions' but you may have to adapt if your SavedModel uses another key
    if "predictions" not in result:
        raise KeyError(f"No 'predictions' key in TF Serving response: {result.keys()}")
    probs = result["predictions"][0]
    if len(probs) != len(labels):
        # Warn but try to proceed (could be shape mismatch or label file problem)
        raise ValueError(f"Number of classes in model output ({len(probs)}) does not match labels ({len(labels)})")

    topk = top_k_from_probs(probs, labels, k=top_k)

    # Return a structured dict useful for logging / downstream
    response_obj = {
        "request_id": req_id,
        "latency_seconds": latency,
        "top_k": topk,
        "all_probabilities": [float(x) for x in probs]
    }
    return response_obj

if __name__ == "__main__":
    pred = get_prediction("data/sample_image2.jpg", label_file="labels.txt", top_k=5)
    print("Request ID:", pred["request_id"])
    print(f"Latency: {pred['latency_seconds']:.3f}s")
    print("Top-k:", pred["top_k"])
