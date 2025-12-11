#!/bin/sh
set -euo pipefail

# fallback env vars
: "${MODEL_NAME:=image_classifier}"
: "${MODEL_BASE_PATH:=/models/${MODEL_NAME}}"
: "${MODEL_S3_URI:=}"

echo "[tf_entrypoint] MODEL_NAME=${MODEL_NAME}"
echo "[tf_entrypoint] MODEL_BASE_PATH=${MODEL_BASE_PATH}"
echo "[tf_entrypoint] MODEL_S3_URI=${MODEL_S3_URI:-<not-provided>}"

if [ -n "${MODEL_S3_URI}" ]; then
  echo "[tf_entrypoint] Syncing model from ${MODEL_S3_URI} to ${MODEL_BASE_PATH} ..."
  mkdir -p "${MODEL_BASE_PATH}"
  # retry a few times for transient errors
  n=0
  until [ $n -ge 5 ]
  do
    aws s3 sync "${MODEL_S3_URI}" "${MODEL_BASE_PATH}" --exact-timestamps && break
    n=$((n+1))
    echo "[tf_entrypoint] Sync attempt $n failed — retrying in 5s..."
    sleep 5
  done
  if [ $n -ge 5 ]; then
    echo "[tf_entrypoint] ERROR: Failed to sync model from S3 after 5 attempts" >&2
    exit 1
  fi
  echo "[tf_entrypoint] Sync complete."
else
  echo "[tf_entrypoint] No MODEL_S3_URI provided — assuming model is already present at ${MODEL_BASE_PATH}"
fi

echo "[tf_entrypoint] Starting tensorflow_model_server..."
exec tensorflow_model_server \
  --port=8500 \
  --rest_api_port=8501 \
  --model_name="${MODEL_NAME}" \
  --model_base_path="${MODEL_BASE_PATH}"
