# Direction-1 demo image. CPU-only and deliberately small: no CUDA, no torch (the demo path
# uses onnxruntime for the user tower and FAISS for retrieval).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/mpl \
    PYTHONPATH=/app/src

WORKDIR /app

# libgomp is required by both faiss-cpu and onnxruntime on slim images
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

# Install runtime deps only - notably NOT torch, NOT pyspark, NOT duckdb.
# kafka-python-ng backs the producer/consumer, prometheus-client the /metrics endpoint.
RUN pip install --no-cache-dir \
      "numpy>=2.0" "pyarrow>=17" "fastapi>=0.115" "uvicorn[standard]>=0.32" \
      "pydantic>=2.9" "redis>=5.1" "kafka-python-ng>=2.2" "prometheus-client>=0.20" \
      "onnxruntime>=1.20" "faiss-cpu>=1.9" "httpx"

COPY src/ /app/src/
COPY artifacts/models/ /app/artifacts/models/
# `load_prepared()` / `ItemVocab` / popularity read the small generated artifacts
# (models ~13 MB + stats ~52 KB). The multi-GB trees stay out of the image and are mounted by
# compose: `artifacts/data/` (4.8 GB) and `TMM_DATA/feature_map` (the embedding gather).
COPY artifacts/stats/ /app/artifacts/stats/

EXPOSE 8000

# Liveness, not readiness: losing Redis/Kafka must degrade the service, not restart it.
# `/health` still returns 200 in every mode and carries `mode` for the operator.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/livez || exit 1

# One worker by default: the Recommender costs ~1.4 GB steady per process, so two workers
# would exceed the compose 2 GB limit. Scale with TMM_WORKERS once the memory budget allows,
# remembering that in `minimal` mode sessions/breakers/limiter are per worker.
CMD ["sh", "-c", "exec uvicorn tmm.serve.app:app --host 0.0.0.0 --port ${TMM_PORT:-8000} --workers ${TMM_WORKERS:-1}"]
