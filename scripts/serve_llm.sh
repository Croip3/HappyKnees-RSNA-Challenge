#!/usr/bin/env bash
# Serve a Qwen3.8-27B GGUF with llama.cpp (CUDA, Docker) as an OpenAI-compatible endpoint on :8080.
# Then run: python scripts/llm_extract.py --set llm.backend=openai --set llm.base_url=http://127.0.0.1:8080 \
#             --set llm.model=qwen3.8-27b --set llm.concurrency=$PARALLEL
#
# Env overrides: MODEL_DIR, MODEL_FILE, PARALLEL (slots), CTX_PER_SLOT, NGL (GPU layers), PORT, IMAGE
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-"$(cd "$(dirname "$0")/../.." && pwd)/models/qwen3.8-27b-gguf"}
MODEL_FILE=${MODEL_FILE:-Qwen3.8-27B-UD-IQ4_XS.gguf}
PARALLEL=${PARALLEL:-4}
CTX_PER_SLOT=${CTX_PER_SLOT:-6144}
NGL=${NGL:-999}
PORT=${PORT:-8080}
IMAGE=${IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}

docker rm -f knee-llm >/dev/null 2>&1 || true
docker run -d --name knee-llm --gpus all -p "127.0.0.1:${PORT}:8080" -v "${MODEL_DIR}:/models:ro" "${IMAGE}" \
  -m "/models/${MODEL_FILE}" --alias qwen3.8-27b \
  -ngl "${NGL}" -c $((PARALLEL * CTX_PER_SLOT)) -np "${PARALLEL}" \
  --jinja --reasoning-budget 0 -fa on --cache-type-k q8_0 --cache-type-v q8_0 \
  --host 0.0.0.0 --port 8080 --metrics
echo "started knee-llm (logs: docker logs -f knee-llm); waiting for /health ..."
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then echo "ready"; exit 0; fi
  sleep 5
done
echo "server did not become healthy; see docker logs knee-llm" >&2
exit 1
