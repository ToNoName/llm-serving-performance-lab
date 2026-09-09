#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH:-}}"
VLLM_URL="${VLLM_URL:-http://localhost:8001}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/results/raw/e4}"

if [[ -z "$TOKENIZER_PATH" ]]; then
  echo "Set TOKENIZER_PATH or MODEL_PATH first" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"
cd "$PROJECT_ROOT"
for concurrency in 1 4 8 16 32; do
  before="$RESULT_DIR/e4-c${concurrency}-before.txt"
  after="$RESULT_DIR/e4-c${concurrency}-after.txt"
  requests="$RESULT_DIR/e4-c${concurrency}.csv"
  curl -fsS "$VLLM_URL/metrics" > "$before"
  python -m src.benchmark.benchmark_streaming \
    --url "$VLLM_URL/v1/chat/completions" \
    --model qwen \
    --tokenizer "$TOKENIZER_PATH" \
    --prompt-tokens 128 \
    --max-tokens 512 \
    --concurrency "$concurrency" \
    --num-requests 40 \
    --output "$requests" \
    --summary-output "$RESULT_DIR/e4-c${concurrency}-summary.json"
  curl -fsS "$VLLM_URL/metrics" > "$after"
  python -m src.benchmark.summarize_metrics \
    --before "$before" \
    --after "$after" \
    --requests "$requests" \
    --experiment E4 \
    --case "e4-c${concurrency}" \
    --output "$RESULT_DIR/e4-native-summary.csv"
done
