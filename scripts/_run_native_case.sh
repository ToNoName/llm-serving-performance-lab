#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 EXPERIMENT CASE INPUT_TOKENS CONCURRENCY MAX_TOKENS" >&2
  exit 2
fi

EXPERIMENT="$1"
CASE_NAME="$2"
INPUT_TOKENS="$3"
CONCURRENCY="$4"
MAX_TOKENS="$5"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH:-}}"
VLLM_URL="${VLLM_URL:-http://localhost:8001}"
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/results/raw}"

if [[ -z "$TOKENIZER_PATH" ]]; then
  echo "Set TOKENIZER_PATH or MODEL_PATH first" >&2
  exit 2
fi

mkdir -p "$RESULT_DIR"
before="$RESULT_DIR/${CASE_NAME}-before.txt"
after="$RESULT_DIR/${CASE_NAME}-after.txt"
requests="$RESULT_DIR/${CASE_NAME}.csv"
summary="$RESULT_DIR/${EXPERIMENT,,}-summary.csv"

curl -fsS "$VLLM_URL/metrics" > "$before"
cd "$PROJECT_ROOT"
python -m src.benchmark.benchmark_native \
  --experiment "$EXPERIMENT" \
  --url "$VLLM_URL/v1/chat/completions" \
  --model qwen \
  --tokenizer "$TOKENIZER_PATH" \
  --input-tokens "$INPUT_TOKENS" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS" \
  --num-requests 60 \
  --output "$requests"
curl -fsS "$VLLM_URL/metrics" > "$after"

python -m src.benchmark.summarize_metrics \
  --before "$before" \
  --after "$after" \
  --requests "$requests" \
  --experiment "$EXPERIMENT" \
  --case "$CASE_NAME" \
  --output "$summary"
