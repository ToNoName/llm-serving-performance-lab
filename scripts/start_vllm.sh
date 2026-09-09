#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the AWQ model directory}"
PROFILE="${1:-baseline}"

gpu_memory_utilization=0.85
max_num_seqs=32
case "$PROFILE" in
  baseline|e1|e2|e4) ;;
  e3a-seq8) max_num_seqs=8 ;;
  e3b-mem040) gpu_memory_utilization=0.40 ;;
  e3b-mem035) gpu_memory_utilization=0.35 ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    exit 2
    ;;
esac

args=(
  serve "$MODEL_PATH"
  --served-model-name qwen
  --host 0.0.0.0
  --port 8001
  --gpu-memory-utilization "$gpu_memory_utilization"
  --max-num-seqs "$max_num_seqs"
  --max-model-len 32768
  --trust-remote-code
  --no-enable-prefix-caching
  --enable-chunked-prefill
  --max-num-batched-tokens 2048
)

echo "vLLM version: $(vllm --version)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
printf 'Starting: vllm'
printf ' %q' "${args[@]}"
printf '\n'
exec vllm "${args[@]}"
