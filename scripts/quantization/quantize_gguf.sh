#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <llama.cpp-dir> <hf-model-dir> <output-dir> <Q4_K_M|Q8_0>" >&2
  exit 2
fi

LLAMA_CPP_DIR=$1
HF_MODEL_DIR=$2
OUTPUT_DIR=$3
QUANT_TYPE=$4

case "$QUANT_TYPE" in
  Q4_K_M|Q8_0) ;;
  *)
    echo "Unsupported quantization type: $QUANT_TYPE" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_DIR"
FP16_GGUF="$OUTPUT_DIR/model-f16.gguf"
QUANT_GGUF="$OUTPUT_DIR/model-${QUANT_TYPE}.gguf"

python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$HF_MODEL_DIR" \
  --outfile "$FP16_GGUF" \
  --outtype f16

"$LLAMA_CPP_DIR/build_cuda/bin/llama-quantize" \
  "$FP16_GGUF" \
  "$QUANT_GGUF" \
  "$QUANT_TYPE"

echo "$QUANT_GGUF"
