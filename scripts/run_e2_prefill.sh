#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for input_tokens in 128 512 1024 2048; do
  "$SCRIPT_DIR/_run_native_case.sh" E2 "e2-p${input_tokens}" "$input_tokens" 4 256
done
