#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for concurrency in 1 8 16 32; do
  "$SCRIPT_DIR/_run_native_case.sh" E3A "e3a-seq8-c${concurrency}" 1024 "$concurrency" 256
done
