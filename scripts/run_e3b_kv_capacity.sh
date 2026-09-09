#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ( "$1" != "mem040" && "$1" != "mem035" ) ]]; then
  echo "usage: $0 mem040|mem035" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for concurrency in 8 16 32; do
  "$SCRIPT_DIR/_run_native_case.sh" E3B "e3b-$1-c${concurrency}" 1024 "$concurrency" 256
done
