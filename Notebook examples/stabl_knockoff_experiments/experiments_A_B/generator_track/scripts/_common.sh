#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AB_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STABL_ROOT="${STABL_ROOT:-$(cd "$AB_ROOT/../../.." && pwd)}"
cd "$AB_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
