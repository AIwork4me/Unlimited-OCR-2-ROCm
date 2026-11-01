#!/usr/bin/env bash
# Physical Linux gfx1151 validation sequence (issue #6 P1, primary Linux target).
# Every command runs in a fresh process and writes to a fresh evidence path.
# Usage: HF_CACHE=<dir> VENV_PYTHON=<python> scripts/run_linux_gfx1151_validation.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${VENV_PYTHON:-../envs/linux-gfx1151-rocm/bin/python}"
CACHE="${HF_CACHE:-../hf-cache}"
EVID="eval/results/Linux gfx1151 verified"
SNAP="$("$PY" - <<EOF
import sys; sys.path.insert(0, "src")
from unlimited_ocr_2_rocm.config import read_lock
print(read_lock()["model_revision"])
EOF
)"

cd "$ROOT"
mkdir -p "$EVID"

echo "== Gate A: doctor (stack identity + real BF16 GPU matmul) =="
"$PY" -m unlimited_ocr_2_rocm doctor --output "$EVID/doctor.json"

echo "== Gate B: Triton compile+execute on gfx1151 =="
"$PY" scripts/probe_triton.py --output "$EVID/triton.json"

echo "== Gate C: DASH sparse prefill kernel vs independent math =="
"$PY" scripts/probe_sparse.py --source "$CACHE/models--baidu--Unlimited-OCR-2/snapshots/$SNAP" \
  --output "$EVID/sparse.json"

echo "== Gate D: vision SDPA scoped MATH regression =="
"$PY" scripts/probe_vision_sdpa.py --source "$CACHE/models--baidu--Unlimited-OCR-2/snapshots/$SNAP" \
  --mode math --output "$EVID/vision_sdpa.json"

echo "== Gate E: single-page OCR suite (5 cases + unicode recovery) =="
"$PY" -m unlimited_ocr_2_rocm smoke-single --fixtures tests/fixtures \
  --cache-dir "$CACHE" --output "$EVID/single" --max-new-tokens 1024

echo "== Gate F: multi-page native OCR suite (2/3/5 pages x2 + table) =="
"$PY" -m unlimited_ocr_2_rocm smoke-multi --fixtures tests/fixtures \
  --cache-dir "$CACHE" --output "$EVID/multi" --max-new-tokens 1024

echo "All gates completed — evidence under: $EVID"
