#!/usr/bin/env bash
# End-to-end POC run: OCR -> dataset -> train -> merge -> infer(base) -> infer(v1) -> compare.
# Usage: ./run_all.sh [--skip-train] [--engine mineru|pymupdf]
set -uo pipefail

cd "$(dirname "$0")"
PY="${PYTHON:-python}"
SKIP_TRAIN=0
ENGINE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-train) SKIP_TRAIN=1; shift ;;
    --engine) ENGINE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

step() {
  echo ""
  echo "=============================================================="
  echo ">> $*"
  echo "=============================================================="
}

die() {
  echo ""
  echo "FAILED at: $1 (exit $2)" >&2
  exit "$2"
}

run() {
  local label="$1"; shift
  step "$label"
  "$@" || die "$label" $?
}

# Inference failures must not hide the comparison: an unparseable output is itself
# a result the verdict should record, so these steps warn and continue.
run_soft() {
  local label="$1"; shift
  step "$label"
  if ! "$@"; then
    echo ""
    echo "WARNING: $label exited $? - continuing so compare.py can record it" >&2
  fi
}

OCR_ARGS=()
if [ -n "$ENGINE" ]; then OCR_ARGS=(--engine "$ENGINE"); fi

run "1/8 OCR every document"   "$PY" src/run_ocr.py ${OCR_ARGS[@]+"${OCR_ARGS[@]}"}
# Audit the ground truth before training on it. Runs after OCR so it can check
# that every value in the gold actually appears on its page.
run_soft "2/8 audit the golden JSON" "$PY" src/verify_gold.py --ocr
run "3/8 build training corpus" "$PY" src/build_dataset.py
# The base extraction is taken BEFORE training, so the "before" measurement exists
# on disk no matter what happens during the training run.
run_soft "4/8 extract with BASE model" "$PY" src/infer.py --model base
if [ "$SKIP_TRAIN" -eq 0 ]; then
  run "5/8 QLoRA fine-tune"    "$PY" src/train.py
  run "6/8 merge adapter"      "$PY" src/merge.py --force
else
  step "5-6/8 training and merge skipped (--skip-train)"
fi
run_soft "7/8 extract with V1 (fine-tuned)" "$PY" src/infer.py --model v1

step "8/8 compare base vs v1"
"$PY" src/compare.py
VERDICT_CODE=$?

echo ""
if [ "$VERDICT_CODE" -eq 0 ]; then
  echo "POC RESULT: PASS - v1 is closer to the gold JSON than the base model."
else
  echo "POC RESULT: INVESTIGATE - see outputs/results/comparison.json and the reasons above."
fi
exit "$VERDICT_CODE"
