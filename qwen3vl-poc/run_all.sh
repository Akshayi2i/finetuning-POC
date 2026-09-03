#!/usr/bin/env bash
# End-to-end POC run. Each stage ensures its own inputs: build_dataset OCRs what it
# needs, and train.py rebuilds the corpus when the data or prompt has changed.
# Usage: ./run_all.sh [--skip-train] [--engine mineru|pymupdf]
set -uo pipefail

cd "$(dirname "$0")"
PY="${PYTHON:-python}"
SKIP_TRAIN=0
ENGINE=""
VERSION=""

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-train) SKIP_TRAIN=1; shift ;;
    --engine) ENGINE="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
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
# Which fine-tuned version this run produces. Defaults to model.version in config.yaml.
VER_ARGS=()
if [ -n "$VERSION" ]; then VER_ARGS=(--version "$VERSION"); fi
MODEL_ARG="${VERSION:-$("$PY" -c "import sys;sys.path.insert(0,'src');from common import load_config,model_version;print(model_version(load_config()))")}"
echo "fine-tuned version for this run: $MODEL_ARG"

# build_dataset.py OCRs any document that needs it, so OCR is not a separate step.
run "1/7 build training corpus (OCRs as needed)" "$PY" src/build_dataset.py
# Audit the ground truth before training on it. Runs after OCR so it can check
# that every value in the gold actually appears on its page.
run_soft "2/7 audit the golden JSON" "$PY" src/verify_gold.py --ocr
# The base extraction is taken BEFORE training, so the "before" measurement exists
# on disk no matter what happens during the training run.
run_soft "3/7 extract with BASE model" "$PY" src/infer.py --model base
if [ "$SKIP_TRAIN" -eq 0 ]; then
  run "4/7 QLoRA fine-tune"    "$PY" src/train.py ${VER_ARGS[@]+"${VER_ARGS[@]}"}
  run "5/7 merge adapter"      "$PY" src/merge.py --force ${VER_ARGS[@]+"${VER_ARGS[@]}"}
else
  step "4-5/7 training and merge skipped (--skip-train)"
fi
run_soft "6/7 extract with the fine-tuned model" "$PY" src/infer.py --model "$MODEL_ARG"

step "7/7 compare base vs the fine-tuned model"
"$PY" src/compare.py ${VER_ARGS[@]+"${VER_ARGS[@]}"}
VERDICT_CODE=$?

echo ""
if [ "$VERDICT_CODE" -eq 0 ]; then
  echo "POC RESULT: PASS - $MODEL_ARG is closer to the gold JSON than the base model."
else
  echo "POC RESULT: INVESTIGATE - see results/trained model results/<version>/<doc>/comparison.json and the reasons above."
fi
exit "$VERDICT_CODE"
