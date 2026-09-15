#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage:"
    echo "  $0 DATA TRAIN_OUTPUT_ROOT [IMAGE_DIR] [SPLIT]"
    exit 1
fi

DATA="$1"
TRAIN_OUT="$2"
IMAGE_DIR="${3:-images}"
SPLIT="${4:-20}"

REPO="${REPO:-$HOME/thesis/3dgrut}"

cd "$REPO"

for STEP in 7000 10000 20000 30000; do
    CKPT=$(find "$TRAIN_OUT" \
        -type f \
        -path "*/ours_${STEP}/ckpt_${STEP}.pt" \
        | head -1 || true)

    if [ -z "$CKPT" ]; then
        echo
        echo "STEP $STEP: checkpoint not found; skipping"
        continue
    fi

    echo
    echo "============================================================"
    echo "EVALUATING STEP $STEP"
    echo "CHECKPOINT: $CKPT"
    echo "============================================================"

    EVAL_OUT="${TRAIN_OUT}_EVAL_${STEP}"

    rm -rf "$EVAL_OUT"
    mkdir -p "$EVAL_OUT"

    CC=gcc-11 CXX=g++-11 python3 render.py \
        --checkpoint "$CKPT" \
        --path "$DATA" \
        --out-dir "$EVAL_OUT"

    RUN=$(find "$EVAL_OUT" \
        -type d \
        -path "*/ours_${STEP}" \
        -printf '%h\n' \
        | head -1)

    if [ -z "$RUN" ]; then
        echo "ERROR: rendered run folder not found for step $STEP"
        exit 1
    fi

    echo "Rendered run: $RUN"

    python3 tools/eval_masked_multicam.py \
        --data "$DATA" \
        --run "$RUN" \
        --step "$STEP" \
        --split "$SPLIT" \
        --image-dir "$IMAGE_DIR" \
        | tee "$EVAL_OUT/masked_metrics_${STEP}.txt"

    python3 tools/stratified_dualcam_metrics.py \
        --data "$DATA" \
        --run "$RUN" \
        --step "$STEP" \
        --split "$SPLIT" \
        --image-dir "$IMAGE_DIR" \
        --band-edges 0 30 60 90 \
        --band-names center mid periphery \
        --lpips-max-dim 1024 \
        | tee "$EVAL_OUT/stratified_metrics_${STEP}.txt"
done
