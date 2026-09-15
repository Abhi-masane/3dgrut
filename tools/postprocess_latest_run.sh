#!/usr/bin/env bash

set -e

if [ "$#" -lt 4 ]; then
    echo "Usage:"
    echo "$0 DATA OUTPUT_ROOT STEP SPLIT"
    exit 1
fi

DATA="$1"
OUT="$2"
STEP="$3"
SPLIT="$4"

RUN=$(find "$OUT" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -printf '%T@ %p\n' \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-)

if [ -z "$RUN" ]; then
    echo "ERROR: no run directory found in:"
    echo "$OUT"
    exit 1
fi

echo
echo "Detected latest run:"
echo "$RUN"
echo

python3 tools/make_clean_fisheye_renders.py \
    --data "$DATA" \
    --run "$RUN" \
    --step "$STEP" \
    --split "$SPLIT" \
    --fov 180
