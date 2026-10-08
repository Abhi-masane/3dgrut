#!/usr/bin/env bash
set -euo pipefail

# Final MeetingRoom thesis evaluation wrapper.
#
# Final thesis numbering used here:
#   V1 = FIORD perspective / undistorted baseline
#   V2 = native fisheye + SfM initialization
#   V3 = V2 + solid-angle weighting
#   V4 = depth replacement initialization
#   V5 = gap-aware hybrid SfM + depth initialization
#   V6 = V5 + solid-angle weighting
#
# Existing physical folder names from older experiments are kept as legacy
# defaults. You can always pass an explicit dataset as the third argument.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="${TOOLS:-$SCRIPT_DIR}"
DATA_ROOT="${DATA_ROOT:-$HOME/thesis/SCIEBO_FINAL/my3dscene/data/datasets}"
SPLIT="${SPLIT:-20}"
IMAGE_DIR="${IMAGE_DIR:-images}"
SSIM_WIN_SIZE="${SSIM_WIN_SIZE:-7}"
LPIPS_MAX_SIZE="${LPIPS_MAX_SIZE:-1024}"

VARIANT="${1:-}"
INPUT="${2:-}"
DATA_OVERRIDE="${3:-}"
STEP_OVERRIDE="${4:-}"

if [ -z "$VARIANT" ] || [ -z "$INPUT" ]; then
    echo "Usage:"
    echo "  postprocess_meetingroom.sh <variant> <run-or-experiment-folder> [dataset-folder] [step]"
    echo
    echo "Examples:"
    echo "  postprocess_meetingroom.sh v2 /path/to/timestamped_run"
    echo "  postprocess_meetingroom.sh v5 /path/to/run /path/to/MeetingRoom_V5_dataset 60000"
    exit 1
fi

VARIANT=$(echo "$VARIANT" | tr '[:upper:]' '[:lower:]')
INPUT=$(realpath "$INPUT")

if [ -n "$DATA_OVERRIDE" ]; then
    DATA=$(realpath "$DATA_OVERRIDE")
else
    case "$VARIANT" in
        v1)
            # Current V1: FIORD-provided perspective/undistorted reconstruction.
            DATA="${V1_DATA:-$HOME/Downloads/MeetingRoom_In/meetingroom/model_undistorted}"
            ;;
        v2|v3)
            DATA="${V2_DATA:-$DATA_ROOT/MeetingRoom_V2_NATIVE}"
            ;;
        v4)
            # Final V4 is replacement. Legacy folder name may still contain V4A.
            DATA="${V4_DATA:-$DATA_ROOT/MeetingRoom_V4A_REPLACE}"
            ;;
        v5|v6)
            # Final V5 is gap-aware hybrid. Legacy folder may still be named V4_DEPTH.
            DATA="${V5_DATA:-$DATA_ROOT/MeetingRoom_V4_DEPTH}"
            ;;
        *)
            echo "Unknown variant: $VARIANT"
            echo "Expected one of: v1 v2 v3 v4 v5 v6"
            exit 1
            ;;
    esac
fi

DATA=$(realpath "$DATA")

if [ ! -d "$DATA" ]; then
    echo "Dataset directory does not exist: $DATA"
    exit 1
fi

if [ ! -d "$DATA/$IMAGE_DIR" ]; then
    echo "Image directory does not exist: $DATA/$IMAGE_DIR"
    exit 1
fi

if [ ! -d "$DATA/sparse/0" ]; then
    echo "COLMAP model does not exist: $DATA/sparse/0"
    exit 1
fi

for script in eval_masked_multicam.py stratified_dualcam_metrics.py eval_valid_fov_lpips.py; do
    if [ ! -f "$TOOLS/$script" ]; then
        echo "Missing evaluation script: $TOOLS/$script"
        exit 1
    fi
done

# Find the actual timestamped run directory.
if compgen -G "$INPUT/ours_*" > /dev/null; then
    RUN="$INPUT"
else
    matches=()
    while IFS= read -r folder; do
        if compgen -G "$folder/ours_*" > /dev/null; then
            matches+=("$folder")
        fi
    done < <(
        find "$INPUT" -mindepth 1 -maxdepth 1 -type d | sort
    )

    if [ "${#matches[@]}" -ne 1 ]; then
        echo "Expected exactly one training run inside:"
        echo "$INPUT"
        echo
        echo "Found ${#matches[@]} possible runs."
        if [ "${#matches[@]}" -gt 0 ]; then
            printf '  %s\n' "${matches[@]}"
        fi
        echo
        echo "Pass the exact timestamped run folder instead."
        exit 1
    fi
    RUN="${matches[0]}"
fi

if [ -n "$STEP_OVERRIDE" ]; then
    STEP="$STEP_OVERRIDE"
    if [ ! -d "$RUN/ours_$STEP" ]; then
        echo "Requested step does not exist: $RUN/ours_$STEP"
        exit 1
    fi
else
    STEP=$(
        find "$RUN" -maxdepth 1 -type d -name 'ours_*' -printf '%f\n' \
        | sed 's/^ours_//' \
        | grep -E '^[0-9]+$' \
        | sort -n \
        | tail -1
    )
fi

if [ -z "${STEP:-}" ]; then
    echo "No ours_<step> directory found in: $RUN"
    exit 1
fi

# Native fisheye runs must have exact masks. For V1, if the perspective dataset
# has no mask files, all pixels are treated as valid.
if [ "$VARIANT" = "v1" ]; then
    MASK_MODE="auto"
else
    MASK_MODE="required"
fi

RESULTS="$RUN/postprocess_$STEP"
mkdir -p "$RESULTS"

cat > "$RESULTS/evaluation_info.txt" <<INFO
variant=$VARIANT
step=$STEP
split=$SPLIT
dataset=$DATA
run=$RUN
image_directory=$IMAGE_DIR
mask_mode=$MASK_MODE
mask_threshold=127
ssim_window=$SSIM_WIN_SIZE
angular_bands=0-30,30-60,60-90
lpips_protocol=masked_crop_valid_fov
lpips_note=standard_VGG_LPIPS_after_identical_invalid_pixels_and_tight_mask_bbox_crop
lpips_network=vgg
lpips_normalize=true
lpips_max_size=$LPIPS_MAX_SIZE
INFO

{
    echo ""
    echo "script_sha256:"
    sha256sum \
        "$TOOLS/eval_masked_multicam.py" \
        "$TOOLS/stratified_dualcam_metrics.py" \
        "$TOOLS/eval_valid_fov_lpips.py"
    echo ""
    echo "python_version=$(python3 --version 2>&1)"
} >> "$RESULTS/evaluation_info.txt"

echo "============================================================"
echo "MeetingRoom thesis post-processing"
echo "============================================================"
echo "Variant       : $VARIANT"
echo "Dataset       : $DATA"
echo "Run           : $RUN"
echo "Step          : $STEP"
echo "Split         : $SPLIT"
echo "Mask mode     : $MASK_MODE"
echo "SSIM window   : $SSIM_WIN_SIZE"
echo "LPIPS max size: $LPIPS_MAX_SIZE"
echo "Results       : $RESULTS"
echo "============================================================"
echo

echo "[1/3] Overall full + valid-region PSNR / SSIM"
python3 "$TOOLS/eval_masked_multicam.py" \
    --data "$DATA" \
    --run "$RUN" \
    --step "$STEP" \
    --split "$SPLIT" \
    --image-dir "$IMAGE_DIR" \
    --mask-mode "$MASK_MODE" \
    --ssim-win-size "$SSIM_WIN_SIZE" \
    | tee "$RESULTS/masked_metrics.txt"

echo
echo "[2/3] Angular PSNR / SSIM"
python3 "$TOOLS/stratified_dualcam_metrics.py" \
    --data "$DATA" \
    --run "$RUN" \
    --step "$STEP" \
    --split "$SPLIT" \
    --image-dir "$IMAGE_DIR" \
    --mask-mode "$MASK_MODE" \
    --ssim-win-size "$SSIM_WIN_SIZE" \
    --band-edges "0,30,60,90" \
    | tee "$RESULTS/angular_metrics.txt"

echo
echo "[3/3] Masked-crop valid-FoV LPIPS"
python3 "$TOOLS/eval_valid_fov_lpips.py" \
    --data "$DATA" \
    --run "$RUN" \
    --step "$STEP" \
    --split "$SPLIT" \
    --image-dir "$IMAGE_DIR" \
    --mask-mode "$MASK_MODE" \
    --max-size "$LPIPS_MAX_SIZE" \
    | tee "$RESULTS/valid_fov_lpips.txt"

echo
echo "============================================================"
echo "POST-PROCESSING COMPLETE"
echo "============================================================"
echo "$RESULTS"
echo
echo "Files:"
echo "  masked_metrics.txt"
echo "  angular_metrics.txt"
echo "  valid_fov_lpips.txt"
echo "  evaluation_info.txt"
echo "============================================================"
