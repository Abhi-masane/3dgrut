# Final MeetingRoom Thesis Evaluation Scripts

This folder contains the cleaned evaluation package for the FIORD MeetingRoom experiments.

## Final thesis protocol

Primary native-fisheye metrics:

- masked PSNR
- masked SSIM
- masked-crop valid-FoV LPIPS
- per-camera masked PSNR/SSIM
- angular PSNR/SSIM for 0-30, 30-60, and 60-90 degrees

LPIPS is **not** computed separately in angular bands.

## Files

- `eval_masked_multicam.py`: overall and per-camera PSNR/SSIM
- `stratified_dualcam_metrics.py`: angular PSNR/SSIM
- `eval_valid_fov_lpips.py`: masked-crop valid-FoV VGG LPIPS
- `make_clean_fisheye_renders.py`: visualization only, not metric computation
- `postprocess_meetingroom.sh`: wrapper that runs the three metric scripts

## Important corrections relative to the old scripts

1. SSIM `win_size=7` is explicit.
2. SSIM averaging uses a mask eroded by the 7x7 support window, with the image border treated as outside support.
3. Render filenames are checked for the expected contiguous numeric order when numeric filenames are used.
4. Native fisheye variants require the exact `*_mask.png` files.
5. V1 can fall back to an all-valid mask if the perspective dataset has no masks.
6. Angular evaluation supports calibrated `OPENCV_FISHEYE` and ordinary `PINHOLE`/`SIMPLE_PINHOLE` cameras without silently treating one as the other.
7. LPIPS output is explicitly named a `masked-crop` protocol because standard LPIPS has no native arbitrary pixel mask.
8. `make_clean_fisheye_renders.py` no longer hardcodes `images_2`; it uses `--image-dir`.
9. The wrapper uses the final thesis numbering: V4 replacement, V5 gap-aware hybrid, V6 combined.
10. The wrapper writes an `evaluation_info.txt` file with settings and script SHA256 hashes.

## Install/copy

Copy the scripts to the `tools` folder of the 3DGRUT repository you are using. The wrapper automatically finds the three metric scripts in the same folder:

```bash
cp eval_masked_multicam.py \
   stratified_dualcam_metrics.py \
   eval_valid_fov_lpips.py \
   make_clean_fisheye_renders.py \
   postprocess_meetingroom.sh \
   <YOUR_3DGRUT_REPO>/tools/

chmod +x <YOUR_3DGRUT_REPO>/tools/*.py
chmod +x <YOUR_3DGRUT_REPO>/tools/postprocess_meetingroom.sh
```

## Recommended use

Run the wrapper with the exact timestamped run folder:

```bash
<YOUR_3DGRUT_REPO>/tools/postprocess_meetingroom.sh v2 /path/to/timestamped_run
```

To avoid any ambiguity in old physical dataset folder names, the safest final-thesis form is to pass the dataset explicitly:

```bash
<YOUR_3DGRUT_REPO>/tools/postprocess_meetingroom.sh \
  v5 \
  /path/to/timestamped_run \
  /path/to/final_v5_dataset \
  60000
```

Arguments are:

```text
postprocess_meetingroom.sh <variant> <run-or-experiment-folder> [dataset-folder] [step]
```

If `[step]` is omitted, the wrapper evaluates the largest available `ours_<step>` folder.

## Final numbering used by the wrapper

```text
V1 = FIORD perspective / undistorted baseline
V2 = native fisheye + sparse SfM initialization
V3 = V2 + solid-angle weighting
V4 = depth replacement initialization
V5 = gap-aware hybrid SfM + depth initialization
V6 = V5 + solid-angle weighting
```

Because some dataset folders were created before this final numbering was chosen, the wrapper has legacy defaults:

```text
V4 -> MeetingRoom_V4A_REPLACE
V5/V6 -> MeetingRoom_V4_DEPTH
```

Passing the dataset explicitly is therefore preferred for final thesis runs.

## LPIPS wording for the thesis

Use wording such as:

> LPIPS was evaluated using the same valid-region mask for the rendered and reference images. Pixels outside the valid region were made identical, the pair was cropped to the bounding box of the valid region, and VGG-based LPIPS was computed on the resulting images. One value was obtained per held-out view and the mean was reported over the test set. Since LPIPS is patch based and does not natively support an arbitrary pixel mask, this protocol was applied consistently to all variants and LPIPS was not evaluated separately within angular bands.

## What must be rerun

- PSNR: the mathematical definition is unchanged, so old PSNR values should reproduce.
- SSIM: rerun final values because the border handling is now explicit and stricter.
- LPIPS: the established numerical protocol is intentionally preserved, so values should reproduce apart from library/version-level numerical differences.
