#!/usr/bin/env python3
"""Overall full-image and valid-region PSNR/SSIM evaluation.

Designed for the FIORD MeetingRoom thesis experiments.

Protocol
--------
* Test views follow sorted COLMAP image-name order and the configured split.
* For native fisheye variants, each image's exact ``*_mask.png`` is used.
* PSNR is computed per held-out image and then averaged over views.
* Masked SSIM is obtained from the SSIM map and averaged only where the
  complete 7x7 SSIM support lies inside the valid region.
* Camera-specific means are also reported.

For a perspective baseline with no mask files, ``--mask-mode auto`` can be
used. Missing masks are then treated as all-valid. Native fisheye thesis runs
should use ``--mask-mode required``.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def psnr_from_mse(mse: float) -> float:
    if mse <= 0.0:
        return float("inf")
    return -10.0 * np.log10(mse)


def validate_ssim_window(win_size: int) -> None:
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError("--ssim-win-size must be an odd integer >= 3")


def parse_images_txt(path: Path):
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    if len(lines) % 2 != 0:
        raise RuntimeError(f"Unexpected COLMAP images.txt structure: {path}")

    records = []
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            raise RuntimeError(f"Malformed COLMAP image line: {lines[i]}")
        records.append((parts[9], int(parts[8])))

    records.sort(key=lambda x: x[0])
    return records


def locate_render_dir(run_dir: Path, step: int):
    ours_dir = run_dir / f"ours_{step}"
    render_dir = ours_dir / "renders"

    if render_dir.exists():
        return run_dir, ours_dir, render_dir

    candidates = list(run_dir.rglob(f"ours_{step}/renders"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Cannot uniquely locate renders for step {step}; "
            f"found {len(candidates)} candidates"
        )

    render_dir = candidates[0]
    ours_dir = render_dir.parent
    return ours_dir.parent, ours_dir, render_dir


def sorted_render_files(render_dir: Path, expected_count: int):
    files = list(render_dir.glob("*.png"))
    if len(files) != expected_count:
        raise RuntimeError(
            f"Render count={len(files)} but split expects {expected_count}. "
            "Check --split, --step, dataset, and run folder."
        )

    if files and all(p.stem.isdigit() for p in files):
        files.sort(key=lambda p: int(p.stem))
        indices = [int(p.stem) for p in files]
        expected = list(range(expected_count))
        if indices != expected:
            raise RuntimeError(
                "Numeric render filenames are not the expected contiguous "
                f"0..{expected_count - 1} sequence."
            )
    else:
        files.sort()

    return files


def load_mask(mask_path: Path, shape, mode: str):
    h, w = shape

    if mode == "all-valid":
        return np.ones((h, w), dtype=bool), "all-valid"

    if not mask_path.exists():
        if mode == "required":
            raise RuntimeError(f"Missing mask: {mask_path}")
        return np.ones((h, w), dtype=bool), "all-valid (mask missing)"

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        if mode == "required":
            raise RuntimeError(f"Missing mask: {mask_path}")
        return np.ones((h, w), dtype=bool), "all-valid (mask missing)"

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    valid = mask > 127
    if not valid.any():
        raise RuntimeError(f"Mask contains no valid pixels: {mask_path}")

    return valid, str(mask_path)


def erode_for_ssim(mask: np.ndarray, win_size: int) -> np.ndarray:
    # A win_size x win_size erosion removes pixels whose SSIM neighborhood
    # would cross the validity boundary. For win_size=7 this removes 3 pixels.
    kernel = np.ones((win_size, win_size), dtype=np.uint8)
    return cv2.erode(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def array_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--split", type=int, default=20)
    parser.add_argument("--image-dir", default="images")
    parser.add_argument(
        "--mask-mode",
        choices=("required", "auto", "all-valid"),
        default="required",
        help=(
            "required: every image must have *_mask.png; "
            "auto: use mask when present, otherwise all pixels; "
            "all-valid: ignore mask files"
        ),
    )
    parser.add_argument("--ssim-win-size", type=int, default=7)
    args = parser.parse_args()

    validate_ssim_window(args.ssim_win_size)

    data_dir = Path(args.data).expanduser().resolve()
    run_dir = Path(args.run).expanduser().resolve()
    gt_root = data_dir / args.image_dir

    if not gt_root.exists():
        raise RuntimeError(f"Image directory does not exist: {gt_root}")

    run_dir, ours_dir, render_dir = locate_render_dir(run_dir, args.step)

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        subprocess.run(
            [
                "colmap",
                "model_converter",
                "--input_path",
                str(data_dir / "sparse/0"),
                "--output_path",
                str(tmp),
                "--output_type",
                "TXT",
            ],
            check=True,
        )
        records = parse_images_txt(tmp / "images.txt")

    test_records = records[::args.split] if args.split > 0 else records
    render_files = sorted_render_files(render_dir, len(test_records))

    source_map = {}
    source_map_path = data_dir / "source_map.json"
    if source_map_path.exists():
        source_map = json.loads(source_map_path.read_text()).get(
            "virtual_images", {}
        )

    print("=" * 78)
    print("OVERALL PSNR / SSIM EVALUATION")
    print("=" * 78)
    print("Data             :", data_dir)
    print("Run              :", run_dir)
    print("Step             :", args.step)
    print("Registered images:", len(records))
    print("Test views       :", len(test_records))
    print("Split interval   :", args.split)
    print("Mask mode        :", args.mask_mode)
    print("SSIM window      :", args.ssim_win_size)
    print()

    rows = []

    for idx, (render_path, (image_name, camera_id)) in enumerate(
        zip(render_files, test_records)
    ):
        saved_gt = ours_dir / "gt" / render_path.name
        source_gt = gt_root / image_name
        gt_path = saved_gt if saved_gt.exists() else source_gt

        mask_path = source_gt.with_name(
            source_gt.stem + "_mask.png"
        )

        render = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
        ground_truth = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)

        if render is None:
            raise RuntimeError(f"Cannot read render: {render_path}")
        if ground_truth is None:
            raise RuntimeError(f"Cannot read GT: {gt_path}")

        h, w = render.shape[:2]
        if ground_truth.shape[:2] != (h, w):
            raise RuntimeError(
                f"GT/render size mismatch for {image_name}: "
                f"render={render.shape[:2]} gt={ground_truth.shape[:2]}"
            )

        valid, _ = load_mask(mask_path, (h, w), args.mask_mode)

        render = cv2.cvtColor(
            render, cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0
        ground_truth = cv2.cvtColor(
            ground_truth, cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0

        squared_error = (render - ground_truth) ** 2
        full_mse = float(squared_error.mean())
        masked_mse = float(squared_error[valid].mean())

        full_psnr = psnr_from_mse(full_mse)
        masked_psnr = psnr_from_mse(masked_mse)

        full_ssim, ssim_map = structural_similarity(
            ground_truth,
            render,
            channel_axis=2,
            data_range=1.0,
            win_size=args.ssim_win_size,
            full=True,
        )

        if ssim_map.ndim == 3:
            ssim_map = ssim_map.mean(axis=2)

        ssim_valid = erode_for_ssim(valid, args.ssim_win_size)
        if not ssim_valid.any():
            raise RuntimeError(
                f"No valid SSIM support remains after erosion for {image_name}."
            )

        masked_ssim = float(ssim_map[ssim_valid].mean())

        physical_camera = camera_id
        if image_name in source_map:
            physical_camera = int(
                source_map[image_name]["source_camera_id"]
            )

        rows.append(
            {
                "camera": physical_camera,
                "full_psnr": full_psnr,
                "masked_psnr": masked_psnr,
                "full_ssim": float(full_ssim),
                "masked_ssim": masked_ssim,
                "valid_fraction": float(valid.mean()),
            }
        )

        print(
            f"{idx:03d} cam{physical_camera} "
            f"fullPSNR={full_psnr:7.3f} "
            f"maskPSNR={masked_psnr:7.3f} "
            f"fullSSIM={full_ssim:.5f} "
            f"maskSSIM={masked_ssim:.5f} "
            f"valid={valid.mean():.3f} "
            f"{image_name}"
        )

    print()
    print("=" * 78)
    print(f"OVERALL STEP {args.step}")
    print("=" * 78)

    for key, label in (
        ("full_psnr", "full PSNR"),
        ("masked_psnr", "masked PSNR"),
        ("full_ssim", "full SSIM"),
        ("masked_ssim", "masked SSIM"),
    ):
        mean, std = array_stats([row[key] for row in rows])
        print(f"{label:<14}: {mean:.9f}   std={std:.9f}")

    mean_valid, _ = array_stats([row["valid_fraction"] for row in rows])
    print(f"{'valid fraction':<14}: {mean_valid:.6f}")

    for camera in sorted({row["camera"] for row in rows}):
        camera_rows = [row for row in rows if row["camera"] == camera]
        p_mean, p_std = array_stats(
            [row["masked_psnr"] for row in camera_rows]
        )
        s_mean, s_std = array_stats(
            [row["masked_ssim"] for row in camera_rows]
        )

        print()
        print(f"PHYSICAL CAMERA {camera}   N={len(camera_rows)}")
        print(f"  masked PSNR: {p_mean:.9f}   std={p_std:.9f}")
        print(f"  masked SSIM: {s_mean:.9f}   std={s_std:.9f}")


if __name__ == "__main__":
    main()
