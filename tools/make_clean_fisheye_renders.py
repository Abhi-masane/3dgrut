#!/usr/bin/env python3
"""Create clean masked fisheye renders for figures and visual inspection.

This script is NOT used to calculate thesis metrics. Metric scripts read the
original renders and masks directly.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def theta_d(theta, k):
    t2 = theta * theta
    return theta * (
        1.0
        + k[0] * t2
        + k[1] * t2**2
        + k[2] * t2**3
        + k[3] * t2**4
    )


def sorted_render_files(render_dir: Path, expected_count: int):
    files = list(render_dir.glob("*.png"))
    if len(files) != expected_count:
        raise RuntimeError(
            f"Found {len(files)} renders but expected {expected_count} test views"
        )

    if files and all(p.stem.isdigit() for p in files):
        files.sort(key=lambda p: int(p.stem))
        indices = [int(p.stem) for p in files]
        if indices != list(range(expected_count)):
            raise RuntimeError("Render indices are not contiguous from zero")
    else:
        files.sort()
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--split", type=int, default=20)
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--fov", type=float, default=180.0)
    args = parser.parse_args()

    data_dir = Path(args.data).expanduser().resolve()
    run_dir = Path(args.run).expanduser().resolve()
    base = run_dir / f"ours_{args.step}"
    render_dir = base / "renders"

    if not render_dir.exists():
        raise RuntimeError(f"Render folder does not exist: {render_dir}")

    gt_root = data_dir / args.image_dir
    if not gt_root.exists():
        raise RuntimeError(f"Image directory does not exist: {gt_root}")

    masked_out = base / "renders_masked"
    fov_out = base / f"renders_fov{int(args.fov)}"
    masked_out.mkdir(parents=True, exist_ok=True)
    fov_out.mkdir(parents=True, exist_ok=True)

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

        cameras = {}
        for line in (tmp / "cameras.txt").read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            if model != "OPENCV_FISHEYE":
                raise RuntimeError(
                    f"Camera {camera_id}: expected OPENCV_FISHEYE, got {model}"
                )
            if len(params) != 8:
                raise RuntimeError(
                    f"Camera {camera_id}: expected 8 OPENCV_FISHEYE parameters"
                )

            cameras[camera_id] = {
                "width": width,
                "height": height,
                "fx": params[0],
                "fy": params[1],
                "cx": params[2],
                "cy": params[3],
                "k": params[4:8],
            }

        image_lines = [
            line.strip()
            for line in (tmp / "images.txt").read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if len(image_lines) % 2 != 0:
            raise RuntimeError("Unexpected COLMAP images.txt structure")

        records = []
        for i in range(0, len(image_lines), 2):
            parts = image_lines[i].split()
            records.append((parts[9], int(parts[8])))
        records.sort(key=lambda x: x[0])

    test_records = records[::args.split] if args.split > 0 else records
    render_files = sorted_render_files(render_dir, len(test_records))
    theta_max = np.deg2rad(args.fov / 2.0)

    print("=" * 78)
    print("CLEAN FISHEYE RENDER EXPORT")
    print("=" * 78)
    print("Visualization only. Do not compute thesis metrics from these folders.")
    print("Data       :", data_dir)
    print("Run        :", run_dir)
    print("Step       :", args.step)
    print("Image dir  :", args.image_dir)
    print("Test views :", len(test_records))
    print("FoV        :", args.fov)
    print()

    for idx, (render_path, (name, camera_id)) in enumerate(
        zip(render_files, test_records)
    ):
        render = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
        if render is None:
            raise RuntimeError(f"Cannot read render: {render_path}")

        h, w = render.shape[:2]

        source_image = gt_root / name
        mask_path = source_image.with_name(source_image.stem + "_mask.png")
        training_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if training_mask is None:
            raise RuntimeError(f"Missing evaluation mask: {mask_path}")
        if training_mask.shape[:2] != (h, w):
            training_mask = cv2.resize(
                training_mask,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        metric_safe = render.copy()
        metric_safe[training_mask <= 127] = 0
        cv2.imwrite(str(masked_out / render_path.name), metric_safe)

        camera = cameras[camera_id]
        sx = w / camera["width"]
        sy = h / camera["height"]
        fx = camera["fx"] * sx
        fy = camera["fy"] * sy
        cx = camera["cx"] * sx
        cy = camera["cy"] * sy

        rd_max = theta_d(theta_max, camera["k"])
        yy, xx = np.mgrid[0:h, 0:w]
        xx = xx + 0.5
        yy = yy + 0.5
        rd = np.sqrt(
            ((xx - cx) / fx) ** 2
            + ((yy - cy) / fy) ** 2
        )
        fov_mask = rd <= rd_max

        fov_render = render.copy()
        fov_render[~fov_mask] = 0
        cv2.imwrite(str(fov_out / render_path.name), fov_render)

        print(f"{idx:03d} cam{camera_id} {render_path.name} <- {name}")

    print()
    print("DONE")
    print("Exact training/evaluation-mask visualization:", masked_out)
    print("Pure calibrated-FoV visualization          :", fov_out)


if __name__ == "__main__":
    main()
