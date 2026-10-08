#!/usr/bin/env python3
"""Angular PSNR/SSIM evaluation for MeetingRoom held-out views.

Default angular bands are 0-30, 30-60, and 60-90 degrees from the optical
axis. Native fisheye angles are obtained by inverting the calibrated
OPENCV_FISHEYE polynomial. A PINHOLE/SIMPLE_PINHOLE baseline is also supported
for completeness, although bands outside its actual image FoV will naturally
contain no pixels.

LPIPS is intentionally not computed per angular band.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def validate_ssim_window(win_size: int) -> None:
    if win_size < 3 or win_size % 2 == 0:
        raise ValueError("--ssim-win-size must be an odd integer >= 3")


def read_cameras(path: Path):
    cameras = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cameras[int(parts[0])] = {
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": list(map(float, parts[4:])),
        }
    return cameras


def read_images(path: Path):
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    if len(lines) % 2 != 0:
        raise RuntimeError(f"Unexpected COLMAP images.txt structure: {path}")

    images = []
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        if len(parts) < 10:
            raise RuntimeError(f"Malformed COLMAP image line: {lines[i]}")
        images.append((parts[9], int(parts[8])))

    images.sort(key=lambda x: x[0])
    return images


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
            f"Found {len(files)} renders but expected {expected_count} test views"
        )

    if files and all(p.stem.isdigit() for p in files):
        files.sort(key=lambda p: int(p.stem))
        indices = [int(p.stem) for p in files]
        if indices != list(range(expected_count)):
            raise RuntimeError(
                "Numeric render filenames are not the expected contiguous "
                f"0..{expected_count - 1} sequence."
            )
    else:
        files.sort()

    return files


def scaled_intrinsics(height, width, camera):
    sx = width / camera["width"]
    sy = height / camera["height"]
    return sx, sy


def fisheye_angles(height, width, camera):
    if camera["model"] != "OPENCV_FISHEYE":
        raise RuntimeError(
            f"Expected OPENCV_FISHEYE, got {camera['model']}"
        )
    if len(camera["params"]) < 8:
        raise RuntimeError("OPENCV_FISHEYE camera requires 8 parameters")

    fx, fy, cx, cy, k1, k2, k3, k4 = camera["params"][:8]
    sx, sy = scaled_intrinsics(height, width, camera)
    fx, cx = fx * sx, cx * sx
    fy, cy = fy * sy, cy * sy

    u = np.arange(width, dtype=np.float64) + 0.5
    v = np.arange(height, dtype=np.float64) + 0.5
    vv, uu = np.meshgrid(v, u, indexing="ij")

    rd = np.sqrt(
        ((uu - cx) / fx) ** 2
        + ((vv - cy) / fy) ** 2
    )

    # rd = theta_d = theta * (1 + k1 theta^2 + ... + k4 theta^8)
    # Invert with bounded Newton iterations.
    theta = np.clip(rd.copy(), 0.0, np.deg2rad(179.0))

    for _ in range(12):
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4

        f = theta * (
            1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8
        ) - rd
        df = (
            1.0
            + 3.0 * k1 * t2
            + 5.0 * k2 * t4
            + 7.0 * k3 * t6
            + 9.0 * k4 * t8
        )
        df = np.where(np.abs(df) < 1e-8, 1e-8, df)
        step = np.clip(f / df, -0.25, 0.25)
        theta = np.clip(
            theta - step,
            0.0,
            np.deg2rad(179.0),
        )

    return np.degrees(theta)


def pinhole_angles(height, width, camera):
    model = camera["model"]
    params = camera["params"]

    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        fx = fy = f
    else:
        raise RuntimeError(
            f"Angular evaluation supports OPENCV_FISHEYE, PINHOLE, and "
            f"SIMPLE_PINHOLE; got {model}"
        )

    sx, sy = scaled_intrinsics(height, width, camera)
    fx, cx = fx * sx, cx * sx
    fy, cy = fy * sy, cy * sy

    u = np.arange(width, dtype=np.float64) + 0.5
    v = np.arange(height, dtype=np.float64) + 0.5
    vv, uu = np.meshgrid(v, u, indexing="ij")

    x = (uu - cx) / fx
    y = (vv - cy) / fy
    return np.degrees(np.arctan(np.sqrt(x * x + y * y)))


def virtual_to_original_angles(height, width, camera, source_info):
    if camera["model"] not in {"PINHOLE", "SIMPLE_PINHOLE"}:
        raise RuntimeError(
            "source_map angular mapping expects a perspective virtual camera; "
            f"got {camera['model']}"
        )

    model = camera["model"]
    params = camera["params"]
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    else:
        f, cx, cy = params[:3]
        fx = fy = f

    sx, sy = scaled_intrinsics(height, width, camera)
    fx, cx = fx * sx, cx * sx
    fy, cy = fy * sy, cy * sy

    u = np.arange(width, dtype=np.float64) + 0.5
    v = np.arange(height, dtype=np.float64) + 0.5
    vv, uu = np.meshgrid(v, u, indexing="ij")

    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x)

    rotation = np.asarray(
        source_info["A_virtual_to_original"], dtype=np.float64
    )
    if rotation.shape != (3, 3):
        raise RuntimeError("A_virtual_to_original must be a 3x3 matrix")

    ox = rotation[0, 0] * x + rotation[0, 1] * y + rotation[0, 2] * z
    oy = rotation[1, 0] * x + rotation[1, 1] * y + rotation[1, 2] * z
    oz = rotation[2, 0] * x + rotation[2, 1] * y + rotation[2, 2] * z

    return np.degrees(np.arctan2(np.sqrt(ox * ox + oy * oy), oz))


def load_mask(mask_path: Path, shape, mode: str):
    h, w = shape
    if mode == "all-valid":
        return np.ones((h, w), dtype=bool)

    if not mask_path.exists():
        if mode == "required":
            raise RuntimeError(f"Missing mask: {mask_path}")
        return np.ones((h, w), dtype=bool)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        if mode == "required":
            raise RuntimeError(f"Cannot read mask: {mask_path}")
        return np.ones((h, w), dtype=bool)

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

    valid = mask > 127
    if not valid.any():
        raise RuntimeError(f"Mask contains no valid pixels: {mask_path}")
    return valid


def erode_for_ssim(mask: np.ndarray, win_size: int):
    kernel = np.ones((win_size, win_size), dtype=np.uint8)
    return cv2.erode(
        mask.astype(np.uint8),
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def parse_band_edges(text: str):
    try:
        edges = [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError("--band-edges must contain comma-separated numbers") from exc

    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError("--band-edges must be strictly increasing")
    return edges


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
    )
    parser.add_argument("--ssim-win-size", type=int, default=7)
    parser.add_argument(
        "--band-edges",
        default="0,30,60,90",
        help="Comma-separated angular edges in degrees",
    )
    args = parser.parse_args()

    validate_ssim_window(args.ssim_win_size)
    band_edges = parse_band_edges(args.band_edges)
    default_names = ["center", "mid", "periphery"]
    if band_edges == [0.0, 30.0, 60.0, 90.0]:
        band_names = default_names
    else:
        band_names = [
            f"{band_edges[i]:g}-{band_edges[i + 1]:g}deg"
            for i in range(len(band_edges) - 1)
        ]

    data_dir = Path(args.data).expanduser().resolve()
    run_dir = Path(args.run).expanduser().resolve()
    gt_dir = data_dir / args.image_dir

    if not gt_dir.exists():
        raise RuntimeError(f"Image directory does not exist: {gt_dir}")

    run_dir, ours_dir, render_dir = locate_render_dir(run_dir, args.step)

    with tempfile.TemporaryDirectory() as temp_name:
        temp_dir = Path(temp_name)
        subprocess.run(
            [
                "colmap",
                "model_converter",
                "--input_path",
                str(data_dir / "sparse/0"),
                "--output_path",
                str(temp_dir),
                "--output_type",
                "TXT",
            ],
            check=True,
        )
        cameras = read_cameras(temp_dir / "cameras.txt")
        registered = read_images(temp_dir / "images.txt")

    test_images = registered[::args.split] if args.split > 0 else registered
    render_files = sorted_render_files(render_dir, len(test_images))

    source_map = {}
    source_map_file = data_dir / "source_map.json"
    if source_map_file.exists():
        source_map = json.loads(source_map_file.read_text()).get(
            "virtual_images", {}
        )

    results = {band: {} for band in band_names}
    angle_cache = {}
    physical_cameras = set()

    def add_group(group):
        for band in band_names:
            results[band].setdefault(
                group,
                {"psnr": [], "ssim": [], "valid_fraction": []},
            )

    add_group("all")

    print("=" * 86)
    print("ANGULAR PSNR / SSIM EVALUATION")
    print("=" * 86)
    print("Data          :", data_dir)
    print("Run           :", run_dir)
    print("Step          :", args.step)
    print("Views         :", len(test_images))
    print("Split interval:", args.split)
    print("Mask mode     :", args.mask_mode)
    print("SSIM window   :", args.ssim_win_size)
    print("Band edges    :", band_edges)
    print()

    for index, (render_path, (image_name, camera_id)) in enumerate(
        zip(render_files, test_images)
    ):
        saved_gt = ours_dir / "gt" / render_path.name
        source_gt = gt_dir / image_name
        gt_path = saved_gt if saved_gt.exists() else source_gt
        mask_path = source_gt.with_name(source_gt.stem + "_mask.png")

        prediction = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
        ground_truth = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)

        if prediction is None:
            raise RuntimeError(f"Cannot read render: {render_path}")
        if ground_truth is None:
            raise RuntimeError(f"Cannot read GT: {gt_path}")

        height, width = prediction.shape[:2]
        if ground_truth.shape[:2] != (height, width):
            raise RuntimeError(
                f"GT/render size mismatch for {image_name}: "
                f"render={prediction.shape[:2]} gt={ground_truth.shape[:2]}"
            )

        valid = load_mask(mask_path, (height, width), args.mask_mode)
        camera = cameras[camera_id]

        if image_name in source_map:
            info = source_map[image_name]
            physical_camera = int(info["source_camera_id"])
            cache_key = (
                "virtual",
                camera_id,
                info.get("face", image_name),
                height,
                width,
            )
            if cache_key not in angle_cache:
                angle_cache[cache_key] = virtual_to_original_angles(
                    height, width, camera, info
                )
        elif camera["model"] == "OPENCV_FISHEYE":
            physical_camera = camera_id
            cache_key = ("fisheye", camera_id, height, width)
            if cache_key not in angle_cache:
                angle_cache[cache_key] = fisheye_angles(
                    height, width, camera
                )
        elif camera["model"] in {"PINHOLE", "SIMPLE_PINHOLE"}:
            physical_camera = camera_id
            cache_key = ("pinhole", camera_id, height, width)
            if cache_key not in angle_cache:
                angle_cache[cache_key] = pinhole_angles(
                    height, width, camera
                )
        else:
            raise RuntimeError(
                f"Unsupported camera model for angular evaluation: "
                f"{camera['model']} ({image_name})"
            )

        theta = angle_cache[cache_key]
        physical_cameras.add(physical_camera)
        add_group(f"cam{physical_camera}")

        prediction = cv2.cvtColor(
            prediction, cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0
        ground_truth = cv2.cvtColor(
            ground_truth, cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0

        _, ssim_map = structural_similarity(
            ground_truth,
            prediction,
            channel_axis=2,
            data_range=1.0,
            win_size=args.ssim_win_size,
            full=True,
        )
        if ssim_map.ndim == 3:
            ssim_map = ssim_map.mean(axis=2)

        squared_error = (prediction - ground_truth) ** 2

        print(
            f"{index:03d} cam{physical_camera} "
            f"{render_path.name} <- {image_name} "
            f"model={camera['model']}"
        )

        for band_index, band in enumerate(band_names):
            lower = band_edges[band_index]
            upper = band_edges[band_index + 1]

            angular_region = (theta >= lower) & (theta < upper)
            evaluation_mask = angular_region & valid

            total_pixels = int(angular_region.sum())
            valid_pixels = int(evaluation_mask.sum())

            if valid_pixels == 0:
                continue

            mse = float(squared_error[evaluation_mask].mean())
            psnr = -10.0 * np.log10(max(mse, 1e-12))

            ssim_valid = erode_for_ssim(
                evaluation_mask, args.ssim_win_size
            )
            if not ssim_valid.any():
                raise RuntimeError(
                    f"No valid SSIM support for {image_name}, band {band}."
                )
            ssim = float(ssim_map[ssim_valid].mean())

            valid_fraction = (
                valid_pixels / total_pixels if total_pixels > 0 else 0.0
            )

            for group in ("all", f"cam{physical_camera}"):
                values = results[band][group]
                values["psnr"].append(psnr)
                values["ssim"].append(ssim)
                values["valid_fraction"].append(valid_fraction)

    def print_results(group):
        print()
        print("=" * 86)
        print(group.upper())
        print("=" * 86)
        print(
            f"{'Band':<16}"
            f"{'N':<6}"
            f"{'PSNR':<11}"
            f"{'PSNR std':<12}"
            f"{'SSIM':<11}"
            f"{'SSIM std':<12}"
            f"{'valid_frac':<12}"
        )
        print("-" * 86)

        for band in band_names:
            values = results[band].get(group)
            if not values or not values["psnr"]:
                print(f"{band:<16}{0:<6}{'no pixels in this band'}")
                continue

            psnr = np.asarray(values["psnr"], dtype=float)
            ssim = np.asarray(values["ssim"], dtype=float)
            valid_fraction = np.asarray(
                values["valid_fraction"], dtype=float
            )

            print(
                f"{band:<16}"
                f"{len(psnr):<6}"
                f"{psnr.mean():<11.3f}"
                f"{psnr.std():<12.3f}"
                f"{ssim.mean():<11.4f}"
                f"{ssim.std():<12.4f}"
                f"{valid_fraction.mean():<12.3f}"
            )

    print_results("all")
    for camera_id in sorted(physical_cameras):
        print_results(f"cam{camera_id}")


if __name__ == "__main__":
    main()
