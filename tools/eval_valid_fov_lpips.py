#!/usr/bin/env python3
"""Consistent masked-crop LPIPS evaluation for MeetingRoom.

This preserves the thesis protocol used for the existing MeetingRoom results:
1. use the exact per-image validity/rig mask,
2. set invalid pixels to the same constant in prediction and ground truth,
3. crop both images to the tight valid-mask bounding box,
4. optionally downsample so the maximum dimension is at most 1024,
5. compute standard VGG LPIPS once per held-out view,
6. report mean and standard deviation across views.

Important terminology: this is a *masked-crop LPIPS protocol*. LPIPS itself is
not a native pixel-masked metric. The same protocol must therefore be used for
all compared variants.
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


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
        images.append(parts[9])

    images.sort()
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


def resize_image(image: np.ndarray, max_size: int):
    if max_size <= 0:
        return image

    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    if scale >= 1.0:
        return image

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def to_tensor(image: np.ndarray, device):
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(image))
    return tensor.permute(2, 0, 1).unsqueeze(0).to(device)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--split", type=int, default=20)
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument(
        "--mask-mode",
        choices=("required", "auto", "all-valid"),
        default="required",
    )
    args = parser.parse_args()

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
        registered = read_images(temp_dir / "images.txt")

    test_images = registered[::args.split] if args.split > 0 else registered
    render_files = sorted_render_files(render_dir, len(test_images))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        metric = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg",
            normalize=True,
        ).to(device).eval()
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize VGG LPIPS. Ensure torchmetrics/torchvision "
            "are installed and the required pretrained VGG weights are "
            "available in this environment."
        ) from exc

    print("=" * 78)
    print("MASKED-CROP VALID-FOV LPIPS")
    print("=" * 78)
    print("Data          :", data_dir)
    print("Run           :", run_dir)
    print("Step          :", args.step)
    print("Views         :", len(test_images))
    print("Split interval:", args.split)
    print("Mask mode     :", args.mask_mode)
    print("Network       : VGG")
    print("Normalize     : True ([0,1] input)")
    print("Max size      :", args.max_size)
    print("Device        :", device)
    print("Protocol      : identical invalid pixels -> tight crop -> standard LPIPS")
    print("Note          : LPIPS itself is not a native pixel-masked metric")
    print()

    scores = []

    for index, (render_path, image_name) in enumerate(
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

        prediction = cv2.cvtColor(
            prediction, cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0
        ground_truth = cv2.cvtColor(
            ground_truth, cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0

        # Preserve the established thesis protocol exactly: make invalid pixels
        # identical before cropping. The absolute number should be interpreted
        # only under this same protocol across all compared variants.
        prediction[~valid] = 0.0
        ground_truth[~valid] = 0.0

        rows, cols = np.where(valid)
        if len(rows) == 0:
            raise RuntimeError(f"Empty mask for {image_name}")

        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(cols.min()), int(cols.max()) + 1

        prediction = prediction[y0:y1, x0:x1]
        ground_truth = ground_truth[y0:y1, x0:x1]

        prediction = resize_image(prediction, args.max_size)
        ground_truth = resize_image(ground_truth, args.max_size)

        with torch.no_grad():
            score = float(
                metric(
                    to_tensor(prediction, device),
                    to_tensor(ground_truth, device),
                ).item()
            )

        scores.append(score)
        print(f"{index:03d} LPIPS={score:.6f} {image_name}")

    scores = np.asarray(scores, dtype=np.float64)

    print()
    print("=" * 78)
    print(f"OVERALL STEP {args.step}")
    print("=" * 78)
    print(f"masked-crop valid-FoV LPIPS mean: {scores.mean():.9f}")
    print(f"masked-crop valid-FoV LPIPS std : {scores.std():.9f}")


if __name__ == "__main__":
    main()
