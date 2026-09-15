#!/usr/bin/env python3
"""
stratified_dualcam_metrics.py

Comparable angular-band evaluation for V1/V2/V3/V4/V4a.

Native fisheye variants:
- parses every COLMAP camera;
- uses the correct OPENCV_FISHEYE intrinsics/distortion for each held-out image;
- bins pixels by original viewing angle;
- applies the exact dataset *_mask.png.

V1 cubemap variant:
- reads source_map.json;
- maps each virtual PINHOLE ray back to the ORIGINAL fisheye-camera frame;
- therefore 0-30, 30-60, 60-90 degree bands mean the same thing as in
  V2/V3/V4/V4a.

Reports ALL, CAM1 and CAM2.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

try:
    import torch
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except Exception:
    torch = None
    LearnedPerceptualImagePatchSimilarity = None


def parse_cameras_txt(path):
    cams = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        s = line.split()
        cams[int(s[0])] = {
            "model": s[1],
            "width": int(s[2]),
            "height": int(s[3]),
            "params": list(map(float, s[4:])),
        }
    return cams


def parse_images_txt(path):
    lines = [
        x.strip()
        for x in Path(path).read_text().splitlines()
        if x.strip() and not x.startswith("#")
    ]
    records = []
    for i in range(0, len(lines), 2):
        s = lines[i].split()
        if len(s) >= 10:
            records.append((s[9], int(s[8])))
    records.sort(key=lambda x: x[0])
    return records


def fisheye_theta_map(H, W, cam):
    p = cam["params"]
    fx, fy, cx, cy, k1, k2, k3, k4 = p[:8]

    sx = W / cam["width"]
    sy = H / cam["height"]

    fx *= sx
    cx *= sx
    fy *= sy
    cy *= sy

    u = np.arange(W, dtype=np.float64) + 0.5
    v = np.arange(H, dtype=np.float64) + 0.5
    vv, uu = np.meshgrid(v, u, indexing="ij")

    rd = np.sqrt(
        ((uu - cx) / fx) ** 2 +
        ((vv - cy) / fy) ** 2
    )

    theta = np.clip(rd.copy(), 0.0, np.deg2rad(179.0))

    for _ in range(12):
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4

        f = theta * (
            1.0 + k1*t2 + k2*t4 + k3*t6 + k4*t8
        ) - rd

        fp = (
            1.0
            + 3.0*k1*t2
            + 5.0*k2*t4
            + 7.0*k3*t6
            + 9.0*k4*t8
        )

        fp = np.where(np.abs(fp) < 1e-8, 1e-8, fp)
        step = np.clip(f / fp, -0.25, 0.25)
        theta = np.clip(theta - step, 0.0, np.deg2rad(179.0))

    return np.degrees(theta)


def v1_original_theta_map(H, W, cam, source_meta):
    if cam["model"] != "PINHOLE":
        raise RuntimeError(
            f"V1 source_map expects PINHOLE virtual camera, got {cam['model']}"
        )

    fx, fy, cx, cy = cam["params"][:4]

    sx = W / cam["width"]
    sy = H / cam["height"]

    fx *= sx
    cx *= sx
    fy *= sy
    cy *= sy

    u = np.arange(W, dtype=np.float64) + 0.5
    v = np.arange(H, dtype=np.float64) + 0.5
    vv, uu = np.meshgrid(v, u, indexing="ij")

    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x)

    A = np.asarray(
        source_meta["A_virtual_to_original"],
        dtype=np.float64,
    )

    ox = A[0,0]*x + A[0,1]*y + A[0,2]*z
    oy = A[1,0]*x + A[1,1]*y + A[1,2]*z
    oz = A[2,0]*x + A[2,1]*y + A[2,2]*z

    return np.degrees(
        np.arctan2(np.sqrt(ox*ox + oy*oy), oz)
    )


def resize_for_lpips(arr, max_dim):
    if max_dim <= 0:
        return arr

    h, w = arr.shape[:2]
    m = max(h, w)

    if m <= max_dim:
        return arr

    scale = max_dim / m
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    return cv2.resize(
        arr, (nw, nh),
        interpolation=cv2.INTER_AREA,
    )


def to_torch(arr):
    return torch.from_numpy(
        np.ascontiguousarray(arr)
    ).permute(2, 0, 1).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--split", type=int, default=20)
    ap.add_argument("--image-dir", default="images")

    ap.add_argument(
        "--band-edges",
        nargs="+",
        type=float,
        default=[0.0, 30.0, 60.0, 90.0],
    )
    ap.add_argument(
        "--band-names",
        nargs="+",
        default=["center", "mid", "periphery"],
    )

    ap.add_argument("--skip-lpips", action="store_true")
    ap.add_argument("--lpips-max-dim", type=int, default=1024)

    args = ap.parse_args()

    if len(args.band_names) != len(args.band_edges) - 1:
        raise SystemExit(
            "--band-names must have exactly len(band_edges)-1 values"
        )

    DATA = Path(args.data).expanduser().resolve()
    RUN = Path(args.run).expanduser().resolve()
    GT_ROOT = DATA / args.image_dir

    ours = RUN / f"ours_{args.step}"
    renders_dir = ours / "renders"

    if not renders_dir.exists():
        candidates = list(RUN.rglob(f"ours_{args.step}/renders"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Could not uniquely find step {args.step} renders; "
                f"found {len(candidates)}"
            )
        renders_dir = candidates[0]
        ours = renders_dir.parent
        RUN = ours.parent

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        subprocess.run(
            [
                "colmap", "model_converter",
                "--input_path", str(DATA / "sparse/0"),
                "--output_path", str(tmp),
                "--output_type", "TXT",
            ],
            check=True,
        )

        cams = parse_cameras_txt(tmp / "cameras.txt")
        records = parse_images_txt(tmp / "images.txt")

    test = records[::args.split] if args.split > 0 else records
    render_files = sorted(renders_dir.glob("*.png"))

    if len(render_files) != len(test):
        raise RuntimeError(
            f"Render count={len(render_files)} but split expects {len(test)}"
        )

    source_map = {}
    sm_path = DATA / "source_map.json"
    if sm_path.exists():
        source_map = json.loads(
            sm_path.read_text()
        ).get("virtual_images", {})

    lpips_metric = None
    if not args.skip_lpips:
        if torch is None or LearnedPerceptualImagePatchSimilarity is None:
            raise RuntimeError(
                "torchmetrics LPIPS unavailable; use --skip-lpips or install torchmetrics"
            )
        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg",
            normalize=True,
        ).cpu().eval()

    physical_ids = set()
    theta_cache = {}

    data = {name: {} for name in args.band_names}

    def ensure_group(group):
        for band in args.band_names:
            if group not in data[band]:
                data[band][group] = {
                    "psnr": [],
                    "ssim": [],
                    "lpips": [],
                    "valid_frac": [],
                }

    ensure_group("all")

    print("=" * 80)
    print("STRATIFIED ANGULAR EVALUATION")
    print("=" * 80)
    print("Data      :", DATA)
    print("Run       :", RUN)
    print("Step      :", args.step)
    print("Views     :", len(test))
    print("Band edges:", args.band_edges)
    print(
        "Projection:",
        "V1 cubemap/original-fisheye-angle"
        if source_map
        else "native fisheye"
    )
    print()

    for idx, (rp, (name, virtual_cam_id)) in enumerate(
        zip(render_files, test)
    ):
        gp_saved = ours / "gt" / rp.name
        gp = gp_saved if gp_saved.exists() else GT_ROOT / name

        mp = (GT_ROOT / name).with_name(
            Path(name).stem + "_mask.png"
        )

        pred_bgr = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        gt_bgr = cv2.imread(str(gp), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)

        if pred_bgr is None:
            raise RuntimeError(f"Cannot read {rp}")
        if gt_bgr is None:
            raise RuntimeError(f"Cannot read {gp}")
        if mask is None:
            raise RuntimeError(f"Cannot read {mp}")

        H, W = pred_bgr.shape[:2]

        if gt_bgr.shape[:2] != (H, W):
            raise RuntimeError(f"GT/render shape mismatch for {name}")

        if mask.shape[:2] != (H, W):
            mask = cv2.resize(
                mask, (W, H),
                interpolation=cv2.INTER_NEAREST,
            )

        if name in source_map:
            meta = source_map[name]
            physical_cam = int(meta["source_camera_id"])
            cache_key = ("v1", virtual_cam_id, meta["face"], H, W)

            if cache_key not in theta_cache:
                theta_cache[cache_key] = v1_original_theta_map(
                    H, W,
                    cams[virtual_cam_id],
                    meta,
                )
        else:
            physical_cam = virtual_cam_id
            cache_key = ("fisheye", virtual_cam_id, H, W)

            if cache_key not in theta_cache:
                cam = cams[virtual_cam_id]
                if cam["model"] != "OPENCV_FISHEYE":
                    raise RuntimeError(
                        f"Expected OPENCV_FISHEYE, got {cam['model']} for {name}"
                    )
                theta_cache[cache_key] = fisheye_theta_map(H, W, cam)

        theta = theta_cache[cache_key]

        physical_ids.add(physical_cam)
        ensure_group(f"cam{physical_cam}")

        pred = cv2.cvtColor(
            pred_bgr, cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0

        gt = cv2.cvtColor(
            gt_bgr, cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0

        valid = mask > 127

        _, ssim_map = structural_similarity(
            gt, pred,
            channel_axis=2,
            data_range=1.0,
            full=True,
        )

        if ssim_map.ndim == 3:
            ssim_map = ssim_map.mean(axis=2)

        diff2 = (pred - gt) ** 2

        print(
            f"{idx:03d} physical_cam{physical_cam} "
            f"{rp.name} <- {name}"
        )

        for bi, band in enumerate(args.band_names):
            lo = args.band_edges[bi]
            hi = args.band_edges[bi + 1]

            angular = (theta >= lo) & (theta < hi)
            band_mask = angular & valid

            angular_count = int(angular.sum())
            n = int(band_mask.sum())

            if n == 0:
                continue

            valid_frac = n / angular_count if angular_count > 0 else 0.0

            mse = float(diff2[band_mask].mean())
            psnr = -10.0 * np.log10(max(mse, 1e-12))

            eroded = cv2.erode(
                band_mask.astype(np.uint8),
                np.ones((7, 7), np.uint8),
                iterations=1,
            ).astype(bool)

            ssim = float(
                ssim_map[eroded].mean()
                if eroded.any()
                else ssim_map[band_mask].mean()
            )

            lpips = None

            if lpips_metric is not None:
                composite = gt.copy()
                composite[band_mask] = pred[band_mask]

                composite = resize_for_lpips(
                    composite,
                    args.lpips_max_dim,
                )
                gt_small = resize_for_lpips(
                    gt,
                    args.lpips_max_dim,
                )

                with torch.no_grad():
                    lpips = float(
                        lpips_metric(
                            to_torch(composite).cpu(),
                            to_torch(gt_small).cpu(),
                        ).item()
                    )

            for group in ("all", f"cam{physical_cam}"):
                d = data[band][group]
                d["psnr"].append(psnr)
                d["ssim"].append(ssim)
                d["valid_frac"].append(valid_frac)
                if lpips is not None:
                    d["lpips"].append(lpips)

    def print_group(group):
        print()
        print("=" * 92)
        print(group.upper())
        print("=" * 92)

        print(
            f"{'Band':<14}"
            f"{'N':<6}"
            f"{'PSNR':<11}"
            f"{'std':<10}"
            f"{'SSIM':<11}"
            f"{'LPIPS':<11}"
            f"{'valid_frac':<12}"
        )
        print("-" * 92)

        for band in args.band_names:
            d = data[band].get(group)

            if not d or not d["psnr"]:
                print(f"{band:<14}0     -- no valid pixels --")
                continue

            p = np.asarray(d["psnr"], dtype=float)
            s = np.asarray(d["ssim"], dtype=float)
            l = np.asarray(d["lpips"], dtype=float)
            vf = np.asarray(d["valid_frac"], dtype=float)

            lpips_text = f"{l.mean():.4f}" if len(l) else "--"

            print(
                f"{band:<14}"
                f"{len(p):<6}"
                f"{p.mean():<11.3f}"
                f"{p.std():<10.3f}"
                f"{s.mean():<11.4f}"
                f"{lpips_text:<11}"
                f"{vf.mean():<12.3f}"
            )

    print_group("all")
    for cid in sorted(physical_ids):
        print_group(f"cam{cid}")


if __name__ == "__main__":
    main()
