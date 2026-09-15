#!/usr/bin/env python3
"""
eval_masked_multicam.py

Mask-aware evaluator for MeetingRoom V1/V2/V3/V4/V4a.

- Uses the same sorted COLMAP image-name ordering + split interval as the
  MeetingRoom evaluation workflow.
- Uses each image's exact *_mask.png.
- Reports full and masked PSNR / SSIM.
- Reports masked metrics separately for physical cam1 and cam2.
- For V1 cubemap data, source_map.json maps each virtual perspective view back
  to its original physical fisheye camera.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")
    return -10.0 * np.log10(mse)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--split", type=int, default=20)
    ap.add_argument("--image-dir", default="images")
    args = ap.parse_args()

    DATA = Path(args.data).expanduser().resolve()
    RUN = Path(args.run).expanduser().resolve()
    GT_ROOT = DATA / args.image_dir

    ours = RUN / f"ours_{args.step}"
    renders = ours / "renders"

    if not renders.exists():
        candidates = list(RUN.rglob(f"ours_{args.step}/renders"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Cannot uniquely locate renders for step {args.step}; "
                f"found {len(candidates)} candidates"
            )
        renders = candidates[0]
        ours = renders.parent
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

        records = parse_images_txt(tmp / "images.txt")

    test = records[::args.split] if args.split > 0 else records

    render_files = sorted(renders.glob("*.png"))

    if len(render_files) != len(test):
        raise RuntimeError(
            f"Render count={len(render_files)} but split expects {len(test)}. "
            f"Check --split and run."
        )

    source_map = {}
    source_map_path = DATA / "source_map.json"

    if source_map_path.exists():
        source_map = json.loads(
            source_map_path.read_text()
        ).get("virtual_images", {})

    print("Registered images :", len(records))
    print("Test views        :", len(test))
    print("Run               :", RUN)
    print()

    rows = []

    for idx, (rp, (name, camera_id)) in enumerate(
        zip(render_files, test)
    ):
        gp_saved = ours / "gt" / rp.name
        gp = gp_saved if gp_saved.exists() else GT_ROOT / name

        mp = (GT_ROOT / name).with_name(
            Path(name).stem + "_mask.png"
        )

        r = cv2.imread(str(rp), cv2.IMREAD_COLOR)
        g = cv2.imread(str(gp), cv2.IMREAD_COLOR)
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)

        if r is None:
            raise RuntimeError(f"Missing render: {rp}")
        if g is None:
            raise RuntimeError(f"Missing GT: {gp}")
        if m is None:
            raise RuntimeError(f"Missing mask: {mp}")

        h, w = r.shape[:2]

        if g.shape[:2] != (h, w):
            raise RuntimeError(
                f"GT/render size mismatch: {name}: "
                f"render={r.shape[:2]} gt={g.shape[:2]}"
            )

        if m.shape[:2] != (h, w):
            m = cv2.resize(
                m, (w, h),
                interpolation=cv2.INTER_NEAREST
            )

        r = cv2.cvtColor(
            r, cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0

        g = cv2.cvtColor(
            g, cv2.COLOR_BGR2RGB
        ).astype(np.float32) / 255.0

        valid = m > 127
        diff2 = (r - g) ** 2

        full_mse = float(diff2.mean())
        masked_mse = float(diff2[valid].mean())

        full_psnr = psnr_from_mse(full_mse)
        masked_psnr = psnr_from_mse(masked_mse)

        full_ssim, smap = structural_similarity(
            g, r,
            channel_axis=2,
            data_range=1.0,
            full=True,
        )

        if smap.ndim == 3:
            smap = smap.mean(axis=2)

        valid_eroded = cv2.erode(
            valid.astype(np.uint8),
            np.ones((7, 7), np.uint8),
            iterations=1,
        ).astype(bool)

        masked_ssim = float(
            smap[valid_eroded].mean()
            if valid_eroded.any()
            else smap[valid].mean()
        )

        physical_cam = camera_id
        if name in source_map:
            physical_cam = int(
                source_map[name]["source_camera_id"]
            )

        rows.append({
            "camera": physical_cam,
            "full_psnr": full_psnr,
            "masked_psnr": masked_psnr,
            "full_ssim": float(full_ssim),
            "masked_ssim": masked_ssim,
        })

        print(
            f"{idx:03d} cam{physical_cam} "
            f"fullPSNR={full_psnr:6.2f} "
            f"maskPSNR={masked_psnr:6.2f} "
            f"fullSSIM={full_ssim:.4f} "
            f"maskSSIM={masked_ssim:.4f} "
            f"{name}"
        )

    print()
    print("=" * 72)
    print(f"OVERALL STEP {args.step}")
    print("=" * 72)
    print("full PSNR   :", np.mean([x["full_psnr"] for x in rows]))
    print("masked PSNR :", np.mean([x["masked_psnr"] for x in rows]))
    print("full SSIM   :", np.mean([x["full_ssim"] for x in rows]))
    print("masked SSIM :", np.mean([x["masked_ssim"] for x in rows]))

    for cam in sorted(set(x["camera"] for x in rows)):
        rr = [x for x in rows if x["camera"] == cam]
        print()
        print(f"PHYSICAL CAMERA {cam}")
        print(
            "  masked PSNR:",
            np.mean([x["masked_psnr"] for x in rr])
        )
        print(
            "  masked SSIM:",
            np.mean([x["masked_ssim"] for x in rr])
        )


if __name__ == "__main__":
    main()
