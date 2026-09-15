#!/usr/bin/env python3

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--split", type=int, default=20)
    parser.add_argument("--fov", type=float, default=180.0)

    args = parser.parse_args()

    DATA = Path(args.data).expanduser().resolve()
    RUN = Path(args.run).expanduser().resolve()

    base = RUN / f"ours_{args.step}"

    renders = base / "renders"

    if not renders.exists():
        raise RuntimeError(
            f"Render folder does not exist:\n{renders}"
        )

    # -------------------------------------------------------
    # Output folders
    # -------------------------------------------------------

    masked_out = base / "renders_masked"
    fov_out = base / f"renders_fov{int(args.fov)}"

    masked_out.mkdir(parents=True, exist_ok=True)
    fov_out.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------
    # Convert COLMAP model to TXT temporarily
    # -------------------------------------------------------

    with tempfile.TemporaryDirectory() as tmpdir:

        tmp = Path(tmpdir)

        subprocess.run(
            [
                "colmap",
                "model_converter",
                "--input_path",
                str(DATA / "sparse/0"),
                "--output_path",
                str(tmp),
                "--output_type",
                "TXT",
            ],
            check=True,
        )

        camera_txt = tmp / "cameras.txt"
        image_txt = tmp / "images.txt"

        # ---------------------------------------------------
        # Cameras
        # ---------------------------------------------------

        cameras = {}

        for line in camera_txt.read_text().splitlines():

            line = line.strip()

            if not line or line.startswith("#"):
                continue

            s = line.split()

            cid = int(s[0])
            model = s[1]
            W = int(s[2])
            H = int(s[3])
            params = list(map(float, s[4:]))

            if model != "OPENCV_FISHEYE":
                raise RuntimeError(
                    f"Camera {cid}: expected OPENCV_FISHEYE, got {model}"
                )

            if len(params) != 8:
                raise RuntimeError(
                    f"Camera {cid}: expected 8 parameters"
                )

            cameras[cid] = {
                "W": W,
                "H": H,
                "fx": params[0],
                "fy": params[1],
                "cx": params[2],
                "cy": params[3],
                "k": params[4:8],
            }

        # ---------------------------------------------------
        # Registered image list
        # ---------------------------------------------------

        lines = [
            x.strip()
            for x in image_txt.read_text().splitlines()
            if x.strip() and not x.startswith("#")
        ]

        records = []

        for i in range(0, len(lines), 2):

            s = lines[i].split()

            camera_id = int(s[8])
            name = s[9]

            records.append(
                (name, camera_id)
            )

        # This matches the ordering used in your verified
        # MeetingRoom evaluation.
        records.sort(key=lambda x: x[0])

        if args.split > 0:
            test_records = records[::args.split]
        else:
            test_records = records

        print("Registered images :", len(records))
        print("Render/test views :", len(test_records))

        theta_max = np.deg2rad(args.fov / 2.0)

        # ---------------------------------------------------
        # Process each render
        # ---------------------------------------------------

        for idx, (name, camera_id) in enumerate(test_records):

            rp = renders / f"{idx:05d}.png"

            if not rp.exists():
                raise RuntimeError(
                    f"Missing render: {rp}"
                )

            render = cv2.imread(
                str(rp),
                cv2.IMREAD_COLOR
            )

            if render is None:
                raise RuntimeError(
                    f"Cannot read render: {rp}"
                )

            h, w = render.shape[:2]

            # =================================================
            # A. TRAINING/EVALUATION MASK
            #
            # Includes:
            # - 180 degree FoV
            # - tripod/rig exclusion
            #
            # USE THIS FOR METRICS.
            # =================================================

            gt = DATA / "images_2" / name

            train_mask_path = gt.with_name(
                gt.stem + "_mask.png"
            )

            train_mask = cv2.imread(
                str(train_mask_path),
                cv2.IMREAD_GRAYSCALE
            )

            if train_mask is None:
                raise RuntimeError(
                    f"Missing training mask:\n"
                    f"{train_mask_path}"
                )

            if train_mask.shape[:2] != (h, w):
                train_mask = cv2.resize(
                    train_mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

            masked = render.copy()

            masked[train_mask == 0] = 0

            cv2.imwrite(
                str(masked_out / f"{idx:05d}.png"),
                masked
            )

            # =================================================
            # B. PURE FOV MASK
            #
            # Does NOT include tripod exclusion.
            #
            # VISUALIZATION ONLY.
            #
            # Do NOT calculate thesis metrics from this folder.
            # =================================================

            c = cameras[camera_id]

            sx = w / c["W"]
            sy = h / c["H"]

            fx = c["fx"] * sx
            fy = c["fy"] * sy
            cx = c["cx"] * sx
            cy = c["cy"] * sy

            rd_max = theta_d(
                theta_max,
                c["k"]
            )

            yy, xx = np.mgrid[0:h, 0:w]

            xx = xx + 0.5
            yy = yy + 0.5

            rd = np.sqrt(
                ((xx - cx) / fx) ** 2
                +
                ((yy - cy) / fy) ** 2
            )

            fov_mask = rd <= rd_max

            fov_render = render.copy()

            fov_render[~fov_mask] = 0

            cv2.imwrite(
                str(fov_out / f"{idx:05d}.png"),
                fov_render
            )

            print(
                f"{idx:02d} "
                f"cam{camera_id} "
                f"{name}"
            )

    print()
    print("DONE")
    print()
    print("Metric-safe renders:")
    print(masked_out)

    print()
    print("FoV-only visualization:")
    print(fov_out)


if __name__ == "__main__":
    main()
