#!/usr/bin/env python3

import argparse
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--split", type=int, default=20)

    args = parser.parse_args()

    DATA = Path(args.data).expanduser().resolve()
    RUN = Path(args.run).expanduser().resolve()

    RENDERS = RUN / f"ours_{args.step}" / "renders"

    if not RENDERS.exists():
        raise RuntimeError(
            f"Render directory not found:\n{RENDERS}"
        )

    # ------------------------------------------------------
    # Convert COLMAP model to TXT temporarily
    # ------------------------------------------------------

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

        image_txt = tmp / "images.txt"

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

        # This ordering already reproduced the correct
        # MeetingRoom full-frame metrics in our earlier test.
        records.sort(key=lambda x: x[0])

        if args.split > 0:
            test_records = records[::args.split]
        else:
            test_records = records

        print()
        print("Registered images :", len(records))
        print("Test views        :", len(test_records))
        print()

        results = []

        # --------------------------------------------------
        # Evaluate each held-out render
        # --------------------------------------------------

        for idx, (name, camera_id) in enumerate(test_records):

            render_path = RENDERS / f"{idx:05d}.png"

            gt_path = DATA / "images_2" / name

            mask_path = gt_path.with_name(
                gt_path.stem + "_mask.png"
            )

            render = cv2.imread(
                str(render_path),
                cv2.IMREAD_COLOR
            )

            gt = cv2.imread(
                str(gt_path),
                cv2.IMREAD_COLOR
            )

            mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE
            )

            if render is None:
                raise RuntimeError(
                    f"Missing render:\n{render_path}"
                )

            if gt is None:
                raise RuntimeError(
                    f"Missing GT:\n{gt_path}"
                )

            if mask is None:
                raise RuntimeError(
                    f"Missing mask:\n{mask_path}"
                )

            # ------------------------------------------------
            # Match render resolution if needed
            # ------------------------------------------------

            h, w = render.shape[:2]

            if gt.shape[:2] != (h, w):
                gt = cv2.resize(
                    gt,
                    (w, h),
                    interpolation=cv2.INTER_AREA
                )

            if mask.shape[:2] != (h, w):
                mask = cv2.resize(
                    mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

            # ------------------------------------------------
            # Convert BGR uint8 -> RGB float32 [0,1]
            # ------------------------------------------------

            render_rgb = cv2.cvtColor(
                render,
                cv2.COLOR_BGR2RGB
            ).astype(np.float32) / 255.0

            gt_rgb = cv2.cvtColor(
                gt,
                cv2.COLOR_BGR2RGB
            ).astype(np.float32) / 255.0

            valid = mask > 127

            # ------------------------------------------------
            # PSNR
            # ------------------------------------------------

            diff2 = (
                render_rgb - gt_rgb
            ) ** 2

            full_mse = diff2.mean()

            masked_mse = diff2[valid].mean()

            full_psnr = psnr_from_mse(
                full_mse
            )

            masked_psnr = psnr_from_mse(
                masked_mse
            )

            # ------------------------------------------------
            # SSIM
            # ------------------------------------------------

            full_ssim, ssim_map = structural_similarity(
                gt_rgb,
                render_rgb,
                channel_axis=2,
                data_range=1.0,
                full=True
            )

            # Depending on skimage version, full=True with
            # multichannel data may produce HxWx3.
            if ssim_map.ndim == 3:
                ssim_map = ssim_map.mean(axis=2)

            # Erode mask so SSIM windows do not cross the
            # invalid boundary.
            kernel = np.ones(
                (7, 7),
                dtype=np.uint8
            )

            valid_eroded = cv2.erode(
                valid.astype(np.uint8),
                kernel,
                iterations=1
            ).astype(bool)

            masked_ssim = float(
                ssim_map[valid_eroded].mean()
            )

            results.append(
                {
                    "camera": camera_id,
                    "full_psnr": full_psnr,
                    "masked_psnr": masked_psnr,
                    "full_ssim": float(full_ssim),
                    "masked_ssim": masked_ssim,
                }
            )

            print(
                f"{idx:02d} "
                f"cam{camera_id} "
                f"fullPSNR={full_psnr:6.2f} "
                f"maskPSNR={masked_psnr:6.2f} "
                f"fullSSIM={full_ssim:.4f} "
                f"maskSSIM={masked_ssim:.4f} "
                f"{name}"
            )

        # --------------------------------------------------
        # Overall
        # --------------------------------------------------

        print()
        print("=" * 70)
        print(f"OVERALL {args.step}")
        print("=" * 70)

        print(
            "full PSNR   :",
            np.mean(
                [x["full_psnr"] for x in results]
            )
        )

        print(
            "masked PSNR :",
            np.mean(
                [x["masked_psnr"] for x in results]
            )
        )

        print(
            "full SSIM   :",
            np.mean(
                [x["full_ssim"] for x in results]
            )
        )

        print(
            "masked SSIM :",
            np.mean(
                [x["masked_ssim"] for x in results]
            )
        )

        # --------------------------------------------------
        # Camera-wise
        # --------------------------------------------------

        for camera_id in sorted(
            set(x["camera"] for x in results)
        ):

            rr = [
                x
                for x in results
                if x["camera"] == camera_id
            ]

            print()
            print(f"CAMERA {camera_id}")

            print(
                "  masked PSNR:",
                np.mean(
                    [x["masked_psnr"] for x in rr]
                )
            )

            print(
                "  masked SSIM:",
                np.mean(
                    [x["masked_ssim"] for x in rr]
                )
            )


if __name__ == "__main__":
    main()
