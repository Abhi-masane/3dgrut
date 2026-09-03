from pathlib import Path
import argparse
import math
import numpy as np
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source_fov", type=float, default=185.0)
    ap.add_argument("--target_fov", type=float, default=160.0)
    args = ap.parse_args()

    src_dir = Path(args.input)
    out_root = Path(args.output)

    out_img = out_root / "images"
    out_masks = out_root / "masks"

    out_img.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("HallwayD_*.png"))

    if not files:
        raise RuntimeError(f"No images found in {src_dir}")

    first = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)

    if first is None:
        raise RuntimeError("Could not read first image")

    h, w = first.shape[:2]

    if w != h:
        print(f"WARNING: expected square synthetic images, got {w}x{h}")

    # OpenCV pixel-center convention for generating the images.
    cx_src = (w - 1) / 2.0
    cy_src = (h - 1) / 2.0

    cx_dst = (w - 1) / 2.0
    cy_dst = (h - 1) / 2.0

    # Radius of the circular fisheye image.
    R_src = min(w, h) / 2.0
    R_dst = min(w, h) / 2.0

    theta_src_max = math.radians(args.source_fov / 2.0)
    theta_dst_max = math.radians(args.target_fov / 2.0)

    # Destination is an equidistant/OpenCV fisheye with k1..k4 = 0.
    f_dst = R_dst / theta_dst_max

    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )

    dx = xx - cx_dst
    dy = yy - cy_dst

    r_dst = np.sqrt(dx * dx + dy * dy)

    theta = r_dst / f_dst

    valid = theta <= theta_dst_max

    phi = np.arctan2(dy, dx)

    # FAU synthetic images are documented as equisolid.
    #
    # equisolid:
    # r = 2 f sin(theta/2)
    #
    # We do not need the physical sensor focal length because we normalize
    # using the known 185-degree image radius.
    r_src = (
        R_src
        * np.sin(theta / 2.0)
        / np.sin(theta_src_max / 2.0)
    )

    map_x = cx_src + r_src * np.cos(phi)
    map_y = cy_src + r_src * np.sin(phi)

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    map_x[~valid] = -1
    map_y[~valid] = -1

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[valid] = 255

    for idx, src_path in enumerate(files):
        img = cv2.imread(str(src_path), cv2.IMREAD_COLOR)

        if img is None:
            print("Skipping unreadable:", src_path)
            continue

        remapped = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        out_path = out_img / src_path.name
        mask_path = out_masks / src_path.name

        cv2.imwrite(str(out_path), remapped)
        cv2.imwrite(str(mask_path), mask)

        if idx % 50 == 0:
            print(f"{idx}/{len(files)}")

    # COLMAP coordinates use center at width/2, height/2.
    cx_colmap = w / 2.0
    cy_colmap = h / 2.0

    with open(out_root / "camera_160.txt", "w") as f:
        f.write(f"width={w}\n")
        f.write(f"height={h}\n")
        f.write(f"fx={f_dst:.12f}\n")
        f.write(f"fy={f_dst:.12f}\n")
        f.write(f"cx={cx_colmap:.12f}\n")
        f.write(f"cy={cy_colmap:.12f}\n")
        f.write("k1=0.0\n")
        f.write("k2=0.0\n")
        f.write("k3=0.0\n")
        f.write("k4=0.0\n")
        f.write(f"source_fov={args.source_fov}\n")
        f.write(f"target_fov={args.target_fov}\n")

    print()
    print("DONE")
    print("Output:", out_root)
    print()
    print("COLMAP OPENCV_FISHEYE parameters:")
    print(
        f"{f_dst:.12f},{f_dst:.12f},"
        f"{cx_colmap:.12f},{cy_colmap:.12f},0,0,0,0"
    )


if __name__ == "__main__":
    main()