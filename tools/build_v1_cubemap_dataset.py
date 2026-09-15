#!/usr/bin/env python3
"""
build_v1_cubemap_dataset.py

Build a perspective-image V1 baseline that covers the COMPLETE forward
180-degree fisheye hemisphere without trying to force 180 degrees into a
single pinhole image.

Each registered fisheye image is converted to five 90-degree perspective
cube faces:
    front, right, left, up, down

Together these five tangent views cover the full forward hemisphere.
The camera centre is unchanged. Each virtual view gets the correctly rotated
COLMAP world-to-camera pose. The original 180-degree + tripod mask is remapped
into every virtual perspective image.

The output COLMAP model uses one PINHOLE camera and keeps the original sparse
XYZ/RGB/error values with empty tracks, because the old 2D fisheye feature
coordinates are not valid in the new perspective images.

Example:
python3 tools/build_v1_cubemap_dataset.py \
    --source-data ~/thesis/my3dscene/v2/data/MeetingRoom_RAW_MASK180_CLEAN \
    --out-data ~/thesis/my3dscene/v1/data/MeetingRoom_V1_CUBEMAP_FULL \
    --out-size 3264 \
    --face-fov-deg 90
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


FACE_ORDER = ["front", "right", "left", "up", "down"]

# A maps virtual-camera coordinates -> original fisheye-camera coordinates.
# Camera convention: x right, y down, z forward.
FACE_A = {
    "front": np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float64),

    "right": np.array([
        [ 0, 0, 1],
        [ 0, 1, 0],
        [-1, 0, 0],
    ], dtype=np.float64),

    "left": np.array([
        [0, 0, -1],
        [0, 1,  0],
        [1, 0,  0],
    ], dtype=np.float64),

    "up": np.array([
        [1,  0, 0],
        [0,  0,-1],
        [0,  1, 0],
    ], dtype=np.float64),

    "down": np.array([
        [1,  0, 0],
        [0,  0, 1],
        [0, -1, 0],
    ], dtype=np.float64),
}


def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [
            1 - 2*qy*qy - 2*qz*qz,
            2*qx*qy - 2*qz*qw,
            2*qx*qz + 2*qy*qw,
        ],
        [
            2*qx*qy + 2*qz*qw,
            1 - 2*qx*qx - 2*qz*qz,
            2*qy*qz - 2*qx*qw,
        ],
        [
            2*qx*qz - 2*qy*qw,
            2*qy*qz + 2*qx*qw,
            1 - 2*qx*qx - 2*qy*qy,
        ],
    ], dtype=np.float64)


def rotmat2qvec(R):
    # COLMAP-compatible scalar-first quaternion.
    K = np.array([
        [R[0,0] - R[1,1] - R[2,2], R[1,0] + R[0,1], R[2,0] + R[0,2], R[1,2] - R[2,1]],
        [R[1,0] + R[0,1], R[1,1] - R[0,0] - R[2,2], R[2,1] + R[1,2], R[2,0] - R[0,2]],
        [R[2,0] + R[0,2], R[2,1] + R[1,2], R[2,2] - R[0,0] - R[1,1], R[0,1] - R[1,0]],
        [R[1,2] - R[2,1], R[2,0] - R[0,2], R[0,1] - R[1,0], R[0,0] + R[1,1] + R[2,2]],
    ]) / 3.0

    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[[3,0,1,2], np.argmax(eigvals)]

    if q[0] < 0:
        q = -q

    return q / np.linalg.norm(q)


def parse_cameras_txt(path):
    cams = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        s = line.split()
        cid = int(s[0])
        model = s[1]
        width = int(s[2])
        height = int(s[3])
        params = list(map(float, s[4:]))

        if model != "OPENCV_FISHEYE":
            raise RuntimeError(
                f"Camera {cid}: expected OPENCV_FISHEYE, got {model}"
            )

        if len(params) < 8:
            raise RuntimeError(f"Camera {cid}: expected 8 fisheye params")

        cams[cid] = {
            "width": width,
            "height": height,
            "fx": params[0],
            "fy": params[1],
            "cx": params[2],
            "cy": params[3],
            "k": params[4:8],
        }
    return cams


def parse_images_txt(path):
    lines = [
        x.strip()
        for x in Path(path).read_text().splitlines()
        if x.strip() and not x.startswith("#")
    ]

    recs = []
    for i in range(0, len(lines), 2):
        s = lines[i].split()
        if len(s) < 10:
            continue

        recs.append({
            "image_id": int(s[0]),
            "q": np.asarray(list(map(float, s[1:5])), dtype=np.float64),
            "t": np.asarray(list(map(float, s[5:8])), dtype=np.float64),
            "camera_id": int(s[8]),
            "name": s[9],
        })

    recs.sort(key=lambda x: x["name"])
    return recs


def build_map(cam, A, out_size, fov_deg):
    half = np.deg2rad(fov_deg / 2.0)
    f = (out_size / 2.0) / np.tan(half)
    c = out_size / 2.0

    u = np.arange(out_size, dtype=np.float64) + 0.5
    v = np.arange(out_size, dtype=np.float64) + 0.5
    vv, uu = np.meshgrid(v, u, indexing="ij")

    x = (uu - c) / f
    y = (vv - c) / f
    z = np.ones_like(x)

    # d_orig = A @ d_virtual.
    ox = A[0,0]*x + A[0,1]*y + A[0,2]*z
    oy = A[1,0]*x + A[1,1]*y + A[1,2]*z
    oz = A[2,0]*x + A[2,1]*y + A[2,2]*z

    rho = np.sqrt(ox*ox + oy*oy)
    theta = np.arctan2(rho, oz)

    # The V1 baseline covers the same forward 180-degree domain as V2:
    # half-angle <= 90 degrees.
    valid = (theta <= np.deg2rad(90.0) + 1e-10) & (oz >= -1e-10)

    k1, k2, k3, k4 = cam["k"]
    t2 = theta*theta
    t4 = t2*t2
    t6 = t4*t2
    t8 = t4*t4
    rd = theta * (1.0 + k1*t2 + k2*t4 + k3*t6 + k4*t8)

    scale = np.zeros_like(rd)
    nz = rho > 1e-12
    scale[nz] = rd[nz] / rho[nz]

    map_x = cam["fx"] * ox * scale + cam["cx"]
    map_y = cam["fy"] * oy * scale + cam["cy"]

    center = ~nz
    map_x[center] = cam["cx"]
    map_y[center] = cam["cy"]

    valid &= (
        (map_x >= 0) &
        (map_x <= cam["width"] - 1) &
        (map_y >= 0) &
        (map_y <= cam["height"] - 1)
    )

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    map_x[~valid] = -1.0
    map_y[~valid] = -1.0

    K_new = {
        "fx": float(f),
        "fy": float(f),
        "cx": float(c),
        "cy": float(c),
    }

    return map_x, map_y, valid, K_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-data", required=True)
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-size", type=int, default=3264)
    ap.add_argument("--face-fov-deg", type=float, default=90.0)
    args = ap.parse_args()

    if args.face_fov_deg >= 180.0:
        raise SystemExit(
            "A perspective/pinhole face cannot use a 180-degree FoV. "
            "Use multiple tangent faces. Recommended: 5 faces × 90 degrees."
        )

    SRC = Path(args.source_data).expanduser().resolve()
    OUT = Path(args.out_data).expanduser().resolve()

    if OUT.exists():
        shutil.rmtree(OUT)

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "sparse" / "0").mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        src_txt = Path(td) / "source_txt"
        src_txt.mkdir()

        subprocess.run(
            [
                "colmap", "model_converter",
                "--input_path", str(SRC / "sparse/0"),
                "--output_path", str(src_txt),
                "--output_type", "TXT",
            ],
            check=True,
        )

        cams = parse_cameras_txt(src_txt / "cameras.txt")
        images = parse_images_txt(src_txt / "images.txt")

        print("Registered source views:", len(images))
        print("Physical camera IDs    :", sorted(cams))
        print("Virtual faces/source   :", len(FACE_ORDER))
        print("Virtual images total   :", len(images) * len(FACE_ORDER))

        model_txt = OUT / "model_txt"
        model_txt.mkdir(parents=True, exist_ok=True)

        f = (args.out_size / 2.0) / np.tan(
            np.deg2rad(args.face_fov_deg / 2.0)
        )
        c = args.out_size / 2.0

        (model_txt / "cameras.txt").write_text(
            "# Camera list\n"
            "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
            f"1 PINHOLE {args.out_size} {args.out_size} "
            f"{f:.17g} {f:.17g} {c:.17g} {c:.17g}\n"
        )

        source_map = {}
        image_lines = []
        new_image_id = 1

        # cache remaps per physical camera + face
        map_cache = {}

        for src_index, rec in enumerate(images, 1):
            src_img = SRC / "images" / rec["name"]
            src_mask = src_img.with_name(src_img.stem + "_mask.png")

            image = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)

            if image is None:
                raise RuntimeError(f"Cannot read {src_img}")
            if mask is None:
                raise RuntimeError(f"Cannot read mask {src_mask}")

            cam = cams[rec["camera_id"]]

            if image.shape[1] != cam["width"] or image.shape[0] != cam["height"]:
                raise RuntimeError(
                    f"Full-resolution V1 expects native source size for {rec['name']}: "
                    f"image={image.shape[1]}x{image.shape[0]}, "
                    f"camera={cam['width']}x{cam['height']}"
                )

            R_orig = qvec2rotmat(rec["q"])
            t_orig = rec["t"]

            src_rel = Path(rec["name"])
            out_parent = OUT / "images" / src_rel.parent
            out_parent.mkdir(parents=True, exist_ok=True)

            for face in FACE_ORDER:
                A = FACE_A[face]

                key = (rec["camera_id"], face)
                if key not in map_cache:
                    map_cache[key] = build_map(
                        cam,
                        A,
                        args.out_size,
                        args.face_fov_deg,
                    )

                map_x, map_y, geom_valid, _ = map_cache[key]

                persp = cv2.remap(
                    image,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )

                remapped_mask = cv2.remap(
                    mask,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )

                remapped_mask[~geom_valid] = 0
                persp[remapped_mask == 0] = 0

                out_name = (
                    src_rel.parent /
                    f"{src_rel.stem}__{face}.png"
                ).as_posix()

                out_img = OUT / "images" / out_name
                out_mask = out_img.with_name(
                    out_img.stem + "_mask.png"
                )

                cv2.imwrite(
                    str(out_img),
                    persp,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                )
                cv2.imwrite(str(out_mask), remapped_mask)

                # Virtual pose: p_v = A^T p_orig.
                R_v = A.T @ R_orig
                t_v = A.T @ t_orig
                q_v = rotmat2qvec(R_v)

                image_lines.append(
                    f"{new_image_id} "
                    f"{q_v[0]:.17g} {q_v[1]:.17g} {q_v[2]:.17g} {q_v[3]:.17g} "
                    f"{t_v[0]:.17g} {t_v[1]:.17g} {t_v[2]:.17g} "
                    f"1 {out_name}\n\n"
                )

                source_map[out_name] = {
                    "source_image": rec["name"],
                    "source_camera_id": rec["camera_id"],
                    "face": face,
                    "A_virtual_to_original": A.tolist(),
                }

                new_image_id += 1

            if src_index % 10 == 0 or src_index == len(images):
                print(f"{src_index}/{len(images)} source images complete")

        (model_txt / "images.txt").write_text(
            "# Image list with two lines of data per image\n"
            "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
            + "".join(image_lines)
        )

        # Keep original XYZ/RGB/error, drop invalid old fisheye observation tracks.
        point_lines = []
        for line in (src_txt / "points3D.txt").read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            s = line.split()
            if len(s) >= 8:
                point_lines.append(" ".join(s[:8]) + "\n")

        (model_txt / "points3D.txt").write_text(
            "# 3D point list\n"
            "# POINT3D_ID X Y Z R G B ERROR TRACK[]\n"
            + "".join(point_lines)
        )

        (OUT / "source_map.json").write_text(
            json.dumps(
                {
                    "projection": "five_face_forward_cubemap",
                    "face_fov_deg": args.face_fov_deg,
                    "out_size": args.out_size,
                    "faces": FACE_ORDER,
                    "virtual_images": source_map,
                },
                indent=2,
            )
        )

        # Convert the new model to binary COLMAP format.
        sparse_out = OUT / "sparse" / "0"
        # directory already exists, remove any accidental content
        for p in sparse_out.iterdir():
            if p.is_file():
                p.unlink()

        subprocess.run(
            [
                "colmap", "model_converter",
                "--input_path", str(model_txt),
                "--output_path", str(sparse_out),
                "--output_type", "BIN",
            ],
            check=True,
        )

    print()
    print("V1 DATASET COMPLETE")
    print("Output:", OUT)
    print("Perspective images:", OUT / "images")
    print("COLMAP model:", OUT / "sparse/0")
    print("Source map:", OUT / "source_map.json")


if __name__ == "__main__":
    main()
