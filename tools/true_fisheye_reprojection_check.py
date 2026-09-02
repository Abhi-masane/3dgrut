from pathlib import Path
import numpy as np
import math

MODEL_DIR = Path(
    "/home/masane_abhi/thesis/kitchen_3dgut/model_cam1_txt"
)

CAMERAS = MODEL_DIR / "cameras.txt"
IMAGES = MODEL_DIR / "images.txt"
POINTS3D = MODEL_DIR / "points3D.txt"


# ---------------------------------------------------------
# Read OPENCV_FISHEYE camera
# ---------------------------------------------------------

camera = None

with open(CAMERAS, "r") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue

        p = line.split()

        camera_id = int(p[0])
        model = p[1]

        assert model == "OPENCV_FISHEYE", model

        width = int(p[2])
        height = int(p[3])

        fx, fy, cx, cy = map(float, p[4:8])
        k1, k2, k3, k4 = map(float, p[8:12])

        camera = {
            "id": camera_id,
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k": np.array([k1, k2, k3, k4], dtype=np.float64),
        }

assert camera is not None


# ---------------------------------------------------------
# Quaternion -> rotation matrix
# COLMAP qvec = qw qx qy qz
# ---------------------------------------------------------

def qvec_to_rot(q):
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


# ---------------------------------------------------------
# Read 3D points
# ---------------------------------------------------------

points3d = {}

with open(POINTS3D, "r") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue

        p = line.split()

        pid = int(p[0])

        xyz = np.array(
            [float(p[1]), float(p[2]), float(p[3])],
            dtype=np.float64
        )

        points3d[pid] = xyz


# ---------------------------------------------------------
# OpenCV fisheye forward projection
# ---------------------------------------------------------

def project_fisheye(X_cam, cam):
    x, y, z = X_cam

    if z <= 0:
        return None

    r = math.sqrt(x*x + y*y)

    theta = math.atan2(r, z)

    if r < 1e-12:
        xd = 0.0
        yd = 0.0
    else:
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4

        k1, k2, k3, k4 = cam["k"]

        theta_d = theta * (
            1.0
            + k1 * theta2
            + k2 * theta4
            + k3 * theta6
            + k4 * theta8
        )

        scale = theta_d / r

        xd = x * scale
        yd = y * scale

    u = cam["fx"] * xd + cam["cx"]
    v = cam["fy"] * yd + cam["cy"]

    return np.array([u, v], dtype=np.float64)


# ---------------------------------------------------------
# Read images + observations
# ---------------------------------------------------------

results = []

with open(IMAGES, "r") as f:
    lines = [
        l.rstrip("\n")
        for l in f
        if not l.startswith("#")
    ]

i = 0

while i < len(lines):

    if not lines[i].strip():
        i += 1
        continue

    meta = lines[i].split()

    image_id = int(meta[0])

    qvec = np.array(
        list(map(float, meta[1:5])),
        dtype=np.float64
    )

    tvec = np.array(
        list(map(float, meta[5:8])),
        dtype=np.float64
    )

    camera_id = int(meta[8])
    image_name = meta[9]

    obs_line = lines[i + 1].split()

    R = qvec_to_rot(qvec)

    errors = []

    for j in range(0, len(obs_line), 3):

        if j + 2 >= len(obs_line):
            break

        u_obs = float(obs_line[j])
        v_obs = float(obs_line[j + 1])
        point_id = int(obs_line[j + 2])

        if point_id == -1:
            continue

        if point_id not in points3d:
            continue

        X_world = points3d[point_id]

        X_cam = R @ X_world + tvec

        uv_pred = project_fisheye(X_cam, camera)

        if uv_pred is None:
            continue

        err = np.linalg.norm(
            uv_pred - np.array([u_obs, v_obs])
        )

        errors.append(err)

    if errors:
        e = np.array(errors)

        results.append({
            "image_id": image_id,
            "image_name": image_name,
            "n": len(e),
            "mean": float(np.mean(e)),
            "median": float(np.median(e)),
            "p90": float(np.percentile(e, 90)),
            "max": float(np.max(e)),
        })

    i += 2


# ---------------------------------------------------------
# Print worst images
# ---------------------------------------------------------

results_sorted = sorted(
    results,
    key=lambda r: r["mean"],
    reverse=True
)

print("\nWORST IMAGES BY TRUE FISHEYE REPROJECTION ERROR")
print("=" * 120)

print(
    f"{'rank':>4} "
    f"{'id':>5} "
    f"{'pts':>6} "
    f"{'mean':>9} "
    f"{'median':>9} "
    f"{'p90':>9} "
    f"{'max':>9} "
    f"image"
)

for rank, r in enumerate(results_sorted[:50], start=1):
    print(
        f"{rank:4d} "
        f"{r['image_id']:5d} "
        f"{r['n']:6d} "
        f"{r['mean']:9.3f} "
        f"{r['median']:9.3f} "
        f"{r['p90']:9.3f} "
        f"{r['max']:9.3f} "
        f"{r['image_name']}"
    )


# ---------------------------------------------------------
# Validation split mapping
# ---------------------------------------------------------

print("\nVALIDATION FRAME TRUE REPROJECTION ERRORS")
print("=" * 120)

for global_idx, r in enumerate(results):

    if global_idx % 8 == 0:

        val_idx = global_idx // 8

        marker = ""

        if val_idx in {0, 4, 10, 24, 25, 28}:
            marker = " <-- LOW-PSNR"

        print(
            f"val={val_idx:2d} "
            f"id={r['image_id']:4d} "
            f"pts={r['n']:5d} "
            f"mean={r['mean']:.3f}px "
            f"median={r['median']:.3f}px "
            f"p90={r['p90']:.3f}px "
            f"{r['image_name']}"
            f"{marker}"
        )
