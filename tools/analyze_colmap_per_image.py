from pathlib import Path
from collections import defaultdict
import math

MODEL_DIR = Path(
    "/home/masane_abhi/thesis/kitchen_3dgut/model_cam1_txt"
)

images_path = MODEL_DIR / "images.txt"
points_path = MODEL_DIR / "points3D.txt"


# ---------------------------------------------------------
# Read 3D point reprojection errors
# ---------------------------------------------------------

point_error = {}

with open(points_path, "r") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue

        parts = line.strip().split()

        point_id = int(parts[0])
        error = float(parts[7])

        point_error[point_id] = error


# ---------------------------------------------------------
# Read image observations
# ---------------------------------------------------------

results = []

with open(images_path, "r") as f:
    raw = [
        line.rstrip("\n")
        for line in f
        if not line.startswith("#")
    ]

i = 0

while i < len(raw):

    if not raw[i].strip():
        i += 1
        continue

    metadata = raw[i].strip()

    if i + 1 >= len(raw):
        break

    points2d = raw[i + 1].strip()

    parts = metadata.split()

    image_id = int(parts[0])
    camera_id = int(parts[8])
    image_name = parts[9]

    obs = points2d.split()

    errors = []
    valid_points = 0

    # POINTS2D format:
    # X Y POINT3D_ID
    for j in range(0, len(obs), 3):

        if j + 2 >= len(obs):
            break

        point3d_id = int(obs[j + 2])

        if point3d_id == -1:
            continue

        if point3d_id in point_error:
            errors.append(point_error[point3d_id])
            valid_points += 1

    if errors:

        errors_sorted = sorted(errors)

        n = len(errors_sorted)

        mean_error = sum(errors_sorted) / n

        if n % 2 == 1:
            median_error = errors_sorted[n // 2]
        else:
            median_error = (
                errors_sorted[n // 2 - 1]
                + errors_sorted[n // 2]
            ) / 2.0

        p90_index = min(
            n - 1,
            int(math.floor(0.90 * (n - 1)))
        )

        p90_error = errors_sorted[p90_index]

    else:
        mean_error = float("nan")
        median_error = float("nan")
        p90_error = float("nan")

    results.append(
        {
            "image_id": image_id,
            "camera_id": camera_id,
            "image_name": image_name,
            "num_points": valid_points,
            "mean_error": mean_error,
            "median_error": median_error,
            "p90_error": p90_error,
        }
    )

    i += 2


# ---------------------------------------------------------
# Sort worst-first by mean reprojection error
# ---------------------------------------------------------

results_sorted = sorted(
    results,
    key=lambda x: (
        -999 if math.isnan(x["mean_error"])
        else x["mean_error"]
    ),
    reverse=True,
)


print("\n" + "=" * 110)
print("WORST IMAGES BY COLMAP REPROJECTION ERROR")
print("=" * 110)

print(
    f"{'rank':>4} "
    f"{'image_id':>8} "
    f"{'points':>7} "
    f"{'mean_px':>9} "
    f"{'median_px':>10} "
    f"{'p90_px':>9} "
    f"image"
)

for rank, r in enumerate(results_sorted[:40], start=1):

    print(
        f"{rank:4d} "
        f"{r['image_id']:8d} "
        f"{r['num_points']:7d} "
        f"{r['mean_error']:9.3f} "
        f"{r['median_error']:10.3f} "
        f"{r['p90_error']:9.3f} "
        f"{r['image_name']}"
    )


# ---------------------------------------------------------
# Also show images with fewest triangulated observations
# ---------------------------------------------------------

print("\n" + "=" * 110)
print("IMAGES WITH FEWEST VALID 3D OBSERVATIONS")
print("=" * 110)

fewest = sorted(
    results,
    key=lambda x: x["num_points"]
)

print(
    f"{'rank':>4} "
    f"{'image_id':>8} "
    f"{'points':>7} "
    f"{'mean_px':>9} "
    f"image"
)

for rank, r in enumerate(fewest[:40], start=1):

    print(
        f"{rank:4d} "
        f"{r['image_id']:8d} "
        f"{r['num_points']:7d} "
        f"{r['mean_error']:9.3f} "
        f"{r['image_name']}"
    )


# ---------------------------------------------------------
# Validation split mapping
# ---------------------------------------------------------

print("\n" + "=" * 110)
print("VALIDATION FRAME MAPPING (every 8th image)")
print("=" * 110)

for global_idx, r in enumerate(results):

    if global_idx % 8 == 0:

        val_idx = global_idx // 8

        marker = ""

        if val_idx in {0, 4, 10, 24, 25, 28}:
            marker = "  <-- LOW-PSNR VAL FRAME"

        print(
            f"val_frame={val_idx:2d} "
            f"global_idx={global_idx:3d} "
            f"image_id={r['image_id']:4d} "
            f"points={r['num_points']:5d} "
            f"mean_err={r['mean_error']:.3f}px "
            f"{r['image_name']}"
            f"{marker}"
        )
