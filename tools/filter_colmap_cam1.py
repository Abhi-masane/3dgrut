from pathlib import Path
import sys

if len(sys.argv) != 3:
    print(
        "Usage: python filter_colmap_cam1.py "
        "<input_model_txt> <output_model_txt>"
    )
    sys.exit(1)

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Read cameras
# ---------------------------------------------------------

camera_lines = []

with open(src / "cameras.txt", "r") as f:
    for line in f:
        if line.startswith("#"):
            continue

        line = line.strip()

        if line:
            camera_lines.append(line)


# ---------------------------------------------------------
# Read images.txt
#
# COLMAP text format:
# line 1 = image metadata
# line 2 = POINTS2D
# ---------------------------------------------------------

kept_images = []
kept_image_ids = set()
used_camera_ids = set()

with open(src / "images.txt", "r") as f:
    lines = f.readlines()

i = 0

while i < len(lines):

    line = lines[i]

    if line.startswith("#") or not line.strip():
        i += 1
        continue

    metadata = line.strip()

    if i + 1 >= len(lines):
        break

    points2d = lines[i + 1].rstrip("\n")

    parts = metadata.split()

    image_id = int(parts[0])
    camera_id = int(parts[8])
    image_name = parts[9]

    if image_name.startswith("cam1/"):

        kept_images.append(
            (
                metadata,
                points2d,
            )
        )

        kept_image_ids.add(image_id)
        used_camera_ids.add(camera_id)

    i += 2


print("Kept images:", len(kept_images))
print("Image IDs:", len(kept_image_ids))
print("Camera IDs used:", sorted(used_camera_ids))


# ---------------------------------------------------------
# Write cameras.txt
# ---------------------------------------------------------

with open(dst / "cameras.txt", "w") as f:

    f.write("# Camera list filtered for cam1\n")

    for line in camera_lines:

        camera_id = int(line.split()[0])

        if camera_id in used_camera_ids:
            f.write(line + "\n")


# ---------------------------------------------------------
# Write images.txt
# ---------------------------------------------------------

with open(dst / "images.txt", "w") as f:

    f.write("# Image list filtered for cam1\n")

    for metadata, points2d in kept_images:

        f.write(metadata + "\n")
        f.write(points2d + "\n")


# ---------------------------------------------------------
# Filter points3D tracks
#
# Keep point only if it is observed by at least one cam1 image.
# ---------------------------------------------------------

kept_points = 0

with open(src / "points3D.txt", "r") as fin, \
     open(dst / "points3D.txt", "w") as fout:

    fout.write("# 3D points filtered for cam1\n")

    for line in fin:

        if line.startswith("#") or not line.strip():
            continue

        parts = line.strip().split()

        # Standard fields:
        # ID X Y Z R G B ERROR
        fixed = parts[:8]
        track = parts[8:]

        new_track = []

        for j in range(0, len(track), 2):

            image_id = int(track[j])
            point2d_idx = track[j + 1]

            if image_id in kept_image_ids:

                new_track.extend(
                    [
                        str(image_id),
                        point2d_idx,
                    ]
                )

        if len(new_track) >= 2:

            fout.write(
                " ".join(fixed + new_track)
                + "\n"
            )

            kept_points += 1


print("Kept 3D points:", kept_points)

print("\nDone.")
print("Output:", dst)
