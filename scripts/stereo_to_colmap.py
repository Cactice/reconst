"""
Convert stereo test data (or real stereo captures) into COLMAP sparse format
so it can be fed directly into 3DGS training.

Inputs  (from gen_stereo_test_data.py, or your own stereo rig):
  data/stereo_test/
    calib.json
    frames/NNNN_left.png
    frames/NNNN_depth.npy   (metric depth, left camera; 0 = background)
    frames/NNNN_pose.npy    (4x4 c2w, left camera)

Outputs  (COLMAP text format):
  colmap_ws/
    images/         copies of left images
    sparse/0/
      cameras.txt   -- one camera entry (shared intrinsics)
      images.txt    -- one entry per frame (pose)
      points3D.txt  -- sparse point cloud from depth maps

Usage:
  python3 scripts/stereo_to_colmap.py
  python3 scripts/stereo_to_colmap.py --in data/stereo_test --out colmap_ws
"""

import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image


def c2w_to_colmap(c2w: np.ndarray):
    """Convert 4x4 camera-to-world to COLMAP world-to-camera quat + tvec."""
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    # Rotation matrix → quaternion (scalar-first: qw, qx, qy, qz)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return np.array([qw, qx, qy, qz]), t


def depth_to_points(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray,
                    stride: int = 8, max_depth: float = 20.0):
    """
    Back-project depth pixels to 3-D world points (subsampled by stride).
    Returns (N, 3) float32 array.
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    u, v = u.flatten(), v.flatten()
    d = depth[v, u]

    valid = (d > 0) & (d < max_depth)
    u, v, d = u[valid], v[valid], d[valid]

    # Back-project to camera space
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    pts_cam = np.stack([x, y, d, np.ones_like(d)], axis=1)  # (N, 4)

    # Transform to world
    pts_world = (c2w @ pts_cam.T).T[:, :3]
    return pts_world.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",  dest="src", default="data/stereo_test")
    parser.add_argument("--out", dest="dst", default="colmap_ws")
    parser.add_argument("--stride", type=int, default=8,
                        help="pixel stride when sampling depth for point cloud")
    parser.add_argument("--max-depth", type=float, default=20.0)
    args = parser.parse_args()

    frames_dir = os.path.join(args.src, "frames")
    calib_path = os.path.join(args.src, "calib.json")

    with open(calib_path) as f:
        cal = json.load(f)

    K = np.array([
        [cal["fx"],     0,    cal["cx"]],
        [    0,     cal["fy"], cal["cy"]],
        [    0,         0,        1    ],
    ])

    # Discover frames
    left_imgs = sorted(
        p for p in os.listdir(frames_dir) if p.endswith("_left.png")
    )
    if not left_imgs:
        raise FileNotFoundError(f"No *_left.png found in {frames_dir}")

    # Output dirs
    images_dst = os.path.join(args.dst, "images")
    sparse_dst = os.path.join(args.dst, "sparse", "0")
    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(sparse_dst, exist_ok=True)

    all_pts = []
    image_lines = []

    for img_id, fname in enumerate(left_imgs, start=1):
        stem = fname.replace("_left.png", "")
        src_img  = os.path.join(frames_dir, fname)
        src_dep  = os.path.join(frames_dir, stem + "_depth.npy")
        src_pose = os.path.join(frames_dir, stem + "_pose.npy")

        # Copy image
        dst_img = os.path.join(images_dst, fname)
        shutil.copy2(src_img, dst_img)

        # Load pose
        c2w = np.load(src_pose)
        quat, tvec = c2w_to_colmap(c2w)

        image_lines.append(
            f"{img_id} "
            f"{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f} "
            f"{tvec[0]:.9f} {tvec[1]:.9f} {tvec[2]:.9f} "
            f"1 {fname}\n\n"  # camera_id=1, blank line for 2D points
        )

        # Back-project depth
        if os.path.exists(src_dep):
            depth = np.load(src_dep)
            pts = depth_to_points(depth, K, c2w, args.stride, args.max_depth)
            all_pts.append(pts)

        print(f"  [{img_id:3d}/{len(left_imgs)}] {fname}")

    # --- cameras.txt ---
    with open(os.path.join(sparse_dst, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {cal['width']} {cal['height']} "
                f"{cal['fx']:.6f} {cal['fy']:.6f} "
                f"{cal['cx']:.6f} {cal['cy']:.6f}\n")

    # --- images.txt ---
    with open(os.path.join(sparse_dst, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.writelines(image_lines)

    # --- points3D.txt ---
    pts_all = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))
    # Subsample to keep file manageable
    max_pts = 200_000
    if len(pts_all) > max_pts:
        idx = np.random.choice(len(pts_all), max_pts, replace=False)
        pts_all = pts_all[idx]

    with open(os.path.join(sparse_dst, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i, (x, y, z) in enumerate(pts_all, start=1):
            f.write(f"{i} {x:.6f} {y:.6f} {z:.6f} 128 128 128 1.0\n")

    print(f"\nCOLMAP workspace written to {args.dst}/")
    print(f"  {len(left_imgs)} images, {len(pts_all)} points")
    print(f"\nTo train 3DGS (gaussian-splatting repo):")
    print(f"  python train.py -s {os.path.abspath(args.dst)}")


if __name__ == "__main__":
    main()
