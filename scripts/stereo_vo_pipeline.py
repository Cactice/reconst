"""
Stereo Visual Odometry pipeline  →  COLMAP workspace for 3DGS.

Input directory must contain:
  calib.json          camera intrinsics + baseline
  frames/
    NNNN_left.png
    NNNN_right.png    (same numbering, any zero-padded prefix)

Output:
  colmap_ws/
    images/           copies of left images
    sparse/0/
      cameras.txt
      images.txt
      points3D.txt

Steps
-----
1. Stereo SGBM  →  disparity  →  metric depth
2. Shi-Tomasi corners on left frame, tracked with Lucas-Kanade optical flow
3. 3-D position of each tracked point from depth map
4. solvePnPRansac  →  relative pose
5. Integrate poses into world frame
6. Back-project depth for sparse point cloud

Usage
-----
  # on the synthetic test data:
  python3 scripts/stereo_vo_pipeline.py

  # on real footage:
  python3 scripts/stereo_vo_pipeline.py --in /path/to/my_capture --out colmap_ws
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frames(frames_dir):
    """Return sorted list of (left_path, right_path) pairs."""
    files = sorted(os.listdir(frames_dir))
    lefts  = [f for f in files if f.endswith("_left.png")]
    rights = {f.replace("_left.png", "_right.png") for f in lefts}
    pairs  = []
    for l in lefts:
        r = l.replace("_left.png", "_right.png")
        if r in rights:
            pairs.append((
                os.path.join(frames_dir, l),
                os.path.join(frames_dir, r),
                l,           # image name (used in COLMAP images.txt)
            ))
    return pairs


def compute_depth(left_gray, right_gray, fx, baseline,
                  min_disp=1, num_disp=128, block=5):
    """
    SGBM stereo matching  →  metric depth map (float32, 0 = invalid).
    Assumes images are already rectified (horizontal epipolar lines).
    """
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block,
        P1=8  * 3 * block ** 2,
        P2=32 * 3 * block ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    depth = np.zeros_like(disp)
    valid = disp > min_disp
    depth[valid] = fx * baseline / disp[valid]
    return depth


def detect_features(gray, max_corners=2000):
    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=7,
    )
    return pts  # (N,1,2) or None


def track_features(gray_prev, gray_curr, pts_prev):
    """Lucas-Kanade optical flow. Returns (pts_curr, mask)."""
    pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
        gray_prev, gray_curr, pts_prev, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    mask = (status.ravel() == 1)
    return pts_curr, mask


def depth_at(depth, pts_2d):
    """
    Sample depth at sub-pixel 2-D points.
    pts_2d: (N, 2) float  (x, y)
    Returns (N,) depth values; 0 means invalid.
    """
    H, W = depth.shape
    xs = np.clip(np.round(pts_2d[:, 0]).astype(int), 0, W - 1)
    ys = np.clip(np.round(pts_2d[:, 1]).astype(int), 0, H - 1)
    return depth[ys, xs]


def unproject(pts_2d, depths, K):
    """
    pts_2d : (N, 2)  pixel coords
    depths : (N,)    metric depth
    K      : 3x3     intrinsics
    Returns (N, 3) points in camera space.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (pts_2d[:, 0] - cx) / fx * depths
    y = (pts_2d[:, 1] - cy) / fy * depths
    return np.stack([x, y, depths], axis=1)


def estimate_pose(pts3d_prev, pts2d_curr, K):
    """
    PnP RANSAC: 3-D points in prev camera frame matched to 2-D points in
    current frame.  Returns 4×4 relative pose T_curr_from_prev, or None.
    """
    if len(pts3d_prev) < 6:
        return None
    dist = np.zeros(4)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d_prev.astype(np.float64),
        pts2d_curr.astype(np.float64),
        K, dist,
        iterationsCount=1000,
        reprojectionError=2.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inliers is None or len(inliers) < 6:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = tvec.ravel()
    return T   # world-to-camera of current frame relative to prev


def c2w_to_colmap(c2w):
    """4×4 camera-to-world  →  (quat wxyz, tvec)."""
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    trace = R.trace()
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        qw = (R[2,1] - R[1,2]) / s; qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s; qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        qw = (R[0,2] - R[2,0]) / s; qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s;               qz = (R[1,2] + R[2,1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        qw = (R[1,0] - R[0,1]) / s; qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s; qz = 0.25 * s
    return np.array([qw, qx, qy, qz]), t


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", default="data/stereo_test")
    ap.add_argument("--out", dest="dst", default="colmap_ws_vo")
    ap.add_argument("--min-disp",  type=int,   default=1)
    ap.add_argument("--num-disp",  type=int,   default=128,
                    help="must be divisible by 16")
    ap.add_argument("--depth-stride", type=int, default=8,
                    help="pixel stride when sampling depth for point cloud")
    ap.add_argument("--max-depth",  type=float, default=20.0)
    args = ap.parse_args()

    # --- load calibration ---
    with open(os.path.join(args.src, "calib.json")) as f:
        cal = json.load(f)
    fx, fy  = cal["fx"], cal["fy"]
    cx, cy  = cal["cx"], cal["cy"]
    W, H    = cal["width"], cal["height"]
    B       = cal["baseline"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    frames_dir = os.path.join(args.src, "frames")
    pairs = load_frames(frames_dir)
    if not pairs:
        raise FileNotFoundError(f"No stereo pairs found in {frames_dir}")
    print(f"Found {len(pairs)} stereo pairs.")

    # --- output dirs ---
    images_dst = os.path.join(args.dst, "images")
    sparse_dst = os.path.join(args.dst, "sparse", "0")
    os.makedirs(images_dst, exist_ok=True)
    os.makedirs(sparse_dst, exist_ok=True)

    # --- VO state ---
    c2w_world = np.eye(4)   # first frame defines world origin
    all_c2w   = []
    all_pts3d = []          # accumulated world-space points
    all_names = []

    prev_gray  = None
    prev_depth = None
    prev_pts   = None

    for idx, (left_path, right_path, name) in enumerate(pairs):
        left_bgr  = cv2.imread(left_path)
        right_bgr = cv2.imread(right_path)
        left_gray  = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Stereo depth
        depth = compute_depth(left_gray, right_gray, fx, B,
                               args.min_disp, args.num_disp)

        # 2. VO pose estimation (skip for first frame)
        if prev_gray is not None and prev_pts is not None:
            curr_pts, mask = track_features(prev_gray, left_gray, prev_pts)

            prev_2d = prev_pts[mask].reshape(-1, 2)
            curr_2d = curr_pts[mask].reshape(-1, 2)

            depths_prev = depth_at(prev_depth, prev_2d)
            valid = depths_prev > 0.1
            if valid.sum() >= 6:
                pts3d_cam = unproject(prev_2d[valid], depths_prev[valid], K)
                T_rel = estimate_pose(pts3d_cam, curr_2d[valid], K)
                if T_rel is not None:
                    # T_rel maps prev-camera coords to curr-camera coords
                    # c2w_world is prev camera-to-world
                    # new c2w = c2w_world @ inv(T_rel)
                    c2w_world = c2w_world @ np.linalg.inv(T_rel)

        all_c2w.append(c2w_world.copy())
        all_names.append(name)

        # 3. Back-project depth for point cloud (subsampled)
        u, v = np.meshgrid(
            np.arange(0, W, args.depth_stride),
            np.arange(0, H, args.depth_stride),
        )
        u, v = u.flatten(), v.flatten()
        d = depth[v, u]
        valid_d = (d > 0.1) & (d < args.max_depth)
        if valid_d.sum() > 0:
            x = (u[valid_d] - cx) / fx * d[valid_d]
            y = (v[valid_d] - cy) / fy * d[valid_d]
            pts_cam = np.stack([x, y, d[valid_d], np.ones(valid_d.sum())], 1)
            pts_world = (c2w_world @ pts_cam.T).T[:, :3]
            all_pts3d.append(pts_world.astype(np.float32))

        # 4. Refresh features every frame
        prev_pts  = detect_features(left_gray)
        prev_gray  = left_gray
        prev_depth = depth

        # 5. Copy image to workspace
        shutil.copy2(left_path, os.path.join(images_dst, name))
        print(f"  [{idx+1:3d}/{len(pairs)}] {name}  "
              f"pos=({c2w_world[0,3]:.2f},{c2w_world[1,3]:.2f},{c2w_world[2,3]:.2f})")

    # --- write cameras.txt ---
    with open(os.path.join(sparse_dst, "cameras.txt"), "w") as f:
        f.write("# Camera list\n# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        f.write(f"1 PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    # --- write images.txt ---
    with open(os.path.join(sparse_dst, "images.txt"), "w") as f:
        f.write("# Image list\n"
                "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
                "# POINTS2D[] as (X Y POINT3D_ID)\n")
        for img_id, (c2w, name) in enumerate(zip(all_c2w, all_names), 1):
            q, t = c2w_to_colmap(c2w)
            f.write(f"{img_id} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                    f"1 {name}\n\n")

    # --- write points3D.txt ---
    pts_all = np.concatenate(all_pts3d, axis=0) if all_pts3d else np.zeros((0,3))
    max_pts = 200_000
    if len(pts_all) > max_pts:
        idx = np.random.choice(len(pts_all), max_pts, replace=False)
        pts_all = pts_all[idx]

    with open(os.path.join(sparse_dst, "points3D.txt"), "w") as f:
        f.write("# 3D point list\n# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for i, (x, y, z) in enumerate(pts_all, 1):
            f.write(f"{i} {x:.6f} {y:.6f} {z:.6f} 128 128 128 1.0\n")

    print(f"\nDone  →  {args.dst}/")
    print(f"  {len(pairs)} poses,  {len(pts_all)} points")
    print(f"\nTo train 3DGS:")
    print(f"  python train.py -s {os.path.abspath(args.dst)}")


if __name__ == "__main__":
    main()
