"""
Generate synthetic stereo test data for 3DGS/4DGS pipeline development.

Output layout:
  data/stereo_test/
    calib.json          -- camera intrinsics + baseline
    frames/
      NNNN_left.png
      NNNN_right.png
      NNNN_depth.npy    -- metric depth from left camera (ground truth)
      NNNN_pose.npy     -- 4x4 world-to-camera (left cam)

Scene: a cluster of colored spheres at varying depths, camera orbits around them.
"""

import argparse
import json
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "stereo_test")


# ---------------------------------------------------------------------------
# Scene definition
# ---------------------------------------------------------------------------

SPHERES = [
    # (center_xyz, radius, RGB)
    ([0.0,  0.0,  0.0],  0.3, (220,  60,  60)),
    ([0.7,  0.2,  0.4],  0.2, ( 60, 180,  60)),
    ([-0.5, 0.4,  0.6],  0.25,(  60,  90, 220)),
    ([0.2, -0.5,  0.3],  0.18,(220, 180,  40)),
    ([-0.3,-0.2, -0.4],  0.22,(160,  60, 200)),
    ([0.8, -0.3, -0.2],  0.15,(  40, 200, 200)),
]


# ---------------------------------------------------------------------------
# Ray – sphere intersection (vectorised over one image at a time)
# ---------------------------------------------------------------------------

def ray_sphere_intersect(rays_o, rays_d, center, radius):
    """
    rays_o: (H*W, 3)  ray origins
    rays_d: (H*W, 3)  ray directions (unit)
    Returns t (H*W,) with np.inf where no hit.
    """
    oc = rays_o - center
    b = (oc * rays_d).sum(-1)
    c = (oc * oc).sum(-1) - radius * radius
    disc = b * b - c
    hit = disc >= 0
    t = np.full(len(rays_d), np.inf)
    sq = np.sqrt(np.maximum(disc[hit], 0))
    t1 = -b[hit] - sq
    t2 = -b[hit] + sq
    t[hit] = np.where(t1 > 1e-4, t1, np.where(t2 > 1e-4, t2, np.inf))
    return t


def render_frame(K, width, height, c2w, baseline=0.0):
    """
    Render one (left or right) image + depth map.
    baseline > 0  shifts camera right by that many metres.
    Returns (rgb uint8 H×W×3, depth float32 H×W).
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u, v = u.astype(np.float32), v.astype(np.float32)
    dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
    dirs_cam = dirs_cam.reshape(-1, 3)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    # Apply baseline shift in camera space before rotating to world
    R = c2w[:3, :3]
    t = c2w[:3, 3].copy()
    t = t + R @ np.array([baseline, 0.0, 0.0])

    rays_o = np.broadcast_to(t, dirs_cam.shape).copy()
    rays_d = dirs_cam @ R.T

    depth = np.full(height * width, np.inf)
    color = np.zeros((height * width, 3), dtype=np.float32)

    for center, radius, rgb in SPHERES:
        c = np.array(center, dtype=np.float32)
        t_hit = ray_sphere_intersect(rays_o, rays_d, c, radius)
        closer = t_hit < depth
        depth[closer] = t_hit[closer]

        # Simple diffuse shading
        hit_pts = rays_o[closer] + t_hit[closer, None] * rays_d[closer]
        normals = (hit_pts - c) / radius
        light_dir = np.array([0.5, 1.0, -0.5])
        light_dir /= np.linalg.norm(light_dir)
        diffuse = np.clip((normals * light_dir).sum(-1), 0.1, 1.0)
        col = np.array(rgb, dtype=np.float32) / 255.0
        color[closer] = col * diffuse[:, None]

    # Background
    bg = np.array([0.15, 0.15, 0.2], dtype=np.float32)
    miss = np.isinf(depth)
    color[miss] = bg
    depth[miss] = 0.0

    rgb_img = np.clip(color * 255, 0, 255).astype(np.uint8).reshape(height, width, 3)
    depth_map = depth.astype(np.float32).reshape(height, width)
    return rgb_img, depth_map


# ---------------------------------------------------------------------------
# Camera orbit
# ---------------------------------------------------------------------------

def orbit_c2w(theta, phi, radius):
    """Spherical orbit. Returns 4x4 c2w (camera-to-world)."""
    x = radius * math.cos(phi) * math.cos(theta)
    y = radius * math.sin(phi)
    z = radius * math.cos(phi) * math.sin(theta)
    pos = np.array([x, y, z])

    # Look-at (camera looks toward origin)
    forward = -pos / np.linalg.norm(pos)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    # OpenCV convention: camera +z points INTO the scene (forward)
    # camera +y points down (world -y when world is y-up)
    cam_y = -up  # flip: image rows go down, but world y goes up
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = cam_y
    c2w[:3, 2] = forward
    c2w[:3, 3] = pos
    return c2w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30, help="number of frames")
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--baseline", type=float, default=0.06,
                        help="stereo baseline in metres (default 6 cm)")
    parser.add_argument("--orbit-radius", type=float, default=2.5)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

    W, H = args.width, args.height
    fx = fy = 0.8 * W          # reasonable FOV
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx,  0, cx],
                  [ 0, fy, cy],
                  [ 0,  0,  1]], dtype=np.float64)

    calib = {
        "width": W,
        "height": H,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "baseline": args.baseline,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "camera_model": "PINHOLE",
    }
    with open(os.path.join(args.out, "calib.json"), "w") as f:
        json.dump(calib, f, indent=2)

    for i in range(args.frames):
        theta = 2 * math.pi * i / args.frames
        phi   = math.radians(15) * math.sin(2 * math.pi * i / args.frames)
        c2w   = orbit_c2w(theta, phi, args.orbit_radius)

        left_rgb,  left_depth  = render_frame(K, W, H, c2w, baseline=0.0)
        right_rgb, _           = render_frame(K, W, H, c2w, baseline=args.baseline)

        stem = os.path.join(args.out, "frames", f"{i:04d}")
        Image.fromarray(left_rgb).save(f"{stem}_left.png")
        Image.fromarray(right_rgb).save(f"{stem}_right.png")
        np.save(f"{stem}_depth.npy", left_depth)
        np.save(f"{stem}_pose.npy",  c2w)

        print(f"  frame {i:04d}  theta={math.degrees(theta):.1f}°")

    print(f"\nDone. {args.frames} stereo pairs written to {args.out}/")
    print(f"Calibration saved to {args.out}/calib.json")


if __name__ == "__main__":
    main()
