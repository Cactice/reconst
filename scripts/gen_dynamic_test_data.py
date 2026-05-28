"""
Generate synthetic DYNAMIC stereo test data for 4DGS pipeline development.

Scene: colored spheres move smoothly over time while a stereo camera
orbits the scene.  Each timestep has N_VIEWS stereo pairs from different
viewpoints — exactly the format multi-view dynamic 4DGS methods expect.

Output layout:
  data/dynamic_test/
    calib.json
    t{TTT}/
      v{VVV}_left.png
      v{VVV}_right.png
      v{VVV}_depth.npy    (metric depth, left cam, ground truth)
      v{VVV}_pose.npy     (4x4 c2w, left cam)

Time  axis : T timesteps of object motion
View  axis : V camera positions per timestep (different orbit angles)

Compatible with most 4DGS loaders that expect:
  - per-frame images organised by time
  - shared camera calibration
  - (optional) ground-truth depth/poses
"""

import argparse
import json
import math
import os

import numpy as np
from PIL import Image

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dynamic_test")

# ---------------------------------------------------------------------------
# Dynamic scene: each sphere has a sinusoidal orbit in XYZ
# ---------------------------------------------------------------------------

BASE_SPHERES = [
    # (center_xyz, radius, RGB, amp_xyz, freq_xyz)
    ([0.0,  0.0,  0.0],  0.30, (220,  60,  60), [0.4, 0.0, 0.3], [1.0, 0.0, 1.3]),
    ([0.7,  0.2,  0.4],  0.20, ( 60, 180,  60), [0.2, 0.3, 0.0], [0.8, 1.2, 0.0]),
    ([-0.5, 0.4,  0.6],  0.25, ( 60,  90, 220), [0.3, 0.2, 0.3], [1.5, 0.7, 1.0]),
    ([0.2, -0.5,  0.3],  0.18, (220, 180,  40), [0.0, 0.4, 0.2], [0.0, 1.0, 0.8]),
    ([-0.3,-0.2, -0.4],  0.22, (160,  60, 200), [0.3, 0.1, 0.3], [1.2, 1.5, 0.9]),
]


def spheres_at(t_norm: float):
    """Instantiate sphere centers at normalised time t_norm ∈ [0, 1]."""
    t = t_norm * 2 * math.pi
    result = []
    for (cx, cy, cz), radius, rgb, amp, freq in BASE_SPHERES:
        nx = cx + amp[0] * math.sin(freq[0] * t)
        ny = cy + amp[1] * math.sin(freq[1] * t)
        nz = cz + amp[2] * math.cos(freq[2] * t)
        result.append(([nx, ny, nz], radius, rgb))
    return result


# ---------------------------------------------------------------------------
# Shared rendering helpers (mirrors gen_stereo_test_data.py)
# ---------------------------------------------------------------------------

def ray_sphere_intersect(rays_o, rays_d, center, radius):
    oc = rays_o - center
    b  = (oc * rays_d).sum(-1)
    c  = (oc * oc).sum(-1) - radius * radius
    disc = b * b - c
    hit = disc >= 0
    t = np.full(len(rays_d), np.inf)
    sq = np.sqrt(np.maximum(disc[hit], 0))
    t1 = -b[hit] - sq
    t2 = -b[hit] + sq
    t[hit] = np.where(t1 > 1e-4, t1, np.where(t2 > 1e-4, t2, np.inf))
    return t


def render_frame(spheres, K, width, height, c2w, baseline=0.0):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u, v = u.astype(np.float32), v.astype(np.float32)
    dirs_cam = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1)
    dirs_cam = dirs_cam.reshape(-1, 3)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    R = c2w[:3, :3]
    pos = c2w[:3, 3].copy() + R @ np.array([baseline, 0.0, 0.0])
    rays_o = np.broadcast_to(pos, dirs_cam.shape).copy()
    rays_d = dirs_cam @ R.T

    depth = np.full(height * width, np.inf)
    color = np.zeros((height * width, 3), dtype=np.float32)
    light = np.array([0.5, 1.0, -0.5])
    light /= np.linalg.norm(light)

    for center, radius, rgb in spheres:
        c = np.array(center, dtype=np.float32)
        t_hit = ray_sphere_intersect(rays_o, rays_d, c, radius)
        closer = t_hit < depth
        depth[closer] = t_hit[closer]
        hit_pts = rays_o[closer] + t_hit[closer, None] * rays_d[closer]
        normals = (hit_pts - c) / radius
        diff = np.clip((normals * light).sum(-1), 0.1, 1.0)
        col = np.array(rgb, dtype=np.float32) / 255.0
        color[closer] = col * diff[:, None]

    miss = np.isinf(depth)
    color[miss] = np.array([0.15, 0.15, 0.2])
    depth[miss] = 0.0

    rgb_img   = np.clip(color * 255, 0, 255).astype(np.uint8).reshape(height, width, 3)
    depth_map = depth.astype(np.float32).reshape(height, width)
    return rgb_img, depth_map


def orbit_c2w(theta, phi, radius):
    x = radius * math.cos(phi) * math.cos(theta)
    y = radius * math.sin(phi)
    z = radius * math.cos(phi) * math.sin(theta)
    pos = np.array([x, y, z])
    forward = -pos / np.linalg.norm(pos)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up_corrected = np.cross(right, forward)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = -up_corrected  # camera y = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = pos
    return c2w


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=24,
                        help="number of time steps (motion frames)")
    parser.add_argument("--views", type=int, default=6,
                        help="camera viewpoints per timestep")
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--baseline", type=float, default=0.06)
    parser.add_argument("--orbit-radius", type=float, default=2.5)
    parser.add_argument("--out", default=OUT_DIR)
    args = parser.parse_args()

    W, H = args.width, args.height
    fx = fy = 0.8 * W
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    calib = {
        "width": W, "height": H,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "baseline": args.baseline,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "camera_model": "PINHOLE",
        "timesteps": args.timesteps,
        "views_per_timestep": args.views,
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "calib.json"), "w") as f:
        json.dump(calib, f, indent=2)

    total = args.timesteps * args.views
    done = 0
    for ti in range(args.timesteps):
        t_norm   = ti / args.timesteps
        spheres  = spheres_at(t_norm)
        t_dir    = os.path.join(args.out, f"t{ti:03d}")
        os.makedirs(t_dir, exist_ok=True)

        for vi in range(args.views):
            theta = 2 * math.pi * vi / args.views
            phi   = math.radians(10) * math.sin(2 * math.pi * vi / args.views)
            c2w   = orbit_c2w(theta, phi, args.orbit_radius)

            left_rgb,  left_depth = render_frame(spheres, K, W, H, c2w, 0.0)
            right_rgb, _          = render_frame(spheres, K, W, H, c2w, args.baseline)

            stem = os.path.join(t_dir, f"v{vi:03d}")
            Image.fromarray(left_rgb).save(f"{stem}_left.png")
            Image.fromarray(right_rgb).save(f"{stem}_right.png")
            np.save(f"{stem}_depth.npy", left_depth)
            np.save(f"{stem}_pose.npy",  c2w)

            done += 1
            if done % args.views == 0:
                print(f"  t={ti:03d}/{args.timesteps}  ({done}/{total})")

    print(f"\nDone. {total} stereo pairs in {args.out}/")
    print(f"  {args.timesteps} timesteps × {args.views} views")


if __name__ == "__main__":
    main()
