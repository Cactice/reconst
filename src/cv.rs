//! Minimal OpenCV-equivalent algorithms for the stereo → 3DGS pipeline.
//!
//! Sections mirror OpenCV's module layout:
//!   imgproc  — image filtering and gradients
//!   features — corner detection and feature tracking
//!   stereo   — block-matching disparity
//!   calib3d  — back-projection and pose estimation (PnP)

use image::GrayImage;
use nalgebra::{DMatrix, Matrix3, Matrix4};

// ---------------------------------------------------------------------------
// imgproc
// ---------------------------------------------------------------------------

fn sobel(img: &GrayImage) -> (Vec<f32>, Vec<f32>) {
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    let mut ix = vec![0.0f32; w * h];
    let mut iy = vec![0.0f32; w * h];
    let get = |x: usize, y: usize| img.get_pixel(x as u32, y as u32)[0] as f32;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            ix[y * w + x] = (get(x+1,y-1) - get(x-1,y-1)
                           + 2.0*get(x+1,y) - 2.0*get(x-1,y)
                           + get(x+1,y+1) - get(x-1,y+1)) / 8.0;
            iy[y * w + x] = (get(x-1,y+1) - get(x-1,y-1)
                           + 2.0*get(x,y+1) - 2.0*get(x,y-1)
                           + get(x+1,y+1) - get(x+1,y-1)) / 8.0;
        }
    }
    (ix, iy)
}

/// Separable box blur of radius `r` (window = 2r+1).
fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    // Two-pass: horizontal then vertical, using a sliding sum.
    fn pass_1d(src: &[f32], dst: &mut [f32], stride: usize, len: usize, n_lines: usize, r: usize) {
        for line in 0..n_lines {
            let (mut sum, mut cnt) = (0.0f32, 0usize);
            for i in 0..len {
                sum += src[line * stride + i * (stride == 1) as usize
                    + i * (stride != 1) as usize * 0]; // placeholder; see below
                let _ = (sum, cnt); // suppress warning
            }
            // Simple per-line sliding window
            let base = if stride == 1 { line * len } else { line };
            let step = if stride == 1 { 1 } else { stride };
            let (mut s, mut c) = (0.0f32, 0usize);
            for i in 0..len {
                s += src[base + i * step];
                c += 1;
                if i >= r { dst[base + (i - r) * step] = s / c as f32; }
                if i >= 2 * r { s -= src[base + (i - 2 * r) * step]; c -= 1; }
            }
            // trailing tail (right edge)
            for i in len - r..len {
                s -= src[base + (i - r) * step];
                c -= 1;
                dst[base + i * step] = s / c as f32;
            }
        }
    }
    let mut tmp = vec![0.0f32; w * h];
    let mut out = vec![0.0f32; w * h];
    // Horizontal pass (stride-1, line=row)
    pass_1d(src, &mut tmp, 1, w, h, r);
    // Vertical pass (stride=w, line=col)
    pass_1d(&tmp, &mut out, w, h, w, r);
    out
}

// ---------------------------------------------------------------------------
// features — Harris corner detection + NCC patch tracker
// ---------------------------------------------------------------------------

/// Harris corner detector. Returns up to `max_corners` pixel coordinates [x, y].
pub fn detect_corners(img: &GrayImage, max_corners: usize) -> Vec<[f32; 2]> {
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);

    let (ix, iy) = sobel(img);
    let ixx = ix.iter().zip(ix.iter()).map(|(a, b)| a * b).collect::<Vec<_>>();
    let ixy = ix.iter().zip(iy.iter()).map(|(a, b)| a * b).collect::<Vec<_>>();
    let iyy = iy.iter().zip(iy.iter()).map(|(a, b)| a * b).collect::<Vec<_>>();

    let r = 2usize;
    let ixx = box_blur(&ixx, w, h, r);
    let ixy = box_blur(&ixy, w, h, r);
    let iyy = box_blur(&iyy, w, h, r);

    const K: f32 = 0.04;
    let scores: Vec<f32> = (0..w * h)
        .map(|i| {
            let (a, b, c) = (ixx[i], ixy[i], iyy[i]);
            a * c - b * b - K * (a + c) * (a + c)
        })
        .collect();

    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let thresh = max_s * 0.01;
    let nms = 5usize;

    let mut corners: Vec<([f32; 2], f32)> = Vec::new();
    for y in nms..h - nms {
        for x in nms..w - nms {
            let s = scores[y * w + x];
            if s < thresh {
                continue;
            }
            let is_max = (y.saturating_sub(nms)..=(y + nms).min(h - 1)).all(|ny| {
                (x.saturating_sub(nms)..=(x + nms).min(w - 1))
                    .all(|nx| (ny == y && nx == x) || scores[ny * w + nx] <= s)
            });
            if is_max {
                corners.push(([x as f32, y as f32], s));
            }
        }
    }
    corners.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    corners.truncate(max_corners);
    corners.into_iter().map(|(pt, _)| pt).collect()
}

/// NCC patch tracker. Searches a `±SEARCH_HALF` window in `curr` for each point from `prev`.
/// Returns `None` where tracking fails (flat patch, out-of-bounds, NCC < threshold).
pub fn track_features(prev: &GrayImage, curr: &GrayImage, pts: &[[f32; 2]]) -> Vec<Option<[f32; 2]>> {
    const PATCH_HALF: i32 = 5;
    const SEARCH_HALF: i32 = 80;
    const NCC_THRESH: f32 = 0.70;

    let (w, h) = (prev.width() as i32, prev.height() as i32);
    let ph = PATCH_HALF;
    let pn = ((2 * ph + 1) * (2 * ph + 1)) as usize;

    pts.iter()
        .map(|&[px, py]| {
            let (xi, yi) = (px as i32, py as i32);
            if xi - ph < 0 || xi + ph >= w || yi - ph < 0 || yi + ph >= h {
                return None;
            }
            let patch: Vec<f32> = (-ph..=ph)
                .flat_map(|dy| (-ph..=ph).map(move |dx| (dx, dy)))
                .map(|(dx, dy)| prev.get_pixel((xi+dx) as u32, (yi+dy) as u32)[0] as f32)
                .collect();
            let pm = patch.iter().sum::<f32>() / pn as f32;
            let ps = (patch.iter().map(|&v| (v-pm).powi(2)).sum::<f32>() / pn as f32).sqrt();
            if ps < 1.0 { return None; }

            let mut best_ncc = NCC_THRESH;
            let mut best: Option<[f32; 2]> = None;

            for sy in -SEARCH_HALF..=SEARCH_HALF {
                for sx in -SEARCH_HALF..=SEARCH_HALF {
                    let (cx, cy) = (xi + sx, yi + sy);
                    if cx - ph < 0 || cx + ph >= w || cy - ph < 0 || cy + ph >= h { continue; }
                    let win: Vec<f32> = (-ph..=ph)
                        .flat_map(|dy| (-ph..=ph).map(move |dx| (dx, dy)))
                        .map(|(dx, dy)| curr.get_pixel((cx+dx) as u32, (cy+dy) as u32)[0] as f32)
                        .collect();
                    let wm = win.iter().sum::<f32>() / pn as f32;
                    let ws = (win.iter().map(|&v| (v-wm).powi(2)).sum::<f32>() / pn as f32).sqrt();
                    if ws < 1.0 { continue; }
                    let ncc: f32 = patch.iter().zip(win.iter())
                        .map(|(&p, &w)| (p-pm)*(w-wm))
                        .sum::<f32>() / (pn as f32 * ps * ws);
                    if ncc > best_ncc { best_ncc = ncc; best = Some([cx as f32, cy as f32]); }
                }
            }
            best
        })
        .collect()
}

// ---------------------------------------------------------------------------
// stereo — SAD block-matching disparity
// ---------------------------------------------------------------------------

pub struct StereoConfig {
    pub min_disp: i32,
    pub num_disp: i32,
    pub block_size: i32,
}

impl Default for StereoConfig {
    fn default() -> Self { Self { min_disp: 1, num_disp: 64, block_size: 5 } }
}

/// SAD block-matching on rectified stereo pair. Returns disparity map (0 = invalid).
pub fn compute_disparity(left: &GrayImage, right: &GrayImage, cfg: &StereoConfig) -> Vec<f32> {
    let (w, h) = (left.width() as i32, left.height() as i32);
    let half = cfg.block_size / 2;
    let mut disp = vec![0.0f32; (w * h) as usize];

    for row in half..h - half {
        for col in half..w - half {
            let (mut best_sad, mut best_d) = (u32::MAX, 0i32);
            for d in cfg.min_disp..cfg.min_disp + cfg.num_disp {
                let col_r = col - d;
                if col_r < half || col_r >= w - half { continue; }
                let mut sad = 0u32;
                for dy in -half..=half {
                    for dx in -half..=half {
                        let r = (row + dy) as u32;
                        sad += (left .get_pixel((col   + dx) as u32, r)[0] as u32)
                            .abs_diff(right.get_pixel((col_r + dx) as u32, r)[0] as u32);
                    }
                }
                if sad < best_sad { best_sad = sad; best_d = d; }
            }
            if best_d > cfg.min_disp { disp[(row * w + col) as usize] = best_d as f32; }
        }
    }
    disp
}

pub fn disparity_to_depth(disp: &[f32], fx: f64, baseline: f64, min_disp: f32) -> Vec<f32> {
    disp.iter().map(|&d| if d > min_disp { (fx * baseline / d as f64) as f32 } else { 0.0 }).collect()
}

// ---------------------------------------------------------------------------
// calib3d — back-projection and PnP pose estimation
// ---------------------------------------------------------------------------

/// Sample depth at a 2-D point (nearest-pixel).
pub fn depth_sample(depth: &[f32], width: u32, pt: [f32; 2]) -> f32 {
    let x = (pt[0].round() as u32).min(width - 1);
    let y = (pt[1].round() as u32).min((depth.len() as u32 / width).saturating_sub(1));
    depth[(y * width + x) as usize]
}

/// Back-project a 2-D pixel + metric depth → 3-D point in camera space.
pub fn unproject(pt2d: [f32; 2], depth: f32, k: &Matrix3<f64>) -> [f64; 3] {
    let d = depth as f64;
    [(pt2d[0] as f64 - k[(0,2)]) / k[(0,0)] * d,
     (pt2d[1] as f64 - k[(1,2)]) / k[(1,1)] * d,
     d]
}

/// DLT PnP (≥6 correspondences, known K).
/// Returns w2c: maps prev-camera 3-D points → current camera space.
fn dlt_pnp(pts3d: &[[f64; 3]], pts2d: &[[f64; 2]], k: &Matrix3<f64>) -> Option<Matrix4<f64>> {
    let n = pts3d.len();
    debug_assert!(n >= 6);
    let k_inv = k.try_inverse()?;

    let mut a_data = vec![0.0f64; 2 * n * 12];
    for (i, (&[x, y, z], &[u, v])) in pts3d.iter().zip(pts2d.iter()).enumerate() {
        let pn = k_inv * nalgebra::Vector3::new(u, v, 1.0);
        let (xn, yn) = (pn[0], pn[1]);
        let r0 = &mut a_data[2*i*12..(2*i+1)*12];
        r0[0]=x; r0[1]=y; r0[2]=z; r0[3]=1.0;
        r0[8]=-xn*x; r0[9]=-xn*y; r0[10]=-xn*z; r0[11]=-xn;
        let r1 = &mut a_data[(2*i+1)*12..(2*i+2)*12];
        r1[4]=x; r1[5]=y; r1[6]=z; r1[7]=1.0;
        r1[8]=-yn*x; r1[9]=-yn*y; r1[10]=-yn*z; r1[11]=-yn;
    }

    let a   = DMatrix::from_row_slice(2 * n, 12, &a_data);
    let vt  = a.svd(false, true).v_t?;
    let mut m: Vec<f64> = (0..12).map(|j| vt[(11, j)]).collect();

    // Choose sign: M[2]·X_h > 0 (scene in front of camera)
    if m[8]*pts3d[0][0] + m[9]*pts3d[0][1] + m[10]*pts3d[0][2] + m[11] < 0.0 {
        for v in &mut m { *v = -*v; }
    }

    let r_approx = Matrix3::new(m[0],m[1],m[2], m[4],m[5],m[6], m[8],m[9],m[10]);
    let t_raw    = [m[3], m[7], m[11]];

    let svd2 = r_approx.svd(true, true);
    let u = svd2.u?; let vt2 = svd2.v_t?; let s = &svd2.singular_values;
    let mut diag = Matrix3::identity();
    diag[(2,2)] = (u * vt2).determinant().signum();
    let r = u * diag * vt2;

    let alpha = (s[0] + s[1] + s[2]) / 3.0;
    if alpha.abs() < 1e-10 { return None; }
    let tv = nalgebra::Vector3::new(t_raw[0], t_raw[1], t_raw[2]) / alpha;

    let mut w2c = Matrix4::identity();
    for i in 0..3 { for j in 0..3 { w2c[(i,j)] = r[(i,j)]; } w2c[(i,3)] = tv[i]; }
    Some(w2c)
}

fn repro_err(p3: [f64;3], p2: [f64;2], k: &Matrix3<f64>, w2c: &Matrix4<f64>) -> f64 {
    let p = w2c * nalgebra::Vector4::new(p3[0], p3[1], p3[2], 1.0);
    if p[2] <= 0.0 { return f64::INFINITY; }
    let (pu, pv) = (k[(0,0)]*p[0]/p[2]+k[(0,2)], k[(1,1)]*p[1]/p[2]+k[(1,2)]);
    ((pu-p2[0]).powi(2) + (pv-p2[1]).powi(2)).sqrt()
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn sample6(&mut self, n: usize) -> [usize; 6] {
        let mut out = [0usize; 6];
        let mut k = 0;
        while k < 6 {
            let idx = (self.next() as usize) % n;
            if !out[..k].contains(&idx) { out[k] = idx; k += 1; }
        }
        out
    }
}

/// RANSAC PnP. Returns w2c (prev-camera → current-camera transform).
pub fn solve_pnp_ransac(
    pts3d: &[[f64; 3]],
    pts2d: &[[f64; 2]],
    k: &Matrix3<f64>,
    n_iters: usize,
    thresh_px: f64,
) -> Option<Matrix4<f64>> {
    if pts3d.len() < 6 { return None; }
    let n = pts3d.len();
    let mut rng = Lcg(0xdeadbeef12345678);
    let (mut best_n, mut best_w2c) = (0usize, None);

    for _ in 0..n_iters {
        let idx = rng.sample6(n);
        let s3: Vec<_> = idx.iter().map(|&i| pts3d[i]).collect();
        let s2: Vec<_> = idx.iter().map(|&i| pts2d[i]).collect();
        if let Some(w2c) = dlt_pnp(&s3, &s2, k) {
            let inl = pts3d.iter().zip(pts2d.iter())
                .filter(|(&p3,&p2)| repro_err(p3, p2, k, &w2c) < thresh_px)
                .count();
            if inl > best_n { best_n = inl; best_w2c = Some(w2c); }
        }
    }

    // Refine on all inliers
    if let Some(ref w2c) = best_w2c {
        let (s3, s2): (Vec<_>, Vec<_>) = pts3d.iter().zip(pts2d.iter())
            .filter(|(&p3,&p2)| repro_err(p3, p2, k, w2c) < thresh_px)
            .unzip();
        if s3.len() >= 6 { return dlt_pnp(&s3, &s2, k).or(best_w2c); }
    }
    best_w2c
}
