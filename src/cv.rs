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

/// Separable box blur of radius `r` (window = 2r+1), edges clamped to the
/// available samples (a shrinking window, not zero-padded).
fn box_blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    // One pass over `n_lines` lines, each `len` samples apart by `stride`.
    fn pass_1d(src: &[f32], dst: &mut [f32], stride: usize, len: usize, n_lines: usize, r: usize) {
        for line in 0..n_lines {
            // First element of this line, and the step between samples.
            let (base, step) = if stride == 1 { (line * len, 1) } else { (line, stride) };
            // Prefix-style running sum over [lo, hi); both bounds advance monotonically.
            let (mut sum, mut lo, mut hi) = (0.0f32, 0usize, 0usize);
            for i in 0..len {
                let want_hi = (i + r + 1).min(len); // exclusive upper bound of window
                let want_lo = i.saturating_sub(r);   // inclusive lower bound
                while hi < want_hi { sum += src[base + hi * step]; hi += 1; }
                while lo < want_lo { sum -= src[base + lo * step]; lo += 1; }
                dst[base + i * step] = sum / (hi - lo) as f32;
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

    // Shi-Tomasi-style quality gate relative to the strongest response, kept
    // low so textured interiors aren't starved by a few dominant corners.
    const QUALITY: f32 = 0.005;
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let thresh = max_s * QUALITY;
    let nms = 3usize;

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

/// NCC patch tracker. Searches a `±search_half` window in `curr` for each point
/// from `prev`. Returns `None` where tracking fails (flat patch, out-of-bounds,
/// NCC < threshold). Pick `search_half` to bound the largest expected
/// inter-frame pixel displacement; cost grows with its square.
pub fn track_features(
    prev: &GrayImage,
    curr: &GrayImage,
    pts: &[[f32; 2]],
    search_half: i32,
) -> Vec<Option<[f32; 2]>> {
    const PATCH_HALF: i32 = 5;
    const NCC_THRESH: f32 = 0.70;
    let search_half = search_half.max(1);

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

            for sy in -search_half..=search_half {
                for sx in -search_half..=search_half {
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

    let sad_at = |col: i32, row: i32, d: i32| -> Option<u32> {
        let col_r = col - d;
        if col_r < half || col_r >= w - half { return None; }
        let mut sad = 0u32;
        for dy in -half..=half {
            for dx in -half..=half {
                let r = (row + dy) as u32;
                sad += (left .get_pixel((col   + dx) as u32, r)[0] as u32)
                    .abs_diff(right.get_pixel((col_r + dx) as u32, r)[0] as u32);
            }
        }
        Some(sad)
    };

    for row in half..h - half {
        for col in half..w - half {
            let (mut best_sad, mut best_d) = (u32::MAX, 0i32);
            for d in cfg.min_disp..cfg.min_disp + cfg.num_disp {
                if let Some(sad) = sad_at(col, row, d) {
                    if sad < best_sad { best_sad = sad; best_d = d; }
                }
            }
            if best_d > cfg.min_disp {
                // Sub-pixel refinement: parabola fit through the SAD minimum and
                // its two neighbours (equiangular interpolation).
                let mut sub = best_d as f32;
                if let (Some(sm), Some(sp)) =
                    (sad_at(col, row, best_d - 1), sad_at(col, row, best_d + 1))
                {
                    let (sm, s0, sp) = (sm as f32, best_sad as f32, sp as f32);
                    let denom = sm - 2.0 * s0 + sp;
                    if denom.abs() > 1e-3 {
                        sub += (0.5 * (sm - sp) / denom).clamp(-0.5, 0.5);
                    }
                }
                disp[(row * w + col) as usize] = sub;
            }
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
///
/// Uses Hartley-style isotropic normalization of both the 3-D points and the
/// (K-normalized) image points before the linear solve — without it the DLT is
/// badly conditioned and the recovered pose is unreliable.
fn dlt_pnp(pts3d: &[[f64; 3]], pts2d: &[[f64; 2]], k: &Matrix3<f64>) -> Option<Matrix4<f64>> {
    let n = pts3d.len();
    debug_assert!(n >= 6);
    let k_inv = k.try_inverse()?;

    // K-normalized image points.
    let xn: Vec<[f64; 2]> = pts2d.iter().map(|&[u, v]| {
        let p = k_inv * nalgebra::Vector3::new(u, v, 1.0);
        [p[0] / p[2], p[1] / p[2]]
    }).collect();

    // Isotropic normalization: centroid to origin, mean distance sqrt(dim).
    let centroid3 = pts3d.iter().fold([0.0; 3], |a, p| [a[0]+p[0], a[1]+p[1], a[2]+p[2]])
        .map(|s| s / n as f64);
    let d3 = pts3d.iter().map(|p| {
        let (dx, dy, dz) = (p[0]-centroid3[0], p[1]-centroid3[1], p[2]-centroid3[2]);
        (dx*dx + dy*dy + dz*dz).sqrt()
    }).sum::<f64>() / n as f64;
    if d3 < 1e-9 { return None; }
    let s3 = (3.0f64).sqrt() / d3;

    let centroid2 = xn.iter().fold([0.0; 2], |a, p| [a[0]+p[0], a[1]+p[1]]).map(|s| s / n as f64);
    let d2 = xn.iter().map(|p| {
        let (dx, dy) = (p[0]-centroid2[0], p[1]-centroid2[1]);
        (dx*dx + dy*dy).sqrt()
    }).sum::<f64>() / n as f64;
    if d2 < 1e-9 { return None; }
    let s2 = (2.0f64).sqrt() / d2;

    let mut a_data = vec![0.0f64; 2 * n * 12];
    for (i, (&[x, y, z], &[xi, yi])) in pts3d.iter().zip(xn.iter()).enumerate() {
        // Normalized coordinates.
        let (xh, yh, zh) = (s3*(x-centroid3[0]), s3*(y-centroid3[1]), s3*(z-centroid3[2]));
        let (un, vn) = (s2*(xi-centroid2[0]), s2*(yi-centroid2[1]));
        let r0 = &mut a_data[2*i*12..(2*i+1)*12];
        r0[0]=xh; r0[1]=yh; r0[2]=zh; r0[3]=1.0;
        r0[8]=-un*xh; r0[9]=-un*yh; r0[10]=-un*zh; r0[11]=-un;
        let r1 = &mut a_data[(2*i+1)*12..(2*i+2)*12];
        r1[4]=xh; r1[5]=yh; r1[6]=zh; r1[7]=1.0;
        r1[8]=-vn*xh; r1[9]=-vn*yh; r1[10]=-vn*zh; r1[11]=-vn;
    }

    let a   = DMatrix::from_row_slice(2 * n, 12, &a_data);
    let vt  = a.svd(false, true).v_t?;
    let mn: Vec<f64> = (0..12).map(|j| vt[(11, j)]).collect();

    // Denormalize: M = T2^-1 * M_norm * U3, with
    //   T2 = [[s2,0,-s2 cx2],[0,s2,-s2 cy2],[0,0,1]]   (image)
    //   U3 = diag(s3,s3,s3,1) shifted by -s3*centroid3  (world)
    let m_norm = nalgebra::Matrix3x4::from_row_slice(&mn);
    let t2_inv = Matrix3::new(
        1.0/s2, 0.0,   centroid2[0],
        0.0,   1.0/s2, centroid2[1],
        0.0,   0.0,    1.0,
    );
    let u3 = nalgebra::Matrix4::new(
        s3,  0.0, 0.0, -s3*centroid3[0],
        0.0, s3,  0.0, -s3*centroid3[1],
        0.0, 0.0, s3,  -s3*centroid3[2],
        0.0, 0.0, 0.0, 1.0,
    );
    let m_mat = t2_inv * m_norm * u3; // 3x4, maps world homog -> normalized image homog
    let mut m: [f64; 12] = [0.0; 12];
    for r in 0..3 { for c in 0..4 { m[r*4 + c] = m_mat[(r, c)]; } }

    // Choose sign: depth of first point must be positive (in front of camera).
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

/// Minimum inlier support required to accept a pose. Rejects degenerate /
/// outlier-dominated hypotheses that would otherwise corrupt the trajectory.
const MIN_PNP_INLIERS: usize = 12;

fn count_inliers(
    pts3d: &[[f64; 3]], pts2d: &[[f64; 2]], k: &Matrix3<f64>,
    w2c: &Matrix4<f64>, thresh_px: f64,
) -> usize {
    pts3d.iter().zip(pts2d.iter())
        .filter(|(&p3, &p2)| repro_err(p3, p2, k, w2c) < thresh_px)
        .count()
}

fn inlier_subset(
    pts3d: &[[f64; 3]], pts2d: &[[f64; 2]], k: &Matrix3<f64>,
    w2c: &Matrix4<f64>, thresh_px: f64,
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
    pts3d.iter().zip(pts2d.iter())
        .filter(|(&p3, &p2)| repro_err(p3, p2, k, w2c) < thresh_px)
        .map(|(&p3, &p2)| (p3, p2))
        .unzip()
}

/// RANSAC PnP. Returns w2c (prev-camera → current-camera transform), or `None`
/// if no pose attains `MIN_PNP_INLIERS` support (caller should then hold pose).
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
            let inl = count_inliers(pts3d, pts2d, k, &w2c, thresh_px);
            if inl > best_n { best_n = inl; best_w2c = Some(w2c); }
        }
    }

    let mut best = best_w2c?;
    if best_n < MIN_PNP_INLIERS { return None; }

    // Iteratively refit on the inlier set, but only keep a refit that does not
    // lose support — an unchecked refit can diverge to a worse pose.
    for _ in 0..5 {
        let (s3, s2) = inlier_subset(pts3d, pts2d, k, &best, thresh_px);
        if s3.len() < 6 { break; }
        match dlt_pnp(&s3, &s2, k) {
            Some(cand) => {
                let inl = count_inliers(pts3d, pts2d, k, &cand, thresh_px);
                if inl >= best_n { best = cand; best_n = inl; } else { break; }
            }
            None => break,
        }
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn box_blur_clamped_window_is_correct() {
        // Single row; verify the clamped (shrinking-edge) averages exactly.
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = box_blur(&src, 6, 1, 2);
        let expect = [
            (1.0 + 2.0 + 3.0) / 3.0,             // i=0 -> [0..2]
            (1.0 + 2.0 + 3.0 + 4.0) / 4.0,       // i=1 -> [0..3]
            (1.0 + 2.0 + 3.0 + 4.0 + 5.0) / 5.0, // i=2 -> [0..4]
            (2.0 + 3.0 + 4.0 + 5.0 + 6.0) / 5.0, // i=3 -> [1..5]
            (3.0 + 4.0 + 5.0 + 6.0) / 4.0,       // i=4 -> [2..5]  (right edge)
            (4.0 + 5.0 + 6.0) / 3.0,             // i=5 -> [3..5]
        ];
        for (o, e) in out.iter().zip(expect.iter()) {
            assert!((o - e).abs() < 1e-5, "got {o}, want {e}");
        }
    }

    #[test]
    fn box_blur_does_not_panic_on_non_square() {
        // Regression: the vertical pass must index within a w*h buffer when h<w.
        let (w, h) = (32usize, 8usize);
        let src = vec![1.0f32; w * h];
        let out = box_blur(&src, w, h, 2);
        assert_eq!(out.len(), w * h);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn unproject_roundtrips_through_projection() {
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let p = unproject([400.0, 300.0], 2.5, &k);
        let u = k[(0,0)] * p[0] / p[2] + k[(0,2)];
        let v = k[(1,1)] * p[1] / p[2] + k[(1,2)];
        assert!((u - 400.0).abs() < 1e-6 && (v - 300.0).abs() < 1e-6);
        assert!((p[2] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn pnp_recovers_a_known_pose() {
        // Synthetic: 3-D points in prev-cam, a known w2c, project into curr-cam.
        let k = Matrix3::new(500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0);
        let yaw = 0.05f64;
        let (c, s) = (yaw.cos(), yaw.sin());
        let mut w2c = Matrix4::identity();
        w2c[(0,0)] = c; w2c[(0,2)] = s; w2c[(2,0)] = -s; w2c[(2,2)] = c;
        w2c[(0,3)] = 0.1; w2c[(1,3)] = -0.05; w2c[(2,3)] = 0.2;

        let mut p3 = Vec::new();
        let mut p2 = Vec::new();
        for i in 0..30 {
            let x = (i % 5) as f64 * 0.2 - 0.4;
            let y = (i / 5) as f64 * 0.2 - 0.4;
            // Non-coplanar depth (DLT PnP is degenerate for coplanar points).
            let z = 2.5 + 0.6 * (i as f64 * 1.3).sin();
            let p = w2c * nalgebra::Vector4::new(x, y, z, 1.0);
            p3.push([x, y, z]);
            p2.push([k[(0,0)]*p[0]/p[2]+k[(0,2)], k[(1,1)]*p[1]/p[2]+k[(1,2)]]);
        }
        let est = solve_pnp_ransac(&p3, &p2, &k, 200, 1.0).expect("pose");
        for r in 0..3 {
            assert!((est[(r,3)] - w2c[(r,3)]).abs() < 1e-2, "t[{r}] off");
        }
        // Rotation should match too.
        let rel = w2c * est.try_inverse().unwrap();
        let trace = rel[(0,0)] + rel[(1,1)] + rel[(2,2)];
        let ang = (((trace - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees();
        assert!(ang < 1.0, "rotation error {ang}° too large");
    }

    #[test]
    fn detect_corners_finds_a_checker_intersection() {
        // 2x2 high-contrast blocks → one strong corner near the centre.
        let (w, h) = (40u32, 40u32);
        let mut img = GrayImage::from_pixel(w, h, Luma([0u8]));
        for y in 0..h { for x in 0..w {
            let v = if (x < 20) ^ (y < 20) { 255 } else { 0 };
            img.put_pixel(x, y, Luma([v]));
        }}
        let corners = detect_corners(&img, 10);
        assert!(!corners.is_empty());
        assert!(corners.iter().any(|c| (c[0] - 20.0).abs() < 4.0 && (c[1] - 20.0).abs() < 4.0));
    }
}
