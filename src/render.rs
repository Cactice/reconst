use image::{Rgb, RgbImage};
use nalgebra::{Matrix3, Matrix4, Vector3};

pub struct Sphere {
    pub center: Vector3<f32>,
    pub radius: f32,
    pub color: [u8; 3],
    /// Checkerboard frequency on the surface (0 = solid colour, 6 = good for VO)
    pub checker_freq: u8,
}

fn ray_sphere(ro: Vector3<f32>, rd: Vector3<f32>, c: Vector3<f32>, r: f32) -> f32 {
    let oc = ro - c;
    let b = oc.dot(&rd);
    let disc = b * b - oc.dot(&oc) + r * r;
    if disc < 0.0 {
        return f32::INFINITY;
    }
    let sq = disc.sqrt();
    let t1 = -b - sq;
    let t2 = -b + sq;
    if t1 > 1e-4 { t1 } else if t2 > 1e-4 { t2 } else { f32::INFINITY }
}

/// Render one image and depth map (metres, 0 = background).
/// `baseline` shifts the camera right along its own x-axis (metres).
pub fn render_frame(
    spheres: &[Sphere],
    width: u32,
    height: u32,
    k: &Matrix3<f64>,
    c2w: &Matrix4<f64>,
    baseline: f64,
) -> (RgbImage, Vec<f32>) {
    let fx = k[(0, 0)] as f32;
    let fy = k[(1, 1)] as f32;
    let cx = k[(0, 2)] as f32;
    let cy = k[(1, 2)] as f32;

    // Camera origin with baseline shift along camera-right axis
    let cam_pos = Vector3::new(
        (c2w[(0, 3)] + c2w[(0, 0)] * baseline) as f32,
        (c2w[(1, 3)] + c2w[(1, 0)] * baseline) as f32,
        (c2w[(2, 3)] + c2w[(2, 0)] * baseline) as f32,
    );

    // 3x3 rotation (f32)
    let rot = Matrix3::new(
        c2w[(0,0)] as f32, c2w[(0,1)] as f32, c2w[(0,2)] as f32,
        c2w[(1,0)] as f32, c2w[(1,1)] as f32, c2w[(1,2)] as f32,
        c2w[(2,0)] as f32, c2w[(2,1)] as f32, c2w[(2,2)] as f32,
    );

    let light = Vector3::new(0.5f32, 1.0, -0.5).normalize();
    let bg = [38u8, 38, 51];

    let n = (width * height) as usize;
    let mut pixels = vec![bg; n];
    let mut depth  = vec![0.0f32; n];

    for row in 0..height {
        for col in 0..width {
            let dir_cam = Vector3::new(
                (col as f32 - cx) / fx,
                (row as f32 - cy) / fy,
                1.0,
            )
            .normalize();
            let dir_w = rot * dir_cam;

            let mut best_t = f32::INFINITY;
            let mut best_c = bg;

            for s in spheres {
                let t = ray_sphere(cam_pos, dir_w, s.center, s.radius);
                if t < best_t {
                    best_t = t;
                    let hit    = cam_pos + dir_w * t;
                    let normal = (hit - s.center) / s.radius;
                    let diff   = normal.dot(&light).clamp(0.1, 1.0);

                    // Optional checkerboard texture on sphere surface
                    let tex = if s.checker_freq > 0 {
                        let f = s.checker_freq as f32;
                        let phi   = normal[1].asin();          // latitude
                        let theta = normal[2].atan2(normal[0]); // longitude
                        let even  = ((phi * f).floor() as i32 + (theta * f).floor() as i32) % 2 == 0;
                        if even { 1.0 } else { 0.45 }
                    } else {
                        1.0
                    };

                    best_c = s.color.map(|c| (c as f32 * diff * tex) as u8);
                }
            }

            let idx = (row * width + col) as usize;
            if best_t.is_finite() {
                pixels[idx] = best_c;
                depth[idx]  = best_t;
            }
        }
    }

    let mut img = RgbImage::new(width, height);
    for row in 0..height {
        for col in 0..width {
            img.put_pixel(col, row, Rgb(pixels[(row * width + col) as usize]));
        }
    }
    (img, depth)
}
