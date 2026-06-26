use clap::Parser;
use image::GrayImage;
use nalgebra::Matrix4;
use reconst::{
    camera::Calib,
    colmap::{write_workspace, ColmapFrame},
    cv::{
        compute_disparity, depth_sample, detect_corners, disparity_to_depth,
        solve_pnp_ransac, track_features, unproject, StereoConfig,
    },
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Stereo VO → COLMAP workspace (no COLMAP binary needed)")]
struct Args {
    #[arg(long, default_value = "data/stereo_test")]
    r#in: PathBuf,
    #[arg(long, default_value = "colmap_ws_vo")]
    out: PathBuf,
    #[arg(long, default_value = "8")]
    stride: u32,
    #[arg(long, default_value = "20.0")]
    max_depth: f32,
    #[arg(long, default_value = "2000")]
    max_corners: usize,
}

fn load_frames(dir: &PathBuf) -> Vec<(PathBuf, PathBuf, String)> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .expect("frames dir not found")
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with("_left.png"))
        .collect();
    entries.sort_by_key(|e| e.file_name());
    entries
        .into_iter()
        .map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let right = dir.join(name.replace("_left.png", "_right.png"));
            (e.path(), right, name)
        })
        .collect()
}

fn depth_to_world(
    depth: &[f32],
    k: &nalgebra::Matrix3<f64>,
    c2w: &Matrix4<f64>,
    width: u32, height: u32, stride: u32, max_d: f32,
) -> Vec<[f32; 3]> {
    let (fx, fy, cx, cy) = (k[(0,0)], k[(1,1)], k[(0,2)], k[(1,2)]);
    let mut pts = Vec::new();
    for row in (0..height).step_by(stride as usize) {
        for col in (0..width).step_by(stride as usize) {
            let d = depth[(row * width + col) as usize] as f64;
            if d < 0.1 || d as f32 > max_d { continue; }
            let p = c2w * nalgebra::Vector4::new(
                (col as f64 - cx) / fx * d,
                (row as f64 - cy) / fy * d,
                d, 1.0,
            );
            pts.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        }
    }
    pts
}

fn main() {
    let args = Args::parse();
    let calib: Calib = serde_json::from_reader(
        std::fs::File::open(args.r#in.join("calib.json")).expect("calib.json not found"),
    ).expect("failed to parse calib.json");

    let k      = calib.k();
    let frames = load_frames(&args.r#in.join("frames"));
    println!("Found {} stereo pairs", frames.len());

    let cfg = StereoConfig::default();
    std::fs::create_dir_all(args.out.join("images")).unwrap();

    let mut c2w = Matrix4::<f64>::identity();
    let mut all_c2w:  Vec<Matrix4<f64>> = Vec::new();
    let mut all_names: Vec<String>       = Vec::new();
    let mut all_pts:   Vec<[f32; 3]>    = Vec::new();

    let mut prev_gray:    Option<GrayImage>    = None;
    let mut prev_depth:   Option<Vec<f32>>     = None;
    let mut prev_corners: Option<Vec<[f32;2]>> = None;

    for (idx, (left_path, right_path, name)) in frames.iter().enumerate() {
        let left_gray  = image::open(left_path) .expect("read left") .to_luma8();
        let right_gray = image::open(right_path).expect("read right").to_luma8();

        let disp  = compute_disparity(&left_gray, &right_gray, &cfg);
        let depth = disparity_to_depth(&disp, calib.fx, calib.baseline, cfg.min_disp as f32);

        if let (Some(pg), Some(pd), Some(pc)) = (&prev_gray, &prev_depth, &prev_corners) {
            let tracked = track_features(pg, &left_gray, pc);

            let (mut pts3d, mut pts2d) = (Vec::new(), Vec::new());
            for (&prev_pt, curr_opt) in pc.iter().zip(&tracked) {
                if let Some(curr_pt) = curr_opt {
                    let d = depth_sample(pd, calib.width, prev_pt);
                    if d > 0.1 {
                        pts3d.push(unproject(prev_pt, d, &k));
                        pts2d.push([curr_pt[0] as f64, curr_pt[1] as f64]);
                    }
                }
            }

            if pts3d.len() >= 10 {
                if let Some(t_rel) = solve_pnp_ransac(&pts3d, &pts2d, &k, 500, 2.0) {
                    c2w = c2w * t_rel.try_inverse().unwrap_or_else(Matrix4::identity);
                }
            }
        }

        all_c2w.push(c2w);
        all_names.push(name.clone());
        all_pts.extend(depth_to_world(&depth, &k, &c2w, calib.width, calib.height, args.stride, args.max_depth));

        prev_corners = Some(detect_corners(&left_gray, args.max_corners));
        prev_gray    = Some(left_gray);
        prev_depth   = Some(depth);

        std::fs::copy(left_path, args.out.join("images").join(name)).unwrap();
        println!("  [{:3}/{:3}] {}  pos=({:.2},{:.2},{:.2})",
            idx + 1, frames.len(), name,
            c2w[(0,3)], c2w[(1,3)], c2w[(2,3)]);
    }

    const MAX_PTS: usize = 200_000;
    if all_pts.len() > MAX_PTS {
        let step = all_pts.len() / MAX_PTS;
        all_pts = all_pts.into_iter().step_by(step).collect();
    }

    let colmap_frames: Vec<ColmapFrame> = all_names.iter()
        .zip(all_c2w.iter())
        .map(|(name, &c2w)| ColmapFrame { name: name.clone(), c2w })
        .collect();

    write_workspace(&args.out, &calib, &colmap_frames, &all_pts).unwrap();

    println!("\nDone → {}  ({} poses, {} points)",
        args.out.display(), colmap_frames.len(), all_pts.len());
    println!("To train 3DGS:\n  python train.py -s {}",
        std::fs::canonicalize(&args.out).unwrap_or(args.out.clone()).display());
}
