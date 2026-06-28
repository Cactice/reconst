// Build a COLMAP workspace from the generator's GROUND-TRUTH poses + depth,
// bypassing visual odometry. Useful to (a) get a clean reconstruction and
// (b) isolate VO drift from the rest of the pipeline.
use clap::Parser;
use nalgebra::Matrix4;
use reconst::{
    camera::Calib,
    colmap::{write_workspace, ColmapFrame},
    io::{load_depth, load_pose},
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "COLMAP workspace from ground-truth poses + depth (no VO)")]
struct Args {
    #[arg(long, default_value = "data/stereo_test")]
    r#in: PathBuf,
    #[arg(long, default_value = "colmap_ws_gt")]
    out: PathBuf,
    #[arg(long, default_value = "4")]
    stride: u32,
    #[arg(long, default_value = "20.0")]
    max_depth: f32,
}

fn main() {
    let args = Args::parse();
    let calib: Calib = serde_json::from_reader(
        std::fs::File::open(args.r#in.join("calib.json")).expect("calib.json"),
    ).expect("parse calib");
    let k = calib.k();
    let (fx, fy, cx, cy) = (k[(0,0)], k[(1,1)], k[(0,2)], k[(1,2)]);
    let frames_dir = args.r#in.join("frames");

    let mut stems: Vec<String> = std::fs::read_dir(&frames_dir).expect("frames")
        .flatten()
        .filter_map(|e| e.file_name().to_string_lossy().strip_suffix("_left.png").map(String::from))
        .collect();
    stems.sort();
    println!("Found {} frames", stems.len());

    std::fs::create_dir_all(args.out.join("images")).unwrap();
    let mut colmap_frames = Vec::new();
    let mut all_pts: Vec<([f32; 3], [u8; 3])> = Vec::new();

    for stem in &stems {
        let left_path = frames_dir.join(format!("{stem}_left.png"));
        let rgb = image::open(&left_path).expect("left").to_rgb8();
        let (depth, w, _h) = load_depth(&frames_dir.join(format!("{stem}_depth.bin"))).expect("depth");
        let c2w: Matrix4<f64> = load_pose(&frames_dir.join(format!("{stem}_pose.txt"))).expect("pose");

        for row in (0..calib.height).step_by(args.stride as usize) {
            for col in (0..calib.width).step_by(args.stride as usize) {
                let d = depth[(row * w + col) as usize] as f64;
                if d < 0.1 || d as f32 > args.max_depth { continue; }
                let p = c2w * nalgebra::Vector4::new(
                    (col as f64 - cx) / fx * d,
                    (row as f64 - cy) / fy * d,
                    d, 1.0,
                );
                all_pts.push(([p[0] as f32, p[1] as f32, p[2] as f32], rgb.get_pixel(col, row).0));
            }
        }

        let name = format!("{stem}_left.png");
        std::fs::copy(&left_path, args.out.join("images").join(&name)).unwrap();
        colmap_frames.push(ColmapFrame { name, c2w });
    }

    write_workspace(&args.out, &calib, &colmap_frames, &all_pts).unwrap();
    println!("Done → {}  ({} poses, {} points)",
        args.out.display(), colmap_frames.len(), all_pts.len());
}
