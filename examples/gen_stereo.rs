use clap::Parser;
use nalgebra::{Matrix3, Vector3};
use reconst::{
    camera::Calib,
    io::{save_depth, save_pose},
    orbit::orbit_c2w,
    render::{render_frame, Sphere},
};
use std::f64::consts::TAU;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Generate synthetic static stereo test data")]
struct Args {
    #[arg(long, default_value = "30")]
    frames: usize,
    #[arg(long, default_value = "640")]
    width: u32,
    #[arg(long, default_value = "480")]
    height: u32,
    #[arg(long, default_value = "0.06", help = "stereo baseline metres")]
    baseline: f64,
    #[arg(long, default_value = "2.5")]
    orbit_radius: f64,
    #[arg(long, default_value = "80.0", help = "total azimuth sweep in degrees (small = easy tracking)")]
    arc_deg: f64,
    #[arg(long, default_value = "data/stereo_test")]
    out: PathBuf,
}

fn spheres() -> Vec<Sphere> {
    vec![
        Sphere { center: Vector3::new( 0.0,  0.0,  0.0), radius: 0.30, color: [220,  60,  60], checker_freq: 24 },
        Sphere { center: Vector3::new( 0.7,  0.2,  0.4), radius: 0.20, color: [ 60, 180,  60], checker_freq: 22 },
        Sphere { center: Vector3::new(-0.5,  0.4,  0.6), radius: 0.25, color: [ 60,  90, 220], checker_freq: 20 },
        Sphere { center: Vector3::new( 0.2, -0.5,  0.3), radius: 0.18, color: [220, 180,  40], checker_freq: 26 },
        Sphere { center: Vector3::new(-0.3, -0.2, -0.4), radius: 0.22, color: [160,  60, 200], checker_freq: 22 },
        Sphere { center: Vector3::new( 0.8, -0.3, -0.2), radius: 0.15, color: [ 40, 200, 200], checker_freq: 28 },
    ]
}

fn main() {
    let args = Args::parse();
    let frames_dir = args.out.join("frames");
    std::fs::create_dir_all(&frames_dir).unwrap();

    let fx = 0.8 * args.width as f64;
    let cx = args.width  as f64 / 2.0;
    let cy = args.height as f64 / 2.0;
    let k  = Matrix3::new(fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0);

    let calib = Calib {
        width: args.width, height: args.height,
        fx, fy: fx, cx, cy,
        baseline: args.baseline,
        k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0,
        camera_model: "PINHOLE".into(),
    };
    serde_json::to_writer_pretty(
        std::fs::File::create(args.out.join("calib.json")).unwrap(),
        &calib,
    ).unwrap();

    let scene = spheres();

    let arc = args.arc_deg.to_radians();
    let denom = (args.frames.max(2) - 1) as f64;
    for i in 0..args.frames {
        let frac  = i as f64 / denom; // 0..1 across the sweep
        let theta = arc * (frac - 0.5);
        let phi   = 12f64.to_radians() * (TAU * frac).sin();
        let c2w   = orbit_c2w(theta, phi, args.orbit_radius);

        let (left,  depth) = render_frame(&scene, args.width, args.height, &k, &c2w, 0.0);
        let (right, _)     = render_frame(&scene, args.width, args.height, &k, &c2w, args.baseline);

        let stem = frames_dir.join(format!("{:04}", i));
        left .save(format!("{}_left.png",  stem.display())).unwrap();
        right.save(format!("{}_right.png", stem.display())).unwrap();
        save_depth(&PathBuf::from(format!("{}_depth.bin", stem.display())), &depth, args.width, args.height).unwrap();
        save_pose (&PathBuf::from(format!("{}_pose.txt",  stem.display())), &c2w).unwrap();

        println!("  frame {:04}  θ={:.1}°", i, theta.to_degrees());
    }

    println!("\nDone — {} stereo pairs → {}", args.frames, args.out.display());
}
