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
#[command(about = "Generate synthetic dynamic stereo test data for 4DGS")]
struct Args {
    #[arg(long, default_value = "24", help = "time steps (motion frames)")]
    timesteps: usize,
    #[arg(long, default_value = "6", help = "camera views per timestep")]
    views: usize,
    #[arg(long, default_value = "640")]
    width: u32,
    #[arg(long, default_value = "480")]
    height: u32,
    #[arg(long, default_value = "0.06")]
    baseline: f64,
    #[arg(long, default_value = "2.5")]
    orbit_radius: f64,
    #[arg(long, default_value = "data/dynamic_test")]
    out: PathBuf,
}

// (base_center, radius, color, amplitude_xyz, frequency_xyz)
const DYNAMIC_SPHERES: &[([f64; 3], f32, [u8; 3], [f64; 3], [f64; 3])] = &[
    ([0.0,  0.0,  0.0], 0.30, [220,  60,  60], [0.4, 0.0, 0.3], [1.0, 0.0, 1.3]),
    ([0.7,  0.2,  0.4], 0.20, [ 60, 180,  60], [0.2, 0.3, 0.0], [0.8, 1.2, 0.0]),
    ([-0.5, 0.4,  0.6], 0.25, [ 60,  90, 220], [0.3, 0.2, 0.3], [1.5, 0.7, 1.0]),
    ([0.2, -0.5,  0.3], 0.18, [220, 180,  40], [0.0, 0.4, 0.2], [0.0, 1.0, 0.8]),
    ([-0.3,-0.2, -0.4], 0.22, [160,  60, 200], [0.3, 0.1, 0.3], [1.2, 1.5, 0.9]),
];

fn scene_at(t_norm: f64) -> Vec<Sphere> {
    let t = t_norm * TAU;
    DYNAMIC_SPHERES
        .iter()
        .map(|&([bx, by, bz], radius, color, [ax, ay, az], [fx, fy, fz])| Sphere {
            center: Vector3::new(
                bx + ax * (fx * t).sin(),
                by + ay * (fy * t).sin(),
                bz + az * (fz * t).cos(),
            )
            .cast::<f32>(),
            radius,
            color,
            checker_freq: 5,
        })
        .collect()
}

fn main() {
    let args = Args::parse();

    let fx_cam = 0.8 * args.width as f64;
    let cx = args.width  as f64 / 2.0;
    let cy = args.height as f64 / 2.0;
    let k  = Matrix3::new(fx_cam, 0.0, cx, 0.0, fx_cam, cy, 0.0, 0.0, 1.0);

    let calib = Calib {
        width: args.width, height: args.height,
        fx: fx_cam, fy: fx_cam, cx, cy,
        baseline: args.baseline,
        k1: 0.0, k2: 0.0, p1: 0.0, p2: 0.0,
        camera_model: "PINHOLE".into(),
    };
    std::fs::create_dir_all(&args.out).unwrap();
    serde_json::to_writer_pretty(
        std::fs::File::create(args.out.join("calib.json")).unwrap(),
        &calib,
    ).unwrap();

    let total = args.timesteps * args.views;
    let mut done = 0usize;

    for ti in 0..args.timesteps {
        let scene = scene_at(ti as f64 / args.timesteps as f64);
        let t_dir = args.out.join(format!("t{:03}", ti));
        std::fs::create_dir_all(&t_dir).unwrap();

        for vi in 0..args.views {
            let theta = TAU * vi as f64 / args.views as f64;
            let phi   = 10f64.to_radians() * (TAU * vi as f64 / args.views as f64).sin();
            let c2w   = orbit_c2w(theta, phi, args.orbit_radius);

            let (left,  depth) = render_frame(&scene, args.width, args.height, &k, &c2w, 0.0);
            let (right, _)     = render_frame(&scene, args.width, args.height, &k, &c2w, args.baseline);

            let stem = t_dir.join(format!("v{:03}", vi));
            left .save(format!("{}_left.png",  stem.display())).unwrap();
            right.save(format!("{}_right.png", stem.display())).unwrap();
            save_depth(&PathBuf::from(format!("{}_depth.bin", stem.display())), &depth, args.width, args.height).unwrap();
            save_pose (&PathBuf::from(format!("{}_pose.txt",  stem.display())), &c2w).unwrap();
            done += 1;
        }
        println!("  t={:03}/{} ({}/{})", ti, args.timesteps, done, total);
    }

    println!("\nDone — {} timesteps × {} views → {}", args.timesteps, args.views, args.out.display());
}
