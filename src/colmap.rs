use crate::camera::Calib;
use nalgebra::{Matrix3, Matrix4};
use std::path::Path;

pub struct ColmapFrame {
    pub name: String,
    pub c2w: Matrix4<f64>,
}

fn c2w_to_quat_tvec(c2w: &Matrix4<f64>) -> ([f64; 4], [f64; 3]) {
    let w2c = c2w.try_inverse().unwrap_or_else(Matrix4::identity);
    let r = Matrix3::new(
        w2c[(0,0)], w2c[(0,1)], w2c[(0,2)],
        w2c[(1,0)], w2c[(1,1)], w2c[(1,2)],
        w2c[(2,0)], w2c[(2,1)], w2c[(2,2)],
    );
    let t = [w2c[(0,3)], w2c[(1,3)], w2c[(2,3)]];

    let tr = r[(0,0)] + r[(1,1)] + r[(2,2)];
    let q = if tr > 0.0 {
        let s = 0.5 / (tr + 1.0).sqrt();
        [0.25/s, (r[(2,1)]-r[(1,2)])*s, (r[(0,2)]-r[(2,0)])*s, (r[(1,0)]-r[(0,1)])*s]
    } else if r[(0,0)] > r[(1,1)] && r[(0,0)] > r[(2,2)] {
        let s = 2.0 * (1.0 + r[(0,0)] - r[(1,1)] - r[(2,2)]).sqrt();
        [(r[(2,1)]-r[(1,2)])/s, 0.25*s, (r[(0,1)]+r[(1,0)])/s, (r[(0,2)]+r[(2,0)])/s]
    } else if r[(1,1)] > r[(2,2)] {
        let s = 2.0 * (1.0 + r[(1,1)] - r[(0,0)] - r[(2,2)]).sqrt();
        [(r[(0,2)]-r[(2,0)])/s, (r[(0,1)]+r[(1,0)])/s, 0.25*s, (r[(1,2)]+r[(2,1)])/s]
    } else {
        let s = 2.0 * (1.0 + r[(2,2)] - r[(0,0)] - r[(1,1)]).sqrt();
        [(r[(1,0)]-r[(0,1)])/s, (r[(0,2)]+r[(2,0)])/s, (r[(1,2)]+r[(2,1)])/s, 0.25*s]
    };
    (q, t)
}

pub fn write_workspace(
    dst: &Path,
    calib: &Calib,
    frames: &[ColmapFrame],
    pts3d: &[([f32; 3], [u8; 3])],
) -> std::io::Result<()> {
    let sparse = dst.join("sparse").join("0");
    std::fs::create_dir_all(dst.join("images"))?;
    std::fs::create_dir_all(&sparse)?;

    // cameras.txt
    std::fs::write(
        sparse.join("cameras.txt"),
        format!(
            "# Camera list\n# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n\
             1 PINHOLE {} {} {:.6} {:.6} {:.6} {:.6}\n",
            calib.width, calib.height, calib.fx, calib.fy, calib.cx, calib.cy
        ),
    )?;

    // images.txt
    let mut s = String::from(
        "# Image list\n# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n\
         # POINTS2D[] as (X Y POINT3D_ID)\n",
    );
    for (id, f) in frames.iter().enumerate() {
        let (q, t) = c2w_to_quat_tvec(&f.c2w);
        s.push_str(&format!(
            "{} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} 1 {}\n\n",
            id + 1, q[0], q[1], q[2], q[3], t[0], t[1], t[2], f.name
        ));
    }
    std::fs::write(sparse.join("images.txt"), s)?;

    // points3D.txt
    let mut p = String::from("# 3D point list\n# POINT3D_ID X Y Z R G B ERROR TRACK[]\n");
    for (i, &([x, y, z], [r, g, b])) in pts3d.iter().enumerate() {
        p.push_str(&format!("{} {:.6} {:.6} {:.6} {} {} {} 1.0\n", i + 1, x, y, z, r, g, b));
    }
    std::fs::write(sparse.join("points3D.txt"), p)?;

    Ok(())
}
