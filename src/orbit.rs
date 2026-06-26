use nalgebra::{Matrix4, Vector3};

/// Camera-to-world matrix for a camera on a sphere of `radius`,
/// looking toward the origin. OpenCV convention: +z forward, +y down.
pub fn orbit_c2w(theta: f64, phi: f64, radius: f64) -> Matrix4<f64> {
    let pos = Vector3::new(
        radius * phi.cos() * theta.cos(),
        radius * phi.sin(),
        radius * phi.cos() * theta.sin(),
    );

    let forward = (-pos).normalize();
    let world_up = Vector3::new(0.0, 1.0, 0.0);

    let mut right = forward.cross(&world_up);
    if right.norm() < 1e-6 {
        right = forward.cross(&Vector3::new(0.0, 0.0, 1.0));
    }
    right.normalize_mut();

    // Re-orthogonalise, then flip to camera-y-down (image rows increase downward)
    let up_ortho = right.cross(&forward);
    let cam_y = -up_ortho;

    let mut c2w = Matrix4::identity();
    for i in 0..3 {
        c2w[(i, 0)] = right[i];
        c2w[(i, 1)] = cam_y[i];
        c2w[(i, 2)] = forward[i];
        c2w[(i, 3)] = pos[i];
    }
    c2w
}
