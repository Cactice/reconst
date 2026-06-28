use nalgebra::Matrix4;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Save depth map as raw binary: [u32 width][u32 height][f32 * w*h]
pub fn save_depth(path: &Path, depth: &[f32], width: u32, height: u32) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    f.write_all(&width.to_le_bytes())?;
    f.write_all(&height.to_le_bytes())?;
    for &v in depth {
        f.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

pub fn load_depth(path: &Path) -> std::io::Result<(Vec<f32>, u32, u32)> {
    let mut f = BufReader::new(File::open(path)?);
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    let w = u32::from_le_bytes(b);
    f.read_exact(&mut b)?;
    let h = u32::from_le_bytes(b);
    let mut depth = vec![0.0f32; (w * h) as usize];
    for v in &mut depth {
        f.read_exact(&mut b)?;
        *v = f32::from_le_bytes(b);
    }
    Ok((depth, w, h))
}

/// Save 4x4 pose as 4 space-separated rows of f64.
pub fn save_pose(path: &Path, c2w: &Matrix4<f64>) -> std::io::Result<()> {
    let mut s = String::new();
    for i in 0..4 {
        let row: Vec<_> = (0..4).map(|j| format!("{:.10}", c2w[(i, j)])).collect();
        s.push_str(&row.join(" "));
        s.push('\n');
    }
    std::fs::write(path, s)
}

pub fn load_pose(path: &Path) -> std::io::Result<Matrix4<f64>> {
    let s = std::fs::read_to_string(path)?;
    let vals: Vec<f64> = s
        .split_whitespace()
        .map(|v| {
            v.parse().map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })
        .collect::<Result<_, _>>()?;
    if vals.len() != 16 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "pose file must have exactly 16 values",
        ));
    }
    Ok(Matrix4::from_row_slice(&vals))
}
