use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calib {
    pub width: u32,
    pub height: u32,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub baseline: f64,
    #[serde(default)]
    pub k1: f64,
    #[serde(default)]
    pub k2: f64,
    #[serde(default)]
    pub p1: f64,
    #[serde(default)]
    pub p2: f64,
    #[serde(default = "default_model")]
    pub camera_model: String,
}

fn default_model() -> String {
    "PINHOLE".into()
}

impl Calib {
    pub fn k(&self) -> Matrix3<f64> {
        Matrix3::new(
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0,
        )
    }
}
