//! Optional calibration recipe: temperature scaling and a decision threshold.
//!
//! The file is plain JSON because the tuning scripts write it:
//! `{"temperature": 0.83, "threshold": 0.62}`. Absent fields keep the checkpoint default.

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Default, Deserialize)]
pub struct Recipe {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub threshold: Option<f32>,
    #[serde(default)]
    pub note: Option<String>,
}

impl Recipe {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Input(format!("{}: {e}", path.display())))?;
        let recipe: Recipe = serde_json::from_str(&text)
            .map_err(|e| Error::Input(format!("{}: {e}", path.display())))?;
        if let Some(temperature) = recipe.temperature {
            if temperature <= 0.0 {
                return Err(Error::Input(format!(
                    "{}: temperature must be positive, got {temperature}",
                    path.display()
                )));
            }
        }
        if let Some(threshold) = recipe.threshold {
            if !(0.0..=1.0).contains(&threshold) {
                return Err(Error::Input(format!(
                    "{}: threshold must be in [0, 1], got {threshold}",
                    path.display()
                )));
            }
        }
        Ok(recipe)
    }
}
