//! Device selection with an explicit, observable fallback.

use candle_core::Device;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePreference {
    Auto,
    Cpu,
    Metal,
}

impl DevicePreference {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "cpu" => Some(Self::Cpu),
            "metal" => Some(Self::Metal),
            _ => None,
        }
    }
}

pub struct DeviceChoice {
    pub device: Device,
    pub name: &'static str,
}

impl DeviceChoice {
    pub fn select(preference: DevicePreference) -> Result<Self> {
        match preference {
            DevicePreference::Cpu => Ok(Self {
                device: Device::Cpu,
                name: "cpu",
            }),
            DevicePreference::Metal => Self::metal_or_fail(),
            DevicePreference::Auto => Ok(Self::metal_or_warn().unwrap_or(Self {
                device: Device::Cpu,
                name: "cpu",
            })),
        }
    }

    fn metal_or_fail() -> Result<Self> {
        Self::metal_or_warn().ok_or_else(|| {
            Error::Model("Metal device requested but unavailable in this build".to_string())
        })
    }

    #[cfg(target_os = "macos")]
    fn metal_or_warn() -> Option<Self> {
        match Device::new_metal(0) {
            Ok(device) => Some(Self {
                device,
                name: "metal",
            }),
            Err(err) => {
                eprintln!("eg-judge: Metal unavailable ({err}); continuing on CPU");
                None
            }
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn metal_or_warn() -> Option<Self> {
        eprintln!("eg-judge: Metal support is not compiled in; continuing on CPU");
        None
    }
}
