//! Checkpoint configuration: the encoder block of `encoder/config.json` plus `lazy.json`.
//!
//! The runtime never reads the Hub. Every field it needs is either in the HuggingFace encoder
//! config shipped next to the weights or in the small `lazy.json` that `export_laya.py` writes.

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct EncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub global_attn_every_n_layers: usize,
    #[serde(default = "default_local_attention")]
    pub local_attention: usize,
    #[serde(default = "default_layernorm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub global_rope_theta: Option<f64>,
    #[serde(default)]
    pub local_rope_theta: Option<f64>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default = "default_true")]
    pub norm_bias: bool,
    #[serde(default)]
    pub cls_token_id: Option<u32>,
    #[serde(default)]
    pub sep_token_id: Option<u32>,
    #[serde(default)]
    pub mask_token_id: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub full_attention: Option<RopeVariant>,
    #[serde(default)]
    pub sliding_attention: Option<RopeVariant>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeVariant {
    pub rope_theta: f64,
}

fn default_local_attention() -> usize {
    128
}

fn default_layernorm_eps() -> f64 {
    1e-5
}

fn default_true() -> bool {
    true
}

impl EncoderConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Inclusive sliding-window radius. `local_attention` is the full window (129 positions
    /// for the default 128), so the band is half of it on each side.
    pub fn local_radius(&self) -> usize {
        (self.local_attention / 2).max(1)
    }

    pub fn global_theta(&self) -> f64 {
        self.global_rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(|p| p.full_attention.as_ref())
                    .map(|v| v.rope_theta)
            })
            .unwrap_or(160_000.0)
    }

    pub fn local_theta(&self) -> f64 {
        self.local_rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(|p| p.sliding_attention.as_ref())
                    .map(|v| v.rope_theta)
            })
            .unwrap_or(160_000.0)
    }

    pub fn is_global_layer(&self, index: usize) -> bool {
        self.global_attn_every_n_layers == 0
            || index.is_multiple_of(self.global_attn_every_n_layers)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LazyMeta {
    pub checkpoint: String,
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
    #[serde(default = "default_temperature")]
    pub temperature: [f32; 3],
    #[serde(default = "default_qtype_noul")]
    pub qtype_noul: u32,
    /// Weight prefix in `model.safetensors`: `encoder` for Laya, `model` for a plain ModernBERT.
    #[serde(default)]
    pub weight_prefix: Option<String>,
    /// `cls` or `mean` pooling for the bi-encoder path.
    #[serde(default)]
    pub pooling: Option<String>,
}

fn default_head_layers() -> usize {
    2
}

fn default_temperature() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn default_qtype_noul() -> u32 {
    2
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub encoder: EncoderConfig,
    pub meta: LazyMeta,
}

impl ModelConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        let encoder_path = dir.join("encoder").join("config.json");
        let meta_path = dir.join("lazy.json");
        let encoder_text = std::fs::read_to_string(&encoder_path)
            .map_err(|e| Error::Model(format!("{}: {e}", encoder_path.display())))?;
        let meta_text = std::fs::read_to_string(&meta_path)
            .map_err(|e| Error::Model(format!("{}: {e}", meta_path.display())))?;
        let encoder: EncoderConfig = serde_json::from_str(&encoder_text)
            .map_err(|e| Error::Model(format!("{}: {e}", encoder_path.display())))?;
        let meta: LazyMeta = serde_json::from_str(&meta_text)
            .map_err(|e| Error::Model(format!("{}: {e}", meta_path.display())))?;
        if !encoder
            .hidden_size
            .is_multiple_of(encoder.num_attention_heads)
        {
            return Err(Error::Model(format!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                encoder.hidden_size, encoder.num_attention_heads
            )));
        }
        Ok(Self { encoder, meta })
    }

    pub fn prefix(&self) -> &str {
        self.meta.weight_prefix.as_deref().unwrap_or("encoder")
    }

    pub fn has_head(&self) -> bool {
        self.meta.head_layers > 0
    }

    pub fn temperature_noul(&self) -> f32 {
        self.meta.temperature[self.meta.qtype_noul as usize % 3]
    }
}
