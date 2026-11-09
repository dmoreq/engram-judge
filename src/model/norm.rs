//! Fused layer norm.
//!
//! `candle_nn::LayerNorm` only reaches the single-kernel path when a bias tensor exists; the
//! bias-free ModernBERT norms (`embeddings.norm`, `mlp_norm`, `final_norm`) fall back to a
//! seven-op composition that costs ~220 us per call on Metal instead of ~15 us. This wrapper
//! always calls the fused custom op, materialising a zero bias once at load time.

use candle_core::{Result as CResult, Tensor};
use candle_nn::VarBuilder;

pub struct FusedNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl FusedNorm {
    pub fn load(vb: VarBuilder, name: &str, size: usize, eps: f64) -> CResult<Self> {
        let weight = vb.get(size, &format!("{name}.weight"))?;
        let bias = match vb.get(size, &format!("{name}.bias")) {
            Ok(bias) => bias,
            Err(_) => Tensor::zeros(size, weight.dtype(), weight.device())?,
        };
        Ok(Self {
            weight,
            bias,
            eps: eps as f32,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
        let xs = if xs.is_contiguous() {
            xs.clone()
        } else {
            xs.contiguous()?
        };
        candle_nn::ops::layer_norm(&xs, &self.weight, &self.bias, self.eps)
    }
}
