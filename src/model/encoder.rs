//! ModernBERT encoder forward pass (mmBERT-base / ModernBERT-large compatible).
//!
//! Scope: batch of one, no padding. `eg-laya` judges a single pair per call, so the global
//! mask is always all-ones and only the sliding-window band is materialised. That keeps the
//! kernel path identical for every layer count and avoids a `(seq, seq)` mask for the 22
//! layers whose attention is global.
//!
//! Weight names are the ones the HuggingFace checkpoint ships:
//! `encoder.embeddings.tok_embeddings`, `encoder.embeddings.norm`, `encoder.layers.N.attn.Wqkv`,
//! `encoder.layers.N.attn.Wo`, `encoder.layers.N.attn_norm`, `encoder.layers.N.mlp.Wi`,
//! `encoder.layers.N.mlp.Wo`, `encoder.layers.N.mlp_norm`, `encoder.final_norm`.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result as CResult, Tensor, D};
use candle_nn::{Embedding, Linear, VarBuilder};

use super::norm::FusedNorm;

use crate::config::EncoderConfig;

pub struct Encoder {
    embeddings: Embedding,
    embed_norm: FusedNorm,
    layers: Vec<EncoderLayer>,
    final_norm: FusedNorm,
    local_mask: Option<Arc<Tensor>>,
    local_radius: usize,
    dtype: DType,
}

struct EncoderLayer {
    attn: Attention,
    mlp: Mlp,
    attn_norm: Option<FusedNorm>,
    mlp_norm: FusedNorm,
    local: bool,
}

struct Attention {
    qkv: Linear,
    proj: Linear,
    n_heads: usize,
    head_dim: usize,
    theta: f64,
}

struct Mlp {
    wi: Linear,
    wo: Linear,
}

impl Encoder {
    pub fn load(vb: VarBuilder, cfg: &EncoderConfig, prefix: &str) -> CResult<Self> {
        let embeddings = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp(qualify(prefix, "embeddings.tok_embeddings")),
        )?;
        let hidden = cfg.hidden_size;
        let eps = cfg.layer_norm_eps;
        let embed_norm =
            FusedNorm::load(vb.clone(), &qualify(prefix, "embeddings.norm"), hidden, eps)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            let local = !cfg.is_global_layer(index);
            let theta = if local {
                cfg.local_theta()
            } else {
                cfg.global_theta()
            };
            let layer_vb = vb.pp(qualify(prefix, &format!("layers.{index}")));
            layers.push(EncoderLayer {
                attn: Attention {
                    qkv: candle_nn::linear_no_bias(
                        cfg.hidden_size,
                        cfg.hidden_size * 3,
                        layer_vb.pp("attn").pp("Wqkv"),
                    )?,
                    proj: candle_nn::linear_no_bias(
                        cfg.hidden_size,
                        cfg.hidden_size,
                        layer_vb.pp("attn").pp("Wo"),
                    )?,
                    n_heads: cfg.num_attention_heads,
                    head_dim: cfg.head_dim(),
                    theta,
                },
                mlp: Mlp {
                    wi: candle_nn::linear_no_bias(
                        cfg.hidden_size,
                        cfg.intermediate_size * 2,
                        layer_vb.pp("mlp").pp("Wi"),
                    )?,
                    wo: candle_nn::linear_no_bias(
                        cfg.intermediate_size,
                        cfg.hidden_size,
                        layer_vb.pp("mlp").pp("Wo"),
                    )?,
                },
                // ModernBERT keeps one post-norm layer: layer 0 ships no `attn_norm` tensor.
                // Load it if present and fall back to the post-norm ordering otherwise.
                attn_norm: FusedNorm::load(
                    vb.clone(),
                    &qualify(prefix, &format!("layers.{index}.attn_norm")),
                    hidden,
                    eps,
                )
                .ok(),
                mlp_norm: FusedNorm::load(
                    vb.clone(),
                    &qualify(prefix, &format!("layers.{index}.mlp_norm")),
                    hidden,
                    eps,
                )?,
                local,
            });
        }
        let final_norm = FusedNorm::load(vb.clone(), &qualify(prefix, "final_norm"), hidden, eps)?;
        let dtype = vb.dtype();
        Ok(Self {
            embeddings,
            embed_norm,
            layers,
            final_norm,
            local_mask: None,
            local_radius: cfg.local_radius(),
            dtype,
        })
    }

    /// `ids`: `(1, seq)` u32. Returns `(1, seq, hidden)`.
    pub fn forward(&mut self, ids: &Tensor) -> CResult<Tensor> {
        let seq = ids.dim(1)?;
        let needs_local = self.layers.iter().any(|l| l.local);
        let local_mask = if needs_local {
            match &self.local_mask {
                Some(mask) if mask.dim(0)? == seq => Some(Arc::clone(mask)),
                _ => {
                    let mask =
                        Arc::new(band_mask(seq, self.local_radius, ids.device(), self.dtype)?);
                    self.local_mask = Some(Arc::clone(&mask));
                    Some(mask)
                }
            }
        } else {
            None
        };
        let mut hidden = self.embeddings.forward(ids)?;
        hidden = self.embed_norm.forward(&hidden)?;
        for layer in &self.layers {
            let mask = if layer.local {
                local_mask.as_deref()
            } else {
                None
            };
            hidden = layer.forward(&hidden, mask)?;
        }
        self.final_norm.forward(&hidden)
    }
}

impl EncoderLayer {
    fn forward(&self, xs: &Tensor, local_mask: Option<&Tensor>) -> CResult<Tensor> {
        let residual = xs.clone();
        let input = match &self.attn_norm {
            Some(norm) => norm.forward(xs)?,
            None => xs.clone(),
        };
        let attended = self.attn.forward(&input, local_mask)?;
        let xs = (attended + residual)?;
        let normalized = self.mlp_norm.forward(&xs)?;
        let mlp = self.mlp.forward(&normalized)?;
        &xs + mlp
    }
}

impl Attention {
    fn forward(&self, xs: &Tensor, local_mask: Option<&Tensor>) -> CResult<Tensor> {
        let (b, seq, d) = xs.dims3()?;
        let qkv = xs
            .apply(&self.qkv)?
            .reshape((b, seq, 3, self.n_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;
        let (cos, sin) = rope_tables(seq, self.head_dim, self.theta, xs.device(), xs.dtype())?;
        let q = apply_rope(&q, &cos, &sin)?;
        let k = apply_rope(&k, &cos, &sin)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        // Global layers need no mask in the single-sequence path, so they take candle's fused
        // SDPA kernel on Metal. It has no CPU implementation, and the sliding-window layers
        // still need the additive band mask, so those keep the explicit softmax.
        let out = match (xs.device().is_metal(), local_mask) {
            (true, None) => candle_nn::ops::sdpa(&q, &k, &v, None, false, scale as f32, 1.0)?,
            _ => {
                let att = ((q * scale)?.matmul(&k.transpose(D::Minus2, D::Minus1)?))?;
                let att = match local_mask {
                    Some(mask) => att.broadcast_add(mask)?,
                    None => att,
                };
                let att = candle_nn::ops::softmax_last_dim(&att)?;
                att.matmul(&v)?
            }
        };
        let out = out.transpose(1, 2)?.reshape((b, seq, d))?;
        out.apply(&self.proj)
    }
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
        let projected = xs.apply(&self.wi)?;
        let chunks = projected.chunk(2, D::Minus1)?;
        let gate = chunks[0].gelu_erf()?;
        (&gate * &chunks[1])?.apply(&self.wo)
    }
}

fn qualify(prefix: &str, suffix: &str) -> String {
    if prefix.is_empty() {
        suffix.to_string()
    } else {
        format!("{prefix}.{suffix}")
    }
}

/// Additive band mask: `0` inside `|i - j| <= radius`, `-inf` outside.
fn band_mask(seq: usize, radius: usize, device: &Device, dtype: DType) -> CResult<Tensor> {
    let mut values = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if i.abs_diff(j) > radius {
                values[i * seq + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(values, (seq, seq), device)?.to_dtype(dtype)
}

/// RoPE tables for `seq` positions: `(cos, sin)` of shape `(seq, head_dim / 2)`.
fn rope_tables(
    seq: usize,
    head_dim: usize,
    theta: f64,
    device: &Device,
    dtype: DType,
) -> CResult<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| (1.0f64 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
        .collect();
    let inv = Tensor::from_vec(inv, (1, half), device)?;
    let positions = Tensor::arange(0u32, seq as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((seq, 1))?;
    let freqs = positions.matmul(&inv)?;
    Ok((freqs.cos()?.to_dtype(dtype)?, freqs.sin()?.to_dtype(dtype)?))
}

/// HF `rotate_half` convention: split the head dimension in two, not interleaved.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CResult<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let out1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let out2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
    Tensor::cat(&[&out1, &out2], D::Minus1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_mask_allows_radius_and_rejects_beyond() {
        let device = Device::Cpu;
        let mask = band_mask(5, 1, &device, DType::F32).unwrap();
        let values = mask.to_vec2::<f32>().unwrap();
        assert_eq!(values[0][0], 0.0);
        assert_eq!(values[0][1], 0.0);
        assert!(values[0][2].is_infinite() && values[0][2] < 0.0);
        assert_eq!(values[4][3], 0.0);
        assert!(values[4][2].is_infinite());
    }

    #[test]
    fn rope_preserves_norm() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 1, 1, 4), &device).unwrap();
        let (cos, sin) = rope_tables(1, 4, 10_000.0, &device, DType::F32).unwrap();
        let rotated = apply_rope(&x, &cos, &sin).unwrap();
        let norm = |t: &Tensor| -> f32 {
            t.sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
                .sqrt()
        };
        assert!((norm(&x) - norm(&rotated)).abs() < 1e-5);
        let values = rotated.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((values[0] - 1.0).abs() < 1e-5);
    }
}
