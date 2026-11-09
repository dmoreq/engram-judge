//! Laya decision head for the `noul` question type.
//!
//! Two `nn.TransformerEncoderLayer` blocks (pre-norm, ReLU, 4x feed-forward) run over the whole
//! sequence, then a per-marker scorer turns each `[MASK]` position into one logit. Softmax over
//! the two markers of a `noul` question is the probability of "true".
//!
//! `act_head` is deliberately absent: it only feeds the act/escalate decision that `eg-laya`
//! never reports.

use candle_core::{Result as CResult, Tensor, D};
use candle_nn::{Linear, VarBuilder};

use super::norm::FusedNorm;

pub struct DecisionHead {
    layers: Vec<HeadLayer>,
}

struct HeadLayer {
    attn: MultiHead,
    norm1: FusedNorm,
    norm2: FusedNorm,
    linear1: Linear,
    linear2: Linear,
}

struct MultiHead {
    in_proj: Linear,
    out_proj: Linear,
    n_heads: usize,
    head_dim: usize,
}

pub struct Scorer {
    norm: FusedNorm,
    first: Linear,
    second: Linear,
}

impl DecisionHead {
    pub fn load(
        vb: VarBuilder,
        layers: usize,
        hidden: usize,
        n_heads: usize,
        eps: f64,
    ) -> CResult<Self> {
        let mut head_layers = Vec::with_capacity(layers);
        for index in 0..layers {
            let layer_vb = vb.pp(format!("head.layers.{index}"));
            let attn_vb = layer_vb.pp("self_attn");
            let weight = attn_vb.get((hidden * 3, hidden), "in_proj_weight")?;
            let bias = attn_vb.get(hidden * 3, "in_proj_bias")?;
            head_layers.push(HeadLayer {
                attn: MultiHead {
                    in_proj: Linear::new(weight, Some(bias)),
                    out_proj: candle_nn::linear(hidden, hidden, attn_vb.pp("out_proj"))?,
                    n_heads,
                    head_dim: hidden / n_heads,
                },
                norm1: FusedNorm::load(
                    vb.clone(),
                    &format!("head.layers.{index}.norm1"),
                    hidden,
                    eps,
                )?,
                norm2: FusedNorm::load(
                    vb.clone(),
                    &format!("head.layers.{index}.norm2"),
                    hidden,
                    eps,
                )?,
                linear1: candle_nn::linear(hidden, hidden * 4, layer_vb.pp("linear1"))?,
                linear2: candle_nn::linear(hidden * 4, hidden, layer_vb.pp("linear2"))?,
            });
        }
        Ok(Self {
            layers: head_layers,
        })
    }

    /// `xs`: `(1, seq, hidden)`.
    pub fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
        let mut hidden = xs.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        Ok(hidden)
    }
}

impl HeadLayer {
    fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
        let normalized = self.norm1.forward(xs)?;
        let attended = self.attn.forward(&normalized)?;
        let xs = (xs + attended)?;
        let projected = self
            .norm2
            .forward(&xs)?
            .apply(&self.linear1)?
            .relu()?
            .apply(&self.linear2)?;
        &xs + projected
    }
}

impl MultiHead {
    fn forward(&self, xs: &Tensor) -> CResult<Tensor> {
        let (b, seq, d) = xs.dims3()?;
        let qkv = xs
            .apply(&self.in_proj)?
            .reshape((b, seq, 3, self.n_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let scores = (q * scale)?.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, seq, d))?
            .contiguous()?;
        out.apply(&self.out_proj)
    }
}

impl Scorer {
    pub fn load(vb: VarBuilder, hidden: usize, eps: f64) -> CResult<Self> {
        let scorer = vb.pp("scorer");
        Ok(Self {
            norm: FusedNorm::load(scorer.clone(), "0", hidden, eps)?,
            first: candle_nn::linear(hidden, hidden, scorer.pp("1"))?,
            second: candle_nn::linear(hidden, 1, scorer.pp("3"))?,
        })
    }

    /// `markers`: `(1, k, hidden)`. Returns `(1, k)` logits.
    pub fn forward(&self, markers: &Tensor) -> CResult<Tensor> {
        self.norm
            .forward(markers)?
            .apply(&self.first)?
            .gelu_erf()?
            .apply(&self.second)?
            .squeeze(D::Minus1)
    }
}
