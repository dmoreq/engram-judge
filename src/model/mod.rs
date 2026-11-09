//! Checkpoint loader and the single forward pass `eg-laya` needs.

pub mod encoder;
pub mod head;

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, VarBuilder};

use crate::config::ModelConfig;
use crate::error::{Error, Result};
use encoder::Encoder;
use head::{DecisionHead, Scorer};

pub struct LayaModel {
    encoder: Encoder,
    head: DecisionHead,
    scorer: Scorer,
    type_emb: Embedding,
    config: ModelConfig,
    device: Device,
    device_name: &'static str,
}

impl LayaModel {
    pub fn load(
        dir: &Path,
        device: Device,
        device_name: &'static str,
        dtype: DType,
    ) -> Result<Self> {
        let config = ModelConfig::load(dir)?;
        let weights = dir.join("model.safetensors");
        if !weights.exists() {
            return Err(Error::Model(format!(
                "{} not found; run scripts/export_laya.py",
                weights.display()
            )));
        }
        // `from_mmaped_safetensors` is unsafe in candle; load through the safe reader and
        // hand the tensor map to `from_tensors` instead.
        let raw = candle_core::safetensors::load(&weights, &device)
            .map_err(|e| Error::Model(format!("{}: {e}", weights.display())))?;
        let mut tensors = HashMap::with_capacity(raw.len());
        for (name, tensor) in raw {
            tensors.insert(name, tensor.to_dtype(dtype)?);
        }
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);
        let hidden = config.encoder.hidden_size;
        let eps = config.encoder.layer_norm_eps;
        let encoder = Encoder::load(vb.clone(), &config.encoder)?;
        let head = DecisionHead::load(
            vb.clone(),
            config.meta.head_layers,
            hidden,
            config.encoder.num_attention_heads,
            eps,
        )?;
        let scorer = Scorer::load(vb.clone(), hidden, eps)?;
        let type_emb = candle_nn::embedding(3, hidden, vb.pp("type_emb"))?;
        Ok(Self {
            encoder,
            head,
            scorer,
            type_emb,
            config,
            device,
            device_name,
        })
    }

    pub fn device_name(&self) -> &'static str {
        self.device_name
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    pub fn temperature(&self) -> f32 {
        self.config.temperature_noul()
    }

    pub fn checkpoint(&self) -> &str {
        &self.config.meta.checkpoint
    }

    pub fn hidden_size(&self) -> usize {
        self.config.encoder.hidden_size
    }

    /// Raw `noul` logits for one sequence: `(1, markers)` reduced to `markers.len()` values.
    pub fn logits(&mut self, ids: &[u32], markers: &[usize]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(Error::Input("empty token sequence".to_string()));
        }
        if markers.is_empty() {
            return Err(Error::Input("empty marker list".to_string()));
        }
        let seq = ids.len();
        let token_ids = Tensor::from_slice(ids, (1, seq), &self.device)?;
        let mut hidden = self.encoder.forward(&token_ids)?;
        let qtype = Tensor::from_slice(&[self.config.meta.qtype_noul], (1,), &self.device)?;
        let type_vector = self.type_emb.forward(&qtype)?.unsqueeze(1)?;
        hidden = hidden.broadcast_add(&type_vector)?.contiguous()?;
        let hidden = self.head.forward(&hidden)?;
        let marker_ids: Vec<u32> = markers.iter().map(|m| *m as u32).collect();
        let index = Tensor::from_slice(&marker_ids, marker_ids.len(), &self.device)?;
        let selected = hidden.index_select(&index, 1)?;
        let logits = self.scorer.forward(&selected)?;
        let rows: Vec<Vec<f32>> = logits.to_dtype(DType::F32)?.to_vec2()?;
        Ok(rows.into_iter().next().unwrap_or_default())
    }

    /// `P(true)` for the `noul` question, after temperature scaling.
    pub fn probability_true(&mut self, ids: &[u32], markers: &[usize]) -> Result<f32> {
        let logits = self.logits(ids, markers)?;
        let temperature = f64::from(self.temperature().max(1e-3));
        probability_true(&logits, temperature)
    }
}

/// Softmax over the option markers with temperature scaling, computed in `f64`.
///
/// `p[1]` is the probability of the `true` option because the reference implementation renders
/// the `noul` options in fixed `[false, true]` order and never permutes them.
pub fn probability_true(logits: &[f32], temperature: f64) -> Result<f32> {
    if logits.len() < 2 {
        return Err(Error::Input(format!(
            "noul needs two option markers, got {}",
            logits.len()
        )));
    }
    let scaled: Vec<f64> = logits
        .iter()
        .take(2)
        .map(|l| f64::from(*l) / temperature)
        .collect();
    let max = scaled[0].max(scaled[1]);
    let exp0 = (scaled[0] - max).exp();
    let exp1 = (scaled[1] - max).exp();
    Ok((exp1 / (exp0 + exp1)) as f32)
}

#[cfg(test)]
mod tests {
    use super::probability_true;

    #[test]
    fn probability_is_monotone_in_the_true_logit() {
        let low = probability_true(&[0.0, 1.0], 1.0).unwrap();
        let high = probability_true(&[0.0, 4.0], 1.0).unwrap();
        assert!(low > 0.5 && high > low);
        assert!((low - 0.7310586).abs() < 1e-5);
    }

    #[test]
    fn temperature_one_is_plain_softmax() {
        let p = probability_true(&[0.0, 0.0], 1.0).unwrap();
        assert!((p - 0.5).abs() < 1e-9);
        let cooled = probability_true(&[0.0, 0.0], 0.5).unwrap();
        assert!((cooled - 0.5).abs() < 1e-9);
    }

    #[test]
    fn single_marker_is_an_input_error() {
        assert!(probability_true(&[1.0], 1.0).is_err());
    }
}
