//! Checkpoint loader and the single forward pass `eg-judge` needs.

pub mod encoder;
pub mod head;
pub mod norm;

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Embedding, VarBuilder};

use crate::config::ModelConfig;
use crate::error::{Error, Result};
use encoder::Encoder;
use head::{DecisionHead, Scorer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    Cls,
    Mean,
}

pub struct LayaModel {
    encoder: Encoder,
    head: Option<DecisionHead>,
    scorer: Option<Scorer>,
    type_emb: Option<Embedding>,
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
        let expanded = expand_quantized(raw)?;
        let mut tensors = HashMap::with_capacity(expanded.len());
        for (name, tensor) in expanded {
            tensors.insert(name, tensor.to_dtype(dtype)?);
        }
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);
        let hidden = config.encoder.hidden_size;
        let eps = config.encoder.layer_norm_eps;
        let encoder = Encoder::load(vb.clone(), &config.encoder, config.prefix())?;
        let (head, scorer, type_emb) = if config.has_head() {
            (
                Some(DecisionHead::load(
                    vb.clone(),
                    config.meta.head_layers,
                    hidden,
                    config.encoder.num_attention_heads,
                    eps,
                )?),
                Some(Scorer::load(vb.clone(), hidden, eps)?),
                Some(candle_nn::embedding(3, hidden, vb.pp("type_emb"))?),
            )
        } else {
            (None, None, None)
        };
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

    pub fn pooling(&self) -> Pooling {
        match self.config.meta.pooling.as_deref() {
            Some("mean") => Pooling::Mean,
            _ => Pooling::Cls,
        }
    }

    pub fn has_head(&self) -> bool {
        self.head.is_some()
    }

    /// Pooled, L2-normalised sentence vector for the bi-encoder path.
    pub fn embed(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(Error::Input("empty token sequence".to_string()));
        }
        let token_ids = Tensor::from_slice(ids, (1, ids.len()), &self.device)?;
        let hidden = self.encoder.forward(&token_ids)?;
        let batch = hidden.get(0)?;
        let pooled = match self.pooling() {
            Pooling::Cls => batch.get(0)?,
            Pooling::Mean => batch.mean(0)?,
        };
        let pooled = pooled.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let norm = pooled.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        Ok(pooled.iter().map(|v| v / norm).collect())
    }

    /// Raw `noul` logits for one sequence: `(1, markers)` reduced to `markers.len()` values.
    pub fn logits(&mut self, ids: &[u32], markers: &[usize]) -> Result<Vec<f32>> {
        let (Some(head), Some(scorer), Some(type_emb)) = (&self.head, &self.scorer, &self.type_emb)
        else {
            return Err(Error::Model(
                "checkpoint has no decision head; use the similarity command".to_string(),
            ));
        };
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
        let type_vector = type_emb.forward(&qtype)?.unsqueeze(1)?;
        hidden = hidden.broadcast_add(&type_vector)?.contiguous()?;
        let hidden = head.forward(&hidden)?;
        let marker_ids: Vec<u32> = markers.iter().map(|m| *m as u32).collect();
        let index = Tensor::from_slice(&marker_ids, marker_ids.len(), &self.device)?;
        let selected = hidden.index_select(&index, 1)?;
        let logits = scorer.forward(&selected)?;
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

/// Rebuild dense tensors from the `q8`/`q4` + `scale` pairs written by `scripts/quantize_owl.py`.
///
/// Quantisation only shrinks the file: weights are restored to floating point at load, so the
/// resident floor stays at the F16 size. Norms and other 1-D tensors are stored unquantised and
/// pass through unchanged.
fn expand_quantized(raw: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    // Scale tensors are one-dimensional and small; copy them out first so the weight map can be
    // consumed by value. Cloning the whole map instead triples peak RSS on a 580 MB checkpoint.
    let scales: HashMap<String, Tensor> = raw
        .iter()
        .filter(|(name, _)| name.ends_with(".scale"))
        .map(|(name, tensor)| (name.clone(), tensor.clone()))
        .collect();
    let mut out = HashMap::with_capacity(raw.len());
    for (name, tensor) in raw {
        if name.ends_with(".scale") {
            continue;
        }
        if let Some(base) = name.strip_suffix(".q8") {
            let scale = scales
                .get(&format!("{base}.scale"))
                .ok_or_else(|| Error::Model(format!("{base}: q8 without scale")))?;
            out.insert(base.to_string(), unpack_int8(&tensor, scale)?);
            continue;
        }
        if let Some(base) = name.strip_suffix(".q4") {
            let scale = scales
                .get(&format!("{base}.scale"))
                .ok_or_else(|| Error::Model(format!("{base}: q4 without scale")))?;
            out.insert(base.to_string(), unpack_int4(&tensor, scale)?);
            continue;
        }
        out.insert(name, tensor);
    }
    Ok(out)
}

/// One signed byte per weight (stored as U8 by the exporter), one scale per output row.
fn unpack_int8(quantized: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let (rows, cols) = quantized.dims2()?;
    let bytes = quantized
        .to_device(&candle_core::Device::Cpu)?
        .to_vec2::<u8>()?;
    let scales = scale
        .to_device(&candle_core::Device::Cpu)?
        .to_vec1::<f32>()?;
    let mut values = vec![0f32; rows * cols];
    for row in 0..rows {
        let row_scale = scales[row];
        for col in 0..cols {
            values[row * cols + col] = (bytes[row][col] as i8) as f32 * row_scale;
        }
    }
    Ok(Tensor::from_vec(values, (rows, cols), quantized.device())?)
}

/// Two signed nibbles per byte, low nibble first, one scale per 64-column group.
fn unpack_int4(packed: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let (rows, half) = packed.dims2()?;
    let cols = half * 2;
    let groups = scale.dim(1)?;
    let bytes = packed
        .to_device(&candle_core::Device::Cpu)?
        .to_vec2::<u8>()?;
    let scales = scale
        .to_device(&candle_core::Device::Cpu)?
        .to_vec2::<f32>()?;
    let mut values = vec![0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let byte = bytes[row][col / 2];
            let nibble = if col % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let signed = if nibble >= 8 {
                nibble as i32 - 16
            } else {
                nibble as i32
            };
            let group = col * groups / cols;
            values[row * cols + col] = signed as f32 * scales[row][group];
        }
    }
    Ok(Tensor::from_vec(values, (rows, cols), packed.device())?)
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
