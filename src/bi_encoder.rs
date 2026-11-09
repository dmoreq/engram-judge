//! Bi-encoder path: one pooled vector per text, compared with cosine similarity.
//!
//! This is the shape a sentence-encoder clone detector (for example a fine-tuned ModernBERT)
//! ships in. It also matches how an agent compares a new snippet against an index: encode the
//! new side once, then compare against cached vectors.

use std::path::Path;
use std::time::Instant;

use serde::Serialize;

use crate::device::{DeviceChoice, DevicePreference};
use crate::error::{Error, Result};
use crate::model::LayaModel;
use crate::tokenizer::PromptTokenizer;

pub const DEFAULT_BI_MAX_LEN: usize = 512;

#[derive(Debug, Serialize)]
pub struct SimilarityVerdict {
    pub decision: &'static str,
    pub similarity: f32,
    pub threshold: f32,
    pub latency_ms: f64,
    pub device: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

pub struct BiEncoder {
    model: LayaModel,
    tokenizer: PromptTokenizer,
    threshold: f32,
    max_len: usize,
}

impl BiEncoder {
    pub fn load(
        model_dir: &Path,
        preference: DevicePreference,
        dtype: candle_core::DType,
        threshold: f32,
        max_len: usize,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(Error::Input(format!(
                "threshold must be in [0, 1], got {threshold}"
            )));
        }
        let choice = DeviceChoice::select(preference)?;
        let model = LayaModel::load(model_dir, choice.device, choice.name, dtype)?;
        let tokenizer = PromptTokenizer::load(model_dir, &model.config().encoder)?;
        Ok(Self {
            model,
            tokenizer,
            threshold,
            max_len: max_len.max(8),
        })
    }

    pub fn device_name(&self) -> &'static str {
        self.model.device_name()
    }

    pub fn checkpoint(&self) -> &str {
        self.model.checkpoint()
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let ids = self.tokenizer.encode_with_specials(text, self.max_len)?;
        self.model.embed(&ids)
    }

    pub fn similarity(&mut self, first: &str, second: &str) -> Result<SimilarityVerdict> {
        let started = Instant::now();
        let left = self.embed(first)?;
        let right = self.embed(second)?;
        let similarity = cosine(&left, &right);
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(SimilarityVerdict {
            decision: if similarity >= self.threshold {
                "duplicate"
            } else {
                "distinct"
            },
            similarity: (f64::from(similarity) * 1000.0).round() as f32 / 1000.0,
            threshold: self.threshold,
            latency_ms: (latency_ms * 10.0).round() / 10.0,
            device: self.model.device_name(),
            embedding: None,
        })
    }
}

/// Cosine similarity for already L2-normalised vectors is a dot product; clamp to [-1, 1].
pub fn cosine(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() {
        return 0.0;
    }
    let dot: f32 = left.iter().zip(right).map(|(a, b)| a * b).sum();
    dot.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::cosine;

    #[test]
    fn cosine_of_identical_vectors_is_one() {
        let v = vec![0.6, 0.8];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_of_orthogonal_vectors_is_zero() {
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn length_mismatch_is_zero_not_a_panic() {
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0);
    }
}
