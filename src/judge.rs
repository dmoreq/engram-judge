//! Request layer: one question, one pair, one verdict.

use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::device::{DeviceChoice, DevicePreference};
use crate::error::{Error, Result};
use crate::model::LayaModel;
use crate::recipe::Recipe;
use crate::tokenizer::{noul_options, PromptTokenizer, DEFAULT_HEAD_MAX_LEN, DEFAULT_MAX_LEN};

pub const DUPLICATE_INS: &str = "Both code snippets are pasted below. Do they implement the same \
algorithm, ignoring identifier names, formatting and comment changes?";
pub const DUPLICATE_FALSE: &str = "the two snippets compute different results";
pub const DUPLICATE_TRUE: &str = "the two snippets are behaviourally equivalent";

pub const SUPERSEDE_INS: &str = "Both notes describe the same project fact. Does the new note \
replace the old one, that is, does it make the old note obsolete?";
pub const SUPERSEDE_FALSE: &str = "the notes describe different facts and both stay valid";
pub const SUPERSEDE_TRUE: &str = "the new note supersedes the old note";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Question {
    Duplicate,
    Supersede,
}

impl Question {
    fn instructions(self) -> &'static str {
        match self {
            Question::Duplicate => DUPLICATE_INS,
            Question::Supersede => SUPERSEDE_INS,
        }
    }

    fn descriptions(self) -> (&'static str, &'static str) {
        match self {
            Question::Duplicate => (DUPLICATE_FALSE, DUPLICATE_TRUE),
            Question::Supersede => (SUPERSEDE_FALSE, SUPERSEDE_TRUE),
        }
    }

    fn positive(self) -> &'static str {
        match self {
            Question::Duplicate => "duplicate",
            Question::Supersede => "supersede",
        }
    }

    fn negative(self) -> &'static str {
        match self {
            Question::Duplicate => "distinct",
            Question::Supersede => "keep",
        }
    }

    fn labels(self) -> (&'static str, &'static str) {
        match self {
            Question::Duplicate => ("snippet A", "snippet B"),
            Question::Supersede => ("old note", "new note"),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Verdict {
    pub decision: &'static str,
    pub probability: f32,
    pub calibrated: bool,
    pub latency_ms: f64,
    pub device: &'static str,
    /// Raw `noul` logits; present only with `--with-logits` so the default line stays stable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logits: Option<Vec<f32>>,
}

pub struct Judge {
    model: LayaModel,
    tokenizer: PromptTokenizer,
    threshold: f32,
    temperature: f32,
    calibrated: bool,
    head_max_len: usize,
    max_len: usize,
}

impl Judge {
    pub fn load(
        model_dir: &Path,
        preference: DevicePreference,
        recipe: &Recipe,
        dtype: candle_core::DType,
    ) -> Result<Self> {
        let choice = DeviceChoice::select(preference)?;
        let threshold = recipe.threshold.unwrap_or(0.5);
        if !(0.0..=1.0).contains(&threshold) {
            return Err(Error::Input(format!(
                "threshold must be in [0, 1], got {threshold}"
            )));
        }
        let model = LayaModel::load(model_dir, choice.device, choice.name, dtype)?;
        let tokenizer = PromptTokenizer::load(model_dir, &model.config().encoder)?;
        let temperature = recipe.temperature.unwrap_or_else(|| model.temperature());
        if temperature <= 0.0 {
            return Err(Error::Input(format!(
                "temperature must be positive, got {temperature}"
            )));
        }
        let calibrated = recipe.temperature.is_some() || recipe.threshold.is_some();
        Ok(Self {
            model,
            tokenizer,
            threshold,
            temperature,
            calibrated,
            head_max_len: DEFAULT_HEAD_MAX_LEN,
            max_len: DEFAULT_MAX_LEN,
        })
    }

    pub fn device_name(&self) -> &'static str {
        self.model.device_name()
    }

    pub fn checkpoint(&self) -> &str {
        self.model.checkpoint()
    }

    pub fn has_head(&self) -> bool {
        self.model.has_head()
    }

    pub fn hidden_size(&self) -> usize {
        self.model.hidden_size()
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// `first`/`second` are the two texts the question compares, in the order the question
    /// names them: code A then code B, or old note then new note.
    pub fn judge(&mut self, question: Question, first: &str, second: &str) -> Result<Verdict> {
        let started = Instant::now();
        let (false_desc, true_desc) = question.descriptions();
        let options = noul_options(false_desc, true_desc);
        let layout =
            self.tokenizer
                .question_layout(question.instructions(), &options, self.head_max_len)?;
        let state = pair_state(question, first, second);
        let (ids, markers) = self.tokenizer.full_sequence(&state, layout, self.max_len)?;
        let logits = self.model.logits(&ids, &markers)?;
        let probability = crate::model::probability_true(&logits, f64::from(self.temperature))?;
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(Verdict {
            decision: if probability >= self.threshold {
                question.positive()
            } else {
                question.negative()
            },
            probability: round3(probability),
            calibrated: self.calibrated,
            latency_ms: round1(latency_ms),
            device: self.model.device_name(),
            logits: Some(logits),
        })
    }

    /// Token count of the sequence the model would see, for diagnostics and tests.
    pub fn token_count(&self, question: Question, first: &str, second: &str) -> Result<usize> {
        let (false_desc, true_desc) = question.descriptions();
        let options = noul_options(false_desc, true_desc);
        let layout =
            self.tokenizer
                .question_layout(question.instructions(), &options, self.head_max_len)?;
        let (ids, _) = self.tokenizer.full_sequence(
            &pair_state(question, first, second),
            layout,
            self.max_len,
        )?;
        Ok(ids.len())
    }

    /// Build `ids` and `markers` without running the model. Tests and the parity harness use
    /// this to prove the prompt layout matches the PyTorch reference byte for byte.
    pub fn sequence(
        &self,
        question: Question,
        first: &str,
        second: &str,
    ) -> Result<(Vec<u32>, Vec<usize>)> {
        let (false_desc, true_desc) = question.descriptions();
        let options = noul_options(false_desc, true_desc);
        let layout =
            self.tokenizer
                .question_layout(question.instructions(), &options, self.head_max_len)?;
        self.tokenizer
            .full_sequence(&pair_state(question, first, second), layout, self.max_len)
    }
}

/// The state text both sides of the prompt see: labelled, trimmed, one trailing newline.
pub fn pair_state(question: Question, first: &str, second: &str) -> String {
    let (label_a, label_b) = question.labels();
    format!(
        "{label_a}:\n{}\n{label_b}:\n{}\n",
        first.trim_end(),
        second.trim_end()
    )
}

fn round3(value: f32) -> f32 {
    (f64::from(value) * 1000.0).round() as f32 / 1000.0
}

fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

#[derive(Debug, Deserialize)]
pub struct DuplicateRequest {
    pub code_a: String,
    pub code_b: String,
}

#[derive(Debug, Deserialize)]
pub struct SupersedeRequest {
    pub old_note: String,
    pub new_note: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_layout_keeps_labels_and_trims_trailing_space() {
        let state = pair_state(Question::Duplicate, "fn a() {}\n\n", "fn b() {}");
        assert_eq!(state, "snippet A:\nfn a() {}\nsnippet B:\nfn b() {}\n");
        let state = pair_state(Question::Supersede, "old", "new");
        assert_eq!(state, "old note:\nold\nnew note:\nnew\n");
    }

    #[test]
    fn rounding_matches_the_contract() {
        assert_eq!(round3(0.94159), 0.942);
        assert_eq!(round1(4.14), 4.1);
        assert_eq!(round1(4.16), 4.2);
    }
}
