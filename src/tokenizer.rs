//! BPE tokenizer wrapper plus the exact prompt layout Laya was trained on.
//!
//! Layout (from `laya.common.build_sequence`):
//! `[CLS] "noul question: <ins>" [SEP] [MASK] option0 [MASK] option1 [SEP] <state> [SEP]`
//! Options are capped at 48 tokens each and the instruction is capped to whatever budget the
//! options leave inside `head_max_len`.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::config::EncoderConfig;
use crate::error::{Error, Result};

pub const NOUL_TYPE: &str = "noul";
pub const DEFAULT_HEAD_MAX_LEN: usize = 192;
pub const DEFAULT_MAX_LEN: usize = 1024;
const OPTION_TOKEN_CAP: usize = 48;

pub struct PromptTokenizer {
    inner: Tokenizer,
    mask_token: String,
    pub cls_id: u32,
    pub sep_id: u32,
    pub mask_id: u32,
}

/// Token ids for one encoded question, before the state is appended.
pub struct QuestionLayout {
    pub ids: Vec<u32>,
    pub markers: Vec<usize>,
}

impl PromptTokenizer {
    pub fn load(dir: &Path, cfg: &EncoderConfig) -> Result<Self> {
        let path = dir.join("tokenizer").join("tokenizer.json");
        let inner = Tokenizer::from_file(&path)
            .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
        // `tokenizer_config.json` is authoritative: it maps `cls_token` to `<bos>` and
        // `sep_token` to `<eos>` on the mmBERT/Gemma tokenizer, while `encoder/config.json`
        // advertises 1/1. The reference implementation reads the tokenizer object, so do we.
        let specials = SpecialTokens::load(dir)?;
        let mask_id = specials
            .mask
            .as_deref()
            .and_then(|name| inner.token_to_id(name))
            .or(cfg.mask_token_id)
            .or_else(|| inner.token_to_id("<mask>"))
            .or_else(|| inner.token_to_id("[MASK]"))
            .ok_or_else(|| Error::Model(format!("{}: no mask token", path.display())))?;
        let cls_id = specials
            .cls
            .as_deref()
            .and_then(|name| inner.token_to_id(name))
            .or_else(|| inner.token_to_id("<cls>"))
            .or_else(|| inner.token_to_id("[CLS]"))
            .or_else(|| inner.token_to_id("<bos>"))
            .or(cfg.cls_token_id)
            .unwrap_or(1);
        let sep_id = specials
            .sep
            .as_deref()
            .and_then(|name| inner.token_to_id(name))
            .or_else(|| inner.token_to_id("<sep>"))
            .or_else(|| inner.token_to_id("[SEP]"))
            .or_else(|| inner.token_to_id("<eos>"))
            .or(cfg.sep_token_id)
            .unwrap_or(cls_id);
        let mask_token = inner
            .id_to_token(mask_id)
            .unwrap_or_else(|| "<mask>".to_string());
        Ok(Self {
            inner,
            mask_token,
            cls_id,
            sep_id,
            mask_id,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false)?;
        Ok(encoding.get_ids().to_vec())
    }

    /// BERT-style encoding for the bi-encoder path: `[CLS] tokens [SEP]`, truncated to `max_len`.
    pub fn encode_with_specials(&self, text: &str, max_len: usize) -> Result<Vec<u32>> {
        let mut ids = self.encode(text)?;
        ids.truncate(max_len.saturating_sub(2).max(1));
        let mut out = Vec::with_capacity(ids.len() + 2);
        out.push(self.cls_id);
        out.extend(ids);
        out.push(self.sep_id);
        out.truncate(max_len);
        Ok(out)
    }

    /// Strip the mask token from user text, exactly as the reference implementation does.
    pub fn sanitize(&self, text: &str) -> String {
        if self.mask_token.is_empty() {
            text.to_string()
        } else {
            text.replace(self.mask_token.as_str(), " ")
        }
    }

    /// Build `[CLS] instruction [SEP] [MASK] opt0 [MASK] opt1 [SEP]`.
    pub fn question_layout(
        &self,
        ins: &str,
        options: &[String],
        head_max_len: usize,
    ) -> Result<QuestionLayout> {
        let instruction = format!("{NOUL_TYPE} question: {}", self.sanitize(ins));
        let mut head_ids = self.encode(&instruction)?;
        let mut option_ids = Vec::with_capacity(options.len());
        for option in options {
            let mut ids = self.encode(&format!(" {}", self.sanitize(option)))?;
            ids.truncate(OPTION_TOKEN_CAP);
            let mut with_marker = Vec::with_capacity(ids.len() + 1);
            with_marker.push(self.mask_id);
            with_marker.extend(ids);
            option_ids.push(with_marker);
        }

        let mut option_budget = head_max_len as isize - total_len(&option_ids) as isize;
        if option_budget < 16 {
            let per =
                ((head_max_len as isize - 16) / options.len().max(1) as isize).max(4) as usize;
            for ids in option_ids.iter_mut() {
                ids.truncate(per);
            }
            option_budget = head_max_len as isize - total_len(&option_ids) as isize;
        }
        let keep = option_budget.max(8) as usize;
        head_ids.truncate(keep);

        let mut ids = Vec::with_capacity(head_ids.len() + total_len(&option_ids) + 3);
        ids.push(self.cls_id);
        ids.extend(head_ids);
        ids.push(self.sep_id);
        let mut markers = Vec::with_capacity(option_ids.len());
        for option in option_ids {
            markers.push(ids.len());
            ids.extend(option);
        }
        ids.push(self.sep_id);
        Ok(QuestionLayout { ids, markers })
    }

    /// Append the serialised state, clamp to `max_len`, and keep markers inside the sequence.
    pub fn full_sequence(
        &self,
        state: &str,
        layout: QuestionLayout,
        max_len: usize,
    ) -> Result<(Vec<u32>, Vec<usize>)> {
        let state_ids = self.encode(&self.sanitize(state))?;
        let room = max_len.saturating_sub(layout.ids.len() + 1);
        let mut ids = layout.ids;
        ids.extend(state_ids.iter().take(room));
        ids.push(self.sep_id);
        ids.truncate(max_len);
        let markers = layout
            .markers
            .into_iter()
            .filter(|m| *m < max_len)
            .collect();
        Ok((ids, markers))
    }
}

/// The three special-token names declared by `tokenizer/tokenizer_config.json`.
struct SpecialTokens {
    cls: Option<String>,
    sep: Option<String>,
    mask: Option<String>,
}

impl SpecialTokens {
    fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer").join("tokenizer_config.json");
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(_) => {
                return Ok(Self {
                    cls: None,
                    sep: None,
                    mask: None,
                })
            }
        };
        let value: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
        Ok(Self {
            cls: field(&value, "cls_token"),
            sep: field(&value, "sep_token"),
            mask: field(&value, "mask_token"),
        })
    }
}

fn field(value: &serde_json::Value, key: &str) -> Option<String> {
    let entry = value.get(key)?;
    if let Some(text) = entry.as_str() {
        return Some(text.to_string());
    }
    entry
        .get("content")
        .and_then(|content| content.as_str())
        .map(str::to_string)
}

fn total_len(options: &[Vec<u32>]) -> usize {
    options.iter().map(Vec::len).sum()
}

/// Render the two `noul` options exactly like `laya.common.render_options`.
pub fn noul_options(false_desc: &str, true_desc: &str) -> [String; 2] {
    [
        format!(
            "false: {}",
            described(false_desc, "no, the statement does not hold")
        ),
        format!("true: {}", described(true_desc, "yes, the statement holds")),
    ]
}

fn described(value: &str, fallback: &str) -> String {
    if value.trim().is_empty() {
        fallback.to_string()
    } else {
        value.to_string()
    }
}
