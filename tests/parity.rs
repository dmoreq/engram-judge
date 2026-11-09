//! Numerical parity against the PyTorch reference.
//!
//! `scripts/export_laya.py --fixtures N` writes `tests/fixtures/golden_*.jsonl`, one row per
//! case with the token ids, marker positions, raw logits and calibrated probabilities produced
//! by `laya` 0.4.30 on the same checkpoint. This test rebuilds the prompt with the Rust
//! tokenizer, runs the Rust encoder + head, and requires the probabilities to agree.
//!
//! Gated on `EG_LAYA_MODEL` because the 644 MB checkpoint is not tracked: `cargo test` stays
//! green on a clean clone, and the parity gate runs where the weights live.

use std::path::PathBuf;

use engram_laya::config::ModelConfig;
use engram_laya::model::LayaModel;
use engram_laya::tokenizer::{
    noul_options, PromptTokenizer, DEFAULT_HEAD_MAX_LEN, DEFAULT_MAX_LEN,
};

#[derive(Debug, serde::Deserialize)]
struct GoldenRow {
    id: String,
    state: String,
    ins: String,
    false_desc: String,
    true_desc: String,
    input_ids: Vec<u32>,
    markers: Vec<usize>,
    probs: Vec<f32>,
    temperature: f32,
}

fn fixture_path() -> PathBuf {
    let name = std::env::var("EG_LAYA_FIXTURES")
        .unwrap_or_else(|_| "golden_laya_multilingual.jsonl".to_string());
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

#[test]
fn prompt_layout_matches_reference() {
    let model_dir = match std::env::var("EG_LAYA_MODEL") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => {
            eprintln!("skipping parity: EG_LAYA_MODEL is unset");
            return;
        }
    };
    let path = fixture_path();
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping parity: cannot read {}: {err}", path.display());
            return;
        }
    };
    let config = ModelConfig::load(&model_dir).expect("config");
    let tokenizer = PromptTokenizer::load(&model_dir, &config.encoder).expect("tokenizer");
    let mut checked = 0usize;
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let row: GoldenRow = serde_json::from_str(line).expect("fixture row");
        let options = noul_options(&row.false_desc, &row.true_desc);
        let layout = tokenizer
            .question_layout(&row.ins, &options, DEFAULT_HEAD_MAX_LEN)
            .expect("layout");
        let (ids, markers) = tokenizer
            .full_sequence(&row.state, layout, DEFAULT_MAX_LEN)
            .expect("sequence");
        assert_eq!(ids, row.input_ids, "token ids differ for {}", row.id);
        assert_eq!(
            markers, row.markers,
            "marker positions differ for {}",
            row.id
        );
        checked += 1;
    }
    assert!(checked > 0, "fixture file holds no rows");
    eprintln!("prompt layout parity: {checked} cases match");
}

#[test]
fn probabilities_match_reference() {
    let model_dir = match std::env::var("EG_LAYA_MODEL") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => {
            eprintln!("skipping parity: EG_LAYA_MODEL is unset");
            return;
        }
    };
    let path = fixture_path();
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(err) => {
            eprintln!("skipping parity: cannot read {}: {err}", path.display());
            return;
        }
    };
    let config = ModelConfig::load(&model_dir).expect("config");
    let mut model = LayaModel::load(
        &model_dir,
        candle_core::Device::Cpu,
        "cpu",
        candle_core::DType::F32,
    )
    .expect("model");
    let _ = config;
    let mut worst: f32 = 0.0;
    let mut id = String::new();
    let mut checked = 0usize;
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let row: GoldenRow = serde_json::from_str(line).expect("fixture row");
        let logits = model.logits(&row.input_ids, &row.markers).expect("logits");
        let probability = engram_laya::model::probability_true(&logits, f64::from(row.temperature))
            .expect("prob");
        let reference = row.probs[1];
        let delta = (probability - reference).abs();
        if delta > worst {
            worst = delta;
            id = row.id.clone();
        }
        checked += 1;
    }
    assert!(checked > 0, "fixture file holds no rows");
    eprintln!("probability parity: {checked} cases, worst |delta| = {worst:.6} ({id})");
    assert!(
        worst < 2e-3,
        "worst |delta| {worst:.6} on {id} exceeds 2e-3"
    );
}
