//! Command line surface.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "eg-laya",
    version,
    about = "Offline Laya proposition-head judge for engram anti-reinvention and memory verdicts"
)]
pub struct Cli {
    /// Checkpoint directory (default: $EG_LAYA_MODEL, else ~/.engram/laya/model)
    #[arg(long, global = true, value_name = "DIR", env = "EG_LAYA_MODEL")]
    pub model: Option<PathBuf>,
    /// Execution device
    #[arg(
        long,
        global = true,
        value_name = "auto|cpu|metal",
        default_value = "auto"
    )]
    pub device: String,
    /// Weight dtype: f32 keeps the reference numerics, f16 halves resident memory
    #[arg(long, global = true, value_name = "f32|f16", default_value = "f32")]
    pub dtype: String,
    /// Calibration recipe with "temperature" and "threshold"
    #[arg(long, global = true, value_name = "FILE")]
    pub recipe: Option<PathBuf>,
    /// Print the selected device to stderr before the verdict
    #[arg(long, global = true)]
    pub device_info: bool,
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Decide whether two code snippets implement the same algorithm
    JudgeDuplicate {
        /// Snippet A passed inline
        #[arg(long, value_name = "CODE", conflicts_with = "code_a_file")]
        code_a: Option<String>,
        /// Snippet B passed inline
        #[arg(long, value_name = "CODE", conflicts_with = "code_b_file")]
        code_b: Option<String>,
        /// Read snippet A from a file
        #[arg(long, value_name = "FILE")]
        code_a_file: Option<PathBuf>,
        /// Read snippet B from a file
        #[arg(long, value_name = "FILE")]
        code_b_file: Option<PathBuf>,
        /// Decision threshold for P(duplicate); overrides the recipe
        #[arg(long, value_name = "P")]
        threshold: Option<f32>,
        /// Include the raw option logits in the JSON line
        #[arg(long)]
        with_logits: bool,
    },
    /// Decide whether a new memory note supersedes an old one
    JudgeSupersede {
        /// Old note passed inline
        #[arg(long, value_name = "TEXT", conflicts_with = "old_note_file")]
        old_note: Option<String>,
        /// New note passed inline
        #[arg(long, value_name = "TEXT", conflicts_with = "new_note_file")]
        new_note: Option<String>,
        /// Read the old note from a file
        #[arg(long, value_name = "FILE")]
        old_note_file: Option<PathBuf>,
        /// Read the new note from a file
        #[arg(long, value_name = "FILE")]
        new_note_file: Option<PathBuf>,
        /// Decision threshold for P(supersede); overrides the recipe
        #[arg(long, value_name = "P")]
        threshold: Option<f32>,
        /// Include the raw option logits in the JSON line
        #[arg(long)]
        with_logits: bool,
    },
    /// Run a corpus through the judge and report latency, F1 and ECE
    Bench {
        /// JSONL corpus: {"a": "...", "b": "...", "label": 0|1, "kind": "duplicate"|"supersede"}
        #[arg(long, value_name = "FILE")]
        corpus: PathBuf,
        /// Repeats over the whole corpus
        #[arg(long, default_value_t = 10)]
        iterations: usize,
        /// Decision threshold used for precision, recall and F1
        #[arg(long, value_name = "P")]
        threshold: Option<f32>,
    },
    /// Report the selected device and the loaded checkpoint
    DeviceInfo,
}
