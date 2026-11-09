//! Subcommand execution: resolve inputs, load once, print one JSON line.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::json;

use crate::bench;
use crate::cli::{Cli, Command};
use crate::corpus;
use crate::device::DevicePreference;
use crate::error::{Error, Result};
use crate::judge::{DuplicateRequest, Judge, Question, SupersedeRequest, Verdict};
use crate::recipe::Recipe;

pub fn run(cli: Cli) -> Result<()> {
    let preference = DevicePreference::parse(&cli.device).ok_or_else(|| {
        Error::Input(format!(
            "unknown device {:?}; expected auto, cpu or metal",
            cli.device
        ))
    })?;
    let dtype = match cli.dtype.as_str() {
        "f32" => candle_core::DType::F32,
        "f16" => candle_core::DType::F16,
        other => {
            return Err(Error::Input(format!(
                "unknown dtype {other:?}; expected f32 or f16"
            )))
        }
    };
    let model_dir = resolve_model_dir(cli.model)?;
    let mut recipe = match cli.recipe.as_deref() {
        Some(path) => Recipe::load(path)?,
        None => Recipe::default(),
    };
    match &cli.command {
        Command::DeviceInfo => return device_info(&model_dir, preference, &recipe),
        Command::JudgeDuplicate { threshold, .. } | Command::JudgeSupersede { threshold, .. } => {
            if let Some(value) = threshold {
                recipe.threshold = Some(*value);
            }
        }
        Command::Bench { .. } => {}
    }
    // Read and validate caller input before touching the 644 MB checkpoint: a malformed
    // request must exit 1 without paying the load cost.
    let prepared = match &cli.command {
        Command::DeviceInfo | Command::Bench { .. } => None,
        _ => Some(judge_inputs(&cli.command)?),
    };
    let mut judge = Judge::load(&model_dir, preference, &recipe, dtype)?;
    if cli.device_info {
        eprintln!(
            "{}",
            json!({
                "device": judge.device_name(),
                "checkpoint": judge.checkpoint(),
                "hidden_size": judge.hidden_size(),
                "temperature": judge.temperature(),
                "threshold": judge.threshold(),
            })
        );
    }
    if let Command::Bench {
        corpus,
        iterations,
        threshold,
    } = &cli.command
    {
        let pairs = corpus::load(corpus)?;
        let report = bench::run(&mut judge, &pairs, *iterations, *threshold)?;
        println!("{}", serde_json::to_string(&report)?);
        return Ok(());
    }
    let (question, first, second, with_logits) =
        prepared.expect("judge commands always prepare their input");
    let mut verdict = judge.judge(question, &first, &second)?;
    if !with_logits {
        verdict.logits = None;
    }
    print_verdict(&verdict)
}

fn judge_inputs(command: &Command) -> Result<(Question, String, String, bool)> {
    match command {
        Command::JudgeDuplicate {
            code_a,
            code_b,
            code_a_file,
            code_b_file,
            with_logits,
            ..
        } => {
            let (first, second) = read_pair(code_a, code_b, code_a_file, code_b_file, "code")?;
            checked(&first, &second)?;
            Ok((Question::Duplicate, first, second, *with_logits))
        }
        Command::JudgeSupersede {
            old_note,
            new_note,
            old_note_file,
            new_note_file,
            with_logits,
            ..
        } => {
            let (first, second) =
                read_pair(old_note, new_note, old_note_file, new_note_file, "note")?;
            checked(&first, &second)?;
            Ok((Question::Supersede, first, second, *with_logits))
        }
        Command::DeviceInfo | Command::Bench { .. } => {
            unreachable!("not a judge command")
        }
    }
}

fn device_info(dir: &Path, preference: DevicePreference, recipe: &Recipe) -> Result<()> {
    let choice = crate::device::DeviceChoice::select(preference)?;
    println!(
        "{}",
        json!({
            "device": choice.name,
            "model": dir.display().to_string(),
            "recipe_threshold": recipe.threshold,
            "recipe_temperature": recipe.temperature,
        })
    );
    Ok(())
}

fn print_verdict(verdict: &Verdict) -> Result<()> {
    let line = serde_json::to_string(verdict)?;
    println!("{line}");
    Ok(())
}

fn resolve_model_dir(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(dir) = explicit {
        return Ok(dir);
    }
    if let Ok(dir) = std::env::var("EG_LAYA_MODEL") {
        if !dir.is_empty() {
            return Ok(PathBuf::from(dir));
        }
    }
    let home = std::env::var("HOME")
        .map_err(|_| Error::Input("HOME is unset; pass --model".to_string()))?;
    Ok(PathBuf::from(home)
        .join(".engram")
        .join("laya")
        .join("model"))
}

/// Resolve `--flag`, `--flag-file` or a stdin JSON document.
fn read_pair(
    first_inline: &Option<String>,
    second_inline: &Option<String>,
    first_file: &Option<PathBuf>,
    second_file: &Option<PathBuf>,
    kind: &str,
) -> Result<(String, String)> {
    if first_inline.is_none()
        && second_inline.is_none()
        && first_file.is_none()
        && second_file.is_none()
    {
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .map_err(|e| Error::Input(format!("stdin: {e}")))?;
        if buffer.trim().is_empty() {
            return Err(Error::Input(format!(
                "no input: pass --{kind}-a/--{kind}-b, their *-file forms, or JSON on stdin"
            )));
        }
        return match kind {
            "code" => {
                let request: DuplicateRequest = serde_json::from_str(&buffer)?;
                Ok((request.code_a, request.code_b))
            }
            _ => {
                let request: SupersedeRequest = serde_json::from_str(&buffer)?;
                Ok((request.old_note, request.new_note))
            }
        };
    }
    let first = read_one(first_inline, first_file, &format!("{kind} a"))?;
    let second = read_one(second_inline, second_file, &format!("{kind} b"))?;
    Ok((first, second))
}

fn read_one(inline: &Option<String>, file: &Option<PathBuf>, label: &str) -> Result<String> {
    if let Some(text) = inline {
        return Ok(text.clone());
    }
    if let Some(path) = file {
        return std::fs::read_to_string(path)
            .map_err(|e| Error::Input(format!("{}: {e}", path.display())));
    }
    Err(Error::Input(format!(
        "missing {label}: pass the inline flag or its *-file form"
    )))
}

fn checked<'a>(first: &'a str, second: &'a str) -> Result<(&'a str, &'a str)> {
    if first.trim().is_empty() {
        return Err(Error::Input("the first text is empty".to_string()));
    }
    if second.trim().is_empty() {
        return Err(Error::Input("the second text is empty".to_string()));
    }
    Ok((first, second))
}
