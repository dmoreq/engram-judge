//! CLI contract: exit codes and the single-line JSON shape.
//!
//! Only paths that do not need the 644 MB checkpoint run here, so the suite stays fast and
//! green on a clean clone.

use std::process::{Command, Stdio};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_eg-judge")
}

#[test]
fn help_lists_both_judges() {
    let output = Command::new(binary()).arg("--help").output().expect("run");
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    assert!(text.contains("judge-duplicate"), "{text}");
    assert!(text.contains("judge-supersede"), "{text}");
    assert!(text.contains("device-info"), "{text}");
}

#[test]
fn missing_model_exits_six() {
    let output = Command::new(binary())
        .args([
            "judge-duplicate",
            "--code-a",
            "fn a() {}",
            "--code-b",
            "fn b() {}",
        ])
        .arg("--model")
        .arg("/nonexistent/eg-judge-model")
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(6));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("model error"), "{stderr}");
}

#[test]
fn empty_stdin_exits_one() {
    let output = Command::new(binary())
        .args(["judge-duplicate", "--model", "/nonexistent/eg-judge-model"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn")
        .wait_with_output()
        .expect("wait");
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("invalid input"), "{stderr}");
}

#[test]
fn bad_device_is_usage_error() {
    let output = Command::new(binary())
        .args(["device-info", "--device", "tpu"])
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn bad_threshold_is_input_error() {
    let output = Command::new(binary())
        .args([
            "judge-supersede",
            "--old-note",
            "a",
            "--new-note",
            "b",
            "--threshold",
            "1.5",
        ])
        .arg("--model")
        .arg("/nonexistent/eg-judge-model")
        .output()
        .expect("run");
    assert_eq!(output.status.code(), Some(1));
}
