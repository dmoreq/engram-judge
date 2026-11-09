//! Latency and quality measurement for the keep/reject matrix.

use std::time::Instant;

use serde::Serialize;

use crate::corpus::Pair;
use crate::error::Result;
use crate::judge::Judge;

#[derive(Debug, Serialize)]
pub struct Row {
    pub label: u8,
    pub probability: f32,
    pub logits: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct BenchReport {
    pub cases: usize,
    pub iterations: usize,
    pub device: &'static str,
    pub calibrated: bool,
    pub threshold: f32,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    pub max_ms: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub accuracy: f64,
    pub ece: f64,
    pub rows: Vec<Row>,
}

pub fn run(
    judge: &mut Judge,
    pairs: &[Pair],
    iterations: usize,
    threshold: Option<f32>,
) -> Result<BenchReport> {
    let threshold = threshold.unwrap_or_else(|| judge.threshold());
    let iterations = iterations.max(1);
    // One warm-up pass: the first forward allocates Metal kernels and page-cache-backed tensors,
    // and counting it would inflate every percentile.
    if let Some(first) = pairs.first() {
        judge.judge(first.question()?, &first.a, &first.b)?;
    }
    let mut latencies = Vec::with_capacity(pairs.len() * iterations);
    let mut rows = Vec::with_capacity(pairs.len());
    for iteration in 0..iterations {
        for pair in pairs {
            let started = Instant::now();
            let verdict = judge.judge(pair.question()?, &pair.a, &pair.b)?;
            latencies.push(started.elapsed().as_secs_f64() * 1000.0);
            if iteration == 0 {
                rows.push(Row {
                    label: pair.label,
                    probability: verdict.probability,
                    logits: verdict.logits.unwrap_or_default(),
                });
            }
        }
    }
    let mut sorted = latencies.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean_ms = latencies.iter().sum::<f64>() / latencies.len().max(1) as f64;
    let (precision, recall, f1, accuracy) = classification(&rows, threshold);
    Ok(BenchReport {
        cases: pairs.len(),
        iterations,
        device: judge.device_name(),
        calibrated: judge.threshold() != 0.5 || judge.temperature() != 1.0,
        threshold,
        p50_ms: round1(percentile(&sorted, 50.0)),
        p95_ms: round1(percentile(&sorted, 95.0)),
        p99_ms: round1(percentile(&sorted, 99.0)),
        mean_ms: round1(mean_ms),
        max_ms: round1(*sorted.last().unwrap_or(&0.0)),
        precision,
        recall,
        f1,
        accuracy,
        ece: ece(&rows, 15),
        rows,
    })
}

/// Nearest-rank percentile over an ascending slice.
pub fn percentile(sorted: &[f64], percent: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((percent / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Precision, recall, F1 and accuracy with `probability >= threshold` as the positive call.
pub fn classification(rows: &[Row], threshold: f32) -> (f64, f64, f64, f64) {
    let mut true_positive = 0usize;
    let mut false_positive = 0usize;
    let mut false_negative = 0usize;
    let mut correct = 0usize;
    for row in rows {
        let predicted = row.probability >= threshold;
        let actual = row.label == 1;
        if predicted == actual {
            correct += 1;
        }
        match (predicted, actual) {
            (true, true) => true_positive += 1,
            (true, false) => false_positive += 1,
            (false, true) => false_negative += 1,
            (false, false) => {}
        }
    }
    let precision = ratio(true_positive, true_positive + false_positive);
    let recall = ratio(true_positive, true_positive + false_negative);
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    (precision, recall, f1, ratio(correct, rows.len()))
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Expected calibration error over `bins` equal-width confidence bins.
pub fn ece(rows: &[Row], bins: usize) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    let mut counts = vec![0usize; bins];
    let mut confidence = vec![0f64; bins];
    let mut accuracy = vec![0f64; bins];
    for row in rows {
        let index = ((row.probability as f64 * bins as f64).floor() as usize).min(bins - 1);
        counts[index] += 1;
        confidence[index] += row.probability as f64;
        accuracy[index] += f64::from(row.label);
    }
    let total = rows.len() as f64;
    let mut error = 0.0;
    for bin in 0..bins {
        if counts[bin] == 0 {
            continue;
        }
        let weight = counts[bin] as f64 / total;
        let mean_confidence = confidence[bin] / counts[bin] as f64;
        let mean_accuracy = accuracy[bin] / counts[bin] as f64;
        error += weight * (mean_confidence - mean_accuracy).abs();
    }
    (error * 10_000.0).round() / 10_000.0
}

fn round1(value: f64) -> f64 {
    (value * 10.0).round() / 10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(label: u8, probability: f32) -> Row {
        Row {
            label,
            probability,
            logits: vec![0.0, 0.0],
        }
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&values, 50.0), 3.0);
        assert_eq!(percentile(&values, 95.0), 5.0);
        assert_eq!(percentile(&[], 95.0), 0.0);
    }

    #[test]
    fn classification_counts_both_error_kinds() {
        let rows = [row(1, 0.9), row(1, 0.2), row(0, 0.8), row(0, 0.1)];
        let (precision, recall, f1, accuracy) = classification(&rows, 0.5);
        assert!((precision - 0.5).abs() < 1e-9);
        assert!((recall - 0.5).abs() < 1e-9);
        assert!((f1 - 0.5).abs() < 1e-9);
        assert!((accuracy - 0.5).abs() < 1e-9);
    }

    #[test]
    fn perfect_calibration_has_zero_error() {
        let rows = [row(1, 0.999), row(0, 0.001)];
        assert!(ece(&rows, 15) < 0.01);
        let overconfident = [row(0, 0.99), row(0, 0.99)];
        assert!(ece(&overconfident, 15) > 0.9);
    }
}
