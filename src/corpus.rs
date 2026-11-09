//! Labelled pair corpus: the input the keep/reject gates run on.

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::judge::Question;

#[derive(Debug, Clone, Deserialize)]
pub struct Pair {
    pub a: String,
    pub b: String,
    #[serde(default = "default_kind")]
    pub kind: String,
    pub label: u8,
}

fn default_kind() -> String {
    "duplicate".to_string()
}

impl Pair {
    pub fn question(&self) -> Result<Question> {
        match self.kind.as_str() {
            "duplicate" => Ok(Question::Duplicate),
            "supersede" => Ok(Question::Supersede),
            other => Err(Error::Input(format!(
                "unknown corpus kind {other:?}; expected duplicate or supersede"
            ))),
        }
    }

    pub fn validate(&self, line: usize) -> Result<()> {
        if self.label > 1 {
            return Err(Error::Input(format!("line {line}: label must be 0 or 1")));
        }
        if self.a.trim().is_empty() || self.b.trim().is_empty() {
            return Err(Error::Input(format!("line {line}: empty a or b")));
        }
        self.question()?;
        Ok(())
    }
}

pub fn load(path: &Path) -> Result<Vec<Pair>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Input(format!("{}: {e}", path.display())))?;
    let mut pairs = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let pair: Pair = serde_json::from_str(line)
            .map_err(|e| Error::Input(format!("{}:{}: {e}", path.display(), index + 1)))?;
        pair.validate(index + 1)?;
        pairs.push(pair);
    }
    if pairs.is_empty() {
        return Err(Error::Input(format!("{}: no pairs", path.display())));
    }
    Ok(pairs)
}
