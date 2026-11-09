//! Error type shared by the CLI and the inference engine.
//!
//! Exit codes follow the plan contract:
//! `0` ok, `1` malformed input, `2` invalid CLI usage (clap), `6` model, I/O or device failure.

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// Caller supplied data we cannot interpret (stdin JSON, empty snippet, bad recipe).
    Input(String),
    /// Model directory, tokenizer, configuration or device failure.
    Model(String),
    /// Filesystem or JSON I/O failure.
    Io(String),
}

impl Error {
    pub fn exit_code(&self) -> i32 {
        match self {
            Error::Input(_) => 1,
            Error::Model(_) | Error::Io(_) => 6,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Input(m) => write!(f, "invalid input: {m}"),
            Error::Model(m) => write!(f, "model error: {m}"),
            Error::Io(m) => write!(f, "i/o error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::Io(value.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Error::Input(value.to_string())
    }
}

impl From<candle_core::Error> for Error {
    fn from(value: candle_core::Error) -> Self {
        Error::Model(value.to_string())
    }
}

impl From<tokenizers::Error> for Error {
    fn from(value: tokenizers::Error) -> Self {
        Error::Model(format!("tokenizer: {value}"))
    }
}
