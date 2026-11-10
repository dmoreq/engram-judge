//! `eg-judge`: offline semantic judge for engram banked on a local bi-encoder or
//! proposition-head checkpoint.

pub mod bench;
pub mod bi_encoder;
pub mod cli;
pub mod cli_out;
pub mod config;
pub mod corpus;
pub mod device;
pub mod error;
pub mod judge;
pub mod model;
pub mod recipe;
pub mod run;
pub mod tokenizer;

pub use error::{Error, Result};
