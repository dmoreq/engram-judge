//! The single audited writer in the runtime.
//!
//! `eg-laya bench --out FILE` is an explicit, user-supplied output path; nothing else in the
//! binary writes to disk. Keeping the primitive here lets `tests/scripts/write_path_check.sh`
//! enforce that no other module gains a write.

use std::path::Path;

use crate::error::{Error, Result};

pub fn write_json(path: &Path, line: &str) -> Result<()> {
    std::fs::write(path, format!("{line}\n"))
        .map_err(|e| Error::Io(format!("{}: {e}", path.display())))
}
