//! Small serde + filesystem helpers (JSON read-write).
//!
//! Not artifact-specific — these stayed here because workspace auth, results,
//! and benchmark storage use them.
//!
//! Write helpers create missing parent directories before writing.

use std::fs;
use std::path::Path;

use anyhow::Context;
use serde::{de::DeserializeOwned, Serialize};

use crate::error::{Error, Result as OpsResult};

/// Pretty-print JSON to `path`, creating parent directories if needed.
pub fn write_json_file<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(path, serde_json::to_string_pretty(value)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

/// Read and deserialize JSON from `path`.
pub fn read_json_file<T: DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&raw).with_context(|| format!("failed to parse {}", path.display()))
}

/// Read + JSON-deserialize, mapping failures to typed [`Error`].
pub fn read_json_typed<T: DeserializeOwned>(path: &Path) -> OpsResult<T> {
    let raw = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&raw).map_err(|source| Error::ParseJson {
        path: path.to_path_buf(),
        source,
    })
}

/// Pretty-print JSON to `path` (creates parents), typed [`Error`].
pub fn write_json_typed<T: Serialize>(path: &Path, value: &T) -> OpsResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| Error::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let body = serde_json::to_string_pretty(value).map_err(Error::SerializeJson)?;
    fs::write(path, body).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}
