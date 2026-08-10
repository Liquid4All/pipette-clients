use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;

use crate::state::StateIndex;

pub(super) fn state_file_path(plans_dir: &Path, plan_id: &str) -> PathBuf {
    plans_dir.join(plan_id).join("state.jsonl")
}

fn read_local_state(plans_dir: &Path, plan_id: &str) -> anyhow::Result<Option<String>> {
    let path = state_file_path(plans_dir, plan_id);
    match fs::read_to_string(&path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

pub(super) fn ensure_plans_dir(plans_dir: &Path, plan_id: &str) -> anyhow::Result<()> {
    let path = state_file_path(plans_dir, plan_id);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    Ok(())
}

pub(super) fn append_local_state(
    plans_dir: &Path,
    plan_id: &str,
    line: &str,
) -> anyhow::Result<()> {
    use std::io::Write;
    let path = state_file_path(plans_dir, plan_id);
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    file.write_all(line.as_bytes())
        .with_context(|| format!("writing to {}", path.display()))?;
    Ok(())
}

/// Load the state index for a plan.
pub(crate) fn load_state_index(plans_dir: &Path, plan_id: &str) -> anyhow::Result<StateIndex> {
    let raw = read_local_state(plans_dir, plan_id)?;
    StateIndex::load(raw.as_deref(), plan_id)
}
