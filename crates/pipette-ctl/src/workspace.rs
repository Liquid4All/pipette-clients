use std::path::{Path, PathBuf};

use pipette_workspace::{InitResult, Workspace, storage_root};

/// Workspace directory name: `<work_dir>/.pipette-ctl`.
const MARKER_NAME: &str = "pipette-ctl";

/// From before the rename; a work dir holding only this one keeps using it.
const LEGACY_MARKER_NAME: &str = "pipette-plan";

fn marker_name(work_dir: &Path) -> &'static str {
    if !storage_root(work_dir, MARKER_NAME).exists()
        && storage_root(work_dir, LEGACY_MARKER_NAME).exists()
    {
        LEGACY_MARKER_NAME
    } else {
        MARKER_NAME
    }
}

/// Where this crate keeps everything under `work_dir`, opened or not.
///
/// [`crate::identity`] addresses a subdirectory of it without opening a
/// workspace, since a directory can hold credentials before it holds any plans.
/// Handing it the marker instead would make it restate the leading-dot
/// convention that [`storage_root`] owns.
pub(crate) fn ctl_dir(work_dir: &Path) -> PathBuf {
    storage_root(work_dir, marker_name(work_dir))
}

/// [`pipette_workspace::require_workspace`] bound to this crate's marker.
///
/// `work_dir_arg` is the unresolved `--work-dir` / `PIPETTE_WORK_DIR` value, so
/// the hint offers to `init` the directory the operator actually named.
pub(crate) fn require_workspace(
    work_dir: &Path,
    work_dir_arg: Option<&Path>,
) -> anyhow::Result<()> {
    pipette_workspace::require_workspace(work_dir, work_dir_arg, marker_name(work_dir))
}

#[derive(Debug)]
pub struct PipetteCtlWorkspace {
    inner: Workspace,
}

impl PipetteCtlWorkspace {
    pub fn init(work_dir: &Path) -> anyhow::Result<InitResult> {
        let marker = marker_name(work_dir);
        let root = storage_root(work_dir, marker);
        Workspace::init(work_dir, marker, [root.join("plans")])
    }

    pub fn open(work_dir: &Path) -> anyhow::Result<Self> {
        let inner = Workspace::open(work_dir, marker_name(work_dir))?;
        Ok(Self { inner })
    }

    pub fn root(&self) -> &Path {
        self.inner.root()
    }

    pub fn plans_dir(&self) -> PathBuf {
        self.inner.root().join("plans")
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "pipette-ctl-ws-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    }

    #[test]
    fn init_creates_plans_dir() -> anyhow::Result<()> {
        let work = temp_dir();
        let result = PipetteCtlWorkspace::init(&work)?;
        let root = match result {
            InitResult::Created(p) => p,
            InitResult::AlreadyExists(_) => anyhow::bail!("expected Created"),
        };
        assert!(root.join("plans").is_dir());
        assert!(root.join("manifest.toml").exists());

        let _ = std::fs::remove_dir_all(&work);
        Ok(())
    }

    #[test]
    fn open_after_init() -> anyhow::Result<()> {
        let work = temp_dir();
        PipetteCtlWorkspace::init(&work)?;
        let ws = PipetteCtlWorkspace::open(&work)?;
        assert!(ws.root().ends_with(".pipette-ctl"));
        assert_eq!(ws.plans_dir(), ws.root().join("plans"));

        let _ = std::fs::remove_dir_all(&work);
        Ok(())
    }

    #[rstest]
    #[case::old_alone(false, ".pipette-plan")]
    #[case::new_wins(true, ".pipette-ctl")]
    fn picks_the_dir(#[case] new_too: bool, #[case] expected: &str) -> anyhow::Result<()> {
        let work = tempfile::tempdir()?;
        let work = work.path();
        Workspace::init(work, LEGACY_MARKER_NAME, [])?;
        if new_too {
            Workspace::init(work, MARKER_NAME, [])?;
        }
        require_workspace(work, None)?;
        assert!(PipetteCtlWorkspace::open(work)?.root().ends_with(expected));
        assert_eq!(ctl_dir(work), work.join(expected));
        assert!(matches!(
            PipetteCtlWorkspace::init(work)?,
            InitResult::AlreadyExists(_)
        ));
        Ok(())
    }
}
