use std::path::{Path, PathBuf};

use pipette_workspace::{storage_root, InitResult, Workspace};

const MARKER_NAME: &str = "pipette-plan";

#[derive(Debug)]
pub struct PipettePlanWorkspace {
    inner: Workspace,
}

impl PipettePlanWorkspace {
    pub fn init(work_dir: &Path) -> anyhow::Result<InitResult> {
        let root = storage_root(work_dir, MARKER_NAME);
        Workspace::init(work_dir, MARKER_NAME, [root.join("plans")])
    }

    pub fn open(work_dir: &Path) -> anyhow::Result<Self> {
        let inner = Workspace::open(work_dir, MARKER_NAME)?;
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
    use super::*;

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!(
            "pipette-plan-ws-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    }

    #[test]
    fn init_creates_plans_dir() -> anyhow::Result<()> {
        let work = temp_dir();
        let result = PipettePlanWorkspace::init(&work)?;
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
        PipettePlanWorkspace::init(&work)?;
        let ws = PipettePlanWorkspace::open(&work)?;
        assert!(ws.root().ends_with(".pipette-plan"));
        assert_eq!(ws.plans_dir(), ws.root().join("plans"));

        let _ = std::fs::remove_dir_all(&work);
        Ok(())
    }
}
