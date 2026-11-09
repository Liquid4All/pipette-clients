//! Resolve the sidecar executable from the bound Core AI runtime.
use std::path::PathBuf;

use pipette_plan_types::{AppleCoreAiMacosPipetteSource, RunRequest, Runtime};
pub(crate) fn require_coreai_sidecar(req: &RunRequest) -> anyhow::Result<PathBuf> {
    let Runtime::AppleCoreAiMacosPipette(rt) = &req.runtime.bound else {
        anyhow::bail!("expected Core AI runtime")
    };
    anyhow::ensure!(rt.is_bundled(), "unsupported Core AI Swift stack");
    let AppleCoreAiMacosPipetteSource::AbsoluteDir { dir } = &rt.source else {
        anyhow::bail!("Core AI runtime must be installed and bound before execution")
    };
    let binary = PathBuf::from(dir.as_ref()).join("pipette-coreai-sidecar");
    anyhow::ensure!(
        binary.is_file(),
        "Core AI sidecar missing at {}. Remove the runtime, then pull it again",
        binary.display()
    );
    Ok(binary)
}
