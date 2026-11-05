//! The `RunRequest` the unit tests build: an OVMS runtime, an IR directory that
//! need not exist, and the cell under test.

use std::path::Path;

use pipette_plan_types::{
    AbsolutePath, BenchmarkDefinition, DeclaredBound, Model, ModelSource, Openvino, RunRequest,
    RuntimeFlags,
};

pub(crate) fn request(
    runtime: serde_json::Value,
    runtime_flags: Option<RuntimeFlags>,
    benchmark: BenchmarkDefinition,
) -> anyhow::Result<RunRequest> {
    let ir_dir = if cfg!(windows) {
        r"C:\tmp\ir"
    } else {
        "/tmp/ir"
    };
    Ok(RunRequest {
        runtime: DeclaredBound::already_bound(serde_json::from_value(runtime)?),
        model: DeclaredBound::already_bound(Model::Openvino(Openvino {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(ir_dir.to_owned())?,
            },
        })),
        runtime_flags,
        model_flags: None,
        benchmark_flags: None,
        benchmark,
    })
}

/// A release unpacked at `dir`, as `bind_under` leaves it.
pub(crate) fn bound_at(dir: &Path) -> serde_json::Value {
    serde_json::json!({
        "type": "ovms", "source": "absolute_dir", "dir": dir, "flavor": "windows-x64"
    })
}

pub(crate) fn unbound() -> serde_json::Value {
    serde_json::json!({
        "type": "ovms", "source": "github_release",
        "repository_url": "github.com/openvinotoolkit/model_server",
        "repository_version": "v2026.4.1", "flavor": "windows-x64"
    })
}
