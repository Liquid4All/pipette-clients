//! Plan types shared across pipette clients.
//!
//! The type families live in submodules — `primitives`, `model`,
//! `runtime`, `plan`, `thermal`, `benchmark_type` — but the public API
//! is flat: every type is re-exported here, so external crates keep
//! referencing `pipette_plan_types::Model`, `::Runtime`, `::GgufText`,
//! `::ResourceUrl`, etc. without seeing the submodules. `result`,
//! `benchmark`, `run`, `thermal`, `device` and `eval_id` are the exceptions —
//! see
//! their declarations below.

mod primitives;
pub use primitives::*;

mod model;
pub use model::*;

mod runtime;
pub use runtime::*;

mod benchmark_type;
pub use benchmark_type::{BenchmarkType, UnknownBenchmarkType};

/// What a benchmark *is* — the per-kind parameter structs, the tagged
/// definition over them, and the [`benchmark::eval_id`] an eval names.
/// Namespaced rather than re-exported flat, for the same reason as
/// [`result`]: reading `benchmark::PrefillThroughput` at the use site keeps
/// it apart from the identically-named [`BenchmarkType`] variant.
pub mod benchmark;

mod runtime_flags;
pub use runtime_flags::{
    reserved_flags, LlamacppFlashAttention, OpenvinoGenerateHint, RuntimeFlagError, RuntimeFlagRef,
    RuntimeFlags,
};

mod benchmark_flags;
pub use benchmark_flags::{
    BenchmarkFlagError, BenchmarkFlagRef, BenchmarkFlags, ReadinessOverrides,
};

mod plan;
pub use plan::{
    ClientRunSpec, Matrix, Plan, RetryConfig, RunnableCell, ShellType, TransportConfig, TypedCell,
    Variant, VariantCompatibilityError,
};

pub mod descriptor;

mod scheduler_plan;
pub use scheduler_plan::{Eligibility, SchedulerCell, SchedulerPlan, SchedulerVariant};

/// One cell as a client runs it: the plan coordinate resolved against a
/// catalog and bound to this host. Namespaced rather than re-exported flat, so
/// `run::RunRequest` reads as the run stage at the use site — distinct from the
/// plan-stage [`ClientRunSpec`] it is resolved from.
pub mod run;

/// What the host *is*: the identity fields every submission carries and the
/// trimming newtype they are spelled in. Namespaced for the same reason as
/// [`thermal`] — reading `device::DeviceInfo` keeps it apart from the plan
/// vocabulary, and `device::TrimmedString` apart from [`NonEmptyString`].
pub mod device;

/// Run-environment power and thermal state: the per-vendor leaf types, the
/// per-run series a client accumulates from them, and the flattened telemetry a
/// result carries. Namespaced rather than re-exported flat, for the same reason
/// as [`result`]: reading `thermal::ThermalTelemetry` at the use site keeps
/// these apart from the plan vocabulary the rest of this crate carries.
pub mod thermal;

/// What a finished cell produced. Namespaced rather than re-exported flat:
/// these are result vocabulary, and reading them as `result::BenchmarkEvalCompletionStopReason` at the
/// use site keeps them distinct from the plan types the rest of this crate
/// carries.
pub mod result;

/// True iff `model`'s format matches what `runtime` can load.
///
/// Reading the match below as a table: each row is one accepted
/// (model, runtime) pair. The `_ => false` fail-closed default means
/// a new `Model` or `Runtime` variant has to add a row here to be
/// reachable — pairings involving an un-listed variant are rejected
/// by `Plan::parse`.
// Clippy would prefer `matches!(...)` since every arm returns true,
// but the explicit match reads as a table — that's the point.
#[allow(clippy::match_like_matches_macro)]
pub fn is_compatible(model: &Model, runtime: &Runtime) -> bool {
    use Model::*;
    use Runtime::*;
    match (model, runtime) {
        (GgufText(_), LlamacppCliStockTools(_) | LlamacppApkPipette(_) | LlamacppIosPipette(_)) => {
            true
        }
        (
            GgufVision(_),
            LlamacppCliStockTools(_) | LlamacppApkPipette(_) | LlamacppIosPipette(_),
        ) => true,
        (Mlx(_), MlxMacosPipette(_) | MlxIosPipette(_)) => true,
        (Torch(_), DockerVllm(_) | DockerSglang(_) | UvVllm(_) | UvSglang(_)) => true,
        (AppleFoundationText, AppleFoundation(_)) => true,
        (Openvino(_), UvOpenvino(_)) => true,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// OpenVINO IR pairs with the OpenVINO runtime and nothing else. The
    /// llama.cpp case is the one worth pinning: `linux-x64-openvino` is a
    /// llama.cpp build that uses OpenVINO as a *backend* and still eats GGUF,
    /// so the name collision must not become a compatible pairing.
    #[test]
    fn openvino_ir_pairs_only_with_the_openvino_runtime() -> anyhow::Result<()> {
        let model = Model::Openvino(Openvino {
            source: ModelSource::HuggingFace {
                repo: HfRepo::parse_org_repo("LiquidAI/LFM2.5-350M-ov")?,
                prefix: None,
            },
        });
        let openvino = Runtime::UvOpenvino(UvOpenvino {
            server_version: UvServerVersion::try_new("2026.2.1".to_owned())?,
            python_version: UvPythonVersion::try_new("3.11".to_owned())?,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("openvino-genai==2026.2.1.0\n".to_owned())?,
                install_flags: None,
            },
        });
        assert!(is_compatible(&model, &openvino));

        let llamacpp = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: default_repository_url(),
                repository_version: NonEmptyString::try_new("b9305".to_owned())?,
            }),
            flavor: LlamaCppFlavor::LinuxX64Openvino,
        });
        assert!(!is_compatible(&model, &llamacpp));
        assert!(!is_compatible(
            &model,
            &Runtime::AppleFoundation(Default::default())
        ));

        // And the runtime is equally narrow: it loads IR, not safetensors.
        assert!(!is_compatible(
            &Model::Torch(Torch {
                source: ModelSource::HuggingFace {
                    repo: HfRepo::parse_org_repo("LiquidAI/LFM2.5-350M")?,
                    prefix: None
                },
            }),
            &openvino
        ));
        Ok(())
    }

    #[test]
    fn apple_foundation_compatibility() -> anyhow::Result<()> {
        // AFM text pairs only with the AFM runtime, and the AFM runtime
        // loads only the AFM text model.
        assert!(is_compatible(
            &Model::AppleFoundationText,
            &Runtime::AppleFoundation(Default::default())
        ));
        assert!(!is_compatible(
            &Model::AppleFoundationText,
            &Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
                source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                    repository_url: default_repository_url(),
                    repository_version: NonEmptyString::try_new("b5000".to_owned())?,
                }),
                flavor: LlamaCppFlavor::MacosArm64,
            })
        ));
        assert!(!is_compatible(
            &Model::Torch(Torch {
                source: ModelSource::HuggingFace {
                    repo: HfRepo::parse_org_repo("meta-llama/Llama-3.2-1B")?,
                    prefix: None
                },
            }),
            &Runtime::AppleFoundation(Default::default())
        ));
        Ok(())
    }
}
