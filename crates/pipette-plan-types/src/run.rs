//! One cell as a client runs it: what an engine is handed ([`RunRequest`](crate::run::RunRequest)) and
//! what it answers with ([`RunResponse`](crate::run::RunResponse)) — the two halves of
//! `run(&RunRequest, …, ReadinessGate) -> Result<RunResponse>`.
//!
//! [`ClientRunSpec`](crate::ClientRunSpec) is the request's unbound sibling —
//! what a plan or a claim serves. Resolving one into a [`RunRequest`](crate::run::RunRequest) needs a
//! workspace catalog and artifact stores, so that step lives client-side in
//! `pipette-cli`; this module owns the shapes only.
//!
//! [`RunRequest`](crate::run::RunRequest) is here because both clients hold it: the Swift app mirrors
//! this crate, and independently arrived at the same declared-plus-located
//! pairing in its `ResolvedModel`. [`RunResponse`](crate::run::RunResponse) follows it for cohesion —
//! the two are one signature, and splitting them across crates is what made
//! the seam hard to read. It has no second implementation and no `Serialize`
//! (its `RunThermal` has none): a client that submits builds
//! [`BenchmarkSubmissionPayload`](crate::result::BenchmarkSubmissionPayload)
//! from it rather than sending it. The injected `ReadinessGate` stays with the
//! engines — a function type over them, not a shape.

use serde::Serialize;

use crate::benchmark::BenchmarkDefinition;
use crate::result::{BenchmarkResultData, MemoryObservation};
use crate::thermal::RunThermal;
use crate::{
    BenchmarkFlagRef, BenchmarkFlags, Model, ModelFlags, ModelType, Runtime, RuntimeFlagRef,
    RuntimeFlags, RuntimeType,
};

/// Plan identity vs host-bound form of one artifact (runtime or model).
///
/// `Serialize` is the plan-stable projection: it emits `declared` alone, so a
/// digest or descriptor taken from a whole [`RunRequest`] is portable by
/// construction instead of by a caller remembering which fields to pick. The
/// two halves are the same value in two forms — for a runtime the store holds
/// but cannot relocate (a docker image, Apple Foundation), they are equal.
#[derive(Debug, Clone, Serialize)]
#[serde(transparent)]
pub struct DeclaredBound<T> {
    /// Plan / pull coordinate (storage key, descriptors, record).
    pub declared: T,
    /// After ensure + bind_under — launch form on this host. Meaningless off
    /// it, hence absent from the serialized form.
    #[serde(skip)]
    pub bound: T,
}

impl<T: Clone> DeclaredBound<T> {
    /// The two halves as the same value — nothing relocated it.
    ///
    /// The documented equal case (a docker image, Apple Foundation: held but
    /// not relocatable), and the shape any caller wants before `bind_under`
    /// has run. Names the state instead of restating `declared`/`bound`.
    pub fn already_bound(value: T) -> Self {
        Self {
            declared: value.clone(),
            bound: value,
        }
    }
}

/// Everything a runtime seam needs for one cell: per-axis declared/bound
/// artifacts, flags, and store-resolved benchmark body.
///
/// The client logs this whole struct via `Debug` at dispatch, so a secret it
/// gains has to redact itself the way [`AuthToken`](crate::AuthToken) does — a
/// plain `String` token would be published. `debug_redacts_the_model_auth_token`
/// holds the line.
#[derive(Debug, Clone, Serialize)]
pub struct RunRequest {
    pub runtime: DeclaredBound<Runtime>,
    pub model: DeclaredBound<Model>,
    pub runtime_flags: Option<RuntimeFlags>,
    pub model_flags: Option<ModelFlags>,
    pub benchmark_flags: Option<BenchmarkFlags>,
    pub benchmark: BenchmarkDefinition,
}

impl RunRequest {
    /// The cell's flags in flat form, or an all-unset ref when the run carries
    /// none. Engines start here, set the values their execution overlaid, and
    /// convert back for [`RunResponse::runtime_flags`] — so the
    /// flags they report are routed and validated like an authored entry.
    ///
    /// `prepare` already refuses an entry naming another cell; the axis check
    /// is a backstop against a caller that skipped it.
    pub fn runtime_flags_ref(&self) -> anyhow::Result<RuntimeFlagRef> {
        let axes = (
            self.benchmark.benchmark_type(),
            RuntimeType::of(&self.runtime.declared),
            ModelType::of(&self.model.declared),
        );
        let Some(flags) = self.runtime_flags.as_ref() else {
            return Ok(RuntimeFlagRef::new(axes.0, axes.1, axes.2));
        };
        anyhow::ensure!(
            flags.axes() == axes,
            "runtime flags {flags:?} are not for this cell {axes:?}"
        );
        Ok(RuntimeFlagRef::from(flags.clone()))
    }

    /// The cell's benchmark flags in flat form, or an all-unset ref when the
    /// run carries none. The caller resolves the readiness block into it and
    /// converts back, so what the result reports is routed and validated like
    /// an authored entry — and a cell whose variant has no readiness field
    /// rejects one, which is correct: those cells do not gate.
    pub fn benchmark_flags_ref(&self) -> anyhow::Result<BenchmarkFlagRef> {
        let axes = (
            self.benchmark.benchmark_type(),
            RuntimeType::of(&self.runtime.declared),
            ModelType::of(&self.model.declared),
        );
        let Some(flags) = self.benchmark_flags.as_ref() else {
            return Ok(BenchmarkFlagRef {
                benchmark_type: axes.0,
                runtime_type: axes.1,
                model_type: axes.2,
                http_timeout_seconds: None,
                doomloop: Default::default(),
                readiness: None,
            });
        };
        anyhow::ensure!(
            flags.axes() == axes,
            "benchmark flags {flags:?} are not for this cell {axes:?}"
        );
        Ok(BenchmarkFlagRef::from(flags.clone()))
    }
}

/// What one **run** answers with: metrics, streams, and the request's flags as
/// the run resolved them. The client pairs it with the [`RunRequest`] it was
/// handed, which is what the submitted descriptors come from.
pub struct RunResponse {
    pub result_data: BenchmarkResultData,
    pub stdout: String,
    pub stderr: String,
    /// Thermal snapshots bracketing each measured repetition. Filled by the
    /// caller from the marks the engine made through its thermal gate, not by
    /// the engine — the probe and the series are the caller's.
    pub thermal: RunThermal,
    /// The cell's benchmark flags as the run resolved them — the readiness
    /// block filled in from the policy that actually applied. Filled by the
    /// caller, not the engine, for the same reason as `thermal`: the gating
    /// policy is the caller's, and an engine is handed an opaque gate so it
    /// never learns what the policy was.
    pub benchmark_flags: Option<BenchmarkFlags>,
    /// Invocation preview recorded in the result extras. Empty for a runtime
    /// that doesn't shell out a named command (MLX runs in-process over HTTP).
    pub command: Vec<String>,
    /// The runtime executable when it's a distinct binary (llama.cpp); `None`
    /// for a runtime that has no such binary (MLX/torch-oai launch a server),
    /// so absence is explicit rather than an empty-string sentinel.
    pub executable: Option<String>,
    /// [`RunRequest::runtime_flags`] round-tripped: the same variant, with the
    /// values the engine derived filled in where the cell left them unset. The
    /// diff against the request is what the run decided. `None` for a runtime
    /// that takes no flags. Flags a benchmark fixes for every run can't appear
    /// (see [`RuntimeFlags::submission_value`](crate::RuntimeFlags::submission_value))
    /// — an engine that shells out keeps those in `command`.
    pub runtime_flags: Option<RuntimeFlags>,
    /// What memory the run held while it ran
    /// ([`MemoryObservation`](crate::result::MemoryObservation)), collected on
    /// every benchmark kind rather than only the memory one.
    ///
    /// Filled by the **engine**, unlike `thermal` and `benchmark_flags`:
    /// sampling the child's memory needs the pid, which only the engine that
    /// spawned it has. Default is "observed nothing", which a platform without a
    /// sampler leaves in place.
    pub memory: MemoryObservation,
}

impl RunResponse {
    /// The metrics half, with provenance empty. Engines that record an
    /// invocation fill `command` / `executable` / `runtime_flags` via
    /// struct-update; `thermal` and `benchmark_flags` are the caller's to fill.
    pub fn new(result_data: BenchmarkResultData, stdout: String, stderr: String) -> Self {
        Self {
            result_data,
            stdout,
            stderr,
            thermal: RunThermal::default(),
            benchmark_flags: None,
            command: Vec::new(),
            executable: None,
            runtime_flags: None,
            memory: MemoryObservation::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RunResponse;
    use crate::result::BenchmarkResultData;
    use crate::thermal::RunThermal;

    /// `new` sets the core and leaves provenance empty — the contract the
    /// shared record/submit path relies on for runtimes without an invocation.
    #[test]
    fn new_leaves_provenance_empty() {
        let outcome = RunResponse::new(
            BenchmarkResultData::DecodeThroughput {
                decode_time_ms: 1.0,
                decode_time_ms_stddev: None,
            },
            "out".into(),
            "err".into(),
        );
        assert!(outcome.command.is_empty());
        assert_eq!(outcome.thermal, RunThermal::default());
        assert!(outcome.executable.is_none());
        assert!(outcome.runtime_flags.is_none());
        assert_eq!(outcome.stdout, "out");
        assert_eq!(outcome.stderr, "err");
    }

    use rstest::rstest;

    use super::*;
    use crate::benchmark::EvalBenchmark;
    use crate::{
        default_repository_url, AuthToken, GgufText, GgufTextSource, GgufVision, GgufVisionSource,
        HfRepo, LlamaCppFlavor, LlamacppCliStockTools, LlamacppCliStockToolsSource, Mlx,
        ModelSource, NonEmptyString, RepoSubpath, SourceRepository, Torch,
    };

    const TOKEN: &str = "hf_tokenthatmustnotescape";

    fn gated_repo() -> anyhow::Result<HfRepo> {
        let mut repo = HfRepo::parse_org_repo("liquid-ai/gated")?;
        repo.auth_token = Some(AuthToken::try_new(TOKEN.to_owned())?);
        Ok(repo)
    }

    fn gated_gguf_text() -> anyhow::Result<Model> {
        Ok(Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: gated_repo()?,
                path: RepoSubpath::try_new("model.gguf".to_owned())?,
                sha256: None,
            },
        }))
    }

    /// One repo covering both files, so a single token reaches two paths.
    fn gated_gguf_vision() -> anyhow::Result<Model> {
        Ok(Model::GgufVision(GgufVision {
            source: GgufVisionSource::HuggingFace {
                repo: gated_repo()?,
                model: RepoSubpath::try_new("model.gguf".to_owned())?,
                model_sha256: None,
                mmproj: RepoSubpath::try_new("mmproj.gguf".to_owned())?,
                mmproj_sha256: None,
            },
        }))
    }

    fn gated_mlx() -> anyhow::Result<Model> {
        Ok(Model::Mlx(Mlx {
            source: ModelSource::HuggingFace {
                repo: gated_repo()?,
                prefix: None,
            },
        }))
    }

    fn gated_torch() -> anyhow::Result<Model> {
        Ok(Model::Torch(Torch {
            source: ModelSource::HuggingFace {
                repo: gated_repo()?,
                prefix: None,
            },
        }))
    }

    /// A gated eval cell with every optional group filled, so the dump walks
    /// the flag subtrees too. The token sits on both halves of `model`: it
    /// survives ensure, so a redaction covering only `declared` would pass.
    fn gated_run_request(model: Model) -> anyhow::Result<RunRequest> {
        let runtime = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: default_repository_url(),
                repository_version: NonEmptyString::try_new("b10142".to_owned())?,
            }),
            flavor: LlamaCppFlavor::MacosArm64,
        });
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(runtime),
            model: DeclaredBound::already_bound(model),
            runtime_flags: Some(RuntimeFlags::EvalLlamacppCliStockToolsGgufText {
                threads: Some(4),
                number_gpu_layers: None,
                mmap: None,
                flash_attention: None,
                ctx_size: None,
                no_cache: None,
                raw: vec!["--verbose".to_owned()],
            }),
            model_flags: Some(ModelFlags::EvalGgufText {
                enable_thinking: Some(true),
            }),
            benchmark_flags: Some(BenchmarkFlags::EvalLlamacppCliStockToolsGgufText {
                http_timeout_seconds: Some(15),
                doomloop: Default::default(),
            }),
            benchmark: BenchmarkDefinition::Eval(EvalBenchmark {
                benchmark_id: "eval_gated".to_owned(),
                parameter_eval_id: "gated".into(),
                parameter_dataset_name: "local".to_owned(),
                parameter_max_tokens: 8,
                parameter_mcq_choices: None,
                samples: None,
            }),
        })
    }

    /// The client logs the whole request at dispatch, so `Debug` is a
    /// publishing surface: a token reaching it would land in every run's log.
    /// One case per model shape that can hold one, since each has its own
    /// source enum and only the shared [`AuthToken`] redacts.
    #[rstest]
    #[case::gguf_text(gated_gguf_text)]
    #[case::gguf_vision(gated_gguf_vision)]
    #[case::mlx(gated_mlx)]
    #[case::torch(gated_torch)]
    fn debug_redacts_the_model_auth_token(
        #[case] gated_model: fn() -> anyhow::Result<Model>,
    ) -> anyhow::Result<()> {
        let dumped = format!("{:?}", gated_run_request(gated_model()?)?);
        assert!(
            !dumped.contains(TOKEN),
            "the dispatch log would publish the token: {dumped}"
        );
        assert!(
            dumped.contains("AuthToken(<redacted>)"),
            "no redaction marker, so the token never reached the dump: {dumped}"
        );
        Ok(())
    }

    /// Rebind to a runtime that differs from `declared` in every field, so an
    /// emitted `bound` cannot coincide with the declared one.
    fn rebind_runtime(req: &mut RunRequest) -> anyhow::Result<()> {
        req.runtime.bound = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: default_repository_url(),
                repository_version: NonEmptyString::try_new("b1".to_owned())?,
            }),
            flavor: LlamaCppFlavor::LinuxX64Cpu,
        });
        Ok(())
    }

    /// `Serialize` is the plan-stable projection, so a consumer can hand over
    /// the whole request: the host-bound halves must not reach it, whatever
    /// else changes.
    #[rstest]
    #[case::bound_runtime(rebind_runtime)]
    fn serialize_omits_host_state(
        #[case] mutate: fn(&mut RunRequest) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let baseline = serde_json::to_value(gated_run_request(gated_gguf_text()?)?)?;

        let mut moved = gated_run_request(gated_gguf_text()?)?;
        mutate(&mut moved)?;

        assert_eq!(
            serde_json::to_value(moved)?,
            baseline,
            "host state leaked into the serialized projection"
        );
        Ok(())
    }

    /// The projection keeps the declared axes under their own field names —
    /// `DeclaredBound` is transparent, so `runtime` is a `Runtime`, not a
    /// wrapper object. Callers digesting the request depend on this shape.
    #[test]
    fn serialize_keeps_declared_axes_unwrapped() -> anyhow::Result<()> {
        let req = gated_run_request(gated_gguf_text()?)?;
        let value = serde_json::to_value(&req)?;
        assert_eq!(
            value["runtime"],
            serde_json::to_value(&req.runtime.declared)?
        );
        assert_eq!(value["model"], serde_json::to_value(&req.model.declared)?);
        Ok(())
    }
}
