//! The BenchmarkFlags family: per-cell run-driving knobs, a closed enum with one
//! variant per supported `(benchmark, runtime_type, model)` cell — the variant
//! name *is* the triple (e.g. `PrefillMlxMacosPipetteMlx`), mirroring
//! [`crate::RuntimeFlags`]. Each variant inlines exactly the knobs that cell
//! accepts, so which knob is legal is a type-level fact:
//! - **timing** cells (`prefill`/`decode`/`end_to_end`) carry `readiness` — the
//!   host-thermal gate whose wait moves the measured number.
//! - **`vl`** carries the HTTP-client `http_timeout_seconds` + `readiness`.
//! - **`eval`** carries `http_timeout_seconds` + the `doomloop` monitor.
//! - **iOS** timing cells carry `readiness`, like every other timing cell: the
//!   app gates on the same criteria and a phone is where a thermal wait matters
//!   most. They carry nothing else — an in-process engine has no HTTP client to
//!   time out and no server to watch for a doom loop.
//! - `max_memory_usage`, Apple-Foundation, `eval`/`vl` on iOS, and apk cells have
//!   no variant — a [`BenchmarkFlagError::NoSuchCombination`] (extend as needed).
//!
//! Authored flat via [`BenchmarkFlagRef`]; `TryFrom` routes the triple to its
//! variant and `deny_*`-rejects any knob the target cell doesn't take.

use serde::{Deserialize, Serialize};

use pipette_doomloop::plan::DoomloopOverrides;

use crate::{BenchmarkType, Model, ModelType, Runtime, RuntimeType};

/// Host-readiness gate settings for a cell: how long to wait for the device to
/// reach a nominal thermal/quiet state before a measurement. Authored as a
/// nested table, e.g. `readiness = { max_wait_secs = 1800 }`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReadinessOverrides {
    /// Override the per-platform readiness deadline, in whole seconds. `None`
    /// keeps the built-in default. Delivered to the runner as
    /// `--readiness-max-wait-secs` (which aliases `PIPETTE_READINESS_MAX_WAIT_SECS`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_wait_secs: Option<u64>,
    /// Waive the platform's *thermal* criterion for this cell, keeping the load
    /// criterion. `None`/`false` enforces it. Delivered to the runner as
    /// `--readiness-skip-thermal` (which aliases
    /// `PIPETTE_READINESS_SKIP_THERMAL`).
    ///
    /// For hosts or workloads where the thermal signal costs more than it buys.
    /// The clearest case is macOS, whose thermal enum is a fixed ~318 s hold-off
    /// timed from when the CPU last went quiet rather than a temperature — so it
    /// can add minutes per repetition without describing how hot anything is.
    /// The load check stays on, because it catches a second benchmark running
    /// concurrently, which is a correctness problem unrelated to heat.
    ///
    /// Unlike `max_wait_secs`, this *changes the readiness criteria*: results
    /// from a cell that skipped the thermal gate are not comparable to results
    /// from one that didn't, so set it deliberately and per cell rather than
    /// fleet-wide.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skip_thermal: Option<bool>,
}

/// Resolving a [`BenchmarkFlagRef`] into a [`BenchmarkFlags`] failed.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BenchmarkFlagError {
    /// The `(benchmark, runtime, model)` triple names no supported flag-carrying
    /// cell (`max_memory_usage`, Apple-Foundation, mobile, or an unmodeled pair).
    #[error("no benchmark flags defined for {model:?} running {benchmark:?} on {runtime:?}")]
    NoSuchCombination {
        benchmark: BenchmarkType,
        runtime: RuntimeType,
        model: ModelType,
    },
    /// A knob was set on a cell whose variant has no field for it.
    #[error(
        "{knob} is not a valid benchmark flag for {model:?} running {benchmark:?} on {runtime:?}"
    )]
    KnobNotAllowed {
        knob: &'static str,
        benchmark: BenchmarkType,
        runtime: RuntimeType,
        model: ModelType,
    },
}

/// Per-cell run-driving knobs — one variant per supported cell, named
/// `<Benchmark><RuntimeType><Model>`. Distinct from `ModelFlags` (generation
/// *identity*): these tune how pipette *drives, gates, and monitors* the run.
/// Authored flat via [`BenchmarkFlagRef`]; `TryFrom` routes the triple to its
/// variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize, strum::EnumCount)]
#[serde(try_from = "BenchmarkFlagRef", into = "BenchmarkFlagRef")]
// Only the `eval` variants carry the sizeable `DoomloopOverrides`; the timing
// variants are small. These are authored/parsed per-cell, not on a size-critical
// hot path, so the variance is fine — boxing would only add deref noise.
#[allow(clippy::large_enum_variant)]
pub enum BenchmarkFlags {
    // llama.cpp CLI × gguf-text — timing cells. Prefill/decode drive the CLI
    // directly; end-to-end goes through llama-server, so it also bounds the
    // `/completion` call, which a long context or a slow device can outrun.
    PrefillLlamacppCliStockToolsGgufText {
        readiness: Option<ReadinessOverrides>,
    },
    DecodeLlamacppCliStockToolsGgufText {
        readiness: Option<ReadinessOverrides>,
    },
    EndToEndLlamacppCliStockToolsGgufText {
        http_timeout_seconds: Option<u64>,
        readiness: Option<ReadinessOverrides>,
    },
    // llama.cpp CLI — server cells.
    EvalLlamacppCliStockToolsGgufText {
        http_timeout_seconds: Option<u64>,
        doomloop: DoomloopOverrides,
    },
    VlLlamacppCliStockToolsGgufVision {
        http_timeout_seconds: Option<u64>,
        readiness: Option<ReadinessOverrides>,
    },
    // MLX (macOS) — server for every cell; timing carries readiness, eval the
    // server/monitor knobs.
    PrefillMlxMacosPipetteMlx {
        readiness: Option<ReadinessOverrides>,
    },
    DecodeMlxMacosPipetteMlx {
        readiness: Option<ReadinessOverrides>,
    },
    EndToEndMlxMacosPipetteMlx {
        readiness: Option<ReadinessOverrides>,
    },
    EvalMlxMacosPipetteMlx {
        http_timeout_seconds: Option<u64>,
        doomloop: DoomloopOverrides,
    },
    // iOS (in-process) — timing cells only. `readiness` is the whole surface: the
    // engine runs in this process, so there is no HTTP client to bound and no
    // server to watch. `eval` and `vl` are absent for that reason, not by omission.
    PrefillLlamacppIosPipetteGgufText {
        readiness: Option<ReadinessOverrides>,
    },
    DecodeLlamacppIosPipetteGgufText {
        readiness: Option<ReadinessOverrides>,
    },
    EndToEndLlamacppIosPipetteGgufText {
        readiness: Option<ReadinessOverrides>,
    },
    PrefillMlxIosPipetteMlx {
        readiness: Option<ReadinessOverrides>,
    },
    DecodeMlxIosPipetteMlx {
        readiness: Option<ReadinessOverrides>,
    },
    EndToEndMlxIosPipetteMlx {
        readiness: Option<ReadinessOverrides>,
    },
    // Torch server runtimes — end-to-end latency + eval, both over HTTP.
    EndToEndDockerVllmTorch {
        http_timeout_seconds: Option<u64>,
        readiness: Option<ReadinessOverrides>,
    },
    EndToEndUvVllmTorch {
        http_timeout_seconds: Option<u64>,
        readiness: Option<ReadinessOverrides>,
    },
    EndToEndDockerSglangTorch {
        http_timeout_seconds: Option<u64>,
        readiness: Option<ReadinessOverrides>,
    },
    EndToEndUvSglangTorch {
        http_timeout_seconds: Option<u64>,
        readiness: Option<ReadinessOverrides>,
    },
    EvalDockerVllmTorch {
        http_timeout_seconds: Option<u64>,
        doomloop: DoomloopOverrides,
    },
    EvalUvVllmTorch {
        http_timeout_seconds: Option<u64>,
        doomloop: DoomloopOverrides,
    },
    EvalDockerSglangTorch {
        http_timeout_seconds: Option<u64>,
        doomloop: DoomloopOverrides,
    },
    EvalUvSglangTorch {
        http_timeout_seconds: Option<u64>,
        doomloop: DoomloopOverrides,
    },
}

impl BenchmarkFlags {
    /// The `(benchmark, runtime, model)` triple this variant encodes — its cell
    /// identity, used to match cells and reject duplicate entries in a variant.
    pub fn axes(&self) -> (BenchmarkType, RuntimeType, ModelType) {
        use BenchmarkType as B;
        use ModelType as M;
        use RuntimeType as R;
        match self {
            BenchmarkFlags::PrefillLlamacppCliStockToolsGgufText { .. } => {
                (B::PrefillThroughput, R::LlamacppCliStockTools, M::GgufText)
            }
            BenchmarkFlags::DecodeLlamacppCliStockToolsGgufText { .. } => {
                (B::DecodeThroughput, R::LlamacppCliStockTools, M::GgufText)
            }
            BenchmarkFlags::EndToEndLlamacppCliStockToolsGgufText { .. } => {
                (B::EndToEndLatency, R::LlamacppCliStockTools, M::GgufText)
            }
            BenchmarkFlags::EvalLlamacppCliStockToolsGgufText { .. } => {
                (B::Eval, R::LlamacppCliStockTools, M::GgufText)
            }
            BenchmarkFlags::VlLlamacppCliStockToolsGgufVision { .. } => {
                (B::VlThroughput, R::LlamacppCliStockTools, M::GgufVision)
            }
            BenchmarkFlags::PrefillMlxMacosPipetteMlx { .. } => {
                (B::PrefillThroughput, R::MlxMacosPipette, M::Mlx)
            }
            BenchmarkFlags::DecodeMlxMacosPipetteMlx { .. } => {
                (B::DecodeThroughput, R::MlxMacosPipette, M::Mlx)
            }
            BenchmarkFlags::EndToEndMlxMacosPipetteMlx { .. } => {
                (B::EndToEndLatency, R::MlxMacosPipette, M::Mlx)
            }
            BenchmarkFlags::EvalMlxMacosPipetteMlx { .. } => (B::Eval, R::MlxMacosPipette, M::Mlx),
            BenchmarkFlags::PrefillLlamacppIosPipetteGgufText { .. } => {
                (B::PrefillThroughput, R::LlamacppIosPipette, M::GgufText)
            }
            BenchmarkFlags::DecodeLlamacppIosPipetteGgufText { .. } => {
                (B::DecodeThroughput, R::LlamacppIosPipette, M::GgufText)
            }
            BenchmarkFlags::EndToEndLlamacppIosPipetteGgufText { .. } => {
                (B::EndToEndLatency, R::LlamacppIosPipette, M::GgufText)
            }
            BenchmarkFlags::PrefillMlxIosPipetteMlx { .. } => {
                (B::PrefillThroughput, R::MlxIosPipette, M::Mlx)
            }
            BenchmarkFlags::DecodeMlxIosPipetteMlx { .. } => {
                (B::DecodeThroughput, R::MlxIosPipette, M::Mlx)
            }
            BenchmarkFlags::EndToEndMlxIosPipetteMlx { .. } => {
                (B::EndToEndLatency, R::MlxIosPipette, M::Mlx)
            }
            BenchmarkFlags::EndToEndDockerVllmTorch { .. } => {
                (B::EndToEndLatency, R::DockerVllm, M::Torch)
            }
            BenchmarkFlags::EndToEndUvVllmTorch { .. } => (B::EndToEndLatency, R::UvVllm, M::Torch),
            BenchmarkFlags::EndToEndDockerSglangTorch { .. } => {
                (B::EndToEndLatency, R::DockerSglang, M::Torch)
            }
            BenchmarkFlags::EndToEndUvSglangTorch { .. } => {
                (B::EndToEndLatency, R::UvSglang, M::Torch)
            }
            BenchmarkFlags::EvalDockerVllmTorch { .. } => (B::Eval, R::DockerVllm, M::Torch),
            BenchmarkFlags::EvalUvVllmTorch { .. } => (B::Eval, R::UvVllm, M::Torch),
            BenchmarkFlags::EvalDockerSglangTorch { .. } => (B::Eval, R::DockerSglang, M::Torch),
            BenchmarkFlags::EvalUvSglangTorch { .. } => (B::Eval, R::UvSglang, M::Torch),
        }
    }

    /// Whether this entry applies to a cell running `benchmark` on `runtime`
    /// with `model` — all three axes match.
    pub fn matches(&self, benchmark: BenchmarkType, runtime: &Runtime, model: &Model) -> bool {
        self.axes() == (benchmark, RuntimeType::of(runtime), ModelType::of(model))
    }

    /// The HTTP-client timeout — carried by the variants whose engine talks to a
    /// server: `eval`, `vl`, and the end-to-end cells that are not in-process.
    pub fn http_timeout(&self) -> Option<u64> {
        match self {
            BenchmarkFlags::VlLlamacppCliStockToolsGgufVision {
                http_timeout_seconds,
                ..
            }
            | BenchmarkFlags::EndToEndLlamacppCliStockToolsGgufText {
                http_timeout_seconds,
                ..
            }
            | BenchmarkFlags::EndToEndDockerVllmTorch {
                http_timeout_seconds,
                ..
            }
            | BenchmarkFlags::EndToEndUvVllmTorch {
                http_timeout_seconds,
                ..
            }
            | BenchmarkFlags::EndToEndDockerSglangTorch {
                http_timeout_seconds,
                ..
            }
            | BenchmarkFlags::EndToEndUvSglangTorch {
                http_timeout_seconds,
                ..
            }
            | BenchmarkFlags::EvalLlamacppCliStockToolsGgufText {
                http_timeout_seconds,
                ..
            }
            | BenchmarkFlags::EvalMlxMacosPipetteMlx {
                http_timeout_seconds,
                ..
            }
            | BenchmarkFlags::EvalDockerVllmTorch {
                http_timeout_seconds,
                ..
            }
            | BenchmarkFlags::EvalUvVllmTorch {
                http_timeout_seconds,
                ..
            }
            | BenchmarkFlags::EvalDockerSglangTorch {
                http_timeout_seconds,
                ..
            }
            | BenchmarkFlags::EvalUvSglangTorch {
                http_timeout_seconds,
                ..
            } => *http_timeout_seconds,
            _ => None,
        }
    }

    /// The doom-loop monitor overrides — carried only by the `eval` variants.
    pub fn doomloop(&self) -> Option<&DoomloopOverrides> {
        match self {
            BenchmarkFlags::EvalLlamacppCliStockToolsGgufText { doomloop, .. }
            | BenchmarkFlags::EvalMlxMacosPipetteMlx { doomloop, .. }
            | BenchmarkFlags::EvalDockerVllmTorch { doomloop, .. }
            | BenchmarkFlags::EvalUvVllmTorch { doomloop, .. }
            | BenchmarkFlags::EvalDockerSglangTorch { doomloop, .. }
            | BenchmarkFlags::EvalUvSglangTorch { doomloop, .. } => Some(doomloop),
            _ => None,
        }
    }

    /// The host-readiness gate settings — carried by the timing (and `vl`)
    /// variants; `None` on `eval` and when the cell set no override.
    pub fn readiness(&self) -> Option<&ReadinessOverrides> {
        match self {
            BenchmarkFlags::PrefillLlamacppCliStockToolsGgufText { readiness }
            | BenchmarkFlags::DecodeLlamacppCliStockToolsGgufText { readiness }
            | BenchmarkFlags::EndToEndLlamacppCliStockToolsGgufText { readiness, .. }
            | BenchmarkFlags::VlLlamacppCliStockToolsGgufVision { readiness, .. }
            | BenchmarkFlags::PrefillMlxMacosPipetteMlx { readiness }
            | BenchmarkFlags::DecodeMlxMacosPipetteMlx { readiness }
            | BenchmarkFlags::EndToEndMlxMacosPipetteMlx { readiness }
            | BenchmarkFlags::PrefillLlamacppIosPipetteGgufText { readiness }
            | BenchmarkFlags::DecodeLlamacppIosPipetteGgufText { readiness }
            | BenchmarkFlags::EndToEndLlamacppIosPipetteGgufText { readiness }
            | BenchmarkFlags::PrefillMlxIosPipetteMlx { readiness }
            | BenchmarkFlags::DecodeMlxIosPipetteMlx { readiness }
            | BenchmarkFlags::EndToEndMlxIosPipetteMlx { readiness }
            | BenchmarkFlags::EndToEndDockerVllmTorch { readiness, .. }
            | BenchmarkFlags::EndToEndUvVllmTorch { readiness, .. }
            | BenchmarkFlags::EndToEndDockerSglangTorch { readiness, .. }
            | BenchmarkFlags::EndToEndUvSglangTorch { readiness, .. } => readiness.as_ref(),
            BenchmarkFlags::EvalLlamacppCliStockToolsGgufText { .. }
            | BenchmarkFlags::EvalMlxMacosPipetteMlx { .. }
            | BenchmarkFlags::EvalDockerVllmTorch { .. }
            | BenchmarkFlags::EvalUvVllmTorch { .. }
            | BenchmarkFlags::EvalDockerSglangTorch { .. }
            | BenchmarkFlags::EvalUvSglangTorch { .. } => None,
        }
    }
}

/// Flat wire form of [`BenchmarkFlags`]: the three axis keys plus the knobs.
/// Exists only at the serde boundary — `TryFrom` validates and lifts it.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkFlagRef {
    pub runtime_type: RuntimeType,
    pub model_type: ModelType,
    pub benchmark_type: BenchmarkType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_timeout_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "doomloop_is_default")]
    pub doomloop: DoomloopOverrides,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub readiness: Option<ReadinessOverrides>,
}

/// The cell coordinates a [`BenchmarkFlags`] variant is tagged with. Stripped
/// before submission: the result names its cell in its own columns.
const AXIS_KEYS: [&str; 3] = ["runtime_type", "model_type", "benchmark_type"];

impl BenchmarkFlags {
    /// What the cell ran under, as the result records it: this value minus the
    /// axis keys, which name *which* cell the flags belong to rather than what
    /// the harness did — and the result already carries that identity.
    ///
    /// Report the value the run **resolved to**, not the one the plan authored:
    /// a `readiness` block whose fields are still `None` describes a request,
    /// and the request is not what happened. A variant with no `readiness`
    /// field reports none, which is also the truth — those cells do not gate.
    pub fn submission_value(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(self)
            // `BenchmarkFlags` serializes infallibly (no maps, floats, or
            // fallible `Serialize` impls), so this is the impossible branch.
            // Fall closed on an empty object rather than dropping the record.
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
        if let Some(fields) = value.as_object_mut() {
            AXIS_KEYS.iter().for_each(|key| {
                fields.remove(*key);
            });
        }
        value
    }
}

fn doomloop_is_default(d: &DoomloopOverrides) -> bool {
    *d == DoomloopOverrides::default()
}

impl BenchmarkFlagRef {
    /// Reject a knob that's set but not accepted by this cell.
    fn deny(&self, set: bool, knob: &'static str) -> Result<(), BenchmarkFlagError> {
        if set {
            Err(BenchmarkFlagError::KnobNotAllowed {
                knob,
                benchmark: self.benchmark_type,
                runtime: self.runtime_type,
                model: self.model_type,
            })
        } else {
            Ok(())
        }
    }

    fn deny_http(&self) -> Result<(), BenchmarkFlagError> {
        self.deny(self.http_timeout_seconds.is_some(), "http_timeout_seconds")
    }

    fn deny_doomloop(&self) -> Result<(), BenchmarkFlagError> {
        self.deny(self.doomloop != DoomloopOverrides::default(), "doomloop")
    }

    fn deny_readiness(&self) -> Result<(), BenchmarkFlagError> {
        self.deny(self.readiness.is_some(), "readiness")
    }
}

impl TryFrom<BenchmarkFlagRef> for BenchmarkFlags {
    type Error = BenchmarkFlagError;

    fn try_from(r: BenchmarkFlagRef) -> Result<Self, Self::Error> {
        use BenchmarkType as B;
        use ModelType as M;
        use RuntimeType as R;

        let flags = match (r.benchmark_type, r.runtime_type, r.model_type) {
            // Timing cells: readiness only.
            (B::PrefillThroughput, R::LlamacppCliStockTools, M::GgufText) => {
                r.deny_http()?;
                r.deny_doomloop()?;
                BenchmarkFlags::PrefillLlamacppCliStockToolsGgufText {
                    readiness: r.readiness,
                }
            }
            (B::DecodeThroughput, R::LlamacppCliStockTools, M::GgufText) => {
                r.deny_http()?;
                r.deny_doomloop()?;
                BenchmarkFlags::DecodeLlamacppCliStockToolsGgufText {
                    readiness: r.readiness,
                }
            }
            (B::EndToEndLatency, R::LlamacppCliStockTools, M::GgufText) => {
                r.deny_doomloop()?;
                BenchmarkFlags::EndToEndLlamacppCliStockToolsGgufText {
                    http_timeout_seconds: r.http_timeout_seconds,
                    readiness: r.readiness,
                }
            }
            (B::PrefillThroughput, R::MlxMacosPipette, M::Mlx) => {
                r.deny_http()?;
                r.deny_doomloop()?;
                BenchmarkFlags::PrefillMlxMacosPipetteMlx {
                    readiness: r.readiness,
                }
            }
            (B::DecodeThroughput, R::MlxMacosPipette, M::Mlx) => {
                r.deny_http()?;
                r.deny_doomloop()?;
                BenchmarkFlags::DecodeMlxMacosPipetteMlx {
                    readiness: r.readiness,
                }
            }
            (B::EndToEndLatency, R::MlxMacosPipette, M::Mlx) => {
                r.deny_http()?;
                r.deny_doomloop()?;
                BenchmarkFlags::EndToEndMlxMacosPipetteMlx {
                    readiness: r.readiness,
                }
            }
            (B::PrefillThroughput, R::LlamacppIosPipette, M::GgufText) => {
                r.deny_http()?;
                r.deny_doomloop()?;
                BenchmarkFlags::PrefillLlamacppIosPipetteGgufText {
                    readiness: r.readiness,
                }
            }
            (B::DecodeThroughput, R::LlamacppIosPipette, M::GgufText) => {
                r.deny_http()?;
                r.deny_doomloop()?;
                BenchmarkFlags::DecodeLlamacppIosPipetteGgufText {
                    readiness: r.readiness,
                }
            }
            (B::EndToEndLatency, R::LlamacppIosPipette, M::GgufText) => {
                r.deny_http()?;
                r.deny_doomloop()?;
                BenchmarkFlags::EndToEndLlamacppIosPipetteGgufText {
                    readiness: r.readiness,
                }
            }
            (B::PrefillThroughput, R::MlxIosPipette, M::Mlx) => {
                r.deny_http()?;
                r.deny_doomloop()?;
                BenchmarkFlags::PrefillMlxIosPipetteMlx {
                    readiness: r.readiness,
                }
            }
            (B::DecodeThroughput, R::MlxIosPipette, M::Mlx) => {
                r.deny_http()?;
                r.deny_doomloop()?;
                BenchmarkFlags::DecodeMlxIosPipetteMlx {
                    readiness: r.readiness,
                }
            }
            (B::EndToEndLatency, R::MlxIosPipette, M::Mlx) => {
                r.deny_http()?;
                r.deny_doomloop()?;
                BenchmarkFlags::EndToEndMlxIosPipetteMlx {
                    readiness: r.readiness,
                }
            }
            (B::EndToEndLatency, R::DockerVllm, M::Torch) => {
                r.deny_doomloop()?;
                BenchmarkFlags::EndToEndDockerVllmTorch {
                    http_timeout_seconds: r.http_timeout_seconds,
                    readiness: r.readiness,
                }
            }
            (B::EndToEndLatency, R::UvVllm, M::Torch) => {
                r.deny_doomloop()?;
                BenchmarkFlags::EndToEndUvVllmTorch {
                    http_timeout_seconds: r.http_timeout_seconds,
                    readiness: r.readiness,
                }
            }
            (B::EndToEndLatency, R::DockerSglang, M::Torch) => {
                r.deny_doomloop()?;
                BenchmarkFlags::EndToEndDockerSglangTorch {
                    http_timeout_seconds: r.http_timeout_seconds,
                    readiness: r.readiness,
                }
            }
            (B::EndToEndLatency, R::UvSglang, M::Torch) => {
                r.deny_doomloop()?;
                BenchmarkFlags::EndToEndUvSglangTorch {
                    http_timeout_seconds: r.http_timeout_seconds,
                    readiness: r.readiness,
                }
            }
            // vl-throughput: HTTP timeout + readiness (no doom-loop).
            (B::VlThroughput, R::LlamacppCliStockTools, M::GgufVision) => {
                r.deny_doomloop()?;
                BenchmarkFlags::VlLlamacppCliStockToolsGgufVision {
                    http_timeout_seconds: r.http_timeout_seconds,
                    readiness: r.readiness,
                }
            }
            // eval: HTTP timeout + doom-loop (no readiness).
            (B::Eval, R::LlamacppCliStockTools, M::GgufText) => {
                r.deny_readiness()?;
                BenchmarkFlags::EvalLlamacppCliStockToolsGgufText {
                    http_timeout_seconds: r.http_timeout_seconds,
                    doomloop: r.doomloop,
                }
            }
            (B::Eval, R::MlxMacosPipette, M::Mlx) => {
                r.deny_readiness()?;
                BenchmarkFlags::EvalMlxMacosPipetteMlx {
                    http_timeout_seconds: r.http_timeout_seconds,
                    doomloop: r.doomloop,
                }
            }
            (B::Eval, R::DockerVllm, M::Torch) => {
                r.deny_readiness()?;
                BenchmarkFlags::EvalDockerVllmTorch {
                    http_timeout_seconds: r.http_timeout_seconds,
                    doomloop: r.doomloop,
                }
            }
            (B::Eval, R::UvVllm, M::Torch) => {
                r.deny_readiness()?;
                BenchmarkFlags::EvalUvVllmTorch {
                    http_timeout_seconds: r.http_timeout_seconds,
                    doomloop: r.doomloop,
                }
            }
            (B::Eval, R::DockerSglang, M::Torch) => {
                r.deny_readiness()?;
                BenchmarkFlags::EvalDockerSglangTorch {
                    http_timeout_seconds: r.http_timeout_seconds,
                    doomloop: r.doomloop,
                }
            }
            (B::Eval, R::UvSglang, M::Torch) => {
                r.deny_readiness()?;
                BenchmarkFlags::EvalUvSglangTorch {
                    http_timeout_seconds: r.http_timeout_seconds,
                    doomloop: r.doomloop,
                }
            }
            _ => {
                return Err(BenchmarkFlagError::NoSuchCombination {
                    benchmark: r.benchmark_type,
                    runtime: r.runtime_type,
                    model: r.model_type,
                })
            }
        };
        Ok(flags)
    }
}

impl From<BenchmarkFlags> for BenchmarkFlagRef {
    fn from(f: BenchmarkFlags) -> Self {
        let (benchmark_type, runtime_type, model_type) = f.axes();
        BenchmarkFlagRef {
            runtime_type,
            model_type,
            benchmark_type,
            http_timeout_seconds: f.http_timeout(),
            doomloop: f.doomloop().cloned().unwrap_or_default(),
            readiness: f.readiness().cloned(),
        }
    }
}

#[cfg(test)]
mod submission_tests {
    use super::*;

    fn prefill_flags(readiness: Option<ReadinessOverrides>) -> BenchmarkFlags {
        BenchmarkFlags::PrefillLlamacppCliStockToolsGgufText { readiness }
    }

    /// The axis keys name which cell the flags belong to; the result already
    /// carries that identity in its own columns.
    #[test]
    fn the_cell_coordinates_are_not_reported() {
        let value = prefill_flags(None).submission_value();

        AXIS_KEYS.iter().for_each(|key| {
            assert_eq!(value.get(*key), None, "{key} should not be reported");
        });
    }

    /// A resolved readiness block reports both fields set — which is what
    /// removes the tri-state, without a second vocabulary for the answer.
    #[test]
    fn a_resolved_readiness_block_reports_both_fields() {
        let flags = prefill_flags(Some(ReadinessOverrides {
            max_wait_secs: Some(300),
            skip_thermal: Some(true),
        }));

        assert_eq!(
            flags.submission_value().get("readiness"),
            Some(&serde_json::json!({"max_wait_secs": 300, "skip_thermal": true}))
        );
    }

    /// An eval cell has no readiness field to fill, because it never gates —
    /// so the conversion refuses one rather than recording a policy the run
    /// did not apply.
    #[test]
    fn an_ungated_cell_refuses_a_readiness_block() {
        let cell = BenchmarkFlagRef {
            benchmark_type: BenchmarkType::Eval,
            runtime_type: RuntimeType::LlamacppCliStockTools,
            model_type: ModelType::GgufText,
            http_timeout_seconds: None,
            doomloop: Default::default(),
            readiness: Some(ReadinessOverrides {
                max_wait_secs: Some(300),
                skip_thermal: Some(false),
            }),
        };

        assert!(matches!(
            BenchmarkFlags::try_from(cell),
            Err(BenchmarkFlagError::KnobNotAllowed {
                knob: "readiness",
                ..
            })
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a ref for one cell with the given knobs.
    fn cell_ref(
        benchmark_type: BenchmarkType,
        runtime_type: RuntimeType,
        model_type: ModelType,
    ) -> BenchmarkFlagRef {
        BenchmarkFlagRef {
            runtime_type,
            model_type,
            benchmark_type,
            http_timeout_seconds: None,
            doomloop: DoomloopOverrides::default(),
            readiness: None,
        }
    }

    #[test]
    fn eval_carries_http_and_doomloop_no_readiness() -> anyhow::Result<()> {
        let mut r = cell_ref(
            BenchmarkType::Eval,
            RuntimeType::LlamacppCliStockTools,
            ModelType::GgufText,
        );
        r.http_timeout_seconds = Some(600);
        let flags = BenchmarkFlags::try_from(r)?;
        assert_eq!(flags.http_timeout(), Some(600));
        assert!(flags.doomloop().is_some());
        assert!(flags.readiness().is_none());
        assert_eq!(
            flags.axes(),
            (
                BenchmarkType::Eval,
                RuntimeType::LlamacppCliStockTools,
                ModelType::GgufText
            )
        );
        Ok(())
    }

    /// Readiness rides an mlx prefill cell (a cell `RuntimeFlags` has no variant
    /// for), round-tripping through the flat wire.
    #[test]
    fn readiness_on_mlx_timing_cell_round_trips() -> anyhow::Result<()> {
        let mut r = cell_ref(
            BenchmarkType::PrefillThroughput,
            RuntimeType::MlxMacosPipette,
            ModelType::Mlx,
        );
        r.readiness = Some(ReadinessOverrides {
            skip_thermal: None,
            max_wait_secs: Some(1800),
        });
        let flags = BenchmarkFlags::try_from(r)?;
        assert_eq!(flags.readiness().and_then(|x| x.max_wait_secs), Some(1800));

        let wire = toml::to_string(&flags)?;
        assert!(
            wire.contains(r#"runtime_type = "mlx_macos_pipette""#),
            "got:\n{wire}"
        );
        assert!(wire.contains("max_wait_secs = 1800"), "got:\n{wire}");
        let round: BenchmarkFlags = toml::from_str(&wire)?;
        assert_eq!(flags, round);
        Ok(())
    }

    /// The torch end-to-end latency cell is readiness-carrying — the override
    /// must survive `TryFrom` and the wire round-trip so the torch seam can gate
    /// on it (parallels the MLX/llama.cpp timing cells).
    #[test]
    fn readiness_on_torch_end_to_end_cell_round_trips() -> anyhow::Result<()> {
        let mut r = cell_ref(
            BenchmarkType::EndToEndLatency,
            RuntimeType::DockerVllm,
            ModelType::Torch,
        );
        r.readiness = Some(ReadinessOverrides {
            skip_thermal: None,
            max_wait_secs: Some(1800),
        });
        let flags = BenchmarkFlags::try_from(r)?;
        assert_eq!(flags.readiness().and_then(|x| x.max_wait_secs), Some(1800));

        let round: BenchmarkFlags = toml::from_str(&toml::to_string(&flags)?)?;
        assert_eq!(flags, round);
        Ok(())
    }

    #[test]
    fn readiness_on_eval_is_rejected() {
        let mut r = cell_ref(
            BenchmarkType::Eval,
            RuntimeType::LlamacppCliStockTools,
            ModelType::GgufText,
        );
        r.readiness = Some(ReadinessOverrides {
            skip_thermal: None,
            max_wait_secs: Some(60),
        });
        assert!(matches!(
            BenchmarkFlags::try_from(r),
            Err(BenchmarkFlagError::KnobNotAllowed {
                knob: "readiness",
                ..
            })
        ));
    }

    #[test]
    fn http_timeout_on_timing_cell_is_rejected() {
        let mut r = cell_ref(
            BenchmarkType::DecodeThroughput,
            RuntimeType::MlxMacosPipette,
            ModelType::Mlx,
        );
        r.http_timeout_seconds = Some(600);
        assert!(matches!(
            BenchmarkFlags::try_from(r),
            Err(BenchmarkFlagError::KnobNotAllowed {
                knob: "http_timeout_seconds",
                ..
            })
        ));
    }

    #[test]
    fn doomloop_off_eval_is_rejected() -> anyhow::Result<()> {
        let mut r = cell_ref(
            BenchmarkType::VlThroughput,
            RuntimeType::LlamacppCliStockTools,
            ModelType::GgufVision,
        );
        r.doomloop = toml::from_str("exact_repeat = { required = 5 }")?;
        assert!(matches!(
            BenchmarkFlags::try_from(r),
            Err(BenchmarkFlagError::KnobNotAllowed {
                knob: "doomloop",
                ..
            })
        ));
        Ok(())
    }

    /// max_memory / Apple / an unmodeled (e.g. mobile) triple has no variant.
    #[test]
    fn unsupported_cells_have_no_variant() {
        for r in [
            cell_ref(
                BenchmarkType::MaxMemoryUsage,
                RuntimeType::LlamacppCliStockTools,
                ModelType::GgufText,
            ),
            cell_ref(
                BenchmarkType::Eval,
                RuntimeType::LlamacppCliStockTools,
                ModelType::AppleFoundationText,
            ),
            cell_ref(
                BenchmarkType::PrefillThroughput,
                RuntimeType::LlamacppApkPipette,
                ModelType::GgufText,
            ),
            // iOS gates its timing cells, but has no HTTP client to bound and no
            // server to watch — so eval and max-memory carry nothing there either.
            cell_ref(
                BenchmarkType::Eval,
                RuntimeType::LlamacppIosPipette,
                ModelType::GgufText,
            ),
            cell_ref(
                BenchmarkType::MaxMemoryUsage,
                RuntimeType::MlxIosPipette,
                ModelType::Mlx,
            ),
        ] {
            assert!(matches!(
                BenchmarkFlags::try_from(r),
                Err(BenchmarkFlagError::NoSuchCombination { .. })
            ));
        }
    }

    #[test]
    fn unset_knobs_omitted_from_wire() -> anyhow::Result<()> {
        let flags = BenchmarkFlags::try_from(cell_ref(
            BenchmarkType::Eval,
            RuntimeType::LlamacppCliStockTools,
            ModelType::GgufText,
        ))?;
        let wire = toml::to_string(&flags)?;
        assert!(!wire.contains("http_timeout_seconds"), "got:\n{wire}");
        assert!(!wire.contains("doomloop"), "got:\n{wire}");
        assert!(!wire.contains("readiness"), "got:\n{wire}");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Exhaustive field-acceptance sweep
    // -----------------------------------------------------------------------

    /// Every settable field on [`BenchmarkFlagRef`], with a valid wire value.
    ///
    /// `doomloop` is probed with a single leaf: it serializes expanded (every
    /// monitor spelled out, unset ones `null`), so only the leaf that was
    /// authored can be compared back.
    const PROBES: &[(&str, &str)] = &[
        ("http_timeout_seconds", "600"),
        ("readiness", "{ max_wait_secs = 120 }"),
        ("doomloop", "{ exact_repeat = { required = 3 } }"),
    ];

    /// Every cell that has a variant, as `(benchmark, runtime, model, accepted
    /// fields)` — transcribed from the variant field lists, not derived from
    /// them, so a field added to or dropped from a variant without a matching
    /// edit here fails the sweep.
    #[rustfmt::skip]
    const CELLS: &[(&str, &str, &str, &[&str])] = &[
        // Timing cells: the readiness gate only — nothing drives generation,
        // except where the cell reaches a server over HTTP and must bound it.
        ("prefill_throughput", "llamacpp_cli_stock_tools", "gguf_text", &["readiness"]),
        ("decode_throughput", "llamacpp_cli_stock_tools", "gguf_text", &["readiness"]),
        ("end_to_end_latency", "llamacpp_cli_stock_tools", "gguf_text",
         &["http_timeout_seconds", "readiness"]),
        ("prefill_throughput", "mlx_macos_pipette", "mlx", &["readiness"]),
        ("decode_throughput", "mlx_macos_pipette", "mlx", &["readiness"]),
        ("end_to_end_latency", "mlx_macos_pipette", "mlx", &["readiness"]),
        // iOS runs in-process: the gate is the only thing a plan can drive.
        ("prefill_throughput", "llamacpp_ios_pipette", "gguf_text", &["readiness"]),
        ("decode_throughput", "llamacpp_ios_pipette", "gguf_text", &["readiness"]),
        ("end_to_end_latency", "llamacpp_ios_pipette", "gguf_text", &["readiness"]),
        ("prefill_throughput", "mlx_ios_pipette", "mlx", &["readiness"]),
        ("decode_throughput", "mlx_ios_pipette", "mlx", &["readiness"]),
        ("end_to_end_latency", "mlx_ios_pipette", "mlx", &["readiness"]),
        ("end_to_end_latency", "docker_vllm", "torch", &["http_timeout_seconds", "readiness"]),
        ("end_to_end_latency", "uv_vllm", "torch", &["http_timeout_seconds", "readiness"]),
        ("end_to_end_latency", "docker_sglang", "torch", &["http_timeout_seconds", "readiness"]),
        ("end_to_end_latency", "uv_sglang", "torch", &["http_timeout_seconds", "readiness"]),
        // vl-throughput: an HTTP server run, but no generation to loop-detect.
        ("vl_throughput", "llamacpp_cli_stock_tools", "gguf_vision",
         &["http_timeout_seconds", "readiness"]),
        // eval: HTTP timeout + doom-loop; the runner owns readiness here.
        ("eval", "llamacpp_cli_stock_tools", "gguf_text",
         &["http_timeout_seconds", "doomloop"]),
        ("eval", "mlx_macos_pipette", "mlx", &["http_timeout_seconds", "doomloop"]),
        ("eval", "docker_vllm", "torch", &["http_timeout_seconds", "doomloop"]),
        ("eval", "uv_vllm", "torch", &["http_timeout_seconds", "doomloop"]),
        ("eval", "docker_sglang", "torch", &["http_timeout_seconds", "doomloop"]),
        ("eval", "uv_sglang", "torch", &["http_timeout_seconds", "doomloop"]),
    ];

    fn axes_wire(benchmark: &str, runtime: &str, model: &str) -> String {
        format!(
            "runtime_type = \"{runtime}\"\nmodel_type = \"{model}\"\nbenchmark_type = \"{benchmark}\"\n"
        )
    }

    fn parse(wire: &str) -> Result<BenchmarkFlags, toml::de::Error> {
        toml::from_str(wire)
    }

    /// The [`CELLS`] table covers every variant exactly once. Without this the
    /// sweep below could silently skip a cell — a new variant would add no
    /// coverage and nothing would say so.
    #[test]
    fn cell_table_covers_every_variant() -> anyhow::Result<()> {
        use strum::EnumCount as _;

        let mut seen = std::collections::HashSet::new();
        for (benchmark, runtime, model, _) in CELLS {
            let flags = parse(&axes_wire(benchmark, runtime, model)).map_err(|e| {
                anyhow::anyhow!("{benchmark}/{runtime}/{model} has no variant: {e}")
            })?;
            anyhow::ensure!(
                seen.insert(std::mem::discriminant(&flags)),
                "{benchmark}/{runtime}/{model} duplicates an earlier row's variant"
            );
        }
        anyhow::ensure!(
            seen.len() == BenchmarkFlags::COUNT,
            "table covers {} variants, enum has {}",
            seen.len(),
            BenchmarkFlags::COUNT
        );
        Ok(())
    }

    /// Every variant accepts exactly the fields it declares, rejects every
    /// other one, and carries the values it accepted.
    ///
    /// Same hand-written-arm hazard as `RuntimeFlags`: a missing `deny` drops
    /// an author's setting, a stray one rejects a legitimate cell, and an arm
    /// that accepts a field without reading it off the ref loses the value
    /// while still parsing. The value check compares against the authored wire,
    /// since a dropped field is absent from both sides of a `parse(to_string)`
    /// round-trip.
    #[test]
    fn every_cell_accepts_exactly_its_declared_fields_and_keeps_their_values() -> anyhow::Result<()>
    {
        for (benchmark, runtime, model, accepted) in CELLS {
            for (field, value) in PROBES {
                let wire = format!(
                    "{}{field} = {value}\n",
                    axes_wire(benchmark, runtime, model)
                );
                let expected = accepted.contains(field);
                let parsed = parse(&wire);
                anyhow::ensure!(
                    parsed.is_ok() == expected,
                    "{benchmark}/{runtime}/{model}: `{field}` should be {} but was {}",
                    if expected { "accepted" } else { "rejected" },
                    if expected { "rejected" } else { "accepted" },
                );
                let Ok(flags) = parsed else { continue };
                let emitted = toml::Value::try_from(&flags)?;
                let authored = toml::from_str::<toml::Value>(&format!("{field} = {value}"))?;
                // `doomloop` comes back expanded, so compare the authored leaf
                // rather than the whole table.
                let (got, want) = if *field == "doomloop" {
                    (
                        emitted
                            .get(field)
                            .and_then(|d| d.get("exact_repeat")?.get("required")),
                        authored
                            .get(field)
                            .and_then(|d| d.get("exact_repeat")?.get("required")),
                    )
                } else {
                    (emitted.get(field), authored.get(field))
                };
                anyhow::ensure!(
                    got == want,
                    "{benchmark}/{runtime}/{model}: `{field}` was accepted but came back as {got:?}, authored {want:?}",
                );
            }
        }
        Ok(())
    }
}
