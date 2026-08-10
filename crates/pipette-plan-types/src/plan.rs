//! Outer [`Plan`] + its supporting types: [`Matrix`] (the variant
//! cartesian), [`RetryConfig`], [`TransportConfig`], [`ShellType`],
//! [`RunnableCell`], and [`TypedCell`]. Re-exported flat from the
//! parent crate's `lib.rs`, so external consumers reference these as
//! `pipette_plan_types::Plan` etc. without seeing the submodule.

use std::{
    collections::{BTreeSet, HashSet},
    fs,
    num::NonZeroUsize,
    path::Path,
};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use pipette_doomloop::plan::DoomloopOverrides;

use crate::{
    is_compatible, AuthToken, BenchmarkFlags, BenchmarkId, BenchmarkType, ClientId, Model,
    ModelFlags, NonEmptyVec, Runtime, RuntimeFlags,
};

// ---------------------------------------------------------------------------
// Variant
// ---------------------------------------------------------------------------

/// One self-consistent block of model deployments, runtimes, clients,
/// and an optional benchmark override. The variant expands at
/// plan-build time to the full cross-product
/// `models × runtimes × clients × benchmarks`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct Variant {
    pub models: NonEmptyVec<Model>,
    pub runtimes: NonEmptyVec<Runtime>,
    pub clients: NonEmptyVec<ClientId>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_benchmark_ids"
    )]
    pub benchmarks: Option<NonEmptyVec<BenchmarkId>>,
    /// Typed per-cell runtime flags authored on this variant. Each entry
    /// names its `(benchmark, runtime, model)` cell and carries typed knobs
    /// (plus a `raw` escape hatch); it's resolved onto matching cells and
    /// shipped to the client as the `--runtime-flags` JSON, which the client
    /// renders to its tool's argv. Validated at parse: every entry must
    /// match a real cell in this variant; which knobs a cell accepts (and
    /// that in-process/mlx/apple runtimes accept none) is enforced
    /// structurally by `RuntimeFlags`' `TryFrom`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runtime_flags: Vec<RuntimeFlags>,
    /// Typed per-cell model-generation flags authored on this variant, keyed
    /// on `(benchmark, model)` (see [`ModelFlags`]). Only eval cells carry
    /// generation flags — `ModelFlags`' `TryFrom` rejects any non-eval entry —
    /// so this is the structural "non-evals carry no model flags" rule.
    /// Resolved onto matching cells; each cell forwards its `enable_thinking`
    /// on the eval invocation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_flags: Vec<ModelFlags>,
    /// Typed per-cell eval-run flags authored on this variant, keyed on
    /// `(benchmark, model)` (see [`BenchmarkFlags`]) — pipette's HTTP-client
    /// timeout and doom-loop monitor. Eval-only (`BenchmarkFlags`' `TryFrom`
    /// rejects non-eval entries). Resolved onto matching cells.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub benchmark_flags: Vec<BenchmarkFlags>,
}

/// Reason a variant fails [`Variant::runnable_pairs`]. A
/// variant is fine as long as every model can pair with at least one
/// runtime *and* every runtime can serve at least one model — mixed-
/// kind models and runtimes are allowed, and the runner just skips
/// (model, runtime) pairs that don't match at cell-expansion time.
/// The two variants below name the "orphan" case: a model or runtime
/// that has no compatible counterpart in this variant.
// Variants boxed because `Runtime` is large (~640 bytes) and the
// resulting size mismatch trips clippy::large_enum_variant. The
// error path is cold, so the indirection is free in practice.
#[derive(Debug, Clone, Error)]
pub enum VariantCompatibilityError {
    #[error("model {0} has no compatible runtime in this variant")]
    ModelHasNoCompatibleRuntime(Box<Model>),
    #[error("runtime {0} has no compatible model in this variant")]
    RuntimeHasNoCompatibleModel(Box<Runtime>),
}

/// The (model, runtime) pairs a variant expands into: the cartesian product
/// filtered by [`is_compatible`]. Shared by both the local-dispatch
/// [`Variant`] and the scheduler-mode variant, since orphan detection depends
/// only on the model/runtime lists, not on how eligibility is expressed.
///
/// Errors if any model or runtime has no compatible counterpart — an "orphan"
/// that would dangle without ever producing or being consumed by a cell,
/// almost certainly an authoring mistake. Mixed kinds are allowed otherwise:
/// incompatible pairs are silently dropped.
pub(crate) fn runnable_pairs_of<'a>(
    models: &'a [Model],
    runtimes: &'a [Runtime],
) -> anyhow::Result<Vec<(&'a Model, &'a Runtime)>, VariantCompatibilityError> {
    if let Some(model) = models
        .iter()
        .find(|m| !runtimes.iter().any(|r| is_compatible(m, r)))
    {
        return Err(VariantCompatibilityError::ModelHasNoCompatibleRuntime(
            Box::new(model.clone()),
        ));
    }
    if let Some(runtime) = runtimes
        .iter()
        .find(|r| !models.iter().any(|m| is_compatible(m, r)))
    {
        return Err(VariantCompatibilityError::RuntimeHasNoCompatibleModel(
            Box::new(runtime.clone()),
        ));
    }
    Ok(models
        .iter()
        .flat_map(|m| {
            runtimes
                .iter()
                .filter(move |r| is_compatible(m, r))
                .map(move |r| (m, r))
        })
        .collect())
}

impl Variant {
    /// Materialize the (model, runtime) pairs the runner should turn
    /// into cells: the cartesian product filtered by [`is_compatible`].
    ///
    /// Errors if any model or runtime in the variant has no
    /// compatible counterpart — that's an "orphan" that would dangle
    /// without ever producing or being consumed by a cell, almost
    /// certainly an authoring mistake. Mixed kinds are allowed
    /// otherwise: incompatible pairs are silently dropped.
    pub fn runnable_pairs(
        &self,
    ) -> anyhow::Result<Vec<(&Model, &Runtime)>, VariantCompatibilityError> {
        runnable_pairs_of(&self.models, &self.runtimes)
    }

    /// The runtime flags authored on this variant that apply to a cell running
    /// `benchmark` on `runtime` with `model`, if any (at most one match).
    fn resolved_runtime_flags(
        &self,
        benchmark: Option<BenchmarkType>,
        runtime: &Runtime,
        model: &Model,
    ) -> Option<RuntimeFlags> {
        resolve_runtime_flags(&self.runtime_flags, benchmark, runtime, model)
    }

    /// The model flags authored on this variant for a cell running `benchmark`
    /// with `model`, if any. `None` on non-eval cells (every `ModelFlags` entry
    /// is eval-only) and when no entry matches the cell.
    fn resolved_model_flags(
        &self,
        benchmark: Option<BenchmarkType>,
        model: &Model,
    ) -> Option<ModelFlags> {
        resolve_model_flags(&self.model_flags, benchmark, model)
    }

    /// The benchmark flags authored on this variant for a cell running
    /// `benchmark` on `runtime` with `model`, if any — matched on the full
    /// `(benchmark, runtime, model)` triple. `None` when no entry matches.
    fn resolved_benchmark_flags(
        &self,
        benchmark: Option<BenchmarkType>,
        runtime: &Runtime,
        model: &Model,
    ) -> Option<BenchmarkFlags> {
        resolve_benchmark_flags(&self.benchmark_flags, benchmark, runtime, model)
    }
}

/// Pick the runtime-flags entry matching a cell's `(benchmark, runtime, model)`,
/// if any (at most one). Shared by both plan formats — resolution depends only
/// on the flag list, not on how the enclosing variant expresses eligibility.
pub(crate) fn resolve_runtime_flags(
    entries: &[RuntimeFlags],
    benchmark: Option<BenchmarkType>,
    runtime: &Runtime,
    model: &Model,
) -> Option<RuntimeFlags> {
    let bt = benchmark?;
    entries
        .iter()
        .find(|f| f.matches(bt, runtime, model))
        .cloned()
}

/// Pick the model-flags entry matching a cell's `(benchmark, model)`, if any.
/// `None` off eval (every entry is eval-only) and when nothing matches.
pub(crate) fn resolve_model_flags(
    entries: &[ModelFlags],
    benchmark: Option<BenchmarkType>,
    model: &Model,
) -> Option<ModelFlags> {
    let bt = benchmark?;
    entries.iter().find(|f| f.matches(bt, model)).cloned()
}

/// Pick the benchmark-flags entry matching a cell's `(benchmark, runtime,
/// model)`, if any.
pub(crate) fn resolve_benchmark_flags(
    entries: &[BenchmarkFlags],
    benchmark: Option<BenchmarkType>,
    runtime: &Runtime,
    model: &Model,
) -> Option<BenchmarkFlags> {
    let bt = benchmark?;
    entries
        .iter()
        .find(|f| f.matches(bt, runtime, model))
        .cloned()
}

// ---------------------------------------------------------------------------
// Matrix (the inner cells of a Plan)
// ---------------------------------------------------------------------------

/// The matrix portion of a plan: an optional default benchmark list,
/// and the variants whose cartesian product expands to executable
/// cells. The outer [`Plan`] flattens this struct alongside `plan_id`,
/// `retry`, and `transports`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub struct Matrix {
    /// Optional default benchmark list. A variant can override this
    /// with `Variant::benchmarks`; when this is absent, every variant
    /// must provide its own benchmark list.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_benchmark_ids"
    )]
    pub benchmarks: Option<NonEmptyVec<BenchmarkId>>,
    pub variants: NonEmptyVec<Variant>,
}

// ---------------------------------------------------------------------------
// ShellType
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
pub enum ShellType {
    #[serde(rename = "posix")]
    Posix,
    #[serde(rename = "powershell")]
    PowerShell,
}

// ---------------------------------------------------------------------------
// Plan (outer)
// ---------------------------------------------------------------------------

/// A validated plan ready for cell expansion.
///
/// `Plan` guarantees:
/// - every `[[transports]]` has a unique `name`,
/// - every variant's `clients` entry references a declared transport.
///
/// The only construction paths from outside this module are
/// [`Plan::parse`], [`Plan::load`], or `Deserialize` (which routes
/// through `PlanIntake`). All three run `validate_plan`. `#[non_exhaustive]`
/// blocks external struct-literal construction, so the validation
/// invariants can be trusted.
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields, try_from = "PlanIntake")]
#[non_exhaustive]
pub struct Plan {
    pub plan_id: String,
    #[serde(default, skip_serializing_if = "RetryConfig::is_default")]
    pub retry: RetryConfig,
    pub transports: NonEmptyVec<TransportConfig>,
    /// Flattened — the optional default `benchmarks` and `variants`
    /// parse at the TOML root level.
    #[serde(flatten)]
    pub matrix: Matrix,
}

/// Private intake type for serde — same fields as `Plan`, but with no
/// validation. The `#[serde(try_from = "PlanIntake")]` attribute on
/// `Plan` forces every deserialization through `TryFrom`, which calls
/// `validate_plan` before producing a `Plan`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanIntake {
    plan_id: String,
    #[serde(default)]
    retry: RetryConfig,
    transports: NonEmptyVec<TransportConfig>,
    #[serde(flatten)]
    matrix: Matrix,
}

impl TryFrom<PlanIntake> for Plan {
    type Error = anyhow::Error;

    fn try_from(intake: PlanIntake) -> anyhow::Result<Self> {
        let plan = Plan {
            plan_id: intake.plan_id,
            retry: intake.retry,
            transports: intake.transports,
            matrix: intake.matrix,
        };
        validate_plan(&plan)?;
        Ok(plan)
    }
}

#[derive(Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RetryConfig {
    #[serde(default)]
    pub max_consecutive_failures: usize,
}

impl RetryConfig {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

impl Plan {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let raw =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&raw).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn parse(toml_str: &str) -> anyhow::Result<Self> {
        toml::from_str(toml_str).map_err(anyhow::Error::from)
    }

    /// Materialize every executable cell (variant × benchmark × model
    /// × runtime), filtered by `Model`/`Runtime` compatibility. Errors
    /// if any variant has an orphan model or runtime (one with no
    /// compatible counterpart) — that's an authoring mistake that
    /// would silently drop cells if we let it through.
    ///
    /// Each cell carries the variant's `clients` list so the runner
    /// can route to the right transports.
    pub fn cells(&self) -> anyhow::Result<Vec<TypedCell<'_>>> {
        self.matrix
            .variants
            .iter()
            .enumerate()
            .map(|(idx, variant)| {
                let pairs = variant
                    .runnable_pairs()
                    .map_err(|e| anyhow::anyhow!("variant {idx}: {e}"))?;
                let benchmarks = self.benchmarks_for_variant(idx, variant)?;
                Ok(benchmarks
                    .iter()
                    .flat_map(|benchmark| {
                        pairs.iter().map(move |(model, runtime)| TypedCell {
                            variant_idx: idx,
                            benchmark,
                            model,
                            runtime,
                            allowed_clients: variant.clients.as_ref(),
                        })
                    })
                    .collect::<Vec<_>>())
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map(|chunks| chunks.into_iter().flatten().collect())
    }

    /// Materialize the deduped set of `RunnableCell`s the runner will
    /// process. See [`Plan::cells`] for the error semantics — the two
    /// share `Variant::runnable_pairs` and surface the same orphan
    /// errors.
    pub fn runnable_cells(&self) -> anyhow::Result<HashSet<RunnableCell>> {
        self.matrix
            .variants
            .iter()
            .enumerate()
            .map(|(idx, variant)| {
                let pairs = variant
                    .runnable_pairs()
                    .map_err(|e| anyhow::anyhow!("variant {idx}: {e}"))?;
                let benchmarks = self.benchmarks_for_variant(idx, variant)?;
                let allowed_clients: Vec<ClientId> = variant.clients.iter().cloned().collect();
                Ok(benchmarks
                    .iter()
                    .flat_map(|benchmark| {
                        let allowed_clients = allowed_clients.clone();
                        let bt = benchmark_type_of(benchmark);
                        pairs.iter().map(move |(model, runtime)| RunnableCell {
                            benchmark: benchmark.clone(),
                            model: (*model).clone(),
                            runtime: (*runtime).clone(),
                            allowed_clients: allowed_clients.clone(),
                            runtime_flags: variant.resolved_runtime_flags(bt, runtime, model),
                            model_flags: variant.resolved_model_flags(bt, model),
                            benchmark_flags: variant.resolved_benchmark_flags(bt, runtime, model),
                        })
                    })
                    .collect::<HashSet<_>>())
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map(|sets| sets.into_iter().flatten().collect())
    }

    /// The first access token any model in the plan carries, if any. Used by
    /// the runner to forward [`HF_TOKEN_ENV`](crate::HF_TOKEN_ENV). (A plan is expected to use a
    /// single HF account; the first token is representative.)
    pub fn auth_token(&self) -> Option<&AuthToken> {
        self.matrix
            .variants
            .iter()
            .flat_map(|v| v.models.iter())
            .find_map(|m| m.auth_token())
    }

    fn benchmarks_for_variant<'a>(
        &'a self,
        variant_idx: usize,
        variant: &'a Variant,
    ) -> anyhow::Result<&'a [BenchmarkId]> {
        resolve_benchmarks(
            variant_idx,
            variant.benchmarks.as_deref(),
            self.matrix.benchmarks.as_deref(),
        )
    }
}

/// Resolve a variant's benchmark list: its own override if set, else the
/// plan-level default; errors when neither is present. Shared by both plan
/// formats so the fallback rule and its error text can't drift apart.
pub(crate) fn resolve_benchmarks<'a>(
    variant_idx: usize,
    variant_benchmarks: Option<&'a [BenchmarkId]>,
    plan_benchmarks: Option<&'a [BenchmarkId]>,
) -> anyhow::Result<&'a [BenchmarkId]> {
    variant_benchmarks.or(plan_benchmarks).ok_or_else(|| {
        anyhow::anyhow!(
            "variant {variant_idx}: benchmarks must be set on the variant or at the plan root"
        )
    })
}

fn validate_plan(plan: &Plan) -> anyhow::Result<()> {
    // try_fold doubles as dup-detection and as the membership index
    // for the variant→client pass below.
    let names: HashSet<&str> = plan
        .transports
        .iter()
        .try_fold(HashSet::new(), |mut acc, t| {
            if acc.insert(t.client_id()) {
                Ok(acc)
            } else {
                Err(anyhow::anyhow!(
                    "duplicate transport client_id {:?}",
                    t.client_id()
                ))
            }
        })?;

    let unknown = plan
        .matrix
        .variants
        .iter()
        .enumerate()
        .flat_map(|(idx, v)| v.clients.iter().map(move |c| (idx, c)))
        .find(|(_, c)| !names.contains(c.as_ref()));
    if let Some((idx, client)) = unknown {
        anyhow::bail!(
            "variant {idx}: clients refers to unknown transport {:?}; \
             declared transports: {:?}",
            client.as_ref(),
            names.iter().copied().collect::<BTreeSet<_>>()
        );
    }

    // Compatibility check: surface orphans at parse time so operators
    // see them up front. `cells()` does the same check during cell
    // expansion; running it here keeps the two consistent and front-
    // loads the failure. Discard the Ok payload — we'll re-materialize
    // cells on demand.
    plan.cells().map(|_| ())?;
    reject_non_posix_adb_hop(plan)?;
    validate_variant_flags(plan)
}

/// `adb_over_ssh` renders the device command for a posix shell on both hops, so
/// a `powershell` device shell would produce a command the device mangles
/// rather than rejects. Fail at load instead.
fn reject_non_posix_adb_hop(plan: &Plan) -> anyhow::Result<()> {
    match plan.transports.iter().find(|t| {
        matches!(
            t,
            TransportConfig::AdbOverSsh {
                shell: ShellType::PowerShell,
                ..
            }
        )
    }) {
        Some(t) => anyhow::bail!(
            "transport {:?}: adb_over_ssh requires shell = \"posix\" \
             (it describes the Android device shell)",
            t.client_id()
        ),
        None => Ok(()),
    }
}

/// The benchmark type for a benchmark id, or `None` when it names no known
/// kind. `pipette-plan` has no synced catalog at this layer, so it falls back
/// to the id-prefix heuristic; an unrecognized id resolves to no typed flags.
pub(crate) fn benchmark_type_of(benchmark: &BenchmarkId) -> Option<BenchmarkType> {
    BenchmarkType::from_id(benchmark.as_ref())
}

/// Reject two `runtime_flags` entries keyed on the same `(runtime, model,
/// benchmark)` cell — the entry's identity is that triple.
fn reject_duplicate_cells(idx: usize, entries: &[RuntimeFlags]) -> anyhow::Result<()> {
    let mut seen = HashSet::new();
    entries.iter().try_for_each(|entry| {
        if seen.insert(entry.axes()) {
            Ok(())
        } else {
            anyhow::bail!("variant {idx}: duplicate runtime_flags entry for the same cell triple")
        }
    })
}

/// Reject two `model_flags` entries keyed on the same `(benchmark, model)`
/// cell — the entry's identity is that pair.
fn reject_duplicate_model_cells(idx: usize, entries: &[ModelFlags]) -> anyhow::Result<()> {
    let mut seen = HashSet::new();
    entries.iter().try_for_each(|entry| {
        if seen.insert(entry.axes()) {
            Ok(())
        } else {
            anyhow::bail!("variant {idx}: duplicate model_flags entry for the same cell")
        }
    })
}

/// Reject two `benchmark_flags` entries keyed on the same
/// `(benchmark, runtime, model)` cell — the entry's identity is that triple.
fn reject_duplicate_benchmark_cells(idx: usize, entries: &[BenchmarkFlags]) -> anyhow::Result<()> {
    let mut seen = HashSet::new();
    entries.iter().try_for_each(|entry| {
        if seen.insert(entry.axes()) {
            Ok(())
        } else {
            anyhow::bail!("variant {idx}: duplicate benchmark_flags entry for the same cell")
        }
    })
}

/// Validate each variant's authored `runtime_flags`: no duplicate cells, and
/// every entry matches a real cell in the variant. (Which knobs a cell accepts
/// — including that in-process/mlx/apple runtimes accept none — is enforced
/// structurally at parse, by `RuntimeFlags`' `TryFrom`.)
fn validate_variant_flags(plan: &Plan) -> anyhow::Result<()> {
    plan.matrix
        .variants
        .iter()
        .enumerate()
        .try_for_each(|(idx, variant)| {
            let pairs = variant
                .runnable_pairs()
                .map_err(|e| anyhow::anyhow!("variant {idx}: {e}"))?;
            let benchmark_types: Vec<BenchmarkType> = plan
                .benchmarks_for_variant(idx, variant)?
                .iter()
                .filter_map(benchmark_type_of)
                .collect();
            validate_variant_flag_cells(
                idx,
                &pairs,
                &benchmark_types,
                &variant.runtime_flags,
                &variant.model_flags,
                &variant.benchmark_flags,
            )
        })
}

/// Validate one variant's authored per-cell flags against its runnable cells:
/// reject duplicate cell keys, and require every flag entry to match at least
/// one real `(benchmark, runtime, model)` cell. Shared by both plan formats —
/// the check is over the model/runtime/benchmark matrix, independent of how
/// the variant expresses eligibility. (Which knobs a cell *accepts* is enforced
/// structurally by each flag type's `TryFrom` at parse.)
pub(crate) fn validate_variant_flag_cells(
    idx: usize,
    pairs: &[(&Model, &Runtime)],
    benchmark_types: &[BenchmarkType],
    runtime_flags: &[RuntimeFlags],
    model_flags: &[ModelFlags],
    benchmark_flags: &[BenchmarkFlags],
) -> anyhow::Result<()> {
    reject_duplicate_cells(idx, runtime_flags)?;

    if !runtime_flags.iter().all(|flag| {
        benchmark_types
            .iter()
            .any(|&bt| pairs.iter().any(|(m, r)| flag.matches(bt, r, m)))
    }) {
        anyhow::bail!(
            "variant {idx}: a runtime_flags entry matches no cell in the variant \
             (no benchmark\u{d7}runtime\u{d7}model triple satisfies it)"
        );
    }

    reject_duplicate_model_cells(idx, model_flags)?;

    if !model_flags.iter().all(|flag| {
        benchmark_types
            .iter()
            .any(|&bt| pairs.iter().any(|(m, _)| flag.matches(bt, m)))
    }) {
        anyhow::bail!(
            "variant {idx}: a model_flags entry matches no cell in the variant \
             (no eval benchmark\u{d7}model pair satisfies it)"
        );
    }

    reject_duplicate_benchmark_cells(idx, benchmark_flags)?;

    if !benchmark_flags.iter().all(|flag| {
        benchmark_types
            .iter()
            .any(|&bt| pairs.iter().any(|(m, r)| flag.matches(bt, r, m)))
    }) {
        anyhow::bail!(
            "variant {idx}: a benchmark_flags entry matches no cell in the variant \
             (no benchmark\u{d7}runtime\u{d7}model triple satisfies it)"
        );
    }
    Ok(())
}

/// One executable unit of work: a single benchmark × model × runtime
/// pulled from one variant, eligible to run on any of
/// `allowed_clients`.
#[derive(Debug)]
pub struct TypedCell<'a> {
    pub variant_idx: usize,
    pub benchmark: &'a BenchmarkId,
    pub model: &'a Model,
    pub runtime: &'a Runtime,
    /// Transport names this cell may dispatch to — a borrowed view
    /// into the variant's `clients` array.
    pub allowed_clients: &'a [ClientId],
}

// ---------------------------------------------------------------------------
// TransportConfig (new shape, with `name`)
// ---------------------------------------------------------------------------

/// Per-transport orchestration config. Each transport has a unique
/// `client_id` (ev1_ hash from `auth register`) that doubles as the
/// in-plan routing handle (referenced by `variants.clients`) and the
/// warehouse join key. The id encodes which registered user owns the
/// device server-side, so no separate `owner` field is needed.
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum TransportConfig {
    #[serde(rename = "adb")]
    Adb {
        client_id: String,
        serial: String,
        binary_path: String,
        work_dir: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        port: Option<u16>,
        #[serde(default = "default_shell")]
        shell: ShellType,
        #[serde(
            default = "default_parallelism",
            skip_serializing_if = "is_default_parallelism"
        )]
        parallelism: NonZeroUsize,
    },
    /// A device attached to *another* box's adb server: the `adb` command
    /// runs on `host` over ssh, so the driver needs neither adb nor a
    /// tunnel — only ssh to the box holding the pairing keys. `Adb`'s
    /// counterpart in the same way `SlurmOverSsh` is `SlurmLocal`'s.
    ///
    /// `port` is the **ssh** port (as on every ssh-reached transport);
    /// `adb_port` is the adb server port on `host`. The `--adb-port`
    /// override deliberately does not apply here: it exists to retarget a
    /// tunnel from the driver, which this variant removes the need for.
    ///
    /// `shell` is still the *device* shell, and must be posix — a
    /// powershell value is rejected at load.
    ///
    /// Limitation: the *intermediate host* is also assumed posix, because the
    /// device command is single-quoted for it. Pointing this at a Windows box
    /// running adb would pass the quotes through literally and garble the
    /// command; expressing that needs a separate `host_shell` field.
    #[serde(rename = "adb_over_ssh")]
    AdbOverSsh {
        client_id: String,
        host: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        port: Option<u16>,
        serial: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        adb_port: Option<u16>,
        /// Shell command run before `adb` on the intermediate host, e.g.
        /// `"export PATH=$PATH:$HOME/Android/Sdk/platform-tools"`. Required
        /// when only the login profile — which non-interactive ssh skips —
        /// puts `adb` on PATH. Same role as slurm's `pre_exec`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pre_exec: Option<String>,
        binary_path: String,
        work_dir: String,
        #[serde(default = "default_shell")]
        shell: ShellType,
        #[serde(
            default = "default_parallelism",
            skip_serializing_if = "is_default_parallelism"
        )]
        parallelism: NonZeroUsize,
    },
    #[serde(rename = "ssh")]
    Ssh {
        client_id: String,
        host: String,
        binary_path: String,
        work_dir: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        port: Option<u16>,
        #[serde(default = "default_shell")]
        shell: ShellType,
        #[serde(
            default = "default_parallelism",
            skip_serializing_if = "is_default_parallelism"
        )]
        parallelism: NonZeroUsize,
    },
    #[serde(rename = "local")]
    Local {
        client_id: String,
        binary_path: String,
        work_dir: String,
        #[serde(default = "default_shell")]
        shell: ShellType,
        #[serde(
            default = "default_parallelism",
            skip_serializing_if = "is_default_parallelism"
        )]
        parallelism: NonZeroUsize,
    },
    /// Dispatch each cell as its own SLURM job via `srun`, running the
    /// `srun` command on the local machine — pipette-plan itself runs on
    /// the cluster login node. `parallelism` caps how many `srun`
    /// allocations run concurrently.
    #[serde(rename = "slurm_local")]
    SlurmLocal {
        client_id: String,
        binary_path: String,
        work_dir: String,
        /// Shell command run before `srun`, e.g.
        /// `". /etc/profile.d/modules.sh && module load slurm"`, to put
        /// `srun` on PATH in a non-login shell. Omit if the launching
        /// environment already has slurm set up.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pre_exec: Option<String>,
        /// `--partition`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        partition: Option<String>,
        /// `--account`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        account: Option<String>,
        /// `--gres=gpu:N`. Omitted when absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gpus: Option<u32>,
        /// `--cpus-per-task`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cpus: Option<u32>,
        /// `--time`, e.g. "02:00:00".
        #[serde(default, skip_serializing_if = "Option::is_none")]
        time_limit: Option<String>,
        /// `--mem`, e.g. "32G".
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mem: Option<String>,
        /// Directory for per-job `--output`/`--error` files (slurm
        /// `%x-%j`). Omit to stream task output to the driver instead.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        log_dir: Option<String>,
        /// Extra `srun` flags appended verbatim.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        extra_srun_args: Vec<String>,
        #[serde(default = "default_shell")]
        shell: ShellType,
        #[serde(
            default = "default_parallelism",
            skip_serializing_if = "is_default_parallelism"
        )]
        parallelism: NonZeroUsize,
    },
    /// Same as [`Self::SlurmLocal`], but reaches the cluster over SSH:
    /// runs `ssh [user@]host srun …` so pipette-plan can drive a remote
    /// cluster from a laptop or CI box that is not the login node.
    #[serde(rename = "slurm_over_ssh")]
    SlurmOverSsh {
        client_id: String,
        host: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        port: Option<u16>,
        binary_path: String,
        work_dir: String,
        /// Shell command run before `srun` on the remote side, e.g.
        /// `". /etc/profile.d/modules.sh && module load slurm"`. Required
        /// when only the login profile (skipped by non-interactive ssh)
        /// puts `srun` on PATH.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pre_exec: Option<String>,
        /// `--partition`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        partition: Option<String>,
        /// `--account`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        account: Option<String>,
        /// `--gres=gpu:N`. Omitted when absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gpus: Option<u32>,
        /// `--cpus-per-task`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cpus: Option<u32>,
        /// `--time`, e.g. "02:00:00".
        #[serde(default, skip_serializing_if = "Option::is_none")]
        time_limit: Option<String>,
        /// `--mem`, e.g. "32G".
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mem: Option<String>,
        /// Directory for per-job `--output`/`--error` files (slurm
        /// `%x-%j`). Omit to stream task output to the driver instead.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        log_dir: Option<String>,
        /// Extra `srun` flags appended verbatim.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        extra_srun_args: Vec<String>,
        #[serde(default = "default_shell")]
        shell: ShellType,
        #[serde(
            default = "default_parallelism",
            skip_serializing_if = "is_default_parallelism"
        )]
        parallelism: NonZeroUsize,
    },
    /// Drive an iOS device from the host Mac via `xcrun devicectl`,
    /// launching the Pipette app in its `headlessrun` mode. Unlike the
    /// other transports there is no pre-provisioned remote binary or
    /// work dir: the app ships as the device build, runs one benchmark
    /// per process launch, and uploads its own results (the synthesized
    /// command carries `submit=1`, so the runner does not append
    /// `--sync`). Intended for on-device runtimes — chiefly Apple
    /// Foundation Models (`Runtime::AppleFoundation(_)`).
    ///
    /// Prerequisites the runner cannot bootstrap: the device build must
    /// be installed and code-signed (once, via Xcode), and the device
    /// must have been registered on-device (`headlessrun register …`)
    /// so `submit=1` uploads are accepted.
    #[serde(rename = "ios")]
    Ios {
        client_id: String,
        /// Device UDID passed to `xcrun devicectl … --device <udid>`.
        device_udid: String,
        /// App bundle id to launch. Defaults to the Pipette app.
        #[serde(default = "default_ios_bundle_id")]
        bundle_id: String,
        #[serde(
            default = "default_parallelism",
            skip_serializing_if = "is_default_parallelism"
        )]
        parallelism: NonZeroUsize,
    },

    /// [`TransportConfig::Ios`] reached through an intermediate Mac, for a
    /// driver that is not the machine the iPhones are paired to.
    ///
    /// `xcrun devicectl` only talks to devices paired with the machine it runs
    /// on, which otherwise forces the whole plan onto that Mac. This renders
    /// the same `devicectl` argv as a shell command and runs it there over ssh,
    /// so any driver can reach the phones.
    ///
    /// Limitation: the intermediate host is assumed posix, because the argv is
    /// quoted for its shell — the same constraint `adb_over_ssh` carries.
    #[serde(rename = "ios_over_ssh")]
    IosOverSsh {
        client_id: String,
        host: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        user: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        port: Option<u16>,
        /// Device UDID passed to `xcrun devicectl … --device <udid>`.
        device_udid: String,
        /// App bundle id to launch. Defaults to the Pipette app.
        #[serde(default = "default_ios_bundle_id")]
        bundle_id: String,
        #[serde(
            default = "default_parallelism",
            skip_serializing_if = "is_default_parallelism"
        )]
        parallelism: NonZeroUsize,
    },
}

impl TransportConfig {
    /// Unique routing handle for this transport (also the warehouse
    /// join key). Variants reference it via `variants.clients`.
    pub fn client_id(&self) -> &str {
        match self {
            TransportConfig::Adb { client_id, .. }
            | TransportConfig::AdbOverSsh { client_id, .. }
            | TransportConfig::Ssh { client_id, .. }
            | TransportConfig::Local { client_id, .. }
            | TransportConfig::SlurmLocal { client_id, .. }
            | TransportConfig::SlurmOverSsh { client_id, .. }
            | TransportConfig::Ios { client_id, .. }
            | TransportConfig::IosOverSsh { client_id, .. } => client_id,
        }
    }

    /// The remote program. For device transports this is the
    /// pre-provisioned pipette binary; iOS has none (the app is
    /// launched via `xcrun`), so its `build_argv` branch never reads
    /// this — `"xcrun"` is returned only so the preview/cwd derivation
    /// render sanely.
    pub fn binary_path(&self) -> &str {
        match self {
            TransportConfig::Adb { binary_path, .. }
            | TransportConfig::AdbOverSsh { binary_path, .. }
            | TransportConfig::Ssh { binary_path, .. }
            | TransportConfig::Local { binary_path, .. }
            | TransportConfig::SlurmLocal { binary_path, .. }
            | TransportConfig::SlurmOverSsh { binary_path, .. } => binary_path,
            TransportConfig::Ios { .. } | TransportConfig::IosOverSsh { .. } => "xcrun",
        }
    }

    pub fn work_dir(&self) -> &str {
        match self {
            TransportConfig::Adb { work_dir, .. }
            | TransportConfig::AdbOverSsh { work_dir, .. }
            | TransportConfig::Ssh { work_dir, .. }
            | TransportConfig::Local { work_dir, .. }
            | TransportConfig::SlurmLocal { work_dir, .. }
            | TransportConfig::SlurmOverSsh { work_dir, .. } => work_dir,
            // iOS has no remote work dir; the app manages its own state.
            TransportConfig::Ios { .. } | TransportConfig::IosOverSsh { .. } => "",
        }
    }

    pub fn shell(&self) -> ShellType {
        match self {
            TransportConfig::Adb { shell, .. }
            | TransportConfig::AdbOverSsh { shell, .. }
            | TransportConfig::Ssh { shell, .. }
            | TransportConfig::Local { shell, .. }
            | TransportConfig::SlurmLocal { shell, .. }
            | TransportConfig::SlurmOverSsh { shell, .. } => *shell,
            // `xcrun` runs on the host Mac — its own, or the intermediate one.
            TransportConfig::Ios { .. } | TransportConfig::IosOverSsh { .. } => ShellType::Posix,
        }
    }

    /// How many concurrent `transport.exec` calls this transport
    /// contributes to its host's budget — and, equivalently, how
    /// many worker threads the runner spawns for this transport's
    /// queue. Defaults to 1 (serial per transport, per box). Raise
    /// to opt this transport into running multiple benchmarks
    /// against the same physical box in parallel; the runner takes
    /// `max(parallelism)` across transports sharing a [`Self::physical_id`]
    /// as the box's effective slot capacity.
    pub fn parallelism(&self) -> NonZeroUsize {
        match self {
            TransportConfig::Adb { parallelism, .. }
            | TransportConfig::AdbOverSsh { parallelism, .. }
            | TransportConfig::Ssh { parallelism, .. }
            | TransportConfig::Local { parallelism, .. }
            | TransportConfig::SlurmLocal { parallelism, .. }
            | TransportConfig::SlurmOverSsh { parallelism, .. }
            | TransportConfig::Ios { parallelism, .. }
            | TransportConfig::IosOverSsh { parallelism, .. } => *parallelism,
        }
    }

    /// Stable identifier for the physical machine this transport reaches.
    /// Used to serialize execution across transports that share a box —
    /// e.g. one host running both `pipette-llamacpp` and `pipette-mlx`
    /// under two separate transport entries.
    ///
    /// SSH → `host`. ADB → `serial`. Local → fixed `"local"`. Bare host
    /// (not `user@host:port`) so SSH transports with different `user`/`port`
    /// to the same machine still serialize together — the contention is
    /// the box's CPU/GPU, not the connection identity.
    ///
    /// Slurm is the deliberate exception: it returns a per-transport id
    /// (`slurm:<client_id>`) rather than a shared box. Each `srun` is
    /// scheduled onto a different compute node, so there is no single
    /// physical box to serialize against — the per-transport id makes
    /// the host budget equal this transport's own `parallelism`, which
    /// is exactly the cap on concurrent `srun` allocations we want.
    pub fn physical_id(&self) -> String {
        match self {
            TransportConfig::Adb { serial, .. } | TransportConfig::AdbOverSsh { serial, .. } => {
                serial.clone()
            }
            TransportConfig::Ssh { host, .. } => host.clone(),
            TransportConfig::Local { .. } => "local".to_string(),
            TransportConfig::SlurmLocal { client_id, .. }
            | TransportConfig::SlurmOverSsh { client_id, .. } => format!("slurm:{client_id}"),
            // One device per UDID; serialize launches against it. The
            // intermediate host is not the contended resource — the phone is —
            // so both variants key on the UDID alone.
            TransportConfig::Ios { device_udid, .. }
            | TransportConfig::IosOverSsh { device_udid, .. } => format!("ios:{device_udid}"),
        }
    }

    /// Whether the runner appends `--sync` to each cell command so the
    /// remote binary uploads results after the run. False for both iOS
    /// variants: the app's `headlessrun` command carries `submit=1` instead,
    /// and would reject the unknown bare `--sync` token. Routing the launch
    /// through ssh does not change the app's grammar.
    pub fn appends_sync_flag(&self) -> bool {
        !matches!(
            self,
            TransportConfig::Ios { .. } | TransportConfig::IosOverSsh { .. }
        )
    }
}

fn default_parallelism() -> NonZeroUsize {
    NonZeroUsize::MIN
}

fn default_ios_bundle_id() -> String {
    "ai.liquid.liquid-pipette".to_string()
}

/// Deserialize a `benchmarks` list, reporting a rejected id as the thing to do
/// about it.
///
/// [`BenchmarkId`] refuses a catalog side, which is what plans used to carry
/// (`"remote/eval_smoke"`). Left to the derive, that surfaces as "BenchmarkId
/// failed the predicate test" against the file's first line — so the message is
/// written here, where the field is known.
fn deserialize_benchmark_ids<'de, D>(
    deserializer: D,
) -> Result<Option<NonEmptyVec<BenchmarkId>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let Some(raw) = Option::<Vec<String>>::deserialize(deserializer)? else {
        return Ok(None);
    };
    let ids = raw
        .into_iter()
        .map(|entry| {
            BenchmarkId::try_new(entry.clone()).map_err(|_| {
                serde::de::Error::custom(format!(
                    "benchmark `{entry}` is not a bare id; a plan distributes ids and \
                     the client resolves them, so drop any `local/` or `remote/` prefix"
                ))
            })
        })
        .collect::<Result<Vec<_>, D::Error>>()?;
    NonEmptyVec::try_new(ids)
        .map(Some)
        .map_err(serde::de::Error::custom)
}

fn is_default_parallelism(n: &NonZeroUsize) -> bool {
    n.get() == 1
}

fn default_shell() -> ShellType {
    ShellType::Posix
}

/// Work payload served to a pipette client: benchmark × model × runtime + flags.
///
/// Lives in plan-types so the plan runner, mgmt claim path, and desktop CLI
/// share one contract. No plan routing ([`RunnableCell::allowed_clients`]), no
/// store-resolved benchmark body, no host-absolute ensured paths — those attach
/// later as ops `RunRequest` after ensure/bind. Plan expansion keeps routing on
/// [`RunnableCell`] (`ClientRunSpec::from`); CLI/`benchmarks run` and the worker
/// build a `ClientRunSpec` directly.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClientRunSpec {
    pub benchmark: BenchmarkId,
    pub model: Model,
    pub runtime: Runtime,
    /// Typed runtime flags for this cell's `(benchmark, runtime, model)`.
    /// At most one entry matches per variant.
    ///
    /// `default` so a job body may omit an unset flag group rather than spell
    /// it `null`; serialization still emits the key, leaving existing state
    /// files byte-identical.
    #[serde(default)]
    pub runtime_flags: Option<RuntimeFlags>,
    /// Model-generation flags for `(benchmark, model)`. Eval-only.
    #[serde(default)]
    pub model_flags: Option<ModelFlags>,
    /// Eval/vl-run flags (HTTP timeout, doom-loop, readiness). `None` when unset.
    #[serde(default)]
    pub benchmark_flags: Option<BenchmarkFlags>,
}

/// Plan-internal executable unit: a [`ClientRunSpec`] plus which clients may run it.
///
/// Cell identity for routing is `(benchmark, model, runtime, allowed_clients)`.
/// Reordering `[[variants]]` in the TOML doesn't invalidate the state file, and
/// two variants producing the same tuple deduplicate to a single unit of work.
/// The work payload alone is [`ClientRunSpec`] via `From` (cell keys hash the
/// work axes, not `allowed_clients`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RunnableCell {
    pub benchmark: BenchmarkId,
    pub model: Model,
    pub runtime: Runtime,
    pub allowed_clients: Vec<ClientId>,
    /// Typed runtime flags resolved from the variant's `runtime_flags` for this
    /// cell's `(benchmark, runtime, model)`. At most one entry matches (the key
    /// is unique per variant, and duplicates are rejected), so this is `Option`,
    /// mirroring `model_flags`/`benchmark_flags`. Shipped to the client as the
    /// `--runtime-flags` JSON (see [`Self::runtime_flags_json`]); the client
    /// renders it to its tool's argv.
    pub runtime_flags: Option<RuntimeFlags>,
    /// Model-generation flags resolved from the variant's `model_flags` for
    /// this cell's `(benchmark, model)`. `None` on every non-eval cell and on
    /// any eval cell the variant didn't author flags for; the `(model_type,
    /// benchmark)` key is unique per variant, so a cell resolves to at most one.
    pub model_flags: Option<ModelFlags>,
    /// Eval-run flags resolved from the variant's `benchmark_flags` for this
    /// cell's `(benchmark, model)` — the HTTP-client timeout and doom-loop
    /// monitor. `None` off eval or when the variant authored none.
    pub benchmark_flags: Option<BenchmarkFlags>,
}

impl From<&RunnableCell> for ClientRunSpec {
    /// Work payload served to a client (drops plan-only `allowed_clients`).
    fn from(cell: &RunnableCell) -> Self {
        Self {
            benchmark: cell.benchmark.clone(),
            model: cell.model.clone(),
            runtime: cell.runtime.clone(),
            runtime_flags: cell.runtime_flags.clone(),
            model_flags: cell.model_flags.clone(),
            benchmark_flags: cell.benchmark_flags.clone(),
        }
    }
}

impl From<RunnableCell> for ClientRunSpec {
    fn from(cell: RunnableCell) -> Self {
        ClientRunSpec::from(&cell)
    }
}

impl RunnableCell {
    /// Plan→client wire encoding for `--runtime-flags` (and cell-key hash).
    /// One-element JSON array, or `None` when unset. Not used on the client
    /// after parse — the client holds structured [`RuntimeFlags`].
    /// Identity, not wire: `pipette-plan` hashes this into the cell key that resume and
    /// rerun match on, so it keeps the axes [`Self::runtime_flags_json`] drops — a cell is
    /// identified by which cell it is, not only by what it loads with. Mirrors
    /// [`ModelFlags::canonical_string`], load-bearing for the same reason.
    ///
    /// The one-element array this used to be is gone: nothing selects among entries here,
    /// and keeping the wrapper meant a format we had deleted survived inside the hash.
    /// Dropping it re-keys flagged cells once — see the pinned digest in `pipette-plan`.
    pub fn runtime_flags_canonical_string(&self) -> Option<String> {
        let flags = self.runtime_flags.as_ref()?;
        // `RuntimeFlags` serializes through `RuntimeFlagRef`, so this is the flat entry
        // with its axes — the same fields the plan authored.
        Some(serde_json::to_string(flags).unwrap_or_default())
    }

    /// The identity string as it was spelled before the array wrapper was dropped.
    ///
    /// Migration only: `pipette-plan` derives the old cell key from it so a plan recorded
    /// by an earlier build still matches and its finished cells are not re-run. Nothing
    /// emits this on any wire. Delete once no state predating the change is in play.
    pub fn runtime_flags_legacy_string(&self) -> Option<String> {
        let flags = self.runtime_flags.as_ref()?;
        Some(serde_json::to_string(std::slice::from_ref(flags)).unwrap_or_default())
    }

    pub fn runtime_flags_json(&self) -> Option<String> {
        let flags = self.runtime_flags.as_ref()?;
        // The knobs alone. A client resolves its cell from `--benchmark`, `--runtime` and
        // `--model` before it reads any flags, so the axes this entry was selected by are
        // its own bookkeeping, not something to restate on the wire — where they could
        // only ever agree with the cell or contradict it.
        //
        // Fall closed on the impossible serialization error: an empty payload makes the
        // client reject `--runtime-flags` loudly rather than silently dropping the flags a
        // bare `.ok()` → `None` would.
        Some(serde_json::to_string(&flags.knobs_json()).unwrap_or_default())
    }

    /// `--sync` is added by the runner — not included here.
    pub fn build_argv(&self, transport: &TransportConfig) -> anyhow::Result<Vec<String>> {
        // iOS speaks a different grammar: the app's `headlessrun` args,
        // not the desktop `benchmarks run …` command. Results upload
        // in-process via `submit=1`, so the runner must not append
        // `--sync` for this transport.
        if matches!(
            transport,
            TransportConfig::Ios { .. } | TransportConfig::IosOverSsh { .. }
        ) {
            return self.ios_headless_args();
        }
        // Canonical `Model` JSON, parsed back by the client's `parse_model_arg`.
        // A `GgufVision` carries its projector inline, so there is no separate
        // `--mmproj`. A serialization failure is surfaced, not defaulted to an
        // empty arg that would produce a silently-broken command line.
        let model = serde_json::to_string(&self.model)?;
        // Canonical `Runtime` JSON, parsed back by the client's `parse_runtime_arg`
        // (the unified CLI) or its JSON-or-legacy `--runtime` path (the standalone
        // binaries). Same fall-closed rationale as `--model` above.
        let runtime = serde_json::to_string(&self.runtime)?;
        let mut argv = vec![
            transport.binary_path().into(),
            "--work-dir".into(),
            transport.work_dir().into(),
            "benchmarks".into(),
            "run".into(),
            "--benchmark".into(),
            self.benchmark.as_ref().into(),
            "--model".into(),
            model,
            "--runtime".into(),
            runtime,
        ];
        // The entry itself, same shape as `--runtime-flags`. Not a one-element array:
        // nothing selects among entries once a cell is resolved, and every client already
        // accepted the bare object.
        if let Some(flags) = &self.model_flags {
            let knobs = serde_json::to_string(flags).unwrap_or_default();
            argv.push("--model-flags".into());
            argv.push(knobs);
        }
        // Run-driving knobs (HTTP-client timeout; doom-loop monitor on eval)
        // resolved onto this cell as `BenchmarkFlags` — carried by any
        // server-driven cell (eval and vl-throughput). `extend_with_eval_knobs`
        // emits nothing for unset knobs.
        if let Some(bf) = &self.benchmark_flags {
            extend_with_eval_knobs(&mut argv, bf.http_timeout(), bf.doomloop());
            // Readiness (timing cells): forward the per-cell deadline as the
            // runner's `benchmarks run --readiness-max-wait-secs` flag, which
            // its run() passes to wait_until_ready as an argument. Note this is
            // the per-cell channel, distinct from PIPETTE_READINESS_MAX_WAIT_SECS
            // — that one is how the plan driver waives fleet-wide.
            if let Some(secs) = bf.readiness().and_then(|r| r.max_wait_secs) {
                argv.push("--readiness-max-wait-secs".into());
                argv.push(secs.to_string());
            }
            // Waiving the thermal criterion is a per-cell decision, forwarded
            // the same way. Only emitted when true, so an unset knob leaves the
            // runner's argv unchanged.
            if bf.readiness().and_then(|r| r.skip_thermal) == Some(true) {
                argv.push("--readiness-skip-thermal".into());
            }
        }
        if let Some(knobs) = self.runtime_flags_json() {
            argv.push("--runtime-flags".into());
            argv.push(knobs);
        }
        Ok(argv)
    }

    /// The `headlessrun` argument vector for an iOS launch — the tokens
    /// after the app bundle in `xcrun devicectl … <bundle> <args>`. The
    /// iOS transport prepends the `devicectl` wrapping; this is the app
    /// command it runs. `submit=1` makes the device upload its own
    /// results (the on-device analogue of the desktop `--sync`).
    ///
    /// `benchmarks=<id>` selects the benchmark by its catalog id (the
    /// same id the plan authored). Apple Foundation Models needs no
    /// `model=` — the OS supplies the model; other on-device runtimes
    /// pass the model ref through.
    fn ios_headless_args(&self) -> anyhow::Result<Vec<String>> {
        let mut args = vec![
            "headlessrun".to_string(),
            // Canonical `Runtime` JSON, for the same reason `model=` below carries the
            // canonical `Model`: a type tag names which runtime but not which build, so
            // the device had nothing to check its own identity against and recorded
            // whatever ran.
            format!("runtime={}", serde_json::to_string(&self.runtime)?),
        ];
        if !matches!(self.runtime, Runtime::AppleFoundation(_)) {
            // Canonical `Model` JSON, the same spelling the desktop `--model` carries and
            // one the client's model parser accepts. `Display` was emitted here before,
            // which no client parses: it is the log/warehouse identifier, and a
            // `GgufText`'s `{repo}:{path}` matched nothing on the device, so a
            // plan-dispatched GGUF cell could not resolve its model at all.
            args.push(format!("model={}", serde_json::to_string(&self.model)?));
        }
        args.push(format!("benchmarks={}", self.benchmark.as_ref()));
        // The knobs, in the app's `key=value` grammar. Omitted entirely before this, so a
        // plan could author `threads` for an iOS cell and the phone would run on its
        // derived P-core count while reporting flags it had never been given.
        if let Some(knobs) = self.runtime_flags_json() {
            args.push(format!("runtime-flags={knobs}"));
        }
        args.push("submit=1".to_string());
        Ok(args)
    }
}

fn extend_with_eval_knobs(
    argv: &mut Vec<String>,
    http_timeout_seconds: Option<u64>,
    doomloop: Option<&DoomloopOverrides>,
) {
    if let Some(t) = http_timeout_seconds {
        argv.push("--http-timeout-seconds".into());
        argv.push(t.to_string());
    }
    if let Some(doomloop) = doomloop {
        doomloop.append_argv(argv);
    }
}

#[cfg(test)]
mod ios_transport_tests {
    use super::*;
    use crate::{
        default_repository_url, HfRepo, LlamaCppFlavor, LlamacppCliStockTools,
        LlamacppCliStockToolsSource, ModelSource, NonEmptyString, SourceRepository, Torch,
    };

    fn ios_cfg() -> anyhow::Result<TransportConfig> {
        Ok(toml::from_str(
            "type = \"ios\"\nclient_id = \"iphone15\"\ndevice_udid = \"UDID-123\"\n",
        )?)
    }

    fn cell(benchmark: &str, model: Model, runtime: Runtime) -> anyhow::Result<RunnableCell> {
        Ok(RunnableCell {
            benchmark: BenchmarkId::try_new(benchmark.to_string())?,
            model,
            runtime,
            allowed_clients: Vec::new(),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
        })
    }

    #[test]
    fn adb_over_ssh_config_defaults_and_accessors() -> anyhow::Result<()> {
        let cfg: TransportConfig = toml::from_str(
            "type = \"adb_over_ssh\"\n\
             client_id = \"phone-1\"\n\
             host = \"controller\"\n\
             user = \"liquid\"\n\
             serial = \"R3GL30CRBGM\"\n\
             binary_path = \"/data/local/tmp/pipette-evals/pipette\"\n\
             work_dir = \"/data/local/tmp/pipette-evals\"\n",
        )?;
        assert_eq!(cfg.client_id(), "phone-1");
        // The contended box is the handset, not the box hosting adb — so the
        // host budget keys on the serial, exactly as for a local `adb`.
        assert_eq!(cfg.physical_id(), "R3GL30CRBGM");
        assert_eq!(cfg.binary_path(), "/data/local/tmp/pipette-evals/pipette");
        assert_eq!(cfg.shell(), ShellType::Posix);
        assert!(cfg.appends_sync_flag());

        let reparsed: TransportConfig = toml::from_str(&toml::to_string(&cfg)?)?;
        assert_eq!(cfg, reparsed);
        Ok(())
    }

    /// `port` is the ssh port and `adb_port` the adb server port; a plan that
    /// mixed them up would silently ssh to the adb port.
    #[test]
    fn adb_over_ssh_separates_ssh_and_adb_ports() -> anyhow::Result<()> {
        let cfg: TransportConfig = toml::from_str(
            "type = \"adb_over_ssh\"\n\
             client_id = \"phone-1\"\n\
             host = \"controller\"\n\
             port = 2222\n\
             serial = \"S1\"\n\
             adb_port = 5038\n\
             binary_path = \"/tmp/pipette\"\n\
             work_dir = \"/tmp\"\n",
        )?;
        assert!(matches!(
            &cfg,
            TransportConfig::AdbOverSsh {
                port: Some(2222),
                adb_port: Some(5038),
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn ios_config_defaults_and_accessors() -> anyhow::Result<()> {
        let cfg = ios_cfg()?;
        assert_eq!(cfg.client_id(), "iphone15");
        assert_eq!(cfg.physical_id(), "ios:UDID-123");
        // Device uploads inline via `submit=1`; no `--sync`.
        assert!(!cfg.appends_sync_flag());
        assert!(matches!(
            &cfg,
            TransportConfig::Ios { bundle_id, .. } if bundle_id == "ai.liquid.liquid-pipette"
        ));

        // Round-trips through TOML unchanged.
        let reparsed: TransportConfig = toml::from_str(&toml::to_string(&cfg)?)?;
        assert_eq!(cfg, reparsed);
        Ok(())
    }

    #[test]
    fn ios_build_argv_afm_omits_model() -> anyhow::Result<()> {
        let cell = cell(
            "decode_throughput_256",
            Model::AppleFoundationText,
            Runtime::AppleFoundation(Default::default()),
        )?;
        // AFM: no `model=` (the OS supplies it), `submit=1`, no `--sync`.
        assert_eq!(
            cell.build_argv(&ios_cfg()?)?,
            vec![
                "headlessrun",
                r#"runtime={"type":"apple_foundation"}"#,
                "benchmarks=decode_throughput_256",
                "submit=1",
            ]
        );
        Ok(())
    }

    /// The spelling that actually crosses the wire to a phone: no other test covers an
    /// on-device runtime, so a field renamed on either side would break every
    /// plan-dispatched iOS cell with both suites green. The device compares this against
    /// its own `Runtime.thisBuild` and refuses a mismatch, which only works byte for byte.
    #[test]
    fn ios_build_argv_carries_the_on_device_runtime_verbatim() -> anyhow::Result<()> {
        let runtime: Runtime = serde_json::from_str(
            r#"{"type":"llamacpp_ios_pipette","repository_version":"b10216","flavor":"ios-arm64"}"#,
        )?;
        // Parsed rather than constructed, so the fixture is the plan spelling itself.
        let model: Model = serde_json::from_str(
            r#"{"type":"gguf_text","source":"huggingface","org":"LiquidAI","repo_name":"LFM2.5-350M-GGUF","path":"LFM2.5-350M-Q4_0.gguf"}"#,
        )?;
        let argv = cell("prefill_throughput_256", model, runtime)?.build_argv(&ios_cfg()?)?;

        assert_eq!(
            argv[1],
            r#"runtime={"type":"llamacpp_ios_pipette","repository_url":"github.com/ggml-org/llama.cpp","repository_version":"b10216","flavor":"ios-arm64"}"#
        );
        Ok(())
    }

    #[test]
    fn ios_build_argv_non_afm_includes_model() -> anyhow::Result<()> {
        let runtime = Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: LlamacppCliStockToolsSource::GithubRelease(SourceRepository {
                repository_url: default_repository_url(),
                repository_version: NonEmptyString::try_new("b5000")?,
            }),
            flavor: LlamaCppFlavor::MacosArm64,
        });
        let model = Model::Torch(Torch {
            source: ModelSource::HuggingFace {
                repo: HfRepo::parse_org_repo("org/repo")?,
                prefix: None,
            },
        });
        assert_eq!(
            cell("eval_gpqa", model, runtime)?.build_argv(&ios_cfg()?)?,
            vec![
                "headlessrun",
                r#"runtime={"type":"llamacpp_cli_stock_tools","source":"github_release","repository_url":"github.com/ggml-org/llama.cpp","repository_version":"b5000","flavor":"macos-arm64"}"#,
                // Canonical JSON, not `Display`: the client parses this spelling, and
                // `Display` (`{repo}[:{path}]`) is the log/warehouse identifier.
                r#"model={"type":"torch","source":"huggingface","org":"org","repo_name":"repo"}"#,
                "benchmarks=eval_gpqa",
                "submit=1",
            ]
        );
        Ok(())
    }
}

/// A baseline valid plan TOML with one llamacpp variant. Tests substitute
/// a single field to exercise specific invariants without rewriting the
/// whole document. Shared with `model.rs`'s parse tests.
#[cfg(test)]
pub(crate) fn plan_toml(variant_block: &str) -> String {
    format!(
        r#"benchmarks = ["prefill_throughput_256"]

[[variants]]
{variant_block}
"#
    )
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use anyhow::Context;
    use rstest::rstest;

    use super::*;
    use crate::runtime::app_source;
    use crate::*;

    /// The CLI variant's source (`LlamacppCliStockToolsSource`): a `GitHubRelease` at the
    /// default repo and the given ref.
    fn cli_source(version: &str) -> anyhow::Result<LlamacppCliStockToolsSource> {
        Ok(LlamacppCliStockToolsSource::GithubRelease(app_source(
            version,
        )?))
    }

    const MINIMAL_VARIANT: &str = r#"models = [{ type = "torch", source = "huggingface", org = "x", repo_name = "y" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b5000", flavor = "windows-x64-vulkan" }]
clients = ["ev1_c"]"#;

    fn minimal_plan_toml() -> String {
        plan_toml(MINIMAL_VARIANT)
    }

    // ---- Model / Runtime builders ----------------------------------------

    fn gguf_text(repo: &str) -> anyhow::Result<Model> {
        Ok(Model::GgufText(GgufText {
            source: GgufTextSource::HuggingFace {
                repo: HfRepo::parse_org_repo(repo)?,
                path: RepoSubpath::try_new("a.gguf".to_owned())?,
                sha256: None,
            },
        }))
    }
    fn gguf_vision(repo: &str) -> anyhow::Result<Model> {
        Ok(Model::GgufVision(GgufVision {
            source: GgufVisionSource::HuggingFace {
                repo: HfRepo::parse_org_repo(repo)?,
                model: RepoSubpath::try_new("a.gguf".to_owned())?,
                model_sha256: None,
                mmproj: RepoSubpath::try_new("mm.gguf".to_owned())?,
                mmproj_sha256: None,
            },
        }))
    }
    fn mlx_model(repo: &str) -> anyhow::Result<Model> {
        Ok(Model::Mlx(Mlx {
            source: ModelSource::HuggingFace {
                repo: HfRepo::parse_org_repo(repo)?,
                prefix: None,
            },
        }))
    }
    fn torch_model(repo: &str) -> anyhow::Result<Model> {
        Ok(Model::Torch(Torch {
            source: ModelSource::HuggingFace {
                repo: HfRepo::parse_org_repo(repo)?,
                prefix: None,
            },
        }))
    }
    fn llamacpp_rt() -> anyhow::Result<Runtime> {
        Ok(Runtime::LlamacppCliStockTools(LlamacppCliStockTools {
            source: cli_source("b9050")?,
            flavor: LlamaCppFlavor::MacosArm64,
        }))
    }
    fn mlx_rt() -> anyhow::Result<Runtime> {
        Ok(Runtime::MlxMacosPipette(MlxMacosPipette {
            version: NonEmptyString::try_new("0.31".to_owned())?,
            flavor: MlxMacosPipetteFlavor::MacosArm64,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("mlx-lm==0.31\n".to_owned())?,
                install_flags: None,
            },
        }))
    }
    fn uv_vllm_rt() -> anyhow::Result<Runtime> {
        Ok(Runtime::UvVllm(UvVllm {
            server_version: UvServerVersion::try_new("0.10.0".to_owned())?,
            build: UvBuild::try_new("cu121".to_owned())?,
            python_version: UvPythonVersion::try_new("3.12".to_owned())?,
            source: UvRuntimeSource::PipRequirementsText {
                contents: NonEmptyString::try_new("vllm==0.10.0\n".to_owned())?,
                install_flags: None,
            },
        }))
    }

    fn make_variant(models: Vec<Model>, runtimes: Vec<Runtime>) -> anyhow::Result<Variant> {
        Ok(Variant {
            models: NonEmptyVec::try_new(models)?,
            runtimes: NonEmptyVec::try_new(runtimes)?,
            clients: NonEmptyVec::try_new(vec![ClientId::try_new("ev1".to_owned())?])?,
            benchmarks: None,
            runtime_flags: vec![],
            model_flags: vec![],
            benchmark_flags: vec![],
        })
    }

    // ---- Outer Plan / RunnableCell builders ------------------------------

    fn client_refs(cs: &[ClientId]) -> Vec<&str> {
        cs.iter().map(|c| c.as_ref()).collect()
    }

    fn single(cells: &std::collections::HashSet<RunnableCell>) -> anyhow::Result<&RunnableCell> {
        assert_eq!(cells.len(), 1, "expected exactly one cell");
        cells.iter().next().context("non-empty after len-check")
    }

    fn minimal_outer_plan() -> &'static str {
        r#"
plan_id          = "test"
benchmarks       = ["prefill_throughput_512"]

[[transports]]
client_id   = "mac-1"
type        = "ssh"
host        = "h"
user        = "u"
binary_path = "/bin/pipette"
work_dir    = "/tmp"
shell       = "posix"

[[variants]]
clients  = ["mac-1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "org", repo_name = "repo", path = "Q4_K_M.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
"#
    }

    fn local_transport() -> anyhow::Result<TransportConfig> {
        Ok(TransportConfig::Local {
            client_id: "t1".into(),
            binary_path: "/bin/pipette".into(),
            work_dir: "/tmp".into(),
            shell: ShellType::Posix,
            parallelism: NonZeroUsize::new(1).ok_or_else(|| anyhow::anyhow!("1 is non-zero"))?,
        })
    }

    // ---- NonEmptyVec invariants enforced at parse time -------------------

    #[test]
    fn non_empty_vec_fields_rejected_when_blank() {
        // (label, variant_block) — each variant block violates exactly one
        // non-empty-vec invariant. All should fail to parse.
        let cases: &[(&str, &str)] = &[
                (
                    "empty models",
                    "models = []\n\
                     runtimes = [{ type = \"llamacpp_cli_stock_tools\", source = \"github_release\", version = \"b5000\", flavor = \"windows-x64-vulkan\" }]\n\
                     clients = [\"ev1_c\"]",
                ),
                (
                    "empty runtimes",
                    "models = [{ type = \"torch\", source = \"huggingface\", org = \"x\", repo_name = \"y\" }]\n\
                     runtimes = []\n\
                     clients = [\"ev1_c\"]",
                ),
                (
                    "empty clients",
                    "models = [{ type = \"torch\", source = \"huggingface\", org = \"x\", repo_name = \"y\" }]\n\
                     runtimes = [{ type = \"llamacpp_cli_stock_tools\", source = \"github_release\", version = \"b5000\", flavor = \"windows-x64-vulkan\" }]\n\
                     clients = []",
                ),
                (
                    "empty per-variant benchmarks",
                    "models = [{ type = \"torch\", source = \"huggingface\", org = \"x\", repo_name = \"y\" }]\n\
                     runtimes = [{ type = \"llamacpp_cli_stock_tools\", source = \"github_release\", version = \"b5000\", flavor = \"windows-x64-vulkan\" }]\n\
                     clients = [\"ev1_c\"]\n\
                     benchmarks = []",
                ),
            ];
        for (label, variant_block) in cases {
            let toml_str = plan_toml(variant_block);
            assert!(
                toml::from_str::<Matrix>(&toml_str).is_err(),
                "{label} must reject"
            );
        }
    }

    #[test]
    fn plan_rejects_empty_top_level_fields() {
        // empty top-level benchmarks
        let toml_str = format!("benchmarks = []\n\n[[variants]]\n{MINIMAL_VARIANT}\n");
        assert!(toml::from_str::<Matrix>(&toml_str).is_err());

        // empty variants
        let toml_str = r#"benchmarks = ["prefill_throughput_256"]
variants = []
"#;
        assert!(toml::from_str::<Matrix>(toml_str).is_err());
    }

    // ---- Per-variant benchmarks happy path -------------------------------

    #[test]
    fn variant_benchmarks_override_when_set() -> anyhow::Result<()> {
        let toml_str = plan_toml(&format!(
            "{MINIMAL_VARIANT}\nbenchmarks = [\"decode_throughput_512_100\"]"
        ));
        let plan: Matrix = toml::from_str(&toml_str)?;
        let overrides = plan.variants[0]
            .benchmarks
            .as_ref()
            .context("override present")?;
        assert_eq!(overrides.len(), 1);
        assert_eq!(overrides[0].as_ref(), "decode_throughput_512_100");
        Ok(())
    }

    // ---- Sanity: baseline TOML always parses ----------------------------

    #[test]
    fn minimal_plan_is_valid() -> anyhow::Result<()> {
        let toml_str = minimal_plan_toml();
        toml::from_str::<Matrix>(&toml_str)?;
        Ok(())
    }

    // ---- Variant compatibility filtering ---------------------------------

    /// Number of runnable pairs we expect this variant to produce
    /// after compatibility filtering, or `None` if validation should
    /// reject the variant.
    #[derive(Debug, Clone, Copy)]
    enum Expected {
        Accept { pair_count: usize },
        Reject,
    }

    #[rstest::rstest]
    // Single-kind: each model variant paired with its sole compatible runtime.
    // Cases pass fallible builders; the test body collects them with `?`
    // since `?` cannot appear inside `#[case(...)]` attribute expressions.
    #[case::gguf_text_with_llamacpp(
            vec![gguf_text("org/repo")],
            vec![llamacpp_rt()],
            Expected::Accept { pair_count: 1 },
        )]
    #[case::gguf_vision_with_llamacpp(
            vec![gguf_vision("org/repo")],
            vec![llamacpp_rt()],
            Expected::Accept { pair_count: 1 },
        )]
    #[case::mlx_with_mlx(
            vec![mlx_model("org/repo")],
            vec![mlx_rt()],
            Expected::Accept { pair_count: 1 },
        )]
    #[case::torch_with_uv_vllm(
            vec![torch_model("org/repo")],
            vec![uv_vllm_rt()],
            Expected::Accept { pair_count: 1 },
        )]
    // Mixed-kind, fully matched: every model finds a runtime, every
    // runtime finds a model. Incompatible pairs are silently filtered
    // out — `pair_count` is the size of the matched intersection, not
    // the full cartesian product (which would be 4).
    #[case::mixed_kinds_all_matched(
            vec![gguf_text("org/repo"), mlx_model("org/repo")],
            vec![llamacpp_rt(), mlx_rt()],
            Expected::Accept { pair_count: 2 },
        )]
    // Reject: a model has no compatible runtime in the variant.
    #[case::gguf_with_only_mlx_runtime(
            vec![gguf_text("org/repo")],
            vec![mlx_rt()],
            Expected::Reject,
        )]
    #[case::mlx_with_only_llamacpp(
            vec![mlx_model("org/repo")],
            vec![llamacpp_rt()],
            Expected::Reject,
        )]
    #[case::torch_with_only_llamacpp(
            vec![torch_model("org/repo")],
            vec![llamacpp_rt()],
            Expected::Reject,
        )]
    // Reject: a model dangles even though some other model finds its match.
    #[case::orphan_mlx_model_among_llamacpp(
            vec![gguf_text("org/repo"), mlx_model("org/repo")],
            vec![llamacpp_rt()],
            Expected::Reject,
        )]
    // Reject: a runtime dangles even though some other runtime finds its match.
    #[case::orphan_mlx_runtime_among_gguf(
            vec![gguf_text("org/repo")],
            vec![llamacpp_rt(), mlx_rt()],
            Expected::Reject,
        )]
    fn variant_runnable_pairs(
        #[case] models: Vec<anyhow::Result<Model>>,
        #[case] runtimes: Vec<anyhow::Result<Runtime>>,
        #[case] expected: Expected,
    ) -> anyhow::Result<()> {
        let models = models.into_iter().collect::<anyhow::Result<Vec<_>>>()?;
        let runtimes = runtimes.into_iter().collect::<anyhow::Result<Vec<_>>>()?;
        let variant = make_variant(models, runtimes)?;
        match (variant.runnable_pairs(), expected) {
            (Ok(pairs), Expected::Accept { pair_count }) => {
                if pairs.len() != pair_count {
                    return Err(anyhow::anyhow!(
                        "expected {pair_count} runnable pairs, got {}",
                        pairs.len()
                    ));
                }
            }
            (Err(violation), Expected::Reject) => {
                let msg = format!("{violation}");
                assert!(
                    msg.contains("no compatible"),
                    "msg missing 'no compatible': {msg}"
                );
            }
            (Ok(pairs), Expected::Reject) => {
                return Err(anyhow::anyhow!("expected rejection, got Ok({pairs:?})"));
            }
            (Err(e), Expected::Accept { .. }) => {
                return Err(anyhow::anyhow!("expected acceptance, got: {e}"));
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------------
    // Outer Plan + RunnableCell tests (moved from pipette-plan/src/plan.rs).
    // ------------------------------------------------------------------------

    #[test]
    fn minimal_outer_plan_parses() -> anyhow::Result<()> {
        let plan = Plan::parse(minimal_outer_plan())?;
        assert_eq!(plan.plan_id, "test");
        assert_eq!(plan.transports.len(), 1);
        assert_eq!(plan.transports[0].client_id(), "mac-1");
        Ok(())
    }

    #[test]
    fn cells_expand_to_cartesian_product_per_variant() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["b1", "b2"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [
  { type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" },
  { type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "b.gguf" },
]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.cells()?;
        // 2 benchmarks × 2 models × 1 runtime = 4 cells
        assert_eq!(cells.len(), 4);
        assert!(cells.iter().all(|c| c.variant_idx == 0));
        assert!(cells
            .iter()
            .all(|c| client_refs(c.allowed_clients) == vec!["t1"]));
        Ok(())
    }

    #[test]
    fn variant_benchmark_override_replaces_plan_benchmarks() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["plan-bench"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }]
benchmarks = ["variant-bench-a", "variant-bench-b"]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.cells()?;
        assert_eq!(cells.len(), 2);
        let names: Vec<_> = cells.iter().map(|c| c.benchmark.as_ref()).collect();
        assert!(names.contains(&"variant-bench-a"));
        assert!(names.contains(&"variant-bench-b"));
        assert!(!names.contains(&"plan-bench"));
        Ok(())
    }

    #[test]
    fn variant_benchmarks_allow_missing_plan_benchmarks() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }]
benchmarks = ["variant-bench"]
"#;
        let plan = Plan::parse(toml_str)?;
        assert!(plan.matrix.benchmarks.is_none());
        let cells = plan.cells()?;
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].benchmark.as_ref(), "variant-bench");
        let runnable_cells = plan.runnable_cells()?;
        assert_eq!(single(&runnable_cells)?.benchmark.as_ref(), "variant-bench");
        Ok(())
    }

    #[test]
    fn variant_without_benchmarks_rejected_when_plan_benchmarks_missing() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }]
"#;
        let err = Plan::parse(toml_str)
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected missing benchmarks to reject"))?;
        let msg = format!("{err:#}");
        assert!(msg.contains("variant 0"), "got: {msg}");
        assert!(msg.contains("benchmarks"), "got: {msg}");
        Ok(())
    }

    #[test]
    fn unknown_client_ref_rejected() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["b1"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["does-not-exist"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }]
"#;
        let err = Plan::parse(toml_str)
            .err()
            .ok_or_else(|| anyhow::anyhow!("should reject unknown client"))?;
        let msg = err.to_string();
        assert!(msg.contains("unknown transport"), "got: {msg}");
        assert!(msg.contains("does-not-exist"), "got: {msg}");
        Ok(())
    }

    #[test]
    fn duplicate_transport_names_rejected() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["b1"]

[[transports]]
client_id = "dup"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[transports]]
client_id = "dup"
type = "local"
binary_path = "/bin2"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["dup"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }]
"#;
        let err = Plan::parse(toml_str)
            .err()
            .ok_or_else(|| anyhow::anyhow!("should reject duplicate client_id"))?;
        assert!(err.to_string().contains("duplicate transport client_id"));
        Ok(())
    }

    /// The device command is quoted for posix on both hops, so a powershell
    /// device shell cannot be honored — it must fail at load rather than reach
    /// the device garbled.
    #[test]
    fn adb_over_ssh_rejects_a_powershell_device_shell() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id    = "x"
benchmarks = ["prefill_throughput_512"]

[[transports]]
client_id   = "phone-1"
type        = "adb_over_ssh"
host        = "controller"
serial      = "S1"
shell       = "powershell"
binary_path = "/data/local/tmp/pipette-evals/pipette"
work_dir    = "/data/local/tmp/pipette-evals"

[[variants]]
clients  = ["phone-1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "android-arm64-v8a" }]
"#;
        let err = Plan::parse(toml_str)
            .err()
            .ok_or_else(|| anyhow::anyhow!("should reject a powershell adb_over_ssh transport"))?;
        assert!(
            err.to_string().contains(r#"requires shell = "posix""#),
            "unexpected error: {err:#}"
        );
        Ok(())
    }

    #[test]
    fn cells_cartesian_includes_runtime_axis() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["b1"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [
  { type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" },
  { type = "llamacpp_cli_stock_tools", source = "github_release", version = "b2", flavor = "macos-arm64" },
  { type = "llamacpp_cli_stock_tools", source = "github_release", version = "b3", flavor = "macos-arm64" },
]
"#;
        let plan = Plan::parse(toml_str)?;
        // 1 benchmark × 1 model × 3 runtimes = 3 cells
        let cells = plan.cells()?;
        assert_eq!(cells.len(), 3);
        let versions: Vec<&str> = cells
            .iter()
            .map(|c| match c.runtime {
                Runtime::LlamacppCliStockTools(rt) => rt.source.reference(),
                _ => "",
            })
            .collect();
        assert!(versions.contains(&"b1"));
        assert!(versions.contains(&"b2"));
        assert!(versions.contains(&"b3"));
        Ok(())
    }

    #[test]
    fn empty_transports_array_rejected() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["b1"]
transports       = []

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }]
"#;
        // NonEmptyVec deserialization rejects `[]` before our
        // cross-field validation runs.
        let err = Plan::parse(toml_str)
            .err()
            .ok_or_else(|| anyhow::anyhow!("should reject empty transports"))?;
        assert!(
            err.to_string().contains("empty") || err.to_string().contains("not be empty"),
            "got: {}",
            err
        );
        Ok(())
    }

    #[test]
    fn missing_transports_rejected() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["b1"]

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }]
"#;
        let err = Plan::parse(toml_str)
            .err()
            .ok_or_else(|| anyhow::anyhow!("should reject missing transports"))?;
        assert!(err.to_string().contains("transports") || err.to_string().contains("missing"));
        Ok(())
    }

    #[test]
    fn outer_plan_round_trips_through_toml() -> anyhow::Result<()> {
        let plan = Plan::parse(minimal_outer_plan())?;
        let emitted = toml::to_string(&plan)?;
        let reparsed = Plan::parse(&emitted)?;
        // Full structural equality on the Plan.
        assert_eq!(plan, reparsed);
        // And byte-equal wire form — proves Serialize/Deserialize are
        // mutually consistent.
        let reemitted = toml::to_string(&reparsed)?;
        assert_eq!(emitted, reemitted);
        Ok(())
    }

    #[test]
    fn serialized_omits_absent_optional_fields() -> anyhow::Result<()> {
        // Minimal SSH transport: no `user`, no `port`, no
        // `parallelism` override (default=1). After round-trip,
        // emitted TOML should not contain those keys.
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["b1"]

[[transports]]
client_id = "t1"
type = "ssh"
host = "h"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }]
"#;
        let plan = Plan::parse(toml_str)?;
        let emitted = toml::to_string(&plan)?;
        // skip_serializing_if = "Option::is_none" suppresses these.
        assert!(!emitted.contains("user ="), "got: {emitted}");
        assert!(!emitted.contains("port ="), "got: {emitted}");
        // RetryConfig::is_default suppresses the whole [retry] table.
        assert!(!emitted.contains("[retry]"), "got: {emitted}");
        assert!(
            !emitted.contains("max_consecutive_failures"),
            "got: {emitted}"
        );
        assert!(
            !emitted.contains("cooldown_seconds"),
            "cooldown_seconds is no longer a plan-types field; got: {emitted}"
        );
        Ok(())
    }

    #[test]
    fn cells_isolate_clients_per_variant() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["b1"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[transports]]
client_id = "t2"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }]

[[variants]]
clients  = ["t2"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "b.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.cells()?;
        assert_eq!(cells.len(), 2);
        let v0 = cells
            .iter()
            .find(|c| c.variant_idx == 0)
            .ok_or_else(|| anyhow::anyhow!("missing variant 0"))?;
        let v1 = cells
            .iter()
            .find(|c| c.variant_idx == 1)
            .ok_or_else(|| anyhow::anyhow!("missing variant 1"))?;
        assert_eq!(client_refs(v0.allowed_clients), vec!["t1"]);
        assert_eq!(client_refs(v1.allowed_clients), vec!["t2"]);
        Ok(())
    }

    #[test]
    fn runnable_cell_gguf_text_model_ref() -> anyhow::Result<()> {
        let plan = Plan::parse(minimal_outer_plan())?;
        let cells = plan.runnable_cells()?;
        let cell = single(&cells)?;
        assert_eq!(cell.model.to_string(), "org/repo:Q4_K_M.gguf");
        // Display carries the full source coordinate (repo@version), not just
        // the ref — repository_url defaults to the upstream repo here.
        assert_eq!(
            cell.runtime.to_string(),
            "github.com/ggml-org/llama.cpp@b9050:macos-arm64"
        );
        Ok(())
    }

    #[test]
    fn runnable_cell_vision_model_identity_includes_mmproj() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["b1"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_vision", source = "huggingface", org = "lab", repo_name = "VL-3B", model = "q4.gguf", mmproj = "mm-f16.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "android-arm64-v8a" }]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.runnable_cells()?;
        let cell = single(&cells)?;
        assert_eq!(
            cell.model.to_string(),
            "lab/VL-3B:q4.gguf+lab/VL-3B:mm-f16.gguf"
        );
        assert!(matches!(cell.model, Model::GgufVision(_)));
        // The renamed flavor wire form (not the kebab-cased variant name),
        // prefixed by the full source coordinate.
        assert_eq!(
            cell.runtime.to_string(),
            "github.com/ggml-org/llama.cpp@b9050:android-arm64-v8a"
        );
        Ok(())
    }

    #[test]
    fn runnable_cell_mlx_model_ref_is_bare_repo() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["b1"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "mlx", source = "huggingface", org = "lab", repo_name = "LFM2.5-350M-MLX-4bit" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.31.2",
              source = { type = "pip_requirements_text", contents = "mlx-lm==0.31.2" },
              flavor = "macos-arm64" }]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.runnable_cells()?;
        let cell = single(&cells)?;
        assert_eq!(cell.model.to_string(), "lab/LFM2.5-350M-MLX-4bit");
        // Bare version — no flavor suffix.
        assert_eq!(cell.runtime.to_string(), "0.31.2");
        assert!(matches!(cell.runtime, Runtime::MlxMacosPipette(_)));
        Ok(())
    }

    /// Collect every shipped example plan path (`examples/plans/*.toml`).
    fn example_plan_paths() -> anyhow::Result<Vec<std::path::PathBuf>> {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/plans");
        let paths = std::fs::read_dir(dir)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("toml"))
            .collect::<Vec<_>>();
        assert!(!paths.is_empty(), "no example plans found in {dir}");
        Ok(paths)
    }

    /// The `examples/plans/*.toml` are the reference authoring surface: parse
    /// each one so a type change that invalidates one can't merge unnoticed.
    /// Parsing runs full validation via the `#[serde(try_from = "PlanIntake")]`
    /// boundary (`validate_plan`), so an authored flag with a bad key or shape
    /// fails here.
    #[test]
    fn shipped_example_plans_parse() -> anyhow::Result<()> {
        example_plan_paths()?.iter().try_for_each(|path| {
            let toml = std::fs::read_to_string(path)?;
            Plan::parse(&toml).map_err(|e| anyhow::anyhow!("{}: {e:#}", path.display()))?;
            anyhow::Ok(())
        })
    }

    /// Beyond parsing, materialize each example plan's matrix — this resolves
    /// `runtime`/`model`/`benchmark` flags onto their cells (the triple match),
    /// so an authored `benchmark_flags`/`runtime_flags` entry that names no real
    /// cell, or a plan that expands to nothing, is caught.
    #[test]
    fn shipped_example_plans_materialize_runnable_cells() -> anyhow::Result<()> {
        example_plan_paths()?.iter().try_for_each(|path| {
            let toml = std::fs::read_to_string(path)?;
            let plan =
                Plan::parse(&toml).map_err(|e| anyhow::anyhow!("{}: {e:#}", path.display()))?;
            let cells = plan
                .runnable_cells()
                .map_err(|e| anyhow::anyhow!("{}: {e:#}", path.display()))?;
            anyhow::ensure!(
                !cells.is_empty(),
                "{}: produced no runnable cells",
                path.display()
            );
            anyhow::Ok(())
        })
    }

    /// The iOS transport carries the plan's knobs, in the app's `key=value` grammar.
    ///
    /// It carried none at all until this change: a plan could author `threads` for an iOS
    /// cell, parse and resolve it, and the phone would still run on its derived P-core
    /// count while reporting flags nobody had sent it.
    #[test]
    fn the_ios_transport_ships_the_resolved_knobs() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["prefill_throughput_256"]

[[transports]]
client_id = "phone"
type = "ios"
device_udid = "00000000-000000000000000A"

[[variants]]
clients  = ["phone"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_ios_pipette", version = "b9050", flavor = "ios-arm64" }]
runtime_flags = [{ benchmark_type = "prefill_throughput", runtime_type = "llamacpp_ios_pipette", model_type = "gguf_text", threads = 4 }]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.runnable_cells()?;
        let cell = single(&cells)?;
        let argv = cell.build_argv(&plan.transports[0])?;

        let knobs = argv
            .iter()
            .find_map(|a| a.strip_prefix("runtime-flags="))
            .ok_or_else(|| anyhow::anyhow!("expected runtime-flags in {argv:?}"))?;
        assert!(knobs.contains(r#""threads":4"#), "got: {knobs}");
        // The app derives the cell from `runtime=`/`model=`/`benchmarks=`, all of which
        // this argv already carries, so the axes stay off the wire here too.
        assert!(!knobs.contains("benchmark_type"), "got: {knobs}");
        Ok(())
    }

    #[test]
    fn runnable_cell_ships_typed_flags_as_json() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["prefill_throughput_256"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
runtime_flags = [{ benchmark_type = "prefill_throughput", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", threads = 4, number_gpu_layers = 99 }]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.runnable_cells()?;
        let cell = single(&cells)?;
        // Rendering to `-t 4 -ngl 99` is the client's job now; the runner ships
        // the validated knobs structurally on `--runtime-flags`.
        let json = cell
            .runtime_flags_json()
            .ok_or_else(|| anyhow::anyhow!("expected runtime flags json"))?;
        assert!(json.contains("\"threads\":4"), "got: {json}");
        assert!(json.contains("\"number_gpu_layers\":99"), "got: {json}");
        // The axes the entry was selected by are not on the wire: the client resolves the
        // cell from `--benchmark`/`--runtime`/`--model` before it reads any flags.
        ["benchmark_type", "runtime_type", "model_type"]
            .iter()
            .for_each(|axis| {
                assert!(
                    !json.contains(axis),
                    "{axis} should be stripped, got: {json}"
                )
            });
        // It round-trips against the cell the client resolved, which is what supplies them.
        let back = RuntimeFlags::from_cell_json(
            &json,
            RuntimeType::LlamacppCliStockTools,
            ModelType::GgufText,
            BenchmarkType::PrefillThroughput,
        )?;
        assert_eq!(back, cell.runtime_flags);
        Ok(())
    }

    #[rstest]
    // The in-process apk has no typed-flag cell, so any runtime_flags entry for
    // it is a NoSuchCombination at parse.
    #[case::in_process_apk(
        r#"
plan_id    = "x"
benchmarks = ["eval_smoke"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_apk_pipette", version = "b9050", flavor = "android-arm64-v8" }]
runtime_flags = [{ benchmark_type = "eval", runtime_type = "llamacpp_apk_pipette", model_type = "gguf_text", ctx_size = 8192 }]
"#,
        "no runtime flags defined"
    )]
    // A flag for a benchmark the variant never runs matches no cell.
    #[case::matches_no_cell(
        r#"
plan_id    = "x"
benchmarks = ["prefill_throughput_512"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
runtime_flags = [{ benchmark_type = "eval", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", ctx_size = 8192 }]
"#,
        "matches no cell"
    )]
    fn rejects_invalid_runtime_flags(#[case] toml_str: &str, #[case] expected: &str) {
        let result = Plan::parse(toml_str);
        assert!(result.is_err(), "expected rejection");
        if let Err(err) = result {
            assert!(format!("{err:#}").contains(expected), "got: {err:#}");
        }
    }

    #[test]
    fn benchmark_flags_resolve_and_emit_eval_run_knobs() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id    = "x"
benchmarks = ["eval_smoke"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp/wd"
shell = "posix"

[[variants]]
clients         = ["t1"]
models          = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes        = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
benchmark_flags = [{ benchmark_type = "eval", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", http_timeout_seconds = 600, doomloop = { exact_repeat = { required = 5, window = 8192 } } }]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.runnable_cells()?;
        let cell = single(&cells)?;
        let bf = cell
            .benchmark_flags
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("benchmark_flags resolved onto the eval cell"))?;
        assert_eq!(bf.http_timeout(), Some(600));
        let doomloop = bf
            .doomloop()
            .ok_or_else(|| anyhow::anyhow!("doomloop present on eval cell"))?;
        let exact = doomloop
            .exact_repeat
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("exact_repeat present"))?;
        assert_eq!(exact.required, Some(5));
        assert_eq!(exact.window, Some(8192));

        let argv = cell.build_argv(&local_transport()?)?;
        assert!(argv
            .windows(2)
            .any(|w| w == ["--http-timeout-seconds", "600"]));
        assert!(argv
            .windows(2)
            .any(|w| w == ["--doomloop-exact-repeat-required", "5"]));
        Ok(())
    }

    #[test]
    fn runnable_cell_build_argv_full() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["prefill_throughput_512"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp/wd"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
runtime_flags = [{ benchmark_type = "prefill_throughput", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", threads = 4, number_gpu_layers = 99 }]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.runnable_cells()?;
        let argv = single(&cells)?.build_argv(&local_transport()?)?;

        // Shared prefix.
        assert_eq!(argv[0], "/bin/pipette");
        assert!(argv.windows(2).any(|w| w == ["--work-dir", "/tmp"]));
        assert!(argv.iter().any(|a| a == "benchmarks"));
        assert!(argv.iter().any(|a| a == "run"));

        // Per-cell args.
        assert!(argv
            .windows(2)
            .any(|w| w == ["--benchmark", "prefill_throughput_512"]));
        // `--model` is the model's canonical JSON; assert it round-trips back to
        // the authored gguf-text model (the client parses it via `parse_model_arg`).
        let model_idx = argv
            .iter()
            .position(|a| a == "--model")
            .ok_or_else(|| anyhow::anyhow!("--model present"))?;
        let model: Model = serde_json::from_str(&argv[model_idx + 1])?;
        assert_eq!(model.to_string(), "o/r:a.gguf");
        // `--runtime` is likewise the runtime's canonical JSON; round-trip it.
        let runtime_idx = argv
            .iter()
            .position(|a| a == "--runtime")
            .ok_or_else(|| anyhow::anyhow!("--runtime present"))?;
        let runtime: Runtime = serde_json::from_str(&argv[runtime_idx + 1])?;
        assert_eq!(runtime.cli_ref(), "b9050:macos-arm64");
        // runtime_flags shipped as the structured JSON payload (the client
        // renders it to `-t 4 -ngl 99`, not the runner).
        let rf_idx = argv
            .iter()
            .position(|a| a == "--runtime-flags")
            .ok_or_else(|| anyhow::anyhow!("--runtime-flags present"))?;
        let payload = &argv[rf_idx + 1];
        assert!(payload.contains("\"threads\":4"), "got: {payload}");
        assert!(
            payload.contains("\"number_gpu_layers\":99"),
            "got: {payload}"
        );

        // No mmproj here.
        assert!(!argv.iter().any(|a| a == "--mmproj"));
        Ok(())
    }

    /// A `gguf_vision` cell inlines its projector in the `--model` JSON and emits
    /// no separate `--mmproj`; the projector round-trips through `parse_model_arg`.
    #[test]
    fn build_argv_gguf_vision_inlines_projector_without_mmproj() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id    = "x"
benchmarks = ["eval_smoke"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp/wd"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_vision", source = "huggingface", org = "o", repo_name = "r", model = "a.gguf", mmproj = "m.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.runnable_cells()?;
        let argv = single(&cells)?.build_argv(&local_transport()?)?;

        assert!(!argv.iter().any(|a| a == "--mmproj"));
        let model_idx = argv
            .iter()
            .position(|a| a == "--model")
            .ok_or_else(|| anyhow::anyhow!("--model present"))?;
        let model: Model = serde_json::from_str(&argv[model_idx + 1])?;
        assert!(matches!(model, Model::GgufVision(_)));
        // Projector is part of the typed model / Display — no separate flag.
        assert!(model.to_string().contains('+'));
        Ok(())
    }

    #[test]
    fn runnable_cell_build_argv_omits_optional_flags_when_unset() -> anyhow::Result<()> {
        // Minimal plan: no enable_thinking, no http_timeout, no doomloop
        // overrides, no runtime flags, no mmproj.
        let plan = Plan::parse(minimal_outer_plan())?;
        let cells = plan.runnable_cells()?;
        let argv = single(&cells)?.build_argv(&local_transport()?)?;
        assert!(!argv.iter().any(|a| a == "--model-flags"));
        assert!(!argv.iter().any(|a| a == "--http-timeout-seconds"));
        assert!(!argv.iter().any(|a| a == "--runtime-flags"));
        assert!(!argv.iter().any(|a| a == "--mmproj"));
        assert!(!argv.iter().any(|a| a.starts_with("--doomloop-")));
        Ok(())
    }

    #[test]
    fn build_argv_emits_model_flags_for_eval_cell() -> anyhow::Result<()> {
        // `model_flags` is authored at the variant level, keyed on
        // (benchmark, model); an eval cell ships plan-form `--model-flags` JSON.
        let toml_str = r#"
plan_id    = "x"
benchmarks = ["eval_smoke"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp/wd"
shell = "posix"

[[variants]]
clients     = ["t1"]
models      = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes    = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
model_flags = [{ benchmark_type = "eval", model_type = "gguf_text", enable_thinking = true }]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.runnable_cells()?;
        let argv = single(&cells)?.build_argv(&local_transport()?)?;
        let flags_json = argv
            .windows(2)
            .find(|w| w[0] == "--model-flags")
            .map(|w| w[1].as_str())
            .ok_or_else(|| anyhow::anyhow!("expected --model-flags"))?;
        // The entry itself, not wrapped in an array.
        let parsed: ModelFlags = serde_json::from_str(flags_json)?;
        assert_eq!(parsed.enable_thinking(), Some(true));
        Ok(())
    }

    #[test]
    fn build_argv_emits_readiness_max_wait_for_timing_cell() -> anyhow::Result<()> {
        // Readiness is authored per-cell on a timing benchmark (prefill) and
        // forwarded to the runner as `--readiness-max-wait-secs`.
        let toml_str = r#"
plan_id    = "x"
benchmarks = ["prefill_throughput_512"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp/wd"
shell = "posix"

[[variants]]
clients         = ["t1"]
models          = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes        = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
benchmark_flags = [{ benchmark_type = "prefill_throughput", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", readiness = { max_wait_secs = 1800 } }]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.runnable_cells()?;
        let argv = single(&cells)?.build_argv(&local_transport()?)?;
        assert!(argv
            .windows(2)
            .any(|w| w == ["--readiness-max-wait-secs", "1800"]));
        Ok(())
    }

    /// A prefill (readiness-carrying) plan whose `readiness` table is filled in
    /// by `argv_for_prefill_with_readiness`. Uses a placeholder rather than
    /// `format!` so the TOML's inline-table braces stay readable.
    const PREFILL_PLAN_WITH_READINESS: &str = r#"
plan_id    = "x"
benchmarks = ["prefill_throughput_512"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp/wd"
shell = "posix"

[[variants]]
clients         = ["t1"]
models          = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes        = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
benchmark_flags = [{ benchmark_type = "prefill_throughput", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", readiness = { __READINESS__ } }]
"#;

    fn argv_for_prefill_with_readiness(readiness: &str) -> anyhow::Result<Vec<String>> {
        let toml_str = PREFILL_PLAN_WITH_READINESS.replace("__READINESS__", readiness);
        let plan = Plan::parse(&toml_str)?;
        let cells = plan.runnable_cells()?;
        single(&cells)?.build_argv(&local_transport()?)
    }

    /// `skip_thermal` reaches the runner as `--readiness-skip-thermal`, and only
    /// when it is actually `true`. An unset or `false` knob must leave the argv
    /// untouched — emitting it either way would silently waive the thermal
    /// criterion on every cell in the plan.
    #[rstest]
    #[case("skip_thermal = true", true)]
    #[case("skip_thermal = false", false)]
    #[case("max_wait_secs = 900", false)]
    fn build_argv_emits_readiness_skip_thermal_only_when_true(
        #[case] readiness: &str,
        #[case] want_flag: bool,
    ) -> anyhow::Result<()> {
        let argv = argv_for_prefill_with_readiness(readiness)?;
        assert_eq!(
            argv.iter().any(|a| a == "--readiness-skip-thermal"),
            want_flag,
            "readiness = {{ {readiness} }} produced argv {argv:?}",
        );
        Ok(())
    }

    /// The flag is valueless: the runner declares it `ArgAction::SetTrue`, so
    /// nothing may follow it. A value pushed here would be consumed as the next
    /// argument and mis-bind whatever came after.
    #[test]
    fn readiness_skip_thermal_is_emitted_as_a_bare_flag() -> anyhow::Result<()> {
        let argv = argv_for_prefill_with_readiness("skip_thermal = true")?;
        let at = argv
            .iter()
            .position(|a| a == "--readiness-skip-thermal")
            .ok_or_else(|| anyhow::anyhow!("expected --readiness-skip-thermal in {argv:?}"))?;
        if let Some(next) = argv.get(at + 1) {
            anyhow::ensure!(
                next.starts_with("--"),
                "expected no value after --readiness-skip-thermal, found {next:?}",
            );
        }
        Ok(())
    }

    /// Both readiness knobs on one cell must both survive into the argv — the
    /// deadline as a valued flag, the waiver as a bare one.
    #[test]
    fn build_argv_emits_both_readiness_knobs_together() -> anyhow::Result<()> {
        let argv = argv_for_prefill_with_readiness("max_wait_secs = 1800, skip_thermal = true")?;
        assert!(argv
            .windows(2)
            .any(|w| w == ["--readiness-max-wait-secs", "1800"]));
        assert!(argv.iter().any(|a| a == "--readiness-skip-thermal"));
        Ok(())
    }

    #[test]
    fn model_flags_resolve_only_to_the_matching_type_and_benchmark() -> anyhow::Result<()> {
        // One variant, two model types, eval + prefill; flags authored for
        // (eval, gguf_text) only. They must land on that one cell and no other
        // (not the eval gguf_vision cell, nor either prefill cell).
        let toml_str = r#"
plan_id    = "x"
benchmarks = ["eval_smoke", "prefill_throughput_512"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp/wd"
shell = "posix"

[[variants]]
clients     = ["t1"]
models      = [
  { type = "gguf_text",   source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" },
  { type = "gguf_vision", source = "huggingface", org = "o", repo_name = "r", model = "a.gguf", mmproj = "m.gguf" },
]
runtimes    = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
model_flags = [{ benchmark_type = "eval", model_type = "gguf_text", enable_thinking = true }]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.runnable_cells()?;
        // All four cells exist; exactly the (eval, gguf_text) one carries flags.
        assert_eq!(cells.len(), 4);
        let flagged: Vec<_> = cells.iter().filter(|c| c.model_flags.is_some()).collect();
        assert_eq!(flagged.len(), 1);
        let cell = flagged[0];
        assert!(cell.benchmark.as_ref().contains("eval"));
        assert!(matches!(cell.model, Model::GgufText(_)));
        assert_eq!(
            cell.model_flags,
            Some(ModelFlags::EvalGgufText {
                enable_thinking: Some(true)
            })
        );
        Ok(())
    }

    /// The plan scaffold for a duplicate-flags test: one local transport, one
    /// variant with a gguf-text model on llama.cpp. `benchmarks` and the flags
    /// block are filled per case.
    fn dup_flag_plan(benchmark: &str, flags_block: &str) -> String {
        format!(
            r#"plan_id    = "x"
benchmarks = ["{benchmark}"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp/wd"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }}]
runtimes = [{{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }}]
{flags_block}
"#
        )
    }

    /// Plans distribute ids, so an addressed benchmark is rejected — and the
    /// message has to carry the fix, since it is the migration path for plan
    /// files written when `benchmarks = ["remote/<id>"]` was the form.
    #[rstest]
    #[case::remote("remote/eval_smoke")]
    #[case::local("local/eval_smoke")]
    fn addressed_benchmark_id_rejected_with_guidance(
        #[case] addressed: &str,
    ) -> anyhow::Result<()> {
        let err = Plan::parse(&dup_flag_plan(addressed, ""))
            .err()
            .context("an addressed benchmark id should be rejected")?;
        let message = format!("{err:#}");
        assert!(message.contains(addressed), "got: {message}");
        assert!(
            message.contains("is not a bare id") && message.contains("drop any"),
            "the message must say what to do; got: {message}"
        );
        Ok(())
    }

    /// A variant may not author two `runtime_flags` entries with the same
    /// `(benchmark, runtime, model)` key — the resolver would silently pick
    /// between them. Rejected at parse.
    #[test]
    fn duplicate_runtime_flags_key_rejected() -> anyhow::Result<()> {
        let toml = dup_flag_plan(
            "prefill_throughput_512",
            r#"runtime_flags = [
  { benchmark_type = "prefill_throughput", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", threads = 4 },
  { benchmark_type = "prefill_throughput", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", threads = 8 },
]"#,
        );
        let err = Plan::parse(&toml)
            .err()
            .context("duplicate runtime_flags key should be rejected")?;
        assert!(
            format!("{err:#}").contains("duplicate runtime_flags"),
            "got: {err:#}"
        );
        Ok(())
    }

    /// Same for `model_flags`, keyed on `(benchmark, model)`.
    #[test]
    fn duplicate_model_flags_key_rejected() -> anyhow::Result<()> {
        let toml = dup_flag_plan(
            "eval_smoke",
            r#"model_flags = [
  { benchmark_type = "eval", model_type = "gguf_text", enable_thinking = true },
  { benchmark_type = "eval", model_type = "gguf_text", enable_thinking = false },
]"#,
        );
        let err = Plan::parse(&toml)
            .err()
            .context("duplicate model_flags key should be rejected")?;
        assert!(
            format!("{err:#}").contains("duplicate model_flags"),
            "got: {err:#}"
        );
        Ok(())
    }

    /// Same for `benchmark_flags`, keyed on `(benchmark, model)`.
    #[test]
    fn duplicate_benchmark_flags_key_rejected() -> anyhow::Result<()> {
        let toml = dup_flag_plan(
            "eval_smoke",
            r#"benchmark_flags = [
  { benchmark_type = "eval", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", http_timeout_seconds = 600 },
  { benchmark_type = "eval", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", http_timeout_seconds = 900 },
]"#,
        );
        let err = Plan::parse(&toml)
            .err()
            .context("duplicate benchmark_flags key should be rejected")?;
        assert!(
            format!("{err:#}").contains("duplicate benchmark_flags"),
            "got: {err:#}"
        );
        Ok(())
    }

    #[test]
    fn runnable_cells_deduplicate_across_variants() -> anyhow::Result<()> {
        // Two variants produce the same (benchmark, model, runtime,
        // clients) tuple. HashSet collapses them into one row.
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["b1"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.runnable_cells()?;
        assert_eq!(cells.len(), 1);
        Ok(())
    }

    #[test]
    fn runnable_cell_cartesian_product() -> anyhow::Result<()> {
        // 1 benchmark × 2 models × 2 runtimes × 1 variant = 4 cells.
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["b1"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [
  { type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" },
  { type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "b.gguf" },
]
runtimes = [
  { type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" },
  { type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9051", flavor = "macos-arm64" },
]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.runnable_cells()?;
        assert_eq!(cells.len(), 4);
        let refs: Vec<(String, String)> = cells
            .iter()
            .map(|c| (c.model.to_string(), c.runtime.to_string()))
            .collect();
        let rt = |v: &str| format!("github.com/ggml-org/llama.cpp@{v}:macos-arm64");
        assert!(refs.contains(&("o/r:a.gguf".into(), rt("b9050"))));
        assert!(refs.contains(&("o/r:a.gguf".into(), rt("b9051"))));
        assert!(refs.contains(&("o/r:b.gguf".into(), rt("b9050"))));
        assert!(refs.contains(&("o/r:b.gguf".into(), rt("b9051"))));
        Ok(())
    }

    #[rstest::rstest]
    #[case::no_model_declares_auth(
        r#"{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }"#,
        false
    )]
    #[case::second_model_declares_auth(
        r#"
  { type = "gguf_text", source = "huggingface", org = "o", repo_name = "r1", path = "a.gguf" },
  { type = "gguf_text", source = "huggingface", org = "o", repo_name = "r2", path = "b.gguf", auth_token = "hf_test_xxx" }
"#,
        true
    )]
    fn plan_auth_token_reflects_any_model(
        #[case] models: &str,
        #[case] expect: bool,
    ) -> anyhow::Result<()> {
        let toml_str = format!(
            r#"
plan_id          = "x"
benchmarks       = ["b"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{models}]
runtimes = [{{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }}]
"#
        );
        let plan = Plan::parse(&toml_str)?;
        assert_eq!(plan.auth_token().is_some(), expect);
        Ok(())
    }

    /// Plan::parse must reject variants where a model has no
    /// runtime it can pair with — here a GGUF model with only an
    /// MLX runtime. See `Model::is_compatible_with` for the full
    /// table.
    #[test]
    fn parse_rejects_orphan_model() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["b"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.31", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.31" } }]
"#;
        let err = Plan::parse(toml_str)
            .err()
            .ok_or_else(|| anyhow::anyhow!("expected Plan::parse to reject orphan model"))?;
        let msg = format!("{err:#}");
        assert!(
            msg.contains("no compatible"),
            "expected error to name 'no compatible': {msg}"
        );
        assert!(
            msg.contains("variant 0"),
            "expected error to name the offending variant: {msg}"
        );
        Ok(())
    }

    /// Inverse of the orphan-model case: when models cover every
    /// runtime's kind and vice-versa, mixed-kind variants are valid
    /// and the cell count is the matched subset of the cartesian
    /// product (not the full product). Operator-facing benefit: one
    /// variant block can host different backends on the same transport
    /// cohort without splitting into separate variants.
    #[test]
    fn parse_accepts_mixed_kinds_when_fully_matched() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id          = "x"
benchmarks       = ["b"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [
  { type = "gguf_text", source = "huggingface", org = "o", repo_name = "r1", path = "a.gguf" },
  { type = "mlx", source = "huggingface",       org = "o", repo_name = "r2" },
]
runtimes = [
  { type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" },
  { type = "mlx_macos_pipette", version = "0.31", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.31" } },
]
"#;
        let plan = Plan::parse(toml_str)?;
        let cells = plan.runnable_cells()?;
        // 2 models × 2 runtimes × 1 benchmark would give 4 cells in the
        // full cartesian product; compatibility filtering drops the two
        // mismatched pairs (gguf×mlx, mlx×llamacpp) and leaves 2.
        assert_eq!(cells.len(), 2);
        Ok(())
    }

    // ----------------------------------------------------------------------
    // parallelism budget
    // ----------------------------------------------------------------------

    #[rstest::rstest]
    #[case::defaults_to_one("", 1)]
    #[case::explicit_value("parallelism = 3", 3)]
    fn parallelism_parses(
        #[case] transport_extra: &str,
        #[case] expected: usize,
    ) -> anyhow::Result<()> {
        let toml_str = format!(
            r#"
plan_id    = "x"
benchmarks = ["b1"]

[[transports]]
client_id   = "t1"
type        = "local"
binary_path = "/bin"
work_dir    = "/tmp"
shell       = "posix"
{transport_extra}

[[variants]]
clients  = ["t1"]
models   = [{{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }}]
runtimes = [{{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }}]
"#
        );
        let plan = Plan::parse(&toml_str)?;
        let t = plan
            .transports
            .iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("expected one transport"))?;
        assert_eq!(t.parallelism().get(), expected);
        Ok(())
    }

    #[test]
    fn parallelism_omitted_from_emitted_toml_when_default() -> anyhow::Result<()> {
        let toml_str = r#"
plan_id    = "x"
benchmarks = ["b1"]

[[transports]]
client_id   = "t1"
type        = "local"
binary_path = "/bin"
work_dir    = "/tmp"
shell       = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }]
"#;
        let plan = Plan::parse(toml_str)?;
        let emitted = toml::to_string(&plan)?;
        assert!(!emitted.contains("parallelism"), "got: {emitted}");
        Ok(())
    }
}
