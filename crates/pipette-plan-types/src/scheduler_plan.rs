//! The scheduler-mode [`SchedulerPlan`]: a plan whose expansion is ingested by
//! the pipette-mgmt scheduler, which leases jobs to registered clients by
//! capability matching. It is a separate format from the local-dispatch
//! [`Plan`](crate::Plan): no `transports`, `plan_id`, or `retry`, and
//! eligibility in place of transport routing. Because each format requires
//! top-level fields the other lacks, a complete document authored for one will
//! not parse as the other — which is what keeps a mis-selected parser a loud
//! failure rather than a silent misread.
//!
//! Instead of routing each variant to declared transports, a scheduler variant
//! declares **eligibility**: a set of canonical capability flags in `requires`
//! and/or an explicit `clients` allowlist of management-server client ids
//! (`ev1_…`). Expansion reuses the local-dispatch matrix machinery — the same
//! [`is_compatible`](crate::is_compatible) pairing, orphan detection, and
//! per-cell flag resolution — and attaches the variant's [`Eligibility`] to
//! each produced [`SchedulerCell`].
//!
//! This module owns the schema and its **static** validation only. Two checks
//! live elsewhere: the capability-requirement *rules* (`any_of` injection,
//! contradictions) are a later `pipette-plan` validation pass (PIP-413), and
//! benchmark-catalog membership is resolved server-side at ingestion.

use std::{collections::HashSet, fs, path::Path};

use anyhow::Context;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::{
    plan::{
        benchmark_type_of, resolve_benchmark_flags, resolve_benchmarks, resolve_model_flags,
        resolve_runtime_flags, runnable_pairs_of, validate_variant_flag_cells,
    },
    BenchmarkFlags, BenchmarkId, BenchmarkType, CapabilityFlag, ClientId, Model, ModelFlags,
    NonEmptyVec, Runtime, RuntimeFlags, VariantCompatibilityError,
};

// ---------------------------------------------------------------------------
// Eligibility
// ---------------------------------------------------------------------------

/// Who a cell may run on. A client is eligible if it is listed in `clients`
/// **or** its effective capability set is a superset of `requires` (the server
/// evaluates the union; this crate only carries the declaration). At least one
/// of the two is non-empty — a variant with neither is rejected at validation.
///
/// Both sets are stored **sorted and deduplicated** (`Eligibility::new`), so
/// eligibility is compared as a set: two variants that list the same flags in a
/// different order — or with a repeat — yield an equal `Eligibility`, and thus
/// identical cells collapse in [`SchedulerPlan::runnable_cells`] instead of
/// minting duplicate jobs. Fields are private to keep that normal form; read
/// them through [`requires`](Self::requires) / [`clients`](Self::clients).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Eligibility {
    requires: Vec<CapabilityFlag>,
    clients: Vec<ClientId>,
}

impl Eligibility {
    /// Build a normalized eligibility: each list sorted and deduplicated, so
    /// set-equal inputs produce equal (and equally-hashing) values.
    fn new(mut requires: Vec<CapabilityFlag>, mut clients: Vec<ClientId>) -> Self {
        requires.sort_unstable_by(|a, b| a.as_ref().cmp(b.as_ref()));
        requires.dedup();
        clients.sort_unstable_by(|a, b| a.as_ref().cmp(b.as_ref()));
        clients.dedup();
        Self { requires, clients }
    }

    /// Canonical capability flags a client must all satisfy (sorted, deduped).
    /// Empty means eligibility is by the `clients` allowlist alone.
    pub fn requires(&self) -> &[CapabilityFlag] {
        &self.requires
    }

    /// Management-server client ids (`ev1_…`) allowlisted directly, eligible
    /// regardless of `requires` (sorted, deduped). Empty means `requires`-only.
    pub fn clients(&self) -> &[ClientId] {
        &self.clients
    }
}

// ---------------------------------------------------------------------------
// SchedulerVariant
// ---------------------------------------------------------------------------

/// One eligibility-carrying block of a scheduler-mode plan: a sub-matrix of
/// benchmarks × models × runtimes paired with the [`Eligibility`] for those
/// runs. Expands to the cross-product of compatible cells, each stamped with
/// this variant's eligibility.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerVariant {
    pub models: NonEmptyVec<Model>,
    pub runtimes: NonEmptyVec<Runtime>,
    /// Canonical capability flags every eligible client must satisfy. Optional
    /// on its own, but a variant must set at least one of `requires`/`clients`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub requires: Vec<CapabilityFlag>,
    /// Explicit management-server client allowlist (`ev1_…`). Optional on its
    /// own; see `requires`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub clients: Vec<ClientId>,
    /// Per-variant benchmark override; falls back to the plan-level list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmarks: Option<NonEmptyVec<BenchmarkId>>,
    /// Typed per-cell runtime flags — same shape and validation as the
    /// local-dispatch variant (see [`Variant`](crate::Variant)).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runtime_flags: Vec<RuntimeFlags>,
    /// Typed per-cell model-generation flags (eval-only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub model_flags: Vec<ModelFlags>,
    /// Typed per-cell eval-run flags.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub benchmark_flags: Vec<BenchmarkFlags>,
}

impl SchedulerVariant {
    /// This variant's declared eligibility, normalized and cloned for stamping
    /// onto every [`SchedulerCell`] it produces.
    pub fn eligibility(&self) -> Eligibility {
        Eligibility::new(self.requires.clone(), self.clients.clone())
    }

    /// Whether this variant declares any way to be eligible. A variant with
    /// neither `requires` nor `clients` matches nobody and is an authoring
    /// mistake, rejected at validation.
    fn declares_eligibility(&self) -> bool {
        !self.requires.is_empty() || !self.clients.is_empty()
    }

    /// The compatible (model, runtime) pairs this variant expands into; errors
    /// on an orphan. Shares the local-dispatch pairing logic.
    fn runnable_pairs(&self) -> anyhow::Result<Vec<(&Model, &Runtime)>, VariantCompatibilityError> {
        runnable_pairs_of(&self.models, &self.runtimes)
    }
}

// ---------------------------------------------------------------------------
// SchedulerPlan
// ---------------------------------------------------------------------------

/// A validated scheduler-mode plan ready for cell expansion.
///
/// `SchedulerPlan` guarantees, for every variant: at least one of
/// `requires`/`clients` is set, no orphan model or runtime, a resolvable
/// benchmark list, and well-formed per-cell flags. Capability flags are
/// canonical by construction ([`CapabilityFlag`]).
///
/// The only construction paths are [`SchedulerPlan::parse`],
/// [`SchedulerPlan::load`], or `Deserialize` (which routes through
/// `SchedulerPlanIntake`); all run validation. `#[non_exhaustive]` blocks
/// external struct-literal construction so the invariants can be trusted.
#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields, try_from = "SchedulerPlanIntake")]
#[non_exhaustive]
pub struct SchedulerPlan {
    /// Optional plan-level expiry (RFC 3339 / ISO 8601), stamped on every job
    /// at generation. Absent means the jobs never expire on their own.
    #[serde(
        with = "time::serde::rfc3339::option",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub expires_at: Option<OffsetDateTime>,
    /// Optional default benchmark list; a variant without its own `benchmarks`
    /// inherits this. When absent, every variant must set its own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmarks: Option<NonEmptyVec<BenchmarkId>>,
    pub variants: NonEmptyVec<SchedulerVariant>,
}

/// Private serde intake: same fields as [`SchedulerPlan`], no validation. The
/// `try_from` attribute forces every deserialization through `TryFrom`, which
/// validates before yielding a `SchedulerPlan`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SchedulerPlanIntake {
    #[serde(deserialize_with = "deserialize_opt_expires_at", default)]
    expires_at: Option<OffsetDateTime>,
    #[serde(default)]
    benchmarks: Option<NonEmptyVec<BenchmarkId>>,
    variants: NonEmptyVec<SchedulerVariant>,
}

/// Deserialize `expires_at` from either a quoted RFC 3339 string
/// (`"2026-08-01T00:00:00Z"`) or a native TOML datetime (the idiomatic
/// unquoted `2026-08-01T00:00:00Z`). Both resolve to the same instant; a naive
/// local datetime with no offset is rejected, since an expiry needs a definite
/// instant.
fn deserialize_opt_expires_at<'de, D>(deserializer: D) -> Result<Option<OffsetDateTime>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum ExpiresAt {
        Str(String),
        Native(toml::value::Datetime),
    }

    let text = match Option::<ExpiresAt>::deserialize(deserializer)? {
        None => return Ok(None),
        Some(ExpiresAt::Str(s)) => s,
        Some(ExpiresAt::Native(dt)) => dt.to_string(),
    };
    OffsetDateTime::parse(&text, &time::format_description::well_known::Rfc3339)
        .map(Some)
        .map_err(serde::de::Error::custom)
}

impl TryFrom<SchedulerPlanIntake> for SchedulerPlan {
    type Error = anyhow::Error;

    fn try_from(intake: SchedulerPlanIntake) -> anyhow::Result<Self> {
        let plan = SchedulerPlan {
            expires_at: intake.expires_at,
            benchmarks: intake.benchmarks,
            variants: intake.variants,
        };
        validate_scheduler_plan(&plan)?;
        Ok(plan)
    }
}

impl SchedulerPlan {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let raw =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&raw).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn parse(toml_str: &str) -> anyhow::Result<Self> {
        toml::from_str(toml_str).map_err(anyhow::Error::from)
    }

    /// Materialize the deduped set of cells this plan expands to: every
    /// (variant × benchmark × model × runtime), filtered by model/runtime
    /// compatibility, each carrying its variant's [`Eligibility`] and resolved
    /// per-cell flags. Errors on an orphan model or runtime, matching the
    /// error surfaced at validation.
    pub fn runnable_cells(&self) -> anyhow::Result<HashSet<SchedulerCell>> {
        self.variants
            .iter()
            .enumerate()
            .map(|(idx, variant)| {
                let pairs = variant
                    .runnable_pairs()
                    .map_err(|e| anyhow::anyhow!("variant {idx}: {e}"))?;
                let benchmarks = self.benchmarks_for_variant(idx, variant)?;
                let eligibility = variant.eligibility();
                Ok(benchmarks
                    .iter()
                    .flat_map(|benchmark| {
                        let eligibility = eligibility.clone();
                        let bt = benchmark_type_of(benchmark);
                        pairs.iter().map(move |(model, runtime)| SchedulerCell {
                            benchmark: benchmark.clone(),
                            model: (*model).clone(),
                            runtime: (*runtime).clone(),
                            eligibility: eligibility.clone(),
                            runtime_flags: resolve_runtime_flags(
                                &variant.runtime_flags,
                                bt,
                                runtime,
                                model,
                            ),
                            model_flags: resolve_model_flags(&variant.model_flags, bt, model),
                            benchmark_flags: resolve_benchmark_flags(
                                &variant.benchmark_flags,
                                bt,
                                runtime,
                                model,
                            ),
                        })
                    })
                    .collect::<HashSet<_>>())
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map(|sets| sets.into_iter().flatten().collect())
    }

    fn benchmarks_for_variant<'a>(
        &'a self,
        variant_idx: usize,
        variant: &'a SchedulerVariant,
    ) -> anyhow::Result<&'a [BenchmarkId]> {
        resolve_benchmarks(
            variant_idx,
            variant.benchmarks.as_deref(),
            self.benchmarks.as_deref(),
        )
    }
}

fn validate_scheduler_plan(plan: &SchedulerPlan) -> anyhow::Result<()> {
    plan.variants
        .iter()
        .enumerate()
        .try_for_each(|(idx, variant)| {
            if !variant.declares_eligibility() {
                anyhow::bail!(
                    "variant {idx}: must declare at least one of `requires` or `clients` \
                     (a variant with neither matches no client)"
                );
            }
            // Orphan / structural compatibility — surfaced up front, and the
            // same check `runnable_cells` performs during expansion.
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

// ---------------------------------------------------------------------------
// SchedulerCell
// ---------------------------------------------------------------------------

/// One expanded unit of scheduler-mode work: a single benchmark × model ×
/// runtime pulled from one variant, eligible per [`Eligibility`]. The owned,
/// deduped analogue of the local-dispatch
/// [`RunnableCell`](crate::RunnableCell) — but it carries eligibility instead
/// of a transport allowlist and knows nothing about argv construction (turning
/// a cell into a job body is `pipette-plan`'s `generate`).
///
/// Cell identity is the full tuple including `eligibility` and resolved flags:
/// two variants producing the same benchmark/model/runtime but different
/// eligibility are distinct cells; identical ones deduplicate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SchedulerCell {
    pub benchmark: BenchmarkId,
    pub model: Model,
    pub runtime: Runtime,
    pub eligibility: Eligibility,
    pub runtime_flags: Option<RuntimeFlags>,
    pub model_flags: Option<ModelFlags>,
    pub benchmark_flags: Option<BenchmarkFlags>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two representative plans from the design doc (`plan-ingestion.md`
    /// §4): an iOS Apple-Foundation variant pinned to a client *and* requiring
    /// `os:ios`, and a macOS MLX variant that is `requires`-only. Top-level
    /// benchmarks with a per-variant override on the AFM block.
    fn representative_plan() -> String {
        r#"
expires_at = "2026-08-01T00:00:00Z"
benchmarks = ["decode_throughput_512_100", "end_to_end_latency_512_256"]

[[variants]]
requires   = ["os:ios"]
clients    = ["ev1_9f2c"]
models     = [{ type = "apple_foundation_text" }]
runtimes   = [{ type = "apple_foundation" }]
benchmarks = ["decode_throughput_512_100"]

[[variants]]
requires = ["os:macos"]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "LFM2.5-350M-MLX-4bit" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.20.0", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" } }]
"#
        .to_owned()
    }

    #[test]
    fn representative_plan_expands_to_expected_cells() -> anyhow::Result<()> {
        let plan = SchedulerPlan::parse(&representative_plan())?;

        // expires_at parses to the RFC-3339 instant.
        let expected = OffsetDateTime::parse(
            "2026-08-01T00:00:00Z",
            &time::format_description::well_known::Rfc3339,
        )?;
        assert_eq!(plan.expires_at, Some(expected));

        // variant 1: 1 benchmark × 1 model × 1 runtime = 1
        // variant 2: 2 benchmarks × 1 model × 1 runtime = 2
        let cells = plan.runnable_cells()?;
        assert_eq!(cells.len(), 3, "expected 3 cells, got {}", cells.len());

        // The AFM cell carries both the client pin and the requires flag.
        let afm = cells
            .iter()
            .find(|c| matches!(c.model, Model::AppleFoundationText))
            .context("AFM cell present")?;
        assert_eq!(afm.eligibility.requires().len(), 1);
        assert_eq!(afm.eligibility.requires()[0].as_ref(), "os:ios");
        assert_eq!(afm.eligibility.clients().len(), 1);
        assert_eq!(afm.eligibility.clients()[0].as_ref(), "ev1_9f2c");
        assert_eq!(afm.benchmark.as_ref(), "decode_throughput_512_100");

        // The MLX cells are requires-only across both top-level benchmarks.
        let mlx: Vec<_> = cells
            .iter()
            .filter(|c| matches!(c.model, Model::Mlx(_)))
            .collect();
        assert_eq!(mlx.len(), 2);
        assert!(mlx.iter().all(|c| c.eligibility.clients().is_empty()
            && c.eligibility
                .requires()
                .iter()
                .any(|f| f.as_ref() == "os:macos")));
        Ok(())
    }

    #[test]
    fn local_dispatch_parser_rejects_a_scheduler_plan() -> anyhow::Result<()> {
        // The other direction of the format split: a scheduler document has no
        // `plan_id` and no `transports`, so the local-dispatch parser rejects
        // it rather than silently misreading eligibility as transport routing.
        // Assert a schema tell (a scheduler-only field, or the missing
        // local-required one) so this can't pass for an incidental reason.
        let err = crate::Plan::parse(&representative_plan())
            .err()
            .context("expected the local-dispatch parser to reject a scheduler plan")?;
        let msg = err.to_string();
        assert!(
            msg.contains("expires_at") || msg.contains("plan_id"),
            "unexpected error: {err}"
        );
        Ok(())
    }

    #[test]
    fn set_equal_eligibility_dedups_to_one_cell() -> anyhow::Result<()> {
        // Two variants expanding to the same benchmark×model×runtime cell, with
        // `requires` flags in a different order and one duplicated. Eligibility
        // is a set, so these collapse to a single cell rather than two jobs.
        let toml = r#"
benchmarks = ["decode_throughput_512_100"]

[[variants]]
requires = ["os:macos", "arch:arm64"]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "r" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.20.0", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" } }]

[[variants]]
requires = ["arch:arm64", "os:macos", "os:macos"]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "r" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.20.0", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" } }]
"#;
        let cells = SchedulerPlan::parse(toml)?.runnable_cells()?;
        assert_eq!(cells.len(), 1, "set-equal eligibility should dedup");
        Ok(())
    }

    #[test]
    fn expires_at_accepts_native_toml_datetime() -> anyhow::Result<()> {
        // The idiomatic unquoted TOML datetime, and its quoted-string form,
        // must parse to the same instant.
        let expected = OffsetDateTime::parse(
            "2026-08-01T00:00:00Z",
            &time::format_description::well_known::Rfc3339,
        )?;
        let native = r#"
expires_at = 2026-08-01T00:00:00Z
benchmarks = ["decode_throughput_512_100"]

[[variants]]
requires = ["os:macos"]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "r" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.20.0", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" } }]
"#;
        assert_eq!(SchedulerPlan::parse(native)?.expires_at, Some(expected));
        Ok(())
    }

    #[test]
    fn round_trips_through_toml() -> anyhow::Result<()> {
        let plan = SchedulerPlan::parse(&representative_plan())?;
        let reserialized = toml::to_string(&plan)?;
        let reparsed = SchedulerPlan::parse(&reserialized)?;
        assert_eq!(plan, reparsed);
        Ok(())
    }

    #[test]
    fn requires_only_variant_is_accepted() -> anyhow::Result<()> {
        let toml = r#"
benchmarks = ["decode_throughput_512_100"]

[[variants]]
requires = ["os:macos"]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "r" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.20.0", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" } }]
"#;
        SchedulerPlan::parse(toml)?;
        Ok(())
    }

    #[test]
    fn clients_only_variant_is_accepted() -> anyhow::Result<()> {
        let toml = r#"
benchmarks = ["decode_throughput_512_100"]

[[variants]]
clients  = ["ev1_abc"]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "r" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.20.0", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" } }]
"#;
        SchedulerPlan::parse(toml)?;
        Ok(())
    }

    /// Every static-validation rejection, one case per rule. Each asserts a
    /// distinguishing substring of the error, so a case can't pass because the
    /// TOML happened to fail for an unrelated reason (e.g. a runtime-table
    /// schema change breaking the parse before the rule under test is reached).
    #[rstest::rstest]
    // A variant declaring neither `requires` nor `clients`.
    #[case::no_eligibility(
        r#"benchmarks = ["decode_throughput_512_100"]
[[variants]]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "r" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.20.0", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" } }]"#,
        &["requires", "clients"]
    )]
    // A capability flag not in canonical form (uppercase).
    #[case::non_canonical_flag(
        r#"benchmarks = ["decode_throughput_512_100"]
[[variants]]
requires = ["os:iOS"]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "r" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.20.0", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" } }]"#,
        &["CapabilityFlag"]
    )]
    // Model and runtime that can't pair — both orphaned.
    #[case::incompatible_model_runtime(
        r#"benchmarks = ["decode_throughput_512_100"]
[[variants]]
requires = ["os:macos"]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "r" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b5000", flavor = "macos-arm64" }]"#,
        &["no compatible runtime"]
    )]
    // A compatible pair plus an orphan runtime no model can serve.
    #[case::orphan_runtime(
        r#"benchmarks = ["decode_throughput_512_100"]
[[variants]]
requires = ["os:macos"]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "r" }]
runtimes = [
  { type = "mlx_macos_pipette", version = "0.20.0", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" } },
  { type = "llamacpp_cli_stock_tools", source = "github_release", version = "b5000", flavor = "macos-arm64" },
]"#,
        &["no compatible model"]
    )]
    // A local-dispatch-only key: the distinct formats reject each other's docs.
    #[case::transports_key(
        r#"benchmarks = ["decode_throughput_512_100"]
[[transports]]
client_id   = "mac-1"
type        = "local"
binary_path = "/bin/pipette"
work_dir    = "/tmp"
[[variants]]
requires = ["os:macos"]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "r" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.20.0", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" } }]"#,
        &["unknown field `transports`"]
    )]
    // Benchmarks set neither at the plan root nor on the variant.
    #[case::missing_benchmarks(
        r#"[[variants]]
requires = ["os:macos"]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "r" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.20.0", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" } }]"#,
        &["benchmarks must be set"]
    )]
    fn rejects_invalid_scheduler_plan(
        #[case] toml: &str,
        #[case] expect_substrings: &[&str],
    ) -> anyhow::Result<()> {
        let err = SchedulerPlan::parse(toml)
            .err()
            .context("expected the plan to be rejected")?;
        let msg = err.to_string();
        for needle in expect_substrings {
            assert!(msg.contains(needle), "error missing {needle:?}: {msg}");
        }
        Ok(())
    }

    /// Every shipped scheduler-mode example (`examples/plans/scheduler/*.toml`).
    /// They live in their own directory because the local-dispatch examples
    /// beside them are a different format, and each set is parsed by the
    /// parser that owns it.
    fn scheduler_example_paths() -> anyhow::Result<Vec<std::path::PathBuf>> {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/plans/scheduler"
        );
        let paths = std::fs::read_dir(dir)?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<Vec<_>>>()?
            .into_iter()
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("toml"))
            .collect::<Vec<_>>();
        assert!(!paths.is_empty(), "no scheduler example plans in {dir}");
        Ok(paths)
    }

    /// The examples are the reference authoring surface, so parse and expand
    /// each one: parsing runs full validation through the `try_from` boundary,
    /// and expanding resolves per-cell flags, catching an authored flag entry
    /// that names no real cell or a plan that expands to nothing.
    #[test]
    fn shipped_scheduler_examples_expand() -> anyhow::Result<()> {
        scheduler_example_paths()?.iter().try_for_each(|path| {
            let toml = std::fs::read_to_string(path)?;
            let plan = SchedulerPlan::parse(&toml)
                .map_err(|e| anyhow::anyhow!("{}: {e:#}", path.display()))?;
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

    /// Each format requires top-level fields the other lacks, which is what
    /// makes a mis-selected parser a loud failure instead of a silent misread.
    /// Asserting it on the shipped examples keeps that true of real documents,
    /// not just of hand-written fragments.
    #[test]
    fn a_scheduler_example_does_not_parse_as_a_local_dispatch_plan() -> anyhow::Result<()> {
        scheduler_example_paths()?.iter().try_for_each(|path| {
            let toml = std::fs::read_to_string(path)?;
            anyhow::ensure!(
                crate::Plan::parse(&toml).is_err(),
                "{} parsed as a local-dispatch plan",
                path.display()
            );
            anyhow::Ok(())
        })
    }
}
