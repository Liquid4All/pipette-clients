//! `generate` (PIP-414): expand a scheduler-mode [`SchedulerPlan`] into the
//! directory of job files `pipette-mgmt plans ingest` consumes (see
//! `pipette-mgmt` `docs/plan-ingestion.md` §7).
//!
//! The handoff is deliberately identity-free: the `plan_id` and every `job_id`
//! are minted by the server at ingestion, so [`JobBody`] has no field for
//! either. Ingestion's "already carries a `job_id` / `plan_id`" rejection is
//! therefore unreachable from here by construction rather than by a check.
//!
//! A body serves two readers at once. The server reads `benchmark_id`,
//! `requires` / `any_of` / `clients`, and `expires_at`; it echoes
//! `model_descriptor` / `runtime_descriptor` into synthetic failure records
//! without interpreting them, and passes everything else — `spec` — through
//! untouched. The client reads `spec` alone: a [`ClientRunSpec`], the shape
//! `pipette-cli` already types a claim into, so a generated job and a job the
//! desktop CLI runs directly are the same contract.
//!
//! Generation writes nothing until the whole plan validates: schema and
//! structural checks at [`SchedulerPlan::load`], hardware policy at
//! [`validate_capability_rules`], then every cell resolved. A plan that fails
//! leaves no half-written directory for a later ingest to pick up.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use serde::Serialize;
use tabled::{settings::Style, Tabled};
use time::format_description::well_known::Rfc3339;

use pipette_plan_types::{
    BenchmarkId, CapabilityFlag, ClientId, ClientRunSpec, SchedulerCell, SchedulerPlan,
};

use crate::capability_rules::{resolve_effective_requirement, validate_capability_rules};

// ---------------------------------------------------------------------------
// The job body
// ---------------------------------------------------------------------------

/// One expanded cell in the form `pipette-mgmt` ingests and stores.
///
/// Field order is load-bearing: it is the order `serde_json` emits, and
/// [`expand`] orders the directory by the rendered body, so leading with
/// `benchmark_id` groups the generated files by benchmark for free.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct JobBody {
    /// The one content field the server reads: resolved against its benchmark
    /// catalog at ingestion, so an id this plan invents is caught there.
    pub benchmark_id: BenchmarkId,
    /// The variant's committed flags plus whatever the hardware rules injected.
    pub requires: Vec<CapabilityFlag>,
    /// Always emitted, empty included, so the field's absence never has to
    /// mean `[]`.
    pub any_of: Vec<Vec<CapabilityFlag>>,
    /// Client ids eligible regardless of `requires`.
    pub clients: Vec<ClientId>,
    /// The plan's expiry, stamped on every job it expands to. Absent means the
    /// job never expires on its own.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    /// Model identity as the warehouse carries it — the server copies this
    /// verbatim into a synthetic failure record without parsing it.
    pub model_descriptor: String,
    /// Runtime identity, same treatment as `model_descriptor`.
    pub runtime_descriptor: String,
    /// The cell to run. Opaque to the server, authoritative for the client.
    pub spec: ClientRunSpec,
}

impl JobBody {
    fn from_cell(cell: &SchedulerCell, expires_at: Option<&str>) -> anyhow::Result<Self> {
        let effective = resolve_effective_requirement(cell.eligibility.requires(), &cell.runtime)
            .with_context(|| {
            format!("resolving capability requirements for {}", cell.runtime)
        })?;

        // A plan-authored HuggingFace token is the *local* runner's to forward
        // over its own transport env. A job body is stored by the server and
        // handed to every client that claims it, so it carries the model
        // without one; each client injects its own at run time
        // (`pipette_cli::hf_auth`).
        let model = cell.model.without_auth_token();

        Ok(Self {
            benchmark_id: cell.benchmark.clone(),
            requires: effective.requires,
            any_of: effective.any_of,
            clients: cell.eligibility.clients().to_vec(),
            expires_at: expires_at.map(str::to_owned),
            model_descriptor: serde_json::to_string(&model)?,
            runtime_descriptor: serde_json::to_string(&cell.runtime)?,
            spec: ClientRunSpec {
                benchmark: cell.benchmark.clone(),
                model,
                runtime: cell.runtime.clone(),
                runtime_flags: cell.runtime_flags.clone(),
                model_flags: cell.model_flags.clone(),
                benchmark_flags: cell.benchmark_flags.clone(),
            },
        })
    }
}

/// A job body paired with the file name it is written under.
#[derive(Debug, Clone)]
pub struct JobFile {
    pub name: String,
    pub body: JobBody,
    /// The bytes written, rendered once: also the key [`expand`] sorts on.
    json: String,
}

// ---------------------------------------------------------------------------
// Expansion
// ---------------------------------------------------------------------------

/// Expand `plan` into its job files, validated and deterministically ordered.
///
/// Split from [`generate`] so a caller — a test, or the eventual `submit` that
/// posts the same bodies to `POST /plans` — can take the expansion without a
/// directory.
pub fn expand(plan: &SchedulerPlan) -> anyhow::Result<Vec<JobFile>> {
    validate_capability_rules(plan)?;

    let expires_at = plan
        .expires_at
        .map(|at| at.format(&Rfc3339))
        .transpose()
        .context("formatting the plan's expires_at")?;

    let mut bodies = plan
        .runnable_cells()?
        .iter()
        .map(|cell| {
            let body = JobBody::from_cell(cell, expires_at.as_deref())?;
            let json = serde_json::to_string_pretty(&body)?;
            Ok((json, body))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // Cells expand out of a `HashSet`, so file names are only reproducible if
    // an order is imposed. Sorting on the rendered body needs no separate key
    // that could drift from the fields.
    bodies.sort_by(|(a, _), (b, _)| a.cmp(b));

    // Pad wide enough that the names sort the same way lexically as they were
    // numbered, with the doc's three digits as the floor.
    let width = bodies.len().saturating_sub(1).to_string().len().max(3);
    Ok(bodies
        .into_iter()
        .enumerate()
        .map(|(index, (json, body))| JobFile {
            name: format!("cell-{index:0width$}.json"),
            body,
            json,
        })
        .collect())
}

/// Expand `plan_path` into `out_dir` and report what was written.
pub fn generate(plan_path: &Path, out_dir: &Path) -> anyhow::Result<()> {
    let plan = SchedulerPlan::load(plan_path)?;
    let files = expand(&plan)?;

    prepare_out_dir(out_dir)?;
    files.iter().try_for_each(|file| -> anyhow::Result<()> {
        let path = out_dir.join(&file.name);
        fs::write(&path, format!("{}\n", file.json))
            .with_context(|| format!("writing {}", path.display()))
    })?;

    report(plan_path, out_dir, &files);
    Ok(())
}

/// Create `out_dir` if absent, and refuse one that already holds an expansion.
///
/// At ingestion the directory *is* the manifest — every `*.json` in it becomes
/// a job of the one plan — so a leftover file from an earlier run would join
/// this plan silently, and a stale one that outlived its benchmark would fail
/// the whole ingest.
fn prepare_out_dir(out_dir: &Path) -> anyhow::Result<()> {
    fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;

    let existing = fs::read_dir(out_dir)
        .with_context(|| format!("reading {}", out_dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<PathBuf>>>()
        .with_context(|| format!("reading {}", out_dir.display()))?
        .into_iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .count();

    if existing > 0 {
        anyhow::bail!(
            "{} already holds {existing} .json file(s), and `plans ingest` stages \
             every .json in a directory as one plan; generate into a fresh or \
             emptied directory so this expansion is ingested on its own",
            out_dir.display()
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// Tabled row for one generated job file.
#[derive(Tabled)]
struct JobRow {
    #[tabled(rename = "FILE")]
    file: String,
    #[tabled(rename = "BENCHMARK")]
    benchmark: String,
    #[tabled(rename = "MODEL")]
    model: String,
    #[tabled(rename = "RUNTIME")]
    runtime: String,
    #[tabled(rename = "ELIGIBILITY")]
    eligibility: String,
}

fn report(plan_path: &Path, out_dir: &Path, files: &[JobFile]) {
    let rows: Vec<JobRow> = files
        .iter()
        .map(|file| JobRow {
            file: file.name.clone(),
            benchmark: file.body.benchmark_id.to_string(),
            model: file.body.spec.model.to_string(),
            runtime: file.body.spec.runtime.to_string(),
            eligibility: render_eligibility(&file.body),
        })
        .collect();

    println!("{}", tabled::Table::new(&rows).with(Style::psql()));
    println!();
    println!(
        "wrote {} job file(s) from {} to {}",
        files.len(),
        plan_path.display(),
        out_dir.display()
    );
    println!();
    println!("next: pipette-mgmt plans ingest {}", out_dir.display());
}

/// One cell's eligibility as a few short lines: the flat requirement, each
/// `any_of` group abridged to its first members, then the client allowlist. A
/// device-family group runs to a dozen-plus flags, which would drown the table
/// spelled out — the authoritative set is in the file.
fn render_eligibility(body: &JobBody) -> String {
    const SHOWN: usize = 3;

    body.requires
        .iter()
        .map(ToString::to_string)
        .chain(body.any_of.iter().map(|group| {
            let head = group
                .iter()
                .take(SHOWN)
                .map(AsRef::as_ref)
                .collect::<Vec<&str>>()
                .join(", ");
            match group.len().saturating_sub(SHOWN) {
                0 => format!("one of: {head}"),
                rest => format!("one of: {head}, +{rest} more"),
            }
        }))
        .chain(
            body.clients
                .iter()
                .map(|client| format!("client: {client}")),
        )
        .collect::<Vec<String>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::*;

    /// The two representative variants of `plan-ingestion.md` §4: an iOS
    /// Apple-Foundation block pinned to a client *and* requiring `os:ios`, and
    /// a `requires`-only macOS MLX block. Expands to 3 cells — 1 benchmark on
    /// the first variant, 2 on the second.
    fn representative_plan() -> anyhow::Result<SchedulerPlan> {
        SchedulerPlan::parse(
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
"#,
        )
    }

    fn bodies(plan: &SchedulerPlan) -> anyhow::Result<Vec<Value>> {
        expand(plan)?
            .iter()
            .map(|file| serde_json::from_str(&file.json).map_err(Into::into))
            .collect()
    }

    #[test]
    fn expands_the_matrix_into_one_file_per_cell() -> anyhow::Result<()> {
        let files = expand(&representative_plan()?)?;
        assert_eq!(files.len(), 3, "1 AFM cell + 2 MLX cells");
        assert_eq!(
            files.iter().map(|f| f.name.as_str()).collect::<Vec<_>>(),
            ["cell-000.json", "cell-001.json", "cell-002.json"],
        );
        Ok(())
    }

    /// File names index a sorted order, so re-expanding the same plan assigns
    /// the same body to the same name — an operator can diff two runs.
    #[test]
    fn expansion_is_reproducible() -> anyhow::Result<()> {
        let first = expand(&representative_plan()?)?;
        let second = expand(&representative_plan()?)?;
        assert_eq!(
            first
                .iter()
                .map(|f| (f.name.clone(), f.json.clone()))
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|f| (f.name.clone(), f.json.clone()))
                .collect::<Vec<_>>(),
        );
        Ok(())
    }

    /// A body carrying either field is rejected outright at ingestion.
    #[test]
    fn bodies_carry_no_server_minted_identity() -> anyhow::Result<()> {
        bodies(&representative_plan()?)?
            .iter()
            .try_for_each(|body| -> anyhow::Result<()> {
                let object = body.as_object().context("a job body is a JSON object")?;
                assert!(
                    !object.contains_key("job_id"),
                    "job_id is minted at ingestion"
                );
                assert!(
                    !object.contains_key("plan_id"),
                    "plan_id lives only in the manifest"
                );
                Ok(())
            })
    }

    /// A job with neither is claimable by nobody, and the server deletes it
    /// from `avail/` — the plan schema guarantees one, this holds the line
    /// after the rules have run.
    #[test]
    fn every_body_declares_a_way_to_be_eligible() -> anyhow::Result<()> {
        bodies(&representative_plan()?)?
            .iter()
            .try_for_each(|body| -> anyhow::Result<()> {
                let requires = body["requires"]
                    .as_array()
                    .context("requires is an array")?;
                let clients = body["clients"].as_array().context("clients is an array")?;
                assert!(
                    !requires.is_empty() || !clients.is_empty(),
                    "neither requires nor clients: {body}"
                );
                assert!(body["any_of"].is_array(), "any_of is always emitted");
                Ok(())
            })
    }

    /// The server reads the expiry once, to encode into the `avail/` filename.
    #[test]
    fn the_plan_expiry_is_stamped_on_every_job() -> anyhow::Result<()> {
        bodies(&representative_plan()?)?
            .iter()
            .try_for_each(|body| -> anyhow::Result<()> {
                assert_eq!(body["expires_at"], "2026-08-01T00:00:00Z");
                Ok(())
            })
    }

    /// A plan without `expires_at` omits the key rather than spelling it
    /// `null`: absent is what "never auto-expires" means on the wire.
    #[test]
    fn an_unexpiring_plan_omits_the_field() -> anyhow::Result<()> {
        let plan = SchedulerPlan::parse(
            r#"
benchmarks = ["decode_throughput_512_100"]

[[variants]]
requires = ["os:macos"]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "LFM2.5-350M-MLX-4bit" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.20.0", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" } }]
"#,
        )?;
        bodies(&plan)?
            .iter()
            .try_for_each(|body| -> anyhow::Result<()> {
                let object = body.as_object().context("a job body is a JSON object")?;
                assert!(!object.contains_key("expires_at"), "{body}");
                Ok(())
            })
    }

    /// A claiming client re-parses `spec` as a `ClientRunSpec` and rejects the
    /// job outright if it names a different benchmark than the envelope
    /// (`pipette_cli::client::claim::run_spec_from_claim`). Both come from one
    /// cell here, so this holds that agreement at the writing end.
    #[test]
    fn the_spec_parses_back_and_agrees_with_the_envelope() -> anyhow::Result<()> {
        bodies(&representative_plan()?)?
            .iter()
            .try_for_each(|body| -> anyhow::Result<()> {
                let spec: ClientRunSpec = serde_json::from_value(body["spec"].clone())?;
                assert_eq!(
                    Value::String(spec.benchmark.as_ref().to_owned()),
                    body["benchmark_id"],
                );
                Ok(())
            })
    }

    /// The rules inject the supported-iPhone family for Apple Foundation on
    /// iOS, so the AFM cell requires a device from a list its author never
    /// wrote — the whole point of resolving requirements at generation.
    #[test]
    fn hardware_rules_reach_the_generated_body() -> anyhow::Result<()> {
        let afm = bodies(&representative_plan()?)?
            .into_iter()
            .find(|body| body["requires"].to_string().contains("os:ios"))
            .context("the plan has an iOS variant")?;

        let groups = afm["any_of"].as_array().context("any_of is an array")?;
        let devices = groups
            .iter()
            .find(|group| group.to_string().contains("device:iphone"))
            .context("the iOS AFM rules inject a supported-device family")?;
        assert!(
            devices.as_array().is_some_and(|g| g.len() > 1),
            "a device family is a disjunction, not a pin: {devices}"
        );
        assert_eq!(
            afm["clients"],
            serde_json::json!(["ev1_9f2c"]),
            "the variant's own allowlist survives rule resolution"
        );
        Ok(())
    }

    /// A body is stored by the server and served to every claiming client, so
    /// a token the plan author inlined must not ride along — neither in the
    /// descriptor the warehouse keeps nor in the spec the client runs.
    #[test]
    fn a_plan_authored_hf_token_never_reaches_a_job_body() -> anyhow::Result<()> {
        let plan = SchedulerPlan::parse(
            r#"
benchmarks = ["decode_throughput_512_100"]

[[variants]]
requires = ["os:macos"]
models   = [{ type = "mlx", source = "huggingface", org = "LiquidAI", repo_name = "gated-repo", auth_token = "hf_supersecret" }]
runtimes = [{ type = "mlx_macos_pipette", version = "0.20.0", flavor = "macos-arm64", source = { type = "pip_requirements_text", contents = "mlx-lm==0.20.0" } }]
"#,
        )?;
        expand(&plan)?
            .iter()
            .try_for_each(|file| -> anyhow::Result<()> {
                assert!(
                    !file.json.contains("hf_supersecret"),
                    "the token leaked into {}: {}",
                    file.name,
                    file.json
                );
                Ok(())
            })
    }

    /// Ingestion stages every `.json` in the directory as one plan, so an
    /// expansion may not be written on top of another.
    #[test]
    fn generating_into_a_populated_directory_is_refused() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        fs::write(dir.path().join("cell-000.json"), "{}")?;

        let err = prepare_out_dir(dir.path())
            .err()
            .context("a directory holding job files must be refused")?;
        assert!(
            format!("{err:#}").contains("plans ingest"),
            "the error should say why: {err:#}"
        );
        Ok(())
    }

    /// A non-JSON neighbour (a README, an operator's notes) is ignored by
    /// ingestion, so it is no reason to refuse.
    #[test]
    fn a_non_json_neighbour_does_not_block_generation() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        fs::write(dir.path().join("NOTES.md"), "why this plan exists")?;
        prepare_out_dir(dir.path())
    }

    /// End to end: the files land, and each one round-trips as a job body.
    #[test]
    fn generate_writes_the_expansion_to_disk() -> anyhow::Result<()> {
        let plan_dir = tempfile::tempdir()?;
        let plan_path = plan_dir.path().join("plan.toml");
        fs::write(
            &plan_path,
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
"#,
        )?;

        let out = tempfile::tempdir()?;
        generate(&plan_path, out.path())?;

        let mut written = fs::read_dir(out.path())?
            .map(|entry| Ok(entry?.file_name().to_string_lossy().into_owned()))
            .collect::<anyhow::Result<Vec<String>>>()?;
        written.sort();
        assert_eq!(written, ["cell-000.json", "cell-001.json", "cell-002.json"]);

        let first: Value =
            serde_json::from_str(&fs::read_to_string(out.path().join("cell-000.json"))?)?;
        assert!(first["spec"]["benchmark"].is_string(), "{first}");
        assert!(first["spec"]["model"].is_object(), "{first}");
        assert!(first["spec"]["runtime"].is_object(), "{first}");
        Ok(())
    }

    /// A plan whose hardware policy is contradicted produces no directory at
    /// all — validation runs before the first write.
    #[test]
    fn a_rule_violating_plan_writes_nothing() -> anyhow::Result<()> {
        let plan_dir = tempfile::tempdir()?;
        let plan_path = plan_dir.path().join("plan.toml");
        fs::write(
            &plan_path,
            r#"
benchmarks = ["decode_throughput_512_100"]

[[variants]]
requires = ["os:android"]
models   = [{ type = "apple_foundation_text" }]
runtimes = [{ type = "apple_foundation" }]
"#,
        )?;

        let out = plan_dir.path().join("jobs");
        assert!(
            generate(&plan_path, &out).is_err(),
            "Apple Foundation on os:android is a contradiction"
        );
        assert!(!out.exists(), "nothing may be written for a rejected plan");
        Ok(())
    }
}
