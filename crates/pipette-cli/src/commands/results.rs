//! `results` — runtime-agnostic; spans every runtime. Each record carries a
//! `runtime_descriptor` (canonical JSON of `pipette_plan_types::Runtime`); the
//! `results list` renders it as a compact `cli_ref` in its RUNTIME column
//! (falling back to the raw descriptor if it can't be parsed). Generic over any
//! workspace [`ResultsStore`].

use anyhow::Context;
use clap::{Args, Subcommand};
use tabled::{settings::Style, Tabled};

use pipette_plan_types::BenchmarkType;
use pipette_plan_types::Runtime;

use crate::benchmarks::SourcedBenchmarkId;
use crate::results::{BenchmarkResultLocation, BenchmarkResultState, ResultsStore};

/// Every location a result can live in. A result occupies exactly one at a time
/// (submit/sync `fs::rename` it forward), so `find_result`'s first-match scan is
/// unambiguous regardless of order.
const LOCATIONS: [BenchmarkResultLocation; 3] = [
    BenchmarkResultLocation::Local,
    BenchmarkResultLocation::RemotePending,
    BenchmarkResultLocation::RemoteSynced,
];

const SHOW_LONG_ABOUT: &str = "\
Show the stored files for one result.

The command prints payload.json and, when present, extras.json and
metrics.json.";

/// Render a stored `runtime_descriptor` as a compact CLI ref for the RUNTIME
/// column. The descriptor is canonical [`Runtime`] JSON; parse it and show
/// [`Runtime::cli_ref`] (e.g. `b10094:macos-arm64`), falling back to the raw
/// descriptor when it doesn't deserialize — a listing must never fail on one
/// odd record.
fn runtime_column(descriptor: Option<String>) -> String {
    let Some(descriptor) = descriptor else {
        return String::new();
    };
    serde_json::from_str::<Runtime>(&descriptor)
        .map(|runtime| runtime.cli_ref())
        .unwrap_or(descriptor)
}

/// Render `rows` as a psql-style table, or print `empty_message` when there are
/// none — so `list` handles the empty case the same way across CLIs.
fn print_table_or<T: Tabled>(rows: &[T], empty_message: &str) {
    if rows.is_empty() {
        println!("{empty_message}");
    } else {
        println!("{}", tabled::Table::new(rows).with(Style::psql()));
    }
}

/// List, inspect, and delete benchmark results
#[derive(Args, Debug)]
pub struct ResultsArgs {
    #[command(subcommand)]
    pub command: ResultsCommand,
}

#[derive(Subcommand, Debug)]
pub enum ResultsCommand {
    /// List stored benchmark results
    List(ResultsListArgs),
    /// Show the full result record
    #[command(long_about = SHOW_LONG_ABOUT)]
    Show(ResultShowArgs),
    /// Delete a stored result
    Delete(ResultDeleteArgs),
}

impl ResultsArgs {
    pub fn execute(self, results: &ResultsStore) -> anyhow::Result<()> {
        self.command.execute(results)
    }
}

impl ResultsCommand {
    pub fn execute(self, results: &ResultsStore) -> anyhow::Result<()> {
        match self {
            ResultsCommand::List(args) => args.execute(results),
            ResultsCommand::Show(args) => args.execute(results),
            ResultsCommand::Delete(args) => args.execute(results),
        }
    }
}

#[derive(Tabled)]
pub struct ResultRow {
    #[tabled(rename = "RESULT")]
    pub result_id: String,
    #[tabled(rename = "STATE")]
    pub state: String,
    #[tabled(rename = "BENCHMARK")]
    pub benchmark: String,
    #[tabled(rename = "RUNTIME")]
    pub runtime: String,
    #[tabled(rename = "CREATED")]
    pub created_at: String,
}

#[derive(Args, Debug)]
pub struct ResultsListArgs {
    /// Filter results by benchmark reference or ID
    #[arg(long)]
    pub benchmark: Option<SourcedBenchmarkId>,

    /// Filter by benchmark type: prefill-throughput, decode-throughput,
    /// end-to-end-latency, max-memory-usage, eval, vl-throughput (the
    /// underscore spellings are accepted too)
    #[arg(long = "type")]
    pub benchmark_type: Option<BenchmarkType>,

    /// Filter by result state: local, submitted, or scored
    #[arg(long)]
    pub state: Option<BenchmarkResultState>,

    /// Maximum number of results to show
    #[arg(long)]
    pub limit: Option<usize>,
}

impl ResultsListArgs {
    fn rows(&self, results: &ResultsStore) -> anyhow::Result<Vec<ResultRow>> {
        let wanted = self.benchmark.as_ref();
        let mut all = LOCATIONS
            .into_iter()
            .map(|location| {
                results
                    .list_ids(location)?
                    .into_iter()
                    .map(|id| {
                        results
                            .load_list_entry(location, &id)
                            .map_err(anyhow::Error::from)
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .flatten()
            .filter(|entry| {
                wanted.is_none_or(|w| {
                    entry
                        .benchmark_ref
                        .parse::<SourcedBenchmarkId>()
                        .map(|c| *w == c)
                        .unwrap_or(false)
                })
            })
            .filter(|entry| {
                self.benchmark_type
                    .is_none_or(|filter| entry.benchmark_type == filter)
            })
            .filter(|entry| self.state.is_none_or(|state| entry.state == state))
            .collect::<Vec<_>>();
        all.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        if let Some(limit) = self.limit {
            all.truncate(limit);
        }
        Ok(all
            .into_iter()
            .map(|entry| ResultRow {
                result_id: entry.result_id,
                state: entry.state.to_string(),
                benchmark: entry.benchmark_id.unwrap_or(entry.benchmark_ref),
                runtime: runtime_column(entry.runtime_descriptor),
                created_at: entry.created_at,
            })
            .collect())
    }

    pub fn execute(self, results: &ResultsStore) -> anyhow::Result<()> {
        print_table_or(&self.rows(results)?, "No results.");
        Ok(())
    }
}

#[derive(Args, Debug)]
pub struct ResultShowArgs {
    /// Result ID to show
    pub result_id: String,
}

impl ResultShowArgs {
    pub fn execute(self, results: &ResultsStore) -> anyhow::Result<()> {
        let (location, _) = find_result(results, &self.result_id)?;
        let payload_path = results.payload_path(location, &self.result_id);
        let payload = std::fs::read_to_string(&payload_path)
            .with_context(|| format!("failed to read {}", payload_path.display()))?;
        println!("{payload}");
        let extras_path = results.extras_path(location, &self.result_id);
        if extras_path.exists() {
            let extras = std::fs::read_to_string(&extras_path)
                .with_context(|| format!("failed to read {}", extras_path.display()))?;
            eprintln!("--- extras ---");
            println!("{extras}");
        }
        let metrics_path = results.metrics_path(location, &self.result_id);
        if metrics_path.exists() {
            let metrics = std::fs::read_to_string(&metrics_path)
                .with_context(|| format!("failed to read {}", metrics_path.display()))?;
            eprintln!("--- metrics ---");
            println!("{metrics}");
        }
        Ok(())
    }
}

#[derive(Args, Debug)]
pub struct ResultDeleteArgs {
    /// Result ID to delete
    pub result_id: String,
}

impl ResultDeleteArgs {
    pub fn execute(self, results: &ResultsStore) -> anyhow::Result<()> {
        let (location, dir) = find_result(results, &self.result_id)?;
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("failed to delete {}", dir.display()))?;
        println!("deleted {} ({})", self.result_id, location.label());
        Ok(())
    }
}

fn find_result(
    results: &ResultsStore,
    result_id: &str,
) -> anyhow::Result<(BenchmarkResultLocation, std::path::PathBuf)> {
    LOCATIONS
        .into_iter()
        .map(|location| (location, results.result_dir(location, result_id)))
        .find(|(_, dir)| dir.exists())
        .with_context(|| format!("result {result_id} not found"))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn store(_prefix: &str) -> anyhow::Result<(tempfile::TempDir, ResultsStore)> {
        let tmp = tempfile::tempdir()?;
        let s = ResultsStore::new(tmp.path().join("results"));
        Ok((tmp, s))
    }

    const RUNTIME_DESCRIPTOR: &str = r#"{"type":"llamacpp_cli_stock_tools","source":"github_release","repository_url":"github.com/ggml-org/llama.cpp","repository_version":"b10094","flavor":"macos-arm64"}"#;

    fn seed_payload(
        results: &ResultsStore,
        location: BenchmarkResultLocation,
        result_id: &str,
        runtime_descriptor: &str,
    ) -> anyhow::Result<()> {
        results
            .save_payload(
                location,
                result_id,
                &json!({
                    "benchmark_id": "prefill_throughput_smoke",
                    "runtime_descriptor": runtime_descriptor,
                    "submitted_at": "2026-03-26T00:00:00Z",
                    "prefill_time_ms": 12.5
                }),
            )
            .context("failed to seed payload")
    }

    fn list_args() -> ResultsListArgs {
        ResultsListArgs {
            benchmark: None,
            benchmark_type: None,
            state: None,
            limit: None,
        }
    }

    #[test]
    fn list_renders_runtime_as_a_compact_ref() -> anyhow::Result<()> {
        let (_tmp, results) = store("runtime-column")?;
        seed_payload(
            &results,
            BenchmarkResultLocation::Local,
            "job-1",
            RUNTIME_DESCRIPTOR,
        )?;
        let rows = list_args().rows(&results)?;
        let row = rows
            .iter()
            .find(|r| r.result_id == "job-1")
            .context("seeded result should list")?;
        assert_eq!(row.runtime, "b10094:macos-arm64");
        Ok(())
    }

    #[test]
    fn runtime_column_falls_back_to_the_raw_descriptor() {
        assert_eq!(
            runtime_column(Some("{not a runtime}".to_string())),
            "{not a runtime}"
        );
        assert_eq!(runtime_column(None), "");
    }

    #[test]
    fn list_filters_by_state() -> anyhow::Result<()> {
        let (_tmp, results) = store("state-filter")?;
        seed_payload(
            &results,
            BenchmarkResultLocation::Local,
            "job-1",
            RUNTIME_DESCRIPTOR,
        )?;
        let rows = ResultsListArgs {
            state: Some(BenchmarkResultState::Scored),
            ..list_args()
        }
        .rows(&results)?;
        assert!(rows.is_empty(), "local result is not scored");
        Ok(())
    }

    #[test]
    fn list_errors_on_corrupt_payload() -> anyhow::Result<()> {
        let (_tmp, results) = store("corrupt")?;
        let result_id = "broken-result";
        let dir = results.result_dir(BenchmarkResultLocation::Local, result_id);
        std::fs::create_dir_all(&dir).context("failed to create result dir")?;
        std::fs::write(
            results.payload_path(BenchmarkResultLocation::Local, result_id),
            "{bad json\n",
        )
        .context("failed to seed broken payload")?;
        let err = list_args()
            .rows(&results)
            .err()
            .context("corrupt payload should fail listing")?;
        assert!(format!("{err:#}").contains("failed to parse"));
        Ok(())
    }

    #[test]
    fn delete_removes_result_dir() -> anyhow::Result<()> {
        let (_tmp, results) = store("delete")?;
        seed_payload(
            &results,
            BenchmarkResultLocation::Local,
            "job-1",
            RUNTIME_DESCRIPTOR,
        )?;
        let dir = results.result_dir(BenchmarkResultLocation::Local, "job-1");
        assert!(dir.exists());
        ResultDeleteArgs {
            result_id: "job-1".to_string(),
        }
        .execute(&results)?;
        assert!(!dir.exists(), "result dir should be gone after delete");
        Ok(())
    }
}
