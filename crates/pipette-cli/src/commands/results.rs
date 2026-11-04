//! `results`: list, show and delete stored results, whatever their runtime.

use anyhow::Context;
use clap::{Args, Subcommand};
use tabled::Tabled;

use pipette_plan_types::BenchmarkType;
use pipette_plan_types::Runtime;

use crate::benchmarks::SourcedBenchmarkId;
use crate::commands::write_table_or;
use crate::console::Console;
use crate::results::{BenchmarkResultLocation, BenchmarkResultState, ResultsStore};

/// A result occupies exactly one location at a time (submit and sync rename it
/// forward), so `find_result`'s first-match scan does not depend on order.
const LOCATIONS: [BenchmarkResultLocation; 3] = [
    BenchmarkResultLocation::Local,
    BenchmarkResultLocation::RemotePending,
    BenchmarkResultLocation::RemoteSynced,
];

const SHOW_LONG_ABOUT: &str = "\
Show the stored files for one result.

The command prints payload.json and, when present, extras.json and
metrics.json.";

/// The RUNTIME column: [`Runtime::cli_ref`] (e.g. `b10094:macos-arm64`), or the
/// raw descriptor when it does not parse, so one odd record cannot fail a
/// listing.
fn runtime_column(descriptor: Option<String>) -> String {
    let Some(descriptor) = descriptor else {
        return String::new();
    };
    serde_json::from_str::<Runtime>(&descriptor)
        .map(|runtime| runtime.cli_ref())
        .unwrap_or(descriptor)
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
    pub fn execute(self, results: &ResultsStore, console: &mut dyn Console) -> anyhow::Result<()> {
        self.command.execute(results, console)
    }
}

impl ResultsCommand {
    pub fn execute(self, results: &ResultsStore, console: &mut dyn Console) -> anyhow::Result<()> {
        match self {
            ResultsCommand::List(args) => args.execute(results, console),
            ResultsCommand::Show(args) => args.execute(results, console),
            ResultsCommand::Delete(args) => args.execute(results, console),
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
    /// end-to-end-latency, max-memory-usage, eval, vlm-eval, vl-throughput,
    /// vl-max-memory, vl-spec-decode, asr-throughput, asr-max-memory (the
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

    pub fn execute(self, results: &ResultsStore, console: &mut dyn Console) -> anyhow::Result<()> {
        write_table_or(console.out(), &self.rows(results)?, "No results.")?;
        Ok(())
    }
}

#[derive(Args, Debug)]
pub struct ResultShowArgs {
    /// Result ID to show
    pub result_id: String,
}

impl ResultShowArgs {
    /// Banners go to `err` so a redirect captures the JSON alone.
    pub fn execute(self, results: &ResultsStore, console: &mut dyn Console) -> anyhow::Result<()> {
        let (location, _) = find_result(results, &self.result_id)?;
        let read = |path: std::path::PathBuf| {
            std::fs::read_to_string(&path)
                .with_context(|| format!("failed to read {}", path.display()))
        };
        writeln!(
            console.out(),
            "{}",
            read(results.payload_path(location, &self.result_id))?
        )?;
        [
            ("extras", results.extras_path(location, &self.result_id)),
            ("metrics", results.metrics_path(location, &self.result_id)),
        ]
        .into_iter()
        .filter(|(_, path)| path.exists())
        .try_for_each(|(banner, path)| {
            writeln!(console.err(), "--- {banner} ---")?;
            writeln!(console.out(), "{}", read(path)?)?;
            anyhow::Ok(())
        })
    }
}

#[derive(Args, Debug)]
pub struct ResultDeleteArgs {
    /// Result ID to delete
    pub result_id: String,
}

impl ResultDeleteArgs {
    pub fn execute(self, results: &ResultsStore, console: &mut dyn Console) -> anyhow::Result<()> {
        let (location, dir) = find_result(results, &self.result_id)?;
        std::fs::remove_dir_all(&dir)
            .with_context(|| format!("failed to delete {}", dir.display()))?;
        writeln!(
            console.out(),
            "deleted {} ({})",
            self.result_id,
            location.label()
        )?;
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
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::test_support::CapturedConsole;

    fn store(_prefix: &str) -> anyhow::Result<(tempfile::TempDir, ResultsStore)> {
        let tmp = tempfile::tempdir()?;
        let s = ResultsStore::new(tmp.path().join("results"));
        Ok((tmp, s))
    }

    const RUNTIME_DESCRIPTOR: &str = r#"{"type":"llamacpp_cli_stock_tools","source":"github_release","repository_url":"github.com/ggml-org/llama.cpp","repository_version":"b10094","flavor":"macos-arm64"}"#;

    fn seed_payload(
        results: &ResultsStore,
        location: BenchmarkResultLocation,
        id: &str,
    ) -> anyhow::Result<()> {
        seed(
            results,
            location,
            Seed {
                id,
                benchmark_id: "prefill_throughput_smoke",
                submitted_at: "2026-03-26T00:00:00Z",
            },
        )
    }

    struct Seed<'a> {
        id: &'a str,
        benchmark_id: &'a str,
        submitted_at: &'a str,
    }

    fn seed(
        results: &ResultsStore,
        location: BenchmarkResultLocation,
        seed: Seed<'_>,
    ) -> anyhow::Result<()> {
        results
            .save_payload(
                location,
                seed.id,
                &json!({
                    "benchmark_id": seed.benchmark_id,
                    "runtime_descriptor": RUNTIME_DESCRIPTOR,
                    "submitted_at": seed.submitted_at,
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

    /// The RESULT column of what `results list` writes: the table is what the
    /// operator reads, so the assertion reads it too.
    fn listed(results: &ResultsStore, args: ResultsListArgs) -> anyhow::Result<Vec<String>> {
        let mut captured = CapturedConsole::default();
        args.execute(results, &mut captured)?;
        Ok(result_column(&captured.out_text()))
    }

    fn result_column(table: &str) -> Vec<String> {
        table
            .lines()
            .skip(2)
            .filter_map(|line| line.split('|').next())
            .map(|cell| cell.trim().to_owned())
            .filter(|cell| !cell.is_empty())
            .collect()
    }

    #[test]
    fn list_renders_runtime_as_a_compact_ref() -> anyhow::Result<()> {
        let (_tmp, results) = store("runtime-column")?;
        seed_payload(&results, BenchmarkResultLocation::Local, "job-1")?;
        let mut captured = CapturedConsole::default();
        list_args().execute(&results, &mut captured)?;
        let out = captured.out_text();
        let row = out
            .lines()
            .find(|line| line.contains("job-1"))
            .context("seeded result should list")?;
        assert!(row.contains("| b10094:macos-arm64 |"), "got {row}");
        assert!(out.starts_with(" RESULT "), "got {out}");
        Ok(())
    }

    #[test]
    fn an_empty_store_lists_no_results() -> anyhow::Result<()> {
        let (_tmp, results) = store("empty")?;
        let mut captured = CapturedConsole::default();
        list_args().execute(&results, &mut captured)?;
        assert_eq!(captured.out_text(), "No results.\n");
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
        let err = listed(&results, list_args())
            .err()
            .context("corrupt payload should fail listing")?;
        assert!(format!("{err:#}").contains("failed to parse"));
        Ok(())
    }

    #[test]
    fn delete_removes_result_dir_and_says_so() -> anyhow::Result<()> {
        let (_tmp, results) = store("delete")?;
        seed_payload(&results, BenchmarkResultLocation::Local, "job-1")?;
        let dir = results.result_dir(BenchmarkResultLocation::Local, "job-1");
        assert!(dir.exists());
        let mut captured = CapturedConsole::default();
        ResultsArgs {
            command: ResultsCommand::Delete(ResultDeleteArgs {
                result_id: "job-1".to_string(),
            }),
        }
        .execute(&results, &mut captured)?;
        assert!(!dir.exists(), "result dir should be gone after delete");
        assert_eq!(captured.out_text(), "deleted job-1 (local)\n");
        assert_eq!(listed(&results, list_args())?, Vec::<String>::new());
        Ok(())
    }

    /// A bare `--benchmark <id>` means the remote catalog, so it must not match
    /// the local record.
    #[rstest]
    #[case::scored_state(ResultsListArgs { state: Some(BenchmarkResultState::Scored), ..list_args() }, &[])]
    #[case::submitted_state(ResultsListArgs { state: Some(BenchmarkResultState::Submitted), ..list_args() }, &["job-remote"])]
    #[case::decode_type(ResultsListArgs { benchmark_type: Some(BenchmarkType::DecodeThroughput), ..list_args() }, &["job-decode"])]
    #[case::local_ref(ResultsListArgs { benchmark: Some("local/prefill_throughput_smoke".parse()?), ..list_args() }, &["job-local"])]
    #[case::bare_ref_is_remote(ResultsListArgs { benchmark: Some("prefill_throughput_smoke".parse()?), ..list_args() }, &["job-remote"])]
    #[case::unmatched_ref(ResultsListArgs { benchmark: Some("local/decode_throughput_nowhere".parse()?), ..list_args() }, &[])]
    fn list_filters_narrow_the_seeded_store(
        #[case] args: ResultsListArgs,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let (_tmp, results) = store("filters")?;
        seed_payload(&results, BenchmarkResultLocation::Local, "job-local")?;
        seed_payload(
            &results,
            BenchmarkResultLocation::RemoteSynced,
            "job-remote",
        )?;
        seed(
            &results,
            BenchmarkResultLocation::Local,
            Seed {
                id: "job-decode",
                benchmark_id: "decode_throughput_smoke",
                submitted_at: "2026-03-26T00:00:00Z",
            },
        )?;
        assert_eq!(listed(&results, args)?, expected);
        Ok(())
    }

    #[test]
    fn list_orders_newest_first_and_truncates_to_the_limit() -> anyhow::Result<()> {
        let (_tmp, results) = store("limit")?;
        [
            ("job-old", "2026-03-01T00:00:00Z"),
            ("job-new", "2026-03-03T00:00:00Z"),
            ("job-mid", "2026-03-02T00:00:00Z"),
        ]
        .into_iter()
        .try_for_each(|(id, at)| {
            seed(
                &results,
                BenchmarkResultLocation::Local,
                Seed {
                    id,
                    benchmark_id: "prefill_throughput_smoke",
                    submitted_at: at,
                },
            )
        })?;
        let args = ResultsListArgs {
            limit: Some(2),
            ..list_args()
        };
        assert_eq!(listed(&results, args)?, ["job-new", "job-mid"]);
        Ok(())
    }

    #[test]
    fn show_writes_the_payload_and_banners_each_present_companion() -> anyhow::Result<()> {
        let (_tmp, results) = store("show")?;
        let location = BenchmarkResultLocation::RemoteSynced;
        seed_payload(&results, location, "job-1")?;
        let show = || ResultShowArgs {
            result_id: "job-1".to_string(),
        };

        let mut payload_only = CapturedConsole::default();
        show().execute(&results, &mut payload_only)?;
        assert!(payload_only.out_text().contains("prefill_throughput_smoke"));
        assert_eq!(payload_only.err_text(), "");

        std::fs::write(results.metrics_path(location, "job-1"), "{\"score\":1}")?;
        let mut with_metrics = CapturedConsole::default();
        show().execute(&results, &mut with_metrics)?;
        assert!(with_metrics.out_text().ends_with("{\"score\":1}\n"));
        assert_eq!(with_metrics.err_text(), "--- metrics ---\n");

        std::fs::write(results.extras_path(location, "job-1"), "{\"extra\":1}")?;
        let mut with_both = CapturedConsole::default();
        show().execute(&results, &mut with_both)?;
        assert_eq!(
            with_both.err_text(),
            "--- extras ---\n--- metrics ---\n",
            "extras come before metrics whichever landed first"
        );
        assert!(
            with_both
                .out_text()
                .contains("{\"extra\":1}\n{\"score\":1}\n")
        );
        Ok(())
    }

    #[test]
    fn show_and_delete_fail_on_an_unknown_result() -> anyhow::Result<()> {
        let (_tmp, results) = store("unknown")?;
        let mut captured = CapturedConsole::default();
        let shown = ResultShowArgs {
            result_id: "nope".to_string(),
        }
        .execute(&results, &mut captured)
        .err()
        .context("show should fail for an unknown result")?;
        assert!(format!("{shown:#}").contains("result nope not found"));
        let deleted = ResultDeleteArgs {
            result_id: "nope".to_string(),
        }
        .execute(&results, &mut captured)
        .err()
        .context("delete should fail for an unknown result")?;
        assert!(format!("{deleted:#}").contains("result nope not found"));
        assert_eq!(captured.out_text(), "", "a refusal writes nothing");
        Ok(())
    }
}
