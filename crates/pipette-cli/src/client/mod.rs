//! Management-server flows: registration, catalog and result sync, claim
//! parsing and the planner worker protocol. Client-only: no runtime crate
//! performs them.

use std::path::Path;

use pipette_ops::eval_completions::EvalCompletionsStore;

use crate::benchmarks::BenchmarkStore;
use crate::error::Result;
use crate::results::{BenchmarkResultLocation, ResultsStore};

pub mod auth;
pub mod claim;
pub mod sync;
pub mod worker;

/// Wipe server-derived caches: remote benchmark catalog, remote results, eval state.
/// Local benchmarks/results are left intact.
pub fn clear_remote_state(
    benchmarks: &BenchmarkStore,
    results: &ResultsStore,
    evals: &EvalCompletionsStore,
) -> Result<()> {
    benchmarks.clear_remote()?;
    [
        results.location_dir(BenchmarkResultLocation::RemotePending),
        results.location_dir(BenchmarkResultLocation::RemoteSynced),
    ]
    .iter()
    .try_for_each(|dir| remove_dir_if_exists(dir))?;
    evals.clear().map_err(Into::into)
}

fn remove_dir_if_exists(dir: &Path) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    std::fs::remove_dir_all(dir)
        .map_err(|source| pipette_ops::Error::Io {
            path: dir.to_path_buf(),
            source,
        })
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use pipette_plan_types::{
        BatchSize, BenchmarkDefinition, BenchmarkId, NonNegativeCount, PrefillThroughputBenchmark,
    };

    use super::*;
    use crate::benchmarks::BenchmarkStore;
    use crate::results::BenchmarkResultLocation;

    fn touch(path: &Path) -> anyhow::Result<()> {
        let parent = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("path has no parent: {}", path.display()))?;
        std::fs::create_dir_all(parent)?;
        std::fs::write(path, b"x")?;
        Ok(())
    }

    #[test]
    fn clear_remote_state_removes_server_caches_and_keeps_local() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        let benchmarks_root = root.path().join("benchmarks");
        let benchmarks = BenchmarkStore::new(&benchmarks_root);
        let results = ResultsStore::new(root.path().join("results"));

        benchmarks.put_remote_index(&[])?;
        crate::test_support::cache_remote(
            &benchmarks,
            serde_json::json!({
                "benchmark_id": "remote-1",
                "benchmark_type": "prefill_throughput",
                "parameter_prefill_tokens": 1,
            }),
        )?;
        assert!(crate::test_support::remote_detail_path(&benchmarks_root, "remote-1").exists());
        touch(&results.payload_path(BenchmarkResultLocation::RemotePending, "r-pending"))?;
        touch(&results.payload_path(BenchmarkResultLocation::RemoteSynced, "r-synced"))?;
        let evals = EvalCompletionsStore::new(root.path().join("evals"));
        touch(&evals.root().join("job-1/state.json"))?;
        benchmarks.put_local(&BenchmarkDefinition::PrefillThroughput(
            PrefillThroughputBenchmark {
                benchmark_id: BenchmarkId::try_new("local-1".to_owned())?,
                parameter_prefill_tokens: NonNegativeCount::try_new(1)?,
                parameter_batch_size: BatchSize::one(),
            },
        ))?;
        touch(&results.payload_path(BenchmarkResultLocation::Local, "r-local"))?;

        clear_remote_state(&benchmarks, &results, &evals)?;

        assert!(!benchmarks_root.join("remote").exists());
        assert!(
            !results
                .location_dir(BenchmarkResultLocation::RemotePending)
                .exists()
        );
        assert!(
            !results
                .location_dir(BenchmarkResultLocation::RemoteSynced)
                .exists()
        );
        assert!(!evals.root().exists());
        assert!(
            benchmarks
                .get(&crate::benchmarks::SourcedBenchmarkId::new(
                    pipette_plan_types::BenchmarkSource::Local,
                    pipette_plan_types::BenchmarkId::try_new("local-1".to_string())
                        .map_err(|e| anyhow::anyhow!(e))?,
                ))?
                .is_some()
        );
        assert!(
            results
                .payload_path(BenchmarkResultLocation::Local, "r-local")
                .exists()
        );
        Ok(())
    }
}
