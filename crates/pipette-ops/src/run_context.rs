//! What the client lends a run besides the request ([`RunContext`]).

use std::path::PathBuf;

use pipette_plan_types::Dataset;

use crate::eval_completions::EvalCompletionsStore;

/// What the client lends a run: the tools its benchmarks call when they need
/// them.
///
/// Injected like [`ReadinessGate`](crate::readiness::ReadinessGate): a dataset
/// is fetched as this device, so an engine asks for one and never holds the
/// server or the key that fetches it.
pub trait RunContext {
    /// The bytes `dataset`'s route serves, fetched as this device.
    fn fetch_dataset(&self, dataset: &Dataset) -> anyhow::Result<Vec<u8>>;

    /// [`Self::fetch_dataset`], written to `dest` as the bytes arrive rather
    /// than held: an asr_eval corpus is gigabytes of audio. Returns how many
    /// bytes were written. Unverified, as `fetch_dataset` is: the caller
    /// hashes what it was handed against the pin.
    ///
    /// By default the bytes are fetched whole and then written, which only a
    /// context that holds its datasets in memory anyway (a test's) should
    /// keep.
    fn fetch_dataset_to(
        &self,
        dataset: &Dataset,
        dest: &mut dyn std::io::Write,
    ) -> anyhow::Result<u64> {
        let bytes = self.fetch_dataset(dataset)?;
        dest.write_all(&bytes)?;
        Ok(u64::try_from(bytes.len())?)
    }

    /// Where a run unpacks what it fetches, one directory per benchmark.
    fn datasets_dir(&self) -> PathBuf;

    /// Where an eval checkpoints its samples, so a killed run resumes.
    fn eval_completions(&self) -> EvalCompletionsStore;
}
