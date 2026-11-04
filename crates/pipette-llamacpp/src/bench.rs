use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use anyhow::Context;
use serde::Deserialize;

use pipette_plan_types::{
    EnvForward, LlamacppCliStockToolsExtraRuntimeConfiguration, LlamacppFlashAttention,
    LlamacppLoadMode,
};

use crate::runtimes::{CommonKnobs, ResolvedLlamaBench, apply_launch_env, canonicalize_flag_order};

/// One rep of a timing cell, a `llama-bench` or `llama-batched-bench` run: a
/// slow phone on a large model with a cold cache can take 20+ minutes, and a
/// wedged GPU driver never returns.
pub(crate) const TIMING_TIMEOUT: Duration = Duration::from_secs(3600);

/// One row of `llama-bench` JSON stdout; fields we don't read are ignored.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct BenchRow {
    pub avg_ns: f64,
    pub n_prompt: u32,
    pub n_gen: u32,
    pub n_depth: u32,
}

/// One llama-bench run. The knobs arrive resolved, the benchmark's defaults
/// already overlaid ([`crate::runtimes::get_load_mode`]).
#[derive(Debug, Clone)]
pub struct BenchCommandBuilder {
    program: ResolvedLlamaBench,
    model: PathBuf,
    workload: BenchWorkload,
    threads: Option<u32>,
    number_gpu_layers: Option<u32>,
    mmap: Option<bool>,
    load_mode: Option<LlamacppLoadMode>,
    flash_attention: Option<LlamacppFlashAttention>,
    raw: Vec<String>,
    envs: Vec<EnvForward>,
    client_config: Option<LlamacppCliStockToolsExtraRuntimeConfiguration>,
}

impl BenchCommandBuilder {
    pub fn new(
        program: ResolvedLlamaBench,
        model: impl Into<PathBuf>,
        workload: BenchWorkload,
    ) -> Self {
        Self {
            program,
            model: model.into(),
            workload,
            threads: None,
            number_gpu_layers: None,
            mmap: None,
            load_mode: None,
            flash_attention: None,
            raw: Vec::new(),
            envs: Vec::new(),
            client_config: None,
        }
    }

    pub fn threads(mut self, threads: Option<u32>) -> Self {
        self.threads = threads;
        self
    }

    pub fn number_gpu_layers(mut self, number_gpu_layers: Option<u32>) -> Self {
        self.number_gpu_layers = number_gpu_layers;
        self
    }

    pub fn mmap(mut self, mmap: Option<bool>) -> Self {
        self.mmap = mmap;
        self
    }

    pub fn load_mode(mut self, load_mode: Option<LlamacppLoadMode>) -> Self {
        self.load_mode = load_mode;
        self
    }

    pub fn flash_attention(mut self, flash_attention: Option<LlamacppFlashAttention>) -> Self {
        self.flash_attention = flash_attention;
        self
    }

    pub fn raw(mut self, raw: Vec<String>) -> Self {
        self.raw = raw;
        self
    }

    pub fn envs(mut self, envs: Vec<EnvForward>) -> Self {
        self.envs = envs;
        self
    }

    pub fn client_config(
        mut self,
        client_config: Option<LlamacppCliStockToolsExtraRuntimeConfiguration>,
    ) -> Self {
        self.client_config = client_config;
        self
    }

    fn knobs(&self) -> CommonKnobs<'_> {
        CommonKnobs {
            threads: self.threads,
            number_gpu_layers: self.number_gpu_layers,
            mmap: self.mmap,
            load_mode: self.load_mode,
            flash_attention: self.flash_attention,
            raw: &self.raw,
        }
    }

    /// The knobs in canonical order, then the workload: everything but the
    /// output format and the model, which [`Self::build`] adds.
    pub fn args(&self) -> Vec<String> {
        let knobs = self.knobs();
        let mut tokens = knobs.scalar_tokens();
        // `--load-mode` wins; the deprecated `--mmap 0|1` renders only alone (llama-bench
        // takes no bare `--no-mmap`). Never both: llama-bench runs a row per load flag it
        // parses, and a single-row benchmark cannot tell the extra row apart.
        match (knobs.load_mode, knobs.mmap) {
            (Some(mode), _) => {
                tokens.extend(["--load-mode".to_owned(), mode.as_cli_word().to_owned()]);
            }
            (None, Some(on)) => tokens.extend(["--mmap".to_owned(), u8::from(on).to_string()]),
            (None, None) => {}
        }
        tokens.extend(knobs.raw.iter().cloned());
        // One rep per invoke; the reps are the typed entry's. `raw` cannot carry one:
        // every bench cell reserves `--repetitions`.
        tokens.extend(["--repetitions".to_owned(), "1".to_owned()]);
        let mut argv = canonicalize_flag_order(&tokens);
        argv.extend(self.workload.tokens());
        argv
    }

    /// One run with [`Self::args`], printing its row as JSON.
    pub fn build(&self) -> Command {
        let mut command = Command::new(self.program.as_os_str());
        apply_launch_env(
            &mut command,
            &self.program,
            &self.envs,
            self.client_config.as_ref(),
        );
        command.arg("--output").arg("json");
        command.arg("--model").arg(&self.model);
        command.args(self.args());
        command
    }
}

/// What one llama-bench run processes, which llama-bench reports as one "test".
#[derive(Debug, Clone, Copy)]
pub enum BenchWorkload {
    Prompt {
        n_prompt: u32,
    },
    /// The generation alone, on `n_depth` tokens already in the KV cache.
    Generation {
        n_gen: u32,
        n_depth: u32,
    },
    /// The generation right after the prompt, on its KV cache.
    PromptThenGeneration {
        n_prompt: u32,
        n_gen: u32,
    },
}

impl BenchWorkload {
    /// The zeros turn off the pp512 and tg128 an unset flag adds.
    fn tokens(self) -> Vec<String> {
        let counts = |n_prompt: u32, n_gen: u32| {
            vec![
                "--n-prompt".to_owned(),
                n_prompt.to_string(),
                "--n-gen".to_owned(),
                n_gen.to_string(),
            ]
        };
        match self {
            Self::Prompt { n_prompt } => counts(n_prompt, 0),
            Self::Generation { n_gen, n_depth } => [
                counts(0, n_gen),
                vec!["--n-depth".to_owned(), n_depth.to_string()],
            ]
            .concat(),
            // `--n-prompt` and `--n-gen` are two tests, each in its own context;
            // `-pg` is one. With either count zero there is one test anyway.
            Self::PromptThenGeneration { n_prompt, n_gen } if n_prompt > 0 && n_gen > 0 => [
                counts(0, 0),
                vec!["-pg".to_owned(), format!("{n_prompt},{n_gen}")],
            ]
            .concat(),
            Self::PromptThenGeneration { n_prompt, n_gen } => counts(n_prompt, n_gen),
        }
    }
}

/// The row a run printed for its one test.
pub fn row(stdout: &str) -> anyhow::Result<BenchRow> {
    let rows = serde_json::from_str::<Vec<BenchRow>>(stdout)
        .context("failed to parse llama-bench JSON")?;
    match rows.as_slice() {
        [row] => Ok(*row),
        [] => Err(anyhow::anyhow!("llama-bench printed no row")),
        rows => Err(anyhow::anyhow!(
            "llama-bench printed {} rows for one test; a comma list or -pg in the cell's \
             flags adds tests",
            rows.len()
        )),
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use pipette_plan_types::{
        LlamacppFlashAttention, LlamacppLoadMode, PrefillThroughputLlamacppCliStockToolsGgufText,
        RuntimeFlags,
    };

    use super::*;
    use crate::runtimes::test_llama_bench;

    fn launch() -> BenchCommandBuilder {
        BenchCommandBuilder::new(
            test_llama_bench(),
            "model.gguf",
            BenchWorkload::Prompt { n_prompt: 512 },
        )
    }

    /// Upstream's full row shape reads; `avg_ns` is an integer there.
    #[test]
    fn a_row_reads_upstreams_json() -> anyhow::Result<()> {
        let stdout = r#"[
          {
            "build_commit": "8f2e7b1", "build_number": 6500, "backends": "Metal,BLAS",
            "model_filename": "m.gguf", "n_batch": 2048, "n_ubatch": 512, "flash_attn": 1,
            "n_prompt": 0, "n_gen": 128, "n_depth": 512,
            "test_time": "2026-09-23T10:00:00Z",
            "avg_ns": 1250000000, "stddev_ns": 0, "avg_ts": 102.4, "stddev_ts": 0.0,
            "samples_ns": [ 1250000000 ], "samples_ts": [ 102.4 ]
          }
        ]"#;
        let row = row(stdout)?;
        assert_eq!(
            (row.n_prompt, row.n_gen, row.n_depth, row.avg_ns),
            (0, 128, 512, 1_250_000_000.0)
        );
        Ok(())
    }

    #[rstest]
    #[case::prompt(BenchWorkload::Prompt { n_prompt: 512 }, &["--n-prompt", "512", "--n-gen", "0"])]
    #[case::generation_at_a_depth(
        BenchWorkload::Generation { n_gen: 128, n_depth: 512 },
        &["--n-prompt", "0", "--n-gen", "128", "--n-depth", "512"]
    )]
    #[case::generation_on_an_empty_cache(
        BenchWorkload::Generation { n_gen: 128, n_depth: 0 },
        &["--n-prompt", "0", "--n-gen", "128", "--n-depth", "0"]
    )]
    #[case::generation_on_a_prompt(
        BenchWorkload::PromptThenGeneration { n_prompt: 512, n_gen: 1 },
        &["--n-prompt", "0", "--n-gen", "0", "-pg", "512,1"]
    )]
    #[case::no_prompt_leaves_the_generation(
        BenchWorkload::PromptThenGeneration { n_prompt: 0, n_gen: 1 },
        &["--n-prompt", "0", "--n-gen", "1"]
    )]
    fn a_workload_renders_as_one_test(#[case] workload: BenchWorkload, #[case] expected: &[&str]) {
        assert_eq!(workload.tokens(), expected);
    }

    #[rstest]
    #[case::not_json("not json", "failed to parse llama-bench JSON")]
    #[case::no_row("[]", "llama-bench printed no row")]
    #[case::a_row_per_value(
        r#"[{"avg_ns": 1, "n_prompt": 512, "n_gen": 0, "n_depth": 0},
            {"avg_ns": 2, "n_prompt": 512, "n_gen": 0, "n_depth": 0}]"#,
        "printed 2 rows"
    )]
    fn anything_but_one_row_is_refused(
        #[case] stdout: &str,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let err = row(stdout)
            .err()
            .context("expected the row to be refused")?;
        assert!(format!("{err:#}").contains(expected), "{err:#}");
        Ok(())
    }

    #[rstest]
    #[case::nothing_set(launch(), &["--repetitions", "1"])]
    #[case::mmap_off(launch().mmap(Some(false)), &["--mmap", "0", "--repetitions", "1"])]
    #[case::mmap_on(launch().mmap(Some(true)), &["--mmap", "1", "--repetitions", "1"])]
    #[case::raw_sorts_among_the_knobs(
        launch().threads(Some(8)).mmap(Some(false)).raw(vec!["--poll".to_owned(), "0".to_owned()]),
        &["--mmap", "0", "--poll", "0", "--repetitions", "1", "--threads", "8"]
    )]
    #[case::load_mode_alone(
        launch().load_mode(Some(LlamacppLoadMode::MmapMlock)),
        &["--load-mode", "mmap+mlock", "--repetitions", "1"]
    )]
    #[case::load_mode_beats_the_boolean(
        launch().mmap(Some(true)).load_mode(Some(LlamacppLoadMode::Read)),
        &["--load-mode", "none", "--repetitions", "1"]
    )]
    // Knobs that share a type, set apart so two cannot trade places.
    #[case::every_knob_set(
        launch()
            .threads(Some(3))
            .number_gpu_layers(Some(7))
            .mmap(Some(false))
            .flash_attention(Some(LlamacppFlashAttention::On)),
        &[
            "--flash-attn", "on", "--mmap", "0", "--n-gpu-layers", "7", "--repetitions", "1",
            "--threads", "3",
        ]
    )]
    fn args_render_the_knobs_then_the_workload(
        #[case] launch: BenchCommandBuilder,
        #[case] expected: &[&str],
    ) {
        assert_eq!(
            launch.args(),
            [expected, &["--n-prompt", "512", "--n-gen", "0"]].concat()
        );
    }

    /// The repetition count is the benchmark's, not the cell's: it reaches argv
    /// but has no field to occupy in the submitted record.
    #[test]
    fn the_fixed_repetition_count_stays_out_of_the_record() {
        let runtime_flags = PrefillThroughputLlamacppCliStockToolsGgufText {
            threads: None,
            number_gpu_layers: None,
            mmap: Some(false),
            load_mode: None,
            flash_attention: Some(LlamacppFlashAttention::On),
            envs: Vec::new(),
            raw: Vec::new(),
        };
        let argv = launch()
            .mmap(runtime_flags.mmap)
            .flash_attention(runtime_flags.flash_attention)
            .raw(runtime_flags.raw.clone())
            .args();
        assert!(argv.contains(&"--repetitions".into()));

        let record = RuntimeFlags::PrefillThroughputLlamacppCliStockToolsGgufText(runtime_flags)
            .submission_value();
        assert!(
            !record.to_string().contains("\"-r\""),
            "reserved flag reached the record: {record}"
        );
    }
}
