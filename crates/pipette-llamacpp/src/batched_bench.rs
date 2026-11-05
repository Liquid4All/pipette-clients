use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use serde::Deserialize;

use pipette_plan_types::{
    BatchSize, EnvForward, LlamacppCliStockToolsExtraRuntimeConfiguration, LlamacppFlashAttention,
    LlamacppLoadMode,
};

use crate::runtimes::{
    CommonKnobs, ResolvedLlamaBatchedBench, apply_launch_env, canonicalize_flag_order,
};

/// One row of `llama-batched-bench` JSONL stdout; fields we don't read are
/// ignored. Counts are per sequence and times are the whole batch's, in
/// seconds.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct BatchedBenchRow {
    pub pp: u32,
    pub tg: u32,
    pub pl: u32,
    pub t_pp: f32,
    pub t_tg: f32,
}

/// One llama-batched-bench run. The tool parses llama.cpp's common arguments,
/// so the knobs render as llama-server's ([`CommonKnobs::common_arg_tokens`]).
#[derive(Debug, Clone)]
pub struct BatchedBenchCommandBuilder {
    program: ResolvedLlamaBatchedBench,
    model: PathBuf,
    workload: BatchedBenchWorkload,
    /// The workload's, checked by [`Self::new`].
    ctx_size: u32,
    threads: Option<u32>,
    number_gpu_layers: Option<u32>,
    mmap: Option<bool>,
    load_mode: Option<LlamacppLoadMode>,
    flash_attention: Option<LlamacppFlashAttention>,
    raw: Vec<String>,
    envs: Vec<EnvForward>,
    client_config: Option<LlamacppCliStockToolsExtraRuntimeConfiguration>,
}

impl BatchedBenchCommandBuilder {
    /// Refuses a `workload` whose context does not fit the tool's `int`.
    pub fn new(
        program: ResolvedLlamaBatchedBench,
        model: impl Into<PathBuf>,
        workload: BatchedBenchWorkload,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            program,
            model: model.into(),
            workload,
            ctx_size: workload.ctx_size()?,
            threads: None,
            number_gpu_layers: None,
            mmap: None,
            load_mode: None,
            flash_attention: None,
            raw: Vec::new(),
            envs: Vec::new(),
            client_config: None,
        })
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

    /// The knobs sorted with `raw`, then the workload: everything but the
    /// output format and the model, which [`Self::build`] adds.
    pub fn args(&self) -> Vec<String> {
        let knobs = self.knobs();
        let mut tokens = knobs.common_arg_tokens();
        tokens.extend(knobs.raw.iter().cloned());
        let mut argv = canonicalize_flag_order(&tokens);
        argv.extend(self.workload.tokens(self.ctx_size));
        argv
    }

    /// One run with [`Self::args`], printing its rows as JSONL.
    pub fn build(&self) -> Command {
        let mut command = Command::new(self.program.as_os_str());
        apply_launch_env(
            &mut command,
            &self.program,
            &self.envs,
            self.client_config.as_ref(),
        );
        command.arg("--output-format").arg("jsonl");
        command.arg("--model").arg(&self.model);
        command.args(self.args());
        command
    }
}

/// The prompt, generation and batch shape a run reads.
#[derive(Debug, Clone, Copy)]
pub struct BatchedBenchWorkload {
    pub n_prompt: u32,
    pub n_gen: u32,
    pub batch_size: BatchSize,
}

impl BatchedBenchWorkload {
    /// The context that holds the whole batch: the tool skips a row that does
    /// not fit without saying so. `--ctx-size` is parsed into an `int`.
    fn ctx_size(self) -> anyhow::Result<u32> {
        let Self {
            n_prompt,
            n_gen,
            batch_size,
        } = self;
        let sequences = batch_size.as_u32();
        n_prompt
            .checked_add(n_gen)
            .and_then(|per_sequence| per_sequence.checked_mul(sequences))
            .filter(|&ctx| i32::try_from(ctx).is_ok())
            .with_context(|| {
                format!("{sequences} sequences of {n_prompt} + {n_gen} tokens overflow --ctx-size")
            })
    }

    /// `-npl` lists the batch twice, so the first row warms the process at the
    /// cell's own shape and the second is the reading, in `ctx_size`
    /// ([`Self::ctx_size`]).
    fn tokens(self, ctx_size: u32) -> [String; 8] {
        let Self {
            n_prompt,
            n_gen,
            batch_size,
        } = self;
        let sequences = batch_size.as_u32();
        [
            "--ctx-size".to_owned(),
            ctx_size.to_string(),
            "-npp".to_owned(),
            n_prompt.to_string(),
            "-ntg".to_owned(),
            n_gen.to_string(),
            "-npl".to_owned(),
            format!("{sequences},{sequences}"),
        ]
    }
}

/// The reading a run printed: the second of the two rows a [`BatchedBenchWorkload`] asks for.
pub fn row(stdout: &str) -> anyhow::Result<BatchedBenchRow> {
    let rows = stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str::<BatchedBenchRow>)
        .collect::<Result<Vec<_>, _>>()
        .context("failed to parse llama-batched-bench JSONL")?;
    match rows.as_slice() {
        [_warm_up, row] => Ok(*row),
        rows => Err(anyhow::anyhow!(
            "expected 2 rows from llama-batched-bench (a warm-up and a reading), got {}",
            rows.len()
        )),
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;
    use crate::runtimes::test_llama_batched_bench;

    /// Upstream's row, as `--output-format jsonl` prints it.
    const ROW: &str = r#"{"n_kv_max": 2304, "n_batch": 2048, "n_ubatch": 512, "flash_attn": -1, "is_pp_shared": 0, "n_gpu_layers": -1, "n_threads": 8, "n_threads_batch": 8, "pp": 512, "tg": 1, "pl": 4, "n_kv": 2052, "t_pp": 0.184623, "speed_pp": 11092.660156, "t_tg": 0.009125, "speed_tg": 438.356171, "t": 0.193748, "speed": 10591.078125}"#;

    /// The tool ends its output with a blank line.
    #[test]
    fn the_second_row_is_the_reading() -> anyhow::Result<()> {
        let warm_up = ROW.replace("0.184623", "0.9");
        let row = row(&format!("{warm_up}\n{ROW}\n\n"))?;
        assert_eq!((row.pp, row.tg, row.pl), (512, 1, 4));
        assert!((row.t_pp - 0.184_623).abs() < 1e-7, "{}", row.t_pp);
        assert!((row.t_tg - 0.009_125).abs() < 1e-7, "{}", row.t_tg);
        Ok(())
    }

    /// Anything but the warm-up and the reading is refused: a row skipped for
    /// its context, or one per value a comma list added.
    #[rstest]
    #[case::none("\n", "got 0")]
    #[case::the_warm_up_alone(ROW, "got 1")]
    #[case::an_extra_row(&format!("{ROW}\n{ROW}\n{ROW}\n"), "got 3")]
    // A rate of 0/0 prints as `nan`, which is no JSON.
    #[case::not_json(&ROW.replace("438.356171", "nan"), "failed to parse")]
    fn anything_but_two_rows_is_refused(
        #[case] stdout: &str,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let err = row(stdout)
            .err()
            .context("expected the rows to be refused")?;
        assert!(format!("{err:#}").contains(expected), "{err:#}");
        Ok(())
    }

    #[test]
    fn args_render_the_knobs_then_the_workload() -> anyhow::Result<()> {
        let workload = BatchedBenchWorkload {
            n_prompt: 512,
            n_gen: 128,
            batch_size: BatchSize::try_new(4)?,
        };
        let launch =
            BatchedBenchCommandBuilder::new(test_llama_batched_bench(), "model.gguf", workload)?
                .threads(Some(3))
                .mmap(Some(false))
                .raw(vec!["--poll".to_owned(), "0".to_owned()]);
        assert_eq!(
            launch.args().join(" "),
            "--no-mmap --poll 0 --threads 3 --ctx-size 2560 -npp 512 -ntg 128 -npl 4,4"
        );
        Ok(())
    }

    #[test]
    fn a_test_lists_the_batch_twice_in_a_context_that_fits_it() -> anyhow::Result<()> {
        let workload = BatchedBenchWorkload {
            n_prompt: 512,
            n_gen: 128,
            batch_size: BatchSize::try_new(4)?,
        };
        assert_eq!(
            workload.tokens(workload.ctx_size()?).join(" "),
            "--ctx-size 2560 -npp 512 -ntg 128 -npl 4,4"
        );
        Ok(())
    }

    #[test]
    fn a_context_past_an_int_is_refused() -> anyhow::Result<()> {
        let workload = BatchedBenchWorkload {
            n_prompt: 1 << 20,
            n_gen: 0,
            batch_size: BatchSize::try_new(1 << 11)?,
        };
        let err = workload
            .ctx_size()
            .err()
            .context("expected the context to be refused")?;
        assert!(
            format!("{err:#}").contains("overflow --ctx-size"),
            "{err:#}"
        );
        Ok(())
    }
}
