//! What an eval cell asks of an engine: the knobs its benchmark and flags set
//! ([`EvalCell`]) and the sampling every request carries ([`Sampling`]). The
//! sample loop itself is `measurement::eval`.

use anyhow::Context;
use serde_json::{Value, json};

use pipette_plan_types::{
    EvalBenchmark, MinP, ModelFlags, PositiveCount, ReasoningEffort, RunRequest,
    SamplingTemperature, TopK, TopP,
};

/// What an eval benchmark asked for, plus the knobs the cell's flags set on
/// it.
pub struct EvalCell<'a> {
    pub benchmark_id: &'a str,
    pub max_tokens: u32,
    /// The answer set of a multiple-choice eval, when the benchmark has one.
    pub mcq_choices: Option<&'a [String]>,
    /// What the requests sample at, benchmark and eval-id policy resolved.
    pub sampling: Sampling<'a>,
    /// `chat_template_kwargs.enable_thinking`, forwarded only when set.
    pub enable_thinking: Option<bool>,
    /// The effort a reasoning model is asked for, forwarded only when set.
    /// Where it goes is the engine's: the request's own field on an
    /// OpenAI-compatible server, a template kwarg plus that same field on
    /// llama-server's `/apply-template`.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Samples in flight at once, clamped to the dataset.
    pub workers: usize,
}

/// The sampling one eval cell runs under, resolved from its benchmark.
///
/// The temperature is always sent: the catalog's when the entry pins one,
/// else the client-side policy keyed on the eval id. A cut the entry left
/// unpinned is not sent, so the runtime's own default decides.
#[derive(Debug, Clone, Copy)]
pub struct Sampling<'a> {
    pub temperature: SamplingTemperature,
    pub top_p: Option<TopP>,
    pub top_k: Option<TopK>,
    pub min_p: Option<MinP>,
    /// Held for [`EvalBenchmark::sample_seed`]: a request carries the
    /// sample's own seed, never the entry's base one.
    benchmark: &'a EvalBenchmark,
}

impl<'a> Sampling<'a> {
    pub fn from_benchmark(benchmark: &'a EvalBenchmark) -> anyhow::Result<Self> {
        Ok(Self {
            temperature: benchmark.sampling_temperature()?,
            top_p: benchmark.parameter_top_p,
            top_k: benchmark.parameter_top_k,
            min_p: benchmark.parameter_min_p,
            benchmark,
        })
    }

    pub fn seed_for(&self, sample_id: &str) -> Option<i64> {
        self.benchmark.sample_seed(sample_id)
    }

    /// Write this sample's sampling into `body`, which must be a JSON object.
    ///
    /// llama.cpp's `/completion` and the OpenAI-compatible servers (vLLM,
    /// SGLang) spell these five the same, so one writer serves every engine
    /// that builds its body as JSON.
    pub fn apply(&self, body: &mut Value, sample_id: &str) -> anyhow::Result<()> {
        let body = body
            .as_object_mut()
            .context("a request body must be a JSON object")?;
        body.insert("temperature".to_owned(), json!(self.temperature.as_f64()));
        if let Some(top_p) = self.top_p {
            body.insert("top_p".to_owned(), json!(top_p.as_f64()));
        }
        if let Some(top_k) = self.top_k {
            body.insert("top_k".to_owned(), json!(top_k.into_inner()));
        }
        if let Some(min_p) = self.min_p {
            body.insert("min_p".to_owned(), json!(min_p.as_f64()));
        }
        if let Some(seed) = self.seed_for(sample_id) {
            body.insert("seed".to_owned(), json!(seed));
        }
        Ok(())
    }
}

impl<'a> EvalCell<'a> {
    /// Read the cell off the request. `parallel` is the engine's: which of
    /// its flag variants carry the knob is a typed match only it can make.
    pub fn from_request(
        req: &'a RunRequest,
        parallel: Option<PositiveCount>,
    ) -> anyhow::Result<Self> {
        Self::new(req.benchmark.as_eval()?, req.model_flags.as_ref(), parallel)
    }

    pub fn new(
        benchmark: &'a EvalBenchmark,
        model_flags: Option<&ModelFlags>,
        parallel: Option<PositiveCount>,
    ) -> anyhow::Result<Self> {
        let samples = benchmark
            .samples
            .as_deref()
            .context("eval benchmark missing samples")?;
        let workers = usize::try_from(parallel.map_or(1, PositiveCount::into_inner))
            .context("parallel does not fit in usize")?
            .clamp(1, samples.len().max(1));
        Ok(Self {
            benchmark_id: benchmark.benchmark_id.as_str(),
            max_tokens: benchmark.parameter_max_tokens.into_inner().unsigned_abs(),
            mcq_choices: benchmark
                .parameter_mcq_choices
                .as_deref()
                .filter(|items| !items.is_empty()),
            sampling: Sampling::from_benchmark(benchmark)?,
            enable_thinking: model_flags.and_then(ModelFlags::enable_thinking),
            reasoning_effort: model_flags.and_then(ModelFlags::reasoning_effort),
            workers,
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use pipette_plan_types::{
        BenchmarkDefinition, BenchmarkId, DeclaredBound, EvalBenchmark, EvalId, MinP, Model,
        Runtime, SamplingSeed, SamplingTemperature, TopK, TopP,
    };

    use super::*;

    fn request(n: usize) -> anyhow::Result<RunRequest> {
        let samples = (0..n)
            .map(|i| json!({"id": format!("s{i}"), "messages": [{"role": "user", "content": "q"}]}))
            .collect();
        Ok(RunRequest {
            runtime: DeclaredBound::already_bound(Runtime::AppleFoundation(Default::default())),
            model: DeclaredBound::already_bound(Model::AppleFoundationText),
            runtime_flags: None,
            model_flags: None,
            benchmark_flags: None,
            benchmark: BenchmarkDefinition::Eval(EvalBenchmark {
                benchmark_id: BenchmarkId::try_new("eval_cell".to_owned())?,
                parameter_eval_id: EvalId::from("math_500"),
                parameter_dataset_name: "d".to_string(),
                parameter_max_tokens: PositiveCount::try_new(8)?,
                parameter_max_context: None,
                parameter_mcq_choices: Some(vec![]),
                parameter_temperature: None,
                parameter_top_p: None,
                parameter_top_k: None,
                parameter_min_p: None,
                parameter_seed: None,
                samples: Some(samples),
            }),
        })
    }

    /// An empty answer set is no answer set; `parallel` clamps to the
    /// dataset.
    #[rstest]
    #[case::unset(None, 1)]
    #[case::one(Some(1), 1)]
    #[case::pool(Some(3), 3)]
    #[case::clamped(Some(16), 4)]
    fn the_cell_reads_its_knobs(
        #[case] parallel: Option<i32>,
        #[case] workers: usize,
    ) -> anyhow::Result<()> {
        let req = request(4)?;
        let parallel = parallel.map(PositiveCount::try_new).transpose()?;
        let cell = EvalCell::from_request(&req, parallel)?;
        assert_eq!(cell.workers, workers);
        assert!(cell.mcq_choices.is_none());
        assert_eq!(cell.max_tokens, 8);
        Ok(())
    }

    #[rstest]
    #[case::no_flags(None, None, None)]
    #[case::thinking_alone(
        Some(ModelFlags::EvalTorch { enable_thinking: Some(false), reasoning_effort: None }),
        Some(false),
        None
    )]
    #[case::effort_alone(
        Some(ModelFlags::EvalTorch { enable_thinking: None, reasoning_effort: Some(ReasoningEffort::High) }),
        None,
        Some(ReasoningEffort::High)
    )]
    #[case::both(
        Some(ModelFlags::EvalGgufText { enable_thinking: Some(true), reasoning_effort: Some(ReasoningEffort::Low) }),
        Some(true),
        Some(ReasoningEffort::Low)
    )]
    fn the_cell_reads_its_model_flags(
        #[case] model_flags: Option<ModelFlags>,
        #[case] enable_thinking: Option<bool>,
        #[case] reasoning_effort: Option<ReasoningEffort>,
    ) -> anyhow::Result<()> {
        let mut req = request(1)?;
        req.model_flags = model_flags;
        let cell = EvalCell::from_request(&req, None)?;
        assert_eq!(cell.enable_thinking, enable_thinking);
        assert_eq!(cell.reasoning_effort, reasoning_effort);
        Ok(())
    }

    fn eval_benchmark() -> anyhow::Result<EvalBenchmark> {
        Ok(EvalBenchmark {
            benchmark_id: BenchmarkId::try_new("eval_math_500_2026.06.1".to_owned())?,
            parameter_eval_id: EvalId::from("math_500"),
            parameter_dataset_name: "2026.06.1".into(),
            parameter_max_tokens: PositiveCount::try_new(8192)?,
            parameter_max_context: None,
            parameter_mcq_choices: None,
            parameter_temperature: None,
            parameter_top_p: None,
            parameter_top_k: None,
            parameter_min_p: None,
            parameter_seed: None,
            samples: None,
        })
    }

    #[test]
    fn an_unpinned_cell_sends_only_its_temperature() -> anyhow::Result<()> {
        let benchmark = eval_benchmark()?;
        let sampling = Sampling::from_benchmark(&benchmark)?;
        let mut body = json!({"prompt": "hi"});
        sampling.apply(&mut body, "row-1#0")?;
        assert_eq!(body, json!({"prompt": "hi", "temperature": 0.6}));
        Ok(())
    }

    /// The seed is the sample's own: the `#k` repeats of one sample have to be
    /// independent draws.
    #[test]
    fn a_pinned_cell_sends_every_knob() -> anyhow::Result<()> {
        let benchmark = EvalBenchmark {
            parameter_temperature: Some(SamplingTemperature::try_new(1.0)?),
            parameter_top_p: Some(TopP::try_new(0.95)?),
            parameter_top_k: Some(TopK::try_new(64)?),
            parameter_min_p: Some(MinP::try_new(0.05)?),
            parameter_seed: Some(SamplingSeed::try_new(7)?),
            ..eval_benchmark()?
        };
        let sampling = Sampling::from_benchmark(&benchmark)?;
        let mut body = json!({"prompt": "hi"});
        sampling.apply(&mut body, "row-1#2")?;
        assert_eq!(
            body,
            json!({
                "prompt": "hi",
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "min_p": 0.05,
                "seed": 9,
            })
        );
        Ok(())
    }

    /// The mobile clients are held to the contract's `sampling.json`.
    #[test]
    fn every_contract_sample_is_written_as_the_contract_records_it() -> anyhow::Result<()> {
        let fixtures =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../pipette-contract/fixtures");
        let recorded: std::collections::BTreeMap<String, serde_json::Map<String, Value>> =
            serde_json::from_str(&std::fs::read_to_string(fixtures.join("sampling.json"))?)?;
        anyhow::ensure!(!recorded.is_empty(), "sampling.json records no eval");
        for (example, samples) in recorded {
            let definition: BenchmarkDefinition =
                serde_json::from_str(&std::fs::read_to_string(fixtures.join(&example))?)?;
            let sampling = Sampling::from_benchmark(definition.as_eval()?)?;
            for (id, drawn) in samples {
                let mut body = json!({});
                sampling.apply(&mut body, &id)?;
                assert_eq!(body, drawn, "{example} {id}");
            }
        }
        Ok(())
    }

    #[test]
    fn the_cell_carries_the_benchmarks_sampling() -> anyhow::Result<()> {
        let mut req = request(2)?;
        let BenchmarkDefinition::Eval(benchmark) = &mut req.benchmark else {
            anyhow::bail!("expected an eval benchmark");
        };
        benchmark.parameter_temperature = Some(SamplingTemperature::try_new(1.0)?);
        benchmark.parameter_seed = Some(SamplingSeed::try_new(7)?);
        let cell = EvalCell::from_request(&req, None)?;
        assert_eq!(
            cell.sampling.temperature,
            SamplingTemperature::try_new(1.0)?
        );
        assert_eq!(cell.sampling.seed_for("s0#3"), Some(10));
        Ok(())
    }
}
