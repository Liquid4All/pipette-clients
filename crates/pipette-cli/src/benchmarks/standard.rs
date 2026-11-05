//! Built-in local ladder + smoke benchmark definitions.

use serde_json::json;

use pipette_plan_types::{
    AsrDecodeTokens, AsrMaxMemoryBenchmark, AsrThroughputBenchmark, AudioChannels,
    AudioDurationMillis, AudioSampleRate, BatchSize, BenchmarkDefinition, BenchmarkId,
    BenchmarkSource, BenchmarkType, ChatMessage, ChatRole, ContentPart, DecodeThroughputBenchmark,
    EndToEndLatencyBenchmark, EvalBenchmark, EvalSample, ImageUrl, MaxMemoryUsageBenchmark,
    MessageContent, NonNegativeCount, PositiveCount, PrefillThroughputBenchmark,
    VlMaxMemoryBenchmark, VlThroughputBenchmark, VlmEvalBenchmark,
};

use super::reference::SourcedBenchmarkId;
use super::store::BenchmarkStore;
use crate::error::Result;

/// Token ladder shared across every kind that has a ladder form.
const LADDER_TOKENS: [i32; 7] = [100, 256, 512, 1024, 2048, 4096, 8192];

/// The ladder over [`LADDER_TOKENS`] plus a `_smoke` set, for the kinds the
/// caller can execute. `kinds` is a set: order and duplicates do not change
/// the output.
///
/// Kinds without a ladder form (`Eval`, `VlmEval`, `VlThroughput`,
/// `VlMaxMemory`, `AsrThroughput`, `AsrMaxMemory`) only contribute a smoke
/// entry. `VlSpecDecode` and `AsrEval` contribute neither: their workload is a
/// corpus the management server serves.
fn standard_local_benchmarks(kinds: &[BenchmarkType]) -> Result<Vec<BenchmarkDefinition>> {
    let mut benchmarks = ladder_local_benchmarks(kinds)?;
    benchmarks.extend(smoke_local_benchmarks(kinds)?);
    Ok(benchmarks)
}

fn ladder_local_benchmarks(kinds: &[BenchmarkType]) -> Result<Vec<BenchmarkDefinition>> {
    LADDER_TOKENS
        .into_iter()
        .flat_map(|tokens| selected_kinds(kinds).map(move |k| ladder_entry(k, tokens)))
        .filter_map(Result::transpose)
        .collect()
}

fn smoke_local_benchmarks(kinds: &[BenchmarkType]) -> Result<Vec<BenchmarkDefinition>> {
    selected_kinds(kinds)
        .map(smoke_entry)
        .filter_map(Result::transpose)
        .collect()
}

fn benchmark_id(id: impl Into<String>) -> Result<BenchmarkId> {
    BenchmarkId::try_new(id.into()).map_err(|e| crate::error::Error::Other(e.into()))
}

fn positive(count: i32) -> Result<PositiveCount> {
    PositiveCount::try_new(count).map_err(|e| crate::error::Error::Other(e.into()))
}

fn non_negative(count: i32) -> Result<NonNegativeCount> {
    NonNegativeCount::try_new(count).map_err(|e| crate::error::Error::Other(e.into()))
}

/// The ASR smoke clip's format: 16 kHz mono, what the audio encoders take.
fn smoke_sample_rate() -> Result<AudioSampleRate> {
    AudioSampleRate::try_new(16_000).map_err(|e| crate::error::Error::Other(e.into()))
}

fn smoke_channels() -> Result<AudioChannels> {
    AudioChannels::try_new(1).map_err(|e| crate::error::Error::Other(e.into()))
}

/// The ASR smoke clip: 2.5 s, short enough for any encoder's first window.
fn smoke_duration() -> Result<AudioDurationMillis> {
    AudioDurationMillis::try_new(2_500).map_err(|e| crate::error::Error::Other(e.into()))
}

fn smoke_decode_tokens() -> Result<AsrDecodeTokens> {
    AsrDecodeTokens::try_new(8).map_err(|e| crate::error::Error::Other(e.into()))
}

/// Iterate the kinds in canonical (declaration) order, restricted to those
/// the caller selected. Driving from [`BenchmarkType::ALL`] makes the
/// output order independent of `kinds` order and tolerant of duplicates;
/// `ALL` is auto-generated via `#[derive(VariantArray)]` so a new enum
/// variant is automatically considered, then forced through the exhaustive
/// `match` in [`ladder_entry`] / [`smoke_entry`].
fn selected_kinds(kinds: &[BenchmarkType]) -> impl Iterator<Item = BenchmarkType> + '_ {
    BenchmarkType::ALL
        .iter()
        .copied()
        .filter(|k| kinds.contains(k))
}

/// `None` when the kind has no ladder form.
fn ladder_entry(kind: BenchmarkType, tokens: i32) -> Result<Option<BenchmarkDefinition>> {
    Ok(match kind {
        BenchmarkType::PrefillThroughput => Some(BenchmarkDefinition::PrefillThroughput(
            PrefillThroughputBenchmark {
                benchmark_id: benchmark_id(format!("prefill_throughput_{tokens}"))?,
                parameter_prefill_tokens: non_negative(tokens)?,
                parameter_batch_size: BatchSize::one(),
            },
        )),
        BenchmarkType::DecodeThroughput => Some(BenchmarkDefinition::DecodeThroughput(
            DecodeThroughputBenchmark {
                benchmark_id: benchmark_id(format!("decode_throughput_{tokens}_100"))?,
                parameter_prefill_tokens: non_negative(tokens)?,
                parameter_decode_tokens: positive(100)?,
                parameter_batch_size: BatchSize::one(),
            },
        )),
        BenchmarkType::EndToEndLatency => Some(BenchmarkDefinition::EndToEndLatency(
            EndToEndLatencyBenchmark {
                benchmark_id: benchmark_id(format!("end_to_end_latency_{tokens}_256"))?,
                parameter_prefill_tokens: non_negative(tokens)?,
                parameter_decode_tokens: positive(256)?,
                parameter_batch_size: BatchSize::one(),
            },
        )),
        BenchmarkType::MaxMemoryUsage => Some(BenchmarkDefinition::MaxMemoryUsage(
            MaxMemoryUsageBenchmark {
                benchmark_id: benchmark_id(format!("max_memory_usage_{tokens}"))?,
                parameter_prefill_tokens: non_negative(tokens)?,
            },
        )),
        BenchmarkType::Eval
        | BenchmarkType::VlmEval
        | BenchmarkType::AsrEval
        | BenchmarkType::VlThroughput
        | BenchmarkType::VlMaxMemory
        | BenchmarkType::VlSpecDecode
        | BenchmarkType::AsrThroughput
        | BenchmarkType::AsrMaxMemory => None,
    })
}

/// `None` for a kind that has no local form.
fn smoke_entry(kind: BenchmarkType) -> Result<Option<BenchmarkDefinition>> {
    Ok(Some(match kind {
        BenchmarkType::PrefillThroughput => {
            BenchmarkDefinition::PrefillThroughput(PrefillThroughputBenchmark {
                benchmark_id: benchmark_id("prefill_throughput_smoke")?,
                parameter_prefill_tokens: non_negative(8)?,
                parameter_batch_size: BatchSize::one(),
            })
        }
        BenchmarkType::DecodeThroughput => {
            BenchmarkDefinition::DecodeThroughput(DecodeThroughputBenchmark {
                benchmark_id: benchmark_id("decode_throughput_smoke")?,
                parameter_prefill_tokens: non_negative(8)?,
                parameter_decode_tokens: positive(8)?,
                parameter_batch_size: BatchSize::one(),
            })
        }
        BenchmarkType::EndToEndLatency => {
            BenchmarkDefinition::EndToEndLatency(EndToEndLatencyBenchmark {
                benchmark_id: benchmark_id("end_to_end_latency_smoke")?,
                parameter_prefill_tokens: non_negative(8)?,
                parameter_decode_tokens: positive(8)?,
                parameter_batch_size: BatchSize::one(),
            })
        }
        BenchmarkType::MaxMemoryUsage => {
            BenchmarkDefinition::MaxMemoryUsage(MaxMemoryUsageBenchmark {
                benchmark_id: benchmark_id("max_memory_usage_smoke")?,
                parameter_prefill_tokens: non_negative(8)?,
            })
        }
        BenchmarkType::Eval => BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: benchmark_id("eval_smoke")?,
            parameter_eval_id: "eval_smoke".into(),
            parameter_dataset_name: "local".to_string(),
            parameter_max_tokens: positive(4)?,
            parameter_max_context: None,
            parameter_mcq_choices: None,
            parameter_temperature: None,
            parameter_top_p: None,
            parameter_top_k: None,
            parameter_min_p: None,
            parameter_seed: None,
            samples: Some(vec![json!({
                "id": "smoke-1",
                "messages": [
                    { "role": "user", "content": "Reply with exactly OK." }
                ]
            })]),
        }),
        BenchmarkType::VlmEval => BenchmarkDefinition::VlmEval(VlmEvalBenchmark {
            benchmark_id: benchmark_id("vlm_eval_smoke")?,
            parameter_lmms_task: "vlm_eval_smoke".into(),
            parameter_dataset_version: "local".to_string(),
            parameter_sample_count: positive(1)?,
            parameter_max_tokens: positive(4)?,
            parameter_mcq_choices: None,
            parameter_temperature: None,
            parameter_top_p: None,
            parameter_top_k: None,
            parameter_min_p: None,
            parameter_presence_penalty: None,
            parameter_repeat_penalty: None,
            parameter_seed: None,
            parameter_seed_rule: None,
            parameter_prompt_profile: None,
            parameter_image_tokens: None,
            dataset: None,
            recipe: None,
            // A 224×224 PNG in the server's content-array shape, so a smoke run
            // exercises the vision tower. A 1×1 image would encode to a
            // degenerate patch grid, not a real image's work.
            samples: Some(vec![EvalSample {
                id: "smoke-1".to_string(),
                messages: vec![ChatMessage {
                    role: ChatRole::User,
                    content: MessageContent::Parts(vec![
                        ContentPart::Text {
                            text: "Reply with exactly OK.".to_string(),
                        },
                        ContentPart::ImageUrl {
                            image_url: ImageUrl {
                                url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAAAAAA/RjU9AAAA3UlEQVR42u3PAQkAAAwDoEVf9MU4HG1g+lwEBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBe8MJSoFv1MoF1AAAAAASUVORK5CYII=".to_string(),
                            },
                        },
                    ]),
                }],
                chat_template_kwargs: None,
            }]),
        }),
        BenchmarkType::VlThroughput => BenchmarkDefinition::VlThroughput(VlThroughputBenchmark {
            benchmark_id: benchmark_id("vl_throughput_smoke")?,
            parameter_image_width: positive(224)?,
            parameter_image_height: positive(224)?,
            parameter_text_tokens: non_negative(8)?,
            parameter_decode_tokens: positive(8)?,
            parameter_num_images: non_negative(1)?,
        }),
        BenchmarkType::VlMaxMemory => BenchmarkDefinition::VlMaxMemory(VlMaxMemoryBenchmark {
            benchmark_id: benchmark_id("vl_max_memory_smoke")?,
            parameter_image_width: positive(224)?,
            parameter_image_height: positive(224)?,
            parameter_text_tokens: non_negative(8)?,
            parameter_decode_tokens: None,
            parameter_num_images: non_negative(1)?,
        }),
        // Their samples come only as the corpus a catalog entry pins, from the
        // management server at run time, so a local definition would have none.
        BenchmarkType::VlSpecDecode | BenchmarkType::AsrEval => return Ok(None),
        BenchmarkType::AsrMaxMemory => BenchmarkDefinition::AsrMaxMemory(AsrMaxMemoryBenchmark {
            benchmark_id: benchmark_id("asr_max_memory_smoke")?,
            parameter_audio_duration_ms: smoke_duration()?,
            parameter_audio_sample_rate_hz: smoke_sample_rate()?,
            parameter_audio_channels: smoke_channels()?,
            parameter_text_tokens: non_negative(8)?,
            parameter_decode_tokens: smoke_decode_tokens()?,
        }),
        BenchmarkType::AsrThroughput => {
            BenchmarkDefinition::AsrThroughput(AsrThroughputBenchmark {
                benchmark_id: benchmark_id("asr_throughput_smoke")?,
                parameter_audio_duration_ms: smoke_duration()?,
                parameter_audio_sample_rate_hz: smoke_sample_rate()?,
                parameter_audio_channels: smoke_channels()?,
                parameter_text_tokens: non_negative(8)?,
                parameter_decode_tokens: smoke_decode_tokens()?,
            })
        }
    }))
}

/// Counts from [`seed_standard_local`].
#[derive(Debug, Clone, Copy)]
pub struct LocalBenchmarkInitSummary {
    pub created: usize,
    pub updated: usize,
}

/// Seed the standard set for `kinds` into the local catalog. Catalog policy
/// lives here; the store only receives ready definitions.
pub fn seed_standard_local(
    store: &BenchmarkStore,
    kinds: &[BenchmarkType],
) -> Result<LocalBenchmarkInitSummary> {
    let mut created = 0usize;
    let mut updated = 0usize;
    for def in standard_local_benchmarks(kinds)? {
        let id = def.benchmark_id().clone();
        let existed = store
            .get(&SourcedBenchmarkId::new(BenchmarkSource::Local, id))?
            .is_some();
        store.put_local(&def)?;
        if existed {
            updated += 1;
        } else {
            created += 1;
        }
    }
    Ok(LocalBenchmarkInitSummary { created, updated })
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::BenchmarkType::*;

    use super::*;

    #[test]
    fn every_kind_seeds_thirty_eight() -> Result<()> {
        assert_eq!(standard_local_benchmarks(BenchmarkType::ALL)?.len(), 38);
        Ok(())
    }

    #[test]
    fn kinds_are_a_set() -> Result<()> {
        let ids = |kinds: &[BenchmarkType]| -> Result<Vec<String>> {
            Ok(standard_local_benchmarks(kinds)?
                .iter()
                .map(|d| d.benchmark_id().to_string())
                .collect())
        };
        assert_eq!(
            ids(&[Eval, PrefillThroughput, Eval])?,
            ids(&[PrefillThroughput, Eval])?
        );
        Ok(())
    }

    /// The ids are what `--benchmark local/<id>` and the docs name, so their
    /// spelling is pinned.
    #[test]
    fn the_ladder_ids_are_stable() -> Result<()> {
        let ids: Vec<String> = standard_local_benchmarks(&[DecodeThroughput, EndToEndLatency])?
            .iter()
            .map(|d| d.benchmark_id().to_string())
            .collect();
        assert_eq!(
            ids.get(..2),
            Some(
                &[
                    "decode_throughput_100_100".to_owned(),
                    "end_to_end_latency_100_256".to_owned()
                ][..]
            )
        );
        assert!(ids.contains(&"decode_throughput_8192_100".to_owned()));
        assert!(ids.contains(&"end_to_end_latency_smoke".to_owned()));
        Ok(())
    }

    /// The corpus comes from the server, so there is nothing local to seed.
    #[test]
    fn vl_spec_decode_seeds_nothing() -> Result<()> {
        assert!(standard_local_benchmarks(&[VlSpecDecode])?.is_empty());
        Ok(())
    }

    #[test]
    fn kinds_without_a_ladder_contribute_only_smoke() -> Result<()> {
        let ids: Vec<String> = standard_local_benchmarks(&[Eval, VlThroughput, VlMaxMemory])?
            .iter()
            .map(|d| d.benchmark_id().to_string())
            .collect();
        assert_eq!(
            ids,
            ["eval_smoke", "vl_throughput_smoke", "vl_max_memory_smoke"]
        );
        Ok(())
    }
}
