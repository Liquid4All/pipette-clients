//! What a benchmark *is*: the per-kind parameter structs and the tagged
//! [`BenchmarkDefinition`](crate::benchmark::BenchmarkDefinition) over them, plus the catalog origin each one
//! arrived by.
//!
//! Pure wire vocabulary — no I/O. [`BenchmarkType`](crate::BenchmarkType)
//! is the closed kind enum this mirrors; `result` is what a finished cell
//! produced.

pub mod eval_id;

use nutype::nutype;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use self::eval_id::EvalId;
use crate::BenchmarkType;

// ---------------------------------------------------------------------------
// BenchmarkSource — local (generated) vs remote (synced) catalog origin
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BenchmarkSource {
    Local,
    Remote,
}

/// Sampling temperature for eval `/completion` requests, bounded to
/// `[0.0, 2.0]`.
///
/// `0.0` is greedy decoding; `2.0` is the maximum the OpenAI-compatible
/// servers (vLLM / SGLang) the torch-oai client targets accept, and
/// llama.cpp / mlx accept the same range. A value outside the range — or
/// a non-finite one — is a configuration error rather than something to
/// silently clamp, so construction is fallible (`try_new`). The range
/// validators reject `NaN`/`±inf` on their own; `finite` is kept for
/// intent.
#[nutype(
    validate(finite, greater_or_equal = 0.0, less_or_equal = 2.0),
    derive(Debug, Clone, Copy, PartialEq, PartialOrd, Display)
)]
pub struct Temperature(f64);

impl Temperature {
    /// Greedy decoding — temperature `0.0`. The historical behavior for
    /// every eval.
    pub fn greedy() -> anyhow::Result<Self> {
        Ok(Self::try_new(0.0)?)
    }

    /// The bare `f64`, for inclusion in `/completion` request bodies.
    pub fn as_f64(self) -> f64 {
        self.into_inner()
    }
}

/// A [`BenchmarkDefinition::as_*`](BenchmarkDefinition) accessor was called for
/// the wrong kind.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("expected {expected} benchmark, got `{got_id}` ({got_type})")]
pub struct UnexpectedBenchmarkType {
    pub expected: &'static str,
    pub got_id: String,
    pub got_type: String,
}

// ---------------------------------------------------------------------------
// BenchmarkDefinition — per-kind structs + tagged enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillThroughput {
    pub benchmark_id: String,
    pub parameter_prefill_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeThroughput {
    pub benchmark_id: String,
    pub parameter_prefill_tokens: u32,
    pub parameter_decode_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EndToEndLatency {
    pub benchmark_id: String,
    pub parameter_prefill_tokens: u32,
    pub parameter_decode_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaxMemoryUsage {
    pub benchmark_id: String,
    pub parameter_prefill_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalBenchmark {
    pub benchmark_id: String,
    pub parameter_eval_id: EvalId,
    pub parameter_dataset_name: String,
    pub parameter_max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameter_mcq_choices: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<Vec<Value>>,
}

impl EvalBenchmark {
    /// Client-side sampling temperature for `/completion` (IFBench/IFStruct 0.6;
    /// unknown eval ids stay greedy at 0.0).
    pub fn sampling_temperature(&self) -> anyhow::Result<Temperature> {
        Ok(Temperature::try_new(
            self.parameter_eval_id.sampling_temperature(),
        )?)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VlThroughput {
    pub benchmark_id: String,
    pub parameter_image_width: u32,
    pub parameter_image_height: u32,
    pub parameter_text_tokens: u32,
    pub parameter_decode_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "benchmark_type", rename_all = "snake_case")]
pub enum BenchmarkDefinition {
    PrefillThroughput(PrefillThroughput),
    DecodeThroughput(DecodeThroughput),
    EndToEndLatency(EndToEndLatency),
    MaxMemoryUsage(MaxMemoryUsage),
    Eval(EvalBenchmark),
    VlThroughput(VlThroughput),
}

impl BenchmarkDefinition {
    pub fn benchmark_id(&self) -> &str {
        match self {
            Self::PrefillThroughput(b) => &b.benchmark_id,
            Self::DecodeThroughput(b) => &b.benchmark_id,
            Self::EndToEndLatency(b) => &b.benchmark_id,
            Self::MaxMemoryUsage(b) => &b.benchmark_id,
            Self::Eval(b) => &b.benchmark_id,
            Self::VlThroughput(b) => &b.benchmark_id,
        }
    }

    pub fn benchmark_type(&self) -> BenchmarkType {
        match self {
            Self::PrefillThroughput(_) => BenchmarkType::PrefillThroughput,
            Self::DecodeThroughput(_) => BenchmarkType::DecodeThroughput,
            Self::EndToEndLatency(_) => BenchmarkType::EndToEndLatency,
            Self::MaxMemoryUsage(_) => BenchmarkType::MaxMemoryUsage,
            Self::Eval(_) => BenchmarkType::Eval,
            Self::VlThroughput(_) => BenchmarkType::VlThroughput,
        }
    }

    /// Eval sampling temperature, or greedy `0.0` for non-eval kinds.
    pub fn eval_temperature(&self) -> anyhow::Result<Temperature> {
        match self {
            Self::Eval(b) => b.sampling_temperature(),
            _ => Ok(Temperature::try_new(0.0)?),
        }
    }

    fn unexpected_type(&self, expected: &'static str) -> UnexpectedBenchmarkType {
        UnexpectedBenchmarkType {
            expected,
            got_id: self.benchmark_id().to_owned(),
            got_type: self.benchmark_type().to_string(),
        }
    }

    pub fn as_prefill_throughput(&self) -> Result<&PrefillThroughput, UnexpectedBenchmarkType> {
        match self {
            Self::PrefillThroughput(b) => Ok(b),
            _ => Err(self.unexpected_type("prefill_throughput")),
        }
    }

    pub fn as_decode_throughput(&self) -> Result<&DecodeThroughput, UnexpectedBenchmarkType> {
        match self {
            Self::DecodeThroughput(b) => Ok(b),
            _ => Err(self.unexpected_type("decode_throughput")),
        }
    }

    pub fn as_end_to_end_latency(&self) -> Result<&EndToEndLatency, UnexpectedBenchmarkType> {
        match self {
            Self::EndToEndLatency(b) => Ok(b),
            _ => Err(self.unexpected_type("end_to_end_latency")),
        }
    }

    pub fn as_max_memory_usage(&self) -> Result<&MaxMemoryUsage, UnexpectedBenchmarkType> {
        match self {
            Self::MaxMemoryUsage(b) => Ok(b),
            _ => Err(self.unexpected_type("max_memory_usage")),
        }
    }

    pub fn as_eval(&self) -> Result<&EvalBenchmark, UnexpectedBenchmarkType> {
        match self {
            Self::Eval(b) => Ok(b),
            _ => Err(self.unexpected_type("eval")),
        }
    }

    pub fn as_vl_throughput(&self) -> Result<&VlThroughput, UnexpectedBenchmarkType> {
        match self {
            Self::VlThroughput(b) => Ok(b),
            _ => Err(self.unexpected_type("vl_throughput")),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn as_eval_errors_on_prefill_definition() -> anyhow::Result<()> {
        let def = BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
            benchmark_id: "prefill_throughput_100".into(),
            parameter_prefill_tokens: 100,
        });
        match def.as_eval() {
            Err(UnexpectedBenchmarkType {
                expected,
                got_id,
                got_type,
            }) => {
                assert_eq!(expected, "eval");
                assert_eq!(got_id, "prefill_throughput_100");
                assert_eq!(got_type, "prefill_throughput");
            }
            other => anyhow::bail!("expected UnexpectedBenchmarkType, got {other:?}"),
        }
        Ok(())
    }

    #[test]
    fn benchmark_definition_round_trip_prefill() -> anyhow::Result<()> {
        let spec = BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
            benchmark_id: "prefill_throughput_512".to_string(),
            parameter_prefill_tokens: 512,
        });
        assert_eq!(
            serde_json::from_value::<BenchmarkDefinition>(serde_json::to_value(&spec)?)?,
            spec
        );
        Ok(())
    }

    #[test]
    fn benchmark_definition_round_trip_eval() -> anyhow::Result<()> {
        let spec = BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: "eval_mmlu_pro_edge_2026.03.1".to_string(),
            parameter_eval_id: "mmlu_pro".into(),
            parameter_dataset_name: "edge_2026.03.1".to_string(),
            parameter_max_tokens: 256,
            parameter_mcq_choices: Some(vec!["A".to_string(), "B".to_string()]),
            samples: Some(vec![Value::from("sample")]),
        });
        assert_eq!(
            serde_json::from_value::<BenchmarkDefinition>(serde_json::to_value(&spec)?)?,
            spec
        );
        Ok(())
    }

    #[test]
    fn benchmark_definition_round_trip_vl_throughput() -> anyhow::Result<()> {
        let spec = BenchmarkDefinition::VlThroughput(VlThroughput {
            benchmark_id: "vl_throughput_224x224_8_64".to_string(),
            parameter_image_width: 224,
            parameter_image_height: 224,
            parameter_text_tokens: 8,
            parameter_decode_tokens: 64,
        });
        assert_eq!(
            serde_json::from_value::<BenchmarkDefinition>(serde_json::to_value(&spec)?)?,
            spec
        );
        Ok(())
    }

    #[test]
    fn benchmark_definition_serializes_to_flat_document() -> anyhow::Result<()> {
        let spec = BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: "eval_mmlu_pro_edge_2026.03.1".to_string(),
            parameter_eval_id: "mmlu_pro".into(),
            parameter_dataset_name: "edge_2026.03.1".to_string(),
            parameter_max_tokens: 256,
            parameter_mcq_choices: Some(vec!["A".to_string(), "B".to_string()]),
            samples: Some(vec![Value::from("sample")]),
        });

        let value = serde_json::to_value(&spec)?;
        assert_eq!(
            value,
            serde_json::json!({
                "benchmark_id": "eval_mmlu_pro_edge_2026.03.1",
                "benchmark_type": "eval",
                "parameter_eval_id": "mmlu_pro",
                "parameter_dataset_name": "edge_2026.03.1",
                "parameter_max_tokens": 256,
                "parameter_mcq_choices": ["A", "B"],
                "samples": ["sample"],
            })
        );
        Ok(())
    }

    // The four scoreable evals are generative pass@k -> 0.6; anything
    // pipette-scores can't score (calibration-only or unknown) is greedy.
    #[rstest]
    #[case("ifbench", 0.6)]
    #[case("ifstruct", 0.6)]
    #[case("gpqa_diamond", 0.6)]
    #[case("math_500", 0.6)]
    #[case("mmlu_pro", 0.0)]
    #[case("totally_unknown_eval", 0.0)]
    fn eval_temperature_by_eval_id(
        #[case] eval_id: &str,
        #[case] expected: f64,
    ) -> anyhow::Result<()> {
        let bench = BenchmarkDefinition::Eval(EvalBenchmark {
            benchmark_id: format!("eval_{eval_id}_2026.06.1"),
            parameter_eval_id: eval_id.into(),
            parameter_dataset_name: "edge_2026.06.1".to_string(),
            parameter_max_tokens: 1024,
            parameter_mcq_choices: None,
            samples: None,
        });
        assert_eq!(
            bench.eval_temperature()?.as_f64(),
            expected,
            "eval_id={eval_id}"
        );
        Ok(())
    }

    #[test]
    fn eval_temperature_non_eval_is_greedy() -> anyhow::Result<()> {
        let prefill = BenchmarkDefinition::PrefillThroughput(PrefillThroughput {
            benchmark_id: "prefill_throughput_512".to_string(),
            parameter_prefill_tokens: 512,
        });
        assert_eq!(prefill.eval_temperature()?, Temperature::greedy()?);
        Ok(())
    }

    #[rstest::rstest]
    #[case(0.0)]
    #[case(0.6)]
    #[case(2.0)]
    fn temperature_accepts_in_range(#[case] value: f64) -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(Temperature::try_new(value)?.as_f64(), value);
        Ok(())
    }

    // Out-of-range and non-finite values are rejected, not clamped.
    #[rstest::rstest]
    #[case(-0.1)]
    #[case(2.1)]
    #[case(f64::NAN)]
    #[case(f64::INFINITY)]
    #[case(f64::NEG_INFINITY)]
    fn temperature_rejects_out_of_range(#[case] value: f64) {
        assert!(
            Temperature::try_new(value).is_err(),
            "temperature {value} must be rejected"
        );
    }

    #[test]
    fn temperature_greedy_is_zero() -> anyhow::Result<()> {
        assert_eq!(Temperature::greedy()?.as_f64(), 0.0);
        Ok(())
    }
}
