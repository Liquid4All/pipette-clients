//! What a finished cell produced — the counterpart to the plan vocabulary that
//! says what to run.
//!
//! These are the client-side mirrors of the management server's submission
//! schema: [`BenchmarkResultData`](crate::result::BenchmarkResultData) is the per-benchmark measurement payload and
//! [`BenchmarkEvalCompletion`](crate::result::BenchmarkEvalCompletion) (with [`BenchmarkEvalCompletionStopReason`](crate::result::BenchmarkEvalCompletionStopReason)) is one eval sample's outcome. Pure
//! data — the engines that measure and the stores that record both name these
//! without depending on each other. Namespaced rather than re-exported flat, so
//! consumers reference them as `pipette_plan_types::result::BenchmarkEvalCompletionStopReason`.

use serde::{Deserialize, Serialize};

use crate::device::DeviceInfo;
use crate::thermal::{DevicePowerState, ThermalTelemetry};
use crate::BenchmarkType;

/// Canonical per-sample stop reason.
///
/// The enum is **owned by pipette-mgmt** (`docs/scoring-service.md`); this
/// is the client-side mirror so runtimes produce values type-safely
/// instead of hand-writing wire strings. Variants serialize to the exact
/// snake_case tokens the mgmt receiver expects. Keep in sync with the
/// canonical definition and with the Swift / Kotlin client mirrors.
///
/// **Provenance caveat.** mgmt records a client-produced `stop_reason` as
/// `recorded` regardless of *how* the client derived it — that axis means
/// "captured at generation by the client," not "authoritative." An
/// engine-signal classification (llama.cpp `stop_type`, MLX EOS break) and
/// a client-side heuristic (AFM: generated-token count vs the cap) both
/// land as `recorded`; the enum does not distinguish authoritative from
/// heuristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkEvalCompletionStopReason {
    /// Model emitted EOS — completion tokens `< cap`.
    Eos,
    /// Hit the output-token cap — completion tokens `== cap`
    /// (`n_predict = parameter_max_tokens`).
    Truncated,
    /// The client doom-loop detector aborted generation. Client-only —
    /// not recoverable from stored text.
    DoomLoop,
    /// Empty completion / runtime crash — the sample never produced a
    /// completion. Set by the client on its failed path, alongside the
    /// legacy `failed` / `failed_reason` fields (which carry the detail).
    Failure,
    /// Attempted but indeterminate. On the client this is the catch-all for
    /// every case that can't be classified at generation — the MCQ arm, an
    /// unrecognized/absent stop signal, a resumed pre-feature checkpoint
    /// sample this binary didn't produce. The client never emits an absent
    /// reason, so a `NULL`/absent `stop_reason` seen downstream means the
    /// submission came from a client predating the field entirely.
    #[default]
    Unknown,
}

/// One eval sample's outcome. A normal completion serializes to the wire
/// format `{"id": "...", "completion": "...", "stop_reason": "..."}`. When
/// the runtime crashed while serving the sample, `failed` is set to `true`
/// and an optional human-readable `failed_reason` is attached; the empty
/// `completion` is excluded from scoring on the server side, and the
/// per-sample row in the warehouse keeps the `failed` flag so downstream
/// UIs can surface it. The local checkpoint file also keeps the entry so
/// future runs skip the same sample without re-triggering the crash.
///
/// Schema is forward-compatible: every non-core field defaults on parse, so
/// pre-feature checkpoint files and submissions load cleanly (`stop_reason`
/// → `unknown`; `failed` → `false`; the rest → absent). `failed`,
/// `failed_reason`, and `completion_tokens` elide on the wire when unset;
/// `stop_reason` is required on the client and always written.
///
/// **`failed_reason` prefix convention.** Producers MAY prefix the
/// reason with an RFC3339 timestamp in square brackets —
/// `[<rfc3339-timestamp>] <free-form>` — to preserve "when did this
/// fail" without a dedicated wire field. The prefix is optional;
/// consumers that want to render the timestamp separately must parse
/// defensively and tolerate bare reasons.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BenchmarkEvalCompletion {
    pub id: String,
    pub completion: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub failed: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failed_reason: Option<String>,
    /// Stop reason captured at generation. **Required by convention** — the
    /// client always writes it (indeterminate → `unknown`, never absent), but
    /// it isn't validated: `#[serde(default)]` decodes a missing value as
    /// `unknown` so a pre-feature checkpoint still loads. mgmt keeps its own
    /// copy nullable for submissions from clients predating the field.
    #[serde(default)]
    pub stop_reason: BenchmarkEvalCompletionStopReason,
    /// Free-form diagnostic paired with `stop_reason` — the *why / raw signal*
    /// behind it, for debugging one sample without re-running: the crash detail
    /// for `failure` (mirrors `failed_reason` during the transition), the raw
    /// runtime signal for `unknown` (e.g. `stop_type=word`, `stream dropped`),
    /// the detector trigger for `doom_loop`. `None` for a clean `eos` /
    /// `truncated`. Generalizes `failed_reason`, which is retired once
    /// consumers move over.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_detail: Option<String>,
    /// Completion (output) token count for this sample, when the runtime
    /// reported it. Pairs with `stop_reason` to separate `eos` (< cap)
    /// from `truncated` (== cap). `None` when not reported. **Basis is
    /// runtime-specific** (llama.cpp: server-reported count; other runtimes
    /// may use a generated-token counter or a re-tokenized estimate), so
    /// treat cross-runtime comparisons as approximate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
}

/// What a run's memory looked like while it ran, on every benchmark kind rather
/// than only the memory one.
///
/// These are **observations, not metrics**, in the same sense as
/// `observation_vl_throughput_prefill_tokens`: measured workload facts that
/// qualify a result instead of scoring it. A decode-throughput row that hit its
/// number while zram held part of the model is a different fact from one that
/// stayed resident, and without this nothing in the row says which happened.
///
/// Unlike the `observation_vl_throughput_*` columns these carry no benchmark
/// prefix, because they are not specific to one benchmark type.
///
/// Every field is optional and per platform: a platform with no sampler
/// contributes no keys, so the wire shape is unchanged for a client that
/// populates nothing. Absence means "not observed", never zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryObservation {
    /// Peak *resident* memory the run held: the kernel's high-water mark for
    /// the pages actually in RAM.
    ///
    /// Resident-only on every arm that can be, so the name means one thing
    /// across platforms — `VmHWM` on Android and Linux, `PeakWorkingSetSize` on
    /// Windows. macOS is the exception it cannot avoid: `phys_footprint` bills
    /// compressed pages to the process and the kernel publishes no resident-only
    /// counter beside it.
    ///
    /// Pages the kernel swapped out are deliberately **not** counted here. What
    /// tells a reader this figure was suppressed by reclaim is
    /// [`Self::max_swap_bytes`] standing beside it.
    #[serde(
        rename = "observation_max_host_bytes",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub max_host_bytes: Option<u64>,
    /// Most the kernel held of this run in swap at once.
    ///
    /// A term in its own right, **not** a component of [`Self::max_host_bytes`]
    /// and not additive with it: the resident watermark and the largest swap
    /// reading need not fall at the same instant, so their sum describes no
    /// moment that happened. Read it as the qualifier on the peak beside it —
    /// a non-zero swap term means the resident figure under-states what the run
    /// required, and by roughly how much.
    ///
    /// `Some(0)` is a real reading, meaning the platform sampled swap and the
    /// run stayed resident, which is what makes the peak beside it trustworthy;
    /// `None` means nothing sampled it.
    #[serde(
        rename = "observation_max_swap_bytes",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub max_swap_bytes: Option<u64>,
}

impl MemoryObservation {
    /// A peak from a sampler that sees no swap term, or nothing if it read
    /// nothing.
    ///
    /// Zero is how every sampler here spells "no read landed", so it is withheld
    /// rather than reported: absence is filterable, and a zero on the row reads
    /// as a measurement. Constructing through this instead of the literal is what
    /// keeps that rule in one place — the arms that built their own once shipped
    /// a `Some(0)`.
    pub fn host_only(host_bytes: u64) -> Self {
        Self {
            max_host_bytes: (host_bytes > 0).then_some(host_bytes),
            max_swap_bytes: None,
        }
    }

    /// A peak with the swap term beside it, from a sampler that reads both.
    ///
    /// Same rule for the peak, and the swap term rides with it: a withheld peak
    /// withholds the swap reading too, since half an observation says the
    /// platform sampled swap while refusing to say what it held. A `Some(0)`
    /// swap beside a real peak is the opposite — a reading, and the one that
    /// makes the peak trustworthy.
    pub fn with_swap(host_bytes: u64, swap_bytes: u64) -> Self {
        match (host_bytes > 0).then_some(host_bytes) {
            Some(host_bytes) => Self {
                max_host_bytes: Some(host_bytes),
                max_swap_bytes: Some(swap_bytes),
            },
            None => Self::default(),
        }
    }

    /// The larger of two observations, term by term.
    ///
    /// A benchmark that runs several repetitions observes each one, and the run's
    /// figure is the worst any rep reached: reporting the last rep would hide a
    /// peak an earlier one hit, and averaging would describe no rep that
    /// happened. Taken per term rather than by picking a winning observation,
    /// because the largest peak and the largest swap need not come from the same
    /// rep.
    ///
    /// `None` is "not observed" rather than zero, so it never wins a maximum: an
    /// arm that sampled one rep and missed another still reports the reading it
    /// has.
    pub fn merge_max(self, other: Self) -> Self {
        fn larger(a: Option<u64>, b: Option<u64>) -> Option<u64> {
            match (a, b) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (found, None) | (None, found) => found,
            }
        }
        Self {
            max_host_bytes: larger(self.max_host_bytes, other.max_host_bytes),
            max_swap_bytes: larger(self.max_swap_bytes, other.max_swap_bytes),
        }
    }
}

/// Result data for a benchmark run.
///
/// **Serde note**: this enum uses `#[serde(untagged)]`, so deserialization tries
/// variants in declaration order and picks the first whose fields match. Every
/// variant must have a unique field set — adding a field name that already
/// exists in an earlier variant will silently misroute deserialization. The
/// round-trip tests below guard against this.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum BenchmarkResultData {
    PrefillThroughput {
        prefill_time_ms: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prefill_time_ms_stddev: Option<f64>,
    },
    DecodeThroughput {
        decode_time_ms: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        decode_time_ms_stddev: Option<f64>,
    },
    EndToEndLatency {
        total_time_ms: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        total_time_ms_stddev: Option<f64>,
    },
    MaxMemoryUsage {
        // Wire names stay `max_ram_bytes` / `max_vram_bytes` for
        // compatibility with the mgmt server's `/benchmarks` handler
        // and `/score` payload validator (which still spell-check the
        // old names). Rust code uses the methodology's
        // `max_host_bytes` / `max_gpu_bytes` terminology — the rename
        // happens via `#[serde(rename)]` on serialization, with
        // `alias`-based deserialization accepting both names. Drop
        // the `rename` once the server-side rename lands so the wire
        // format catches up with the methodology spec.
        #[serde(rename = "max_ram_bytes", alias = "max_host_bytes")]
        max_host_bytes: u64,
        #[serde(
            rename = "max_vram_bytes",
            alias = "max_gpu_bytes",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        max_gpu_bytes: Option<u64>,
        // New field — no historical wire name to preserve.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_npu_bytes: Option<u64>,
    },
    Eval {
        completions: Vec<BenchmarkEvalCompletion>,
    },
    VlThroughput {
        prompt_tokens: u32,
        prompt_ms: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt_ms_stddev: Option<f64>,
        predicted_ms: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        predicted_ms_stddev: Option<f64>,
    },
}

impl BenchmarkResultData {
    /// The kind of benchmark that produced this. The variants correspond one to
    /// one, so a recorded result carries its own type — the reader doesn't need
    /// the request that produced it.
    pub fn benchmark_type(&self) -> BenchmarkType {
        match self {
            Self::PrefillThroughput { .. } => BenchmarkType::PrefillThroughput,
            Self::DecodeThroughput { .. } => BenchmarkType::DecodeThroughput,
            Self::EndToEndLatency { .. } => BenchmarkType::EndToEndLatency,
            Self::MaxMemoryUsage { .. } => BenchmarkType::MaxMemoryUsage,
            Self::Eval { .. } => BenchmarkType::Eval,
            Self::VlThroughput { .. } => BenchmarkType::VlThroughput,
        }
    }
}

// ---------------------------------------------------------------------------
// BenchmarkSubmissionPayload — the `POST /benchmarks` wire shape
// ---------------------------------------------------------------------------

/// Payload sent to the management server via `POST /benchmarks`.
///
/// **Serde note**: this struct uses four `#[serde(flatten)]` fields —
/// [`DeviceInfo`], [`ThermalTelemetry`], [`MemoryObservation`], and
/// [`BenchmarkResultData`].  All field names across the flattened types and the
/// struct itself must remain unique; a collision silently breaks
/// deserialization.  The round-trip test
/// `submission_payload_round_trip_with_device_info` guards this.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkSubmissionPayload {
    pub benchmark_id: String,
    #[serde(flatten)]
    pub device: DeviceInfo,
    // Volatile run-environment power state (laptops throttle on battery / in
    // low-power mode). Optional: a desktop with no battery, or a platform that
    // can't read it, omits them. Mirrors the mobile `device_*` power fields.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_battery_level: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_power_state: Option<DevicePowerState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_power_save_mode: Option<bool>,
    // Per-platform before/after/worst thermal telemetry. Flattened so each
    // `device_*_thermal_*` field lands at the top level next to the power
    // fields, matching the management warehouse columns. All optional: a
    // platform (or run) that reads nothing contributes no keys, so the wire
    // shape is unchanged for clients that don't populate it.
    #[serde(flatten)]
    pub thermal: ThermalTelemetry,
    // What memory the run held while it ran, on every benchmark kind. Flattened
    // beside the thermal and power observations for the same reason: these are
    // conditions the result was measured under, not the result. See
    // [`MemoryObservation`].
    #[serde(flatten)]
    pub memory: MemoryObservation,
    /// Canonical JSON of the run's `pipette_plan_types::Model` — the lossless
    /// model coordinate (`type`/`source`/`org`/`repo_name`/…). Supersedes the
    /// old `model_name`/`model_quant`/`mmproj_quant` grouping fields; the server
    /// re-canonicalizes and stores it opaquely. `#[serde(default)]` keeps
    /// deserialization tolerant of an older on-disk payload that predates this
    /// field — the sync path (`submit_pending_result`) rejects such a payload
    /// explicitly rather than silently submitting the retired shape.
    #[serde(default)]
    pub model_descriptor: String,
    /// Canonical JSON of the run's `pipette_plan_types::Runtime` — the lossless
    /// runtime/build identity, superseding `runtime_name`/`runtime_version`.
    /// Load knobs are not part of it — they ship separately as `runtime_flags`.
    #[serde(default)]
    pub runtime_descriptor: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_flags: Option<String>,
    /// Canonical JSON of the run's [`RuntimeFlags`](crate::RuntimeFlags) in its
    /// flat wire form — the load knobs as the cell actually ran with them, plan
    /// entry plus whatever the client derived. JSON-in-a-string like the
    /// descriptors above, so the server stores it opaquely. Env forwards carry
    /// names only (`RuntimeFlags::without_env_values`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_flags: Option<String>,
    /// Canonical JSON of what the *harness* ran under — readiness gating,
    /// timeouts, loop detection — as
    /// [`BenchmarkFlags::submission_value`](crate::BenchmarkFlags::submission_value)
    /// builds it. Resolved rather than authored: readiness is decided entirely
    /// client-side, so this is the only record of it the server ever sees, and
    /// a waived thermal gate is otherwise invisible in a result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub benchmark_flags: Option<String>,
    /// Runtime-selected CPU kernel variant, interpreted per `runtime_name`.
    /// For llama.cpp/ggml this is the CPU backend variant chosen at load time
    /// by feature-dispatch scoring (the active ggml CPU feature set on the
    /// platforms that do runtime dispatch). Lets result analysis detect when
    /// the kernel variant changed. Optional / `None` when the build ships a
    /// single static CPU backend (no runtime dispatch) — e.g. iOS and desktop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_cpu_variant: Option<String>,
    /// Version of the client build that produced this run — the harness, not
    /// the runtime it drove (that is `runtime_descriptor`). Opaque to the
    /// server, which stores it verbatim as a grouping key: it is how a shift in
    /// the numbers gets attributed to a harness change rather than to the
    /// device. Optional on the wire, but the CLI always populates it; `None`
    /// only for an older on-disk payload that predates the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_version: Option<String>,
    pub submitted_at: String,
    #[serde(flatten)]
    pub result: BenchmarkResultData,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::device::DeviceFormFactor;
    use crate::thermal::{LinuxThermalZone, ThermalReading};
    use crate::ModelFlags;

    fn observed(host: Option<u64>, swap: Option<u64>) -> MemoryObservation {
        MemoryObservation {
            max_host_bytes: host,
            max_swap_bytes: swap,
        }
    }

    /// Folding across repetitions must keep the worst of each term and must never
    /// let "not observed" beat a real reading.
    #[rstest]
    // The worst rep wins, not the last one.
    #[case::worst_rep_wins(
        observed(Some(600), Some(50)),
        observed(Some(400), Some(10)),
        observed(Some(600), Some(50))
    )]
    // Terms are taken independently: the biggest peak and the biggest swap can
    // come from different reps, and picking one rep wholesale would lose one.
    #[case::terms_are_independent(
        observed(Some(600), Some(10)),
        observed(Some(400), Some(90)),
        observed(Some(600), Some(90))
    )]
    // `None` is "nobody looked", so a rep that observed nothing must not erase a
    // rep that did.
    #[case::absence_never_wins(
        observed(Some(600), Some(50)),
        observed(None, None),
        observed(Some(600), Some(50))
    )]
    #[case::absence_never_wins_either_order(
        observed(None, None),
        observed(Some(600), Some(50)),
        observed(Some(600), Some(50))
    )]
    // A sampled zero is a real reading and survives, unlike absence.
    #[case::a_sampled_zero_is_kept(
        observed(Some(600), Some(0)),
        observed(None, None),
        observed(Some(600), Some(0))
    )]
    #[case::nothing_observed_stays_empty(
        observed(None, None),
        observed(None, None),
        observed(None, None)
    )]
    fn merging_reps_keeps_the_worst_of_each_term(
        #[case] a: MemoryObservation,
        #[case] b: MemoryObservation,
        #[case] expected: MemoryObservation,
    ) {
        assert_eq!(a.merge_max(b), expected);
    }

    /// The constructors are where the "zero is not a measurement" rule lives, so
    /// every arm that builds an observation inherits it.
    #[rstest]
    #[case::host_only_keeps_a_real_peak(
        observed(Some(600), None),
        MemoryObservation::host_only(600)
    )]
    #[case::host_only_withholds_a_zero(observed(None, None), MemoryObservation::host_only(0))]
    // A sampled zero swap is a reading and survives beside a real peak.
    #[case::swap_zero_is_a_reading(
        observed(Some(600), Some(0)),
        MemoryObservation::with_swap(600, 0)
    )]
    #[case::swap_rides_with_the_peak(
        observed(Some(600), Some(50)),
        MemoryObservation::with_swap(600, 50)
    )]
    // Never half an observation: no peak means the swap term goes too.
    #[case::a_withheld_peak_withholds_swap(
        observed(None, None),
        MemoryObservation::with_swap(0, 50)
    )]
    fn a_zero_peak_is_withheld_rather_than_reported(
        #[case] expected: MemoryObservation,
        #[case] built: MemoryObservation,
    ) {
        assert_eq!(built, expected);
    }

    /// An observation with nothing in it must contribute no keys, so a client on
    /// a platform without a sampler leaves the wire shape untouched.
    #[test]
    fn an_empty_observation_serializes_to_nothing() -> anyhow::Result<()> {
        let json = serde_json::to_value(MemoryObservation::default())?;
        assert_eq!(json, serde_json::json!({}), "got {json}");
        assert_eq!(MemoryObservation::default(), MemoryObservation::default());
        Ok(())
    }

    /// Every untagged variant survives a round trip, with and without its
    /// optional stddev.
    ///
    /// Untagged deserialization picks the first variant whose field set
    /// matches, so a field name reused across variants misroutes silently. The
    /// stddev-absent cases are where the numeric variants' field sets collapse
    /// closest together, which is where a reused name would first bite.
    #[rstest]
    #[case::prefill(BenchmarkResultData::PrefillThroughput {
        prefill_time_ms: 12.5, prefill_time_ms_stddev: None,
    })]
    #[case::prefill_stddev(BenchmarkResultData::PrefillThroughput {
        prefill_time_ms: 12.5, prefill_time_ms_stddev: Some(0.4),
    })]
    #[case::decode(BenchmarkResultData::DecodeThroughput {
        decode_time_ms: 8.25, decode_time_ms_stddev: None,
    })]
    #[case::decode_stddev(BenchmarkResultData::DecodeThroughput {
        decode_time_ms: 8.25, decode_time_ms_stddev: Some(0.2),
    })]
    #[case::end_to_end(BenchmarkResultData::EndToEndLatency {
        total_time_ms: 940.0, total_time_ms_stddev: None,
    })]
    #[case::end_to_end_stddev(BenchmarkResultData::EndToEndLatency {
        total_time_ms: 940.0, total_time_ms_stddev: Some(11.0),
    })]
    #[case::max_memory(BenchmarkResultData::MaxMemoryUsage {
        max_host_bytes: 8_589_934_592, max_gpu_bytes: None, max_npu_bytes: None,
    })]
    #[case::max_memory_all_peaks(BenchmarkResultData::MaxMemoryUsage {
        max_host_bytes: 8_589_934_592,
        max_gpu_bytes: Some(4_294_967_296),
        max_npu_bytes: Some(1_073_741_824),
    })]
    #[case::eval(BenchmarkResultData::Eval {
        completions: vec![BenchmarkEvalCompletion { id: "s1".into(), ..BenchmarkEvalCompletion::default() }],
    })]
    #[case::vl(BenchmarkResultData::VlThroughput {
        prompt_tokens: 75, prompt_ms: 352.3, prompt_ms_stddev: None,
        predicted_ms: 32.7, predicted_ms_stddev: None,
    })]
    #[case::vl_stddev(BenchmarkResultData::VlThroughput {
        prompt_tokens: 75, prompt_ms: 352.3, prompt_ms_stddev: Some(3.8),
        predicted_ms: 32.7, predicted_ms_stddev: Some(1.2),
    })]
    fn result_data_untagged_variant_round_trips(
        #[case] data: BenchmarkResultData,
    ) -> anyhow::Result<()> {
        let round_tripped: BenchmarkResultData =
            serde_json::from_value(serde_json::to_value(&data)?)?;
        assert_eq!(round_tripped, data);
        Ok(())
    }

    /// Each variant reports the kind that produced it — the mapping a recorded
    /// result relies on to describe itself without the request that ran it.
    #[rstest]
    #[case::prefill(BenchmarkResultData::PrefillThroughput {
        prefill_time_ms: 1.0, prefill_time_ms_stddev: None,
    }, BenchmarkType::PrefillThroughput)]
    #[case::decode(BenchmarkResultData::DecodeThroughput {
        decode_time_ms: 1.0, decode_time_ms_stddev: None,
    }, BenchmarkType::DecodeThroughput)]
    #[case::end_to_end(BenchmarkResultData::EndToEndLatency {
        total_time_ms: 1.0, total_time_ms_stddev: None,
    }, BenchmarkType::EndToEndLatency)]
    #[case::max_memory(BenchmarkResultData::MaxMemoryUsage {
        max_host_bytes: 1, max_gpu_bytes: None, max_npu_bytes: None,
    }, BenchmarkType::MaxMemoryUsage)]
    #[case::eval(BenchmarkResultData::Eval { completions: vec![] }, BenchmarkType::Eval)]
    #[case::vl(BenchmarkResultData::VlThroughput {
        prompt_tokens: 1, prompt_ms: 1.0, prompt_ms_stddev: None,
        predicted_ms: 1.0, predicted_ms_stddev: None,
    }, BenchmarkType::VlThroughput)]
    fn benchmark_type_matches_the_variant(
        #[case] data: BenchmarkResultData,
        #[case] expected: BenchmarkType,
    ) {
        assert_eq!(data.benchmark_type(), expected);
    }

    // Wire-format contract: the canonical struct serializes to the given JSON
    // *and* that JSON parses back to it. Covers: `stop_reason` always on the
    // wire (required on the client) while `failed`/`failed_reason`/`stop_detail`/
    // `completion_tokens` elide when unset; the failure `stop_detail` dual-write;
    // and `stop_detail`/`completion_tokens` round-tripping when present.
    #[rstest]
    #[case::non_failed_carries_stop_reason(
        BenchmarkEvalCompletion {
            id: "s1".into(), completion: "answer".into(), failed: false, failed_reason: None,
            stop_reason: BenchmarkEvalCompletionStopReason::Eos, stop_detail: None, completion_tokens: None,
        },
        serde_json::json!({"id": "s1", "completion": "answer", "stop_reason": "eos"}),
    )]
    #[case::failed_marker_round_trips(
        BenchmarkEvalCompletion {
            id: "s2".into(), completion: String::new(), failed: true,
            failed_reason: Some("stack overflow".into()), stop_reason: BenchmarkEvalCompletionStopReason::Failure,
            stop_detail: Some("stack overflow".into()), completion_tokens: None,
        },
        serde_json::json!({
            "id": "s2", "completion": "", "failed": true, "failed_reason": "stack overflow",
            "stop_reason": "failure", "stop_detail": "stack overflow",
        }),
    )]
    #[case::stop_reason_and_tokens_round_trip(
        BenchmarkEvalCompletion {
            id: "s3".into(), completion: "truncated output".into(), failed: false,
            failed_reason: None, stop_reason: BenchmarkEvalCompletionStopReason::Truncated, stop_detail: None,
            completion_tokens: Some(8192),
        },
        serde_json::json!({
            "id": "s3", "completion": "truncated output", "stop_reason": "truncated",
            "completion_tokens": 8192,
        }),
    )]
    #[case::unknown_carries_stop_detail(
        BenchmarkEvalCompletion {
            id: "s4".into(), completion: "partial".into(), failed: false, failed_reason: None,
            stop_reason: BenchmarkEvalCompletionStopReason::Unknown,
            stop_detail: Some("stream ended without a terminal stop event".into()),
            completion_tokens: None,
        },
        serde_json::json!({
            "id": "s4", "completion": "partial", "stop_reason": "unknown",
            "stop_detail": "stream ended without a terminal stop event",
        }),
    )]
    fn benchmark_eval_completion_serde_contract(
        #[case] canonical: BenchmarkEvalCompletion,
        #[case] expected_json: serde_json::Value,
    ) -> anyhow::Result<()> {
        let serialized = serde_json::to_value(&canonical)?;
        assert_eq!(serialized, expected_json, "serialized shape");
        let parsed: BenchmarkEvalCompletion = serde_json::from_value(expected_json.clone())?;
        assert_eq!(parsed, canonical, "parsed value");
        Ok(())
    }

    #[test]
    fn pre_feature_payload_parses_stop_reason_as_unknown() -> anyhow::Result<()> {
        // A pre-feature payload (no `stop_reason`) still parses — the required
        // field defaults to `unknown` rather than failing the decode.
        let legacy: BenchmarkEvalCompletion =
            serde_json::from_value(serde_json::json!({"id": "old", "completion": "x"}))?;
        assert_eq!(
            legacy.stop_reason,
            BenchmarkEvalCompletionStopReason::Unknown
        );
        Ok(())
    }

    #[test]
    fn max_memory_usage_serializes_with_legacy_wire_field_names() -> anyhow::Result<()> {
        // The wire schema deliberately keeps `max_ram_bytes` /
        // `max_vram_bytes` for compatibility with the mgmt server's
        // `/benchmarks` handler. The Rust code uses the methodology's
        // `max_host_bytes` / `max_gpu_bytes` terminology — the rename
        // happens via #[serde(rename)] at the boundary. This test
        // guards that contract.
        let data = BenchmarkResultData::MaxMemoryUsage {
            max_host_bytes: 452_755_456,
            max_gpu_bytes: Some(123_456_789),
            max_npu_bytes: None,
        };
        let json = serde_json::to_value(&data)?;
        assert_eq!(json["max_ram_bytes"].as_u64(), Some(452_755_456));
        assert_eq!(json["max_vram_bytes"].as_u64(), Some(123_456_789));
        assert!(
            json.get("max_host_bytes").is_none(),
            "must not emit `max_host_bytes` on the wire: {json}"
        );
        assert!(
            json.get("max_gpu_bytes").is_none(),
            "must not emit `max_gpu_bytes` on the wire: {json}"
        );
        Ok(())
    }

    #[test]
    fn max_memory_usage_deserializes_both_legacy_and_methodology_field_names() -> anyhow::Result<()>
    {
        // Legacy payloads (what the server emits today, and what
        // `data/results/local/*/payload.json` files contain) use
        // `max_ram_bytes`/`max_vram_bytes`. The alias accepts both
        // those and the methodology names, so historical payloads
        // and any forward-compatible writer round-trip cleanly. The
        // enum is `#[serde(untagged)]`, so the field names alone
        // pick the variant — no discriminator needed in the input.
        let legacy = serde_json::json!({
            "max_ram_bytes": 1_000_000_u64,
            "max_vram_bytes": 500_000_u64
        });
        let from_legacy: BenchmarkResultData = serde_json::from_value(legacy)?;
        let methodology = serde_json::json!({
            "max_host_bytes": 1_000_000_u64,
            "max_gpu_bytes": 500_000_u64
        });
        let from_methodology: BenchmarkResultData = serde_json::from_value(methodology)?;
        assert_eq!(from_legacy, from_methodology);
        // And both decode to the expected variant + values.
        match from_legacy {
            BenchmarkResultData::MaxMemoryUsage {
                max_host_bytes,
                max_gpu_bytes,
                max_npu_bytes,
            } => {
                assert_eq!(max_host_bytes, 1_000_000);
                assert_eq!(max_gpu_bytes, Some(500_000));
                assert_eq!(max_npu_bytes, None);
            }
            other => anyhow::bail!("expected MaxMemoryUsage, got {other:?}"),
        }
        Ok(())
    }

    // ---- BenchmarkSubmissionPayload: the flattened wire shape ----

    // Representative canonical descriptors for payload tests. Content is opaque
    // to these tests (the payload round-trips them as strings); the per-runtime
    // crates assert the real `Model`/`Runtime` JSON.
    const MODEL_DESC: &str = r#"{"type":"gguf_text","source":"huggingface","org":"o","repo_name":"r","path":"m-Q4_0.gguf"}"#;
    const RT_DESC: &str = r#"{"type":"llamacpp_cli_stock_tools","source":"repository"}"#;

    fn sample_payload() -> BenchmarkSubmissionPayload {
        BenchmarkSubmissionPayload {
            benchmark_id: "decode_throughput_256_128".to_string(),
            device: DeviceInfo {
                device_name: "test".into(),
                device_form_factor: DeviceFormFactor::Embedded,
                device_os_name: "Linux".into(),
                device_os_version: "22.04".into(),
                device_os_build: None,
                device_os_security_patch: None,
                device_chip_model: "test".into(),
                device_ram_bytes: 0,
                device_gpu_model: None,
                device_gpu_vram_bytes: None,
                device_npu_model: None,
                device_npu_vram_bytes: None,
            },
            device_battery_level: None,
            device_power_state: None,
            device_power_save_mode: None,
            thermal: ThermalTelemetry::default(),
            memory: MemoryObservation::default(),
            model_descriptor: MODEL_DESC.to_string(),
            runtime_descriptor: RT_DESC.to_string(),
            model_flags: None,
            runtime_flags: None,
            benchmark_flags: None,
            runtime_cpu_variant: None,
            client_version: None,
            submitted_at: "2026-01-01T00:00:00Z".to_string(),
            result: BenchmarkResultData::DecodeThroughput {
                decode_time_ms: 50.0,
                decode_time_ms_stddev: None,
            },
        }
    }

    #[test]
    fn submission_payload_flattens_device_info() -> anyhow::Result<()> {
        let payload = BenchmarkSubmissionPayload {
            benchmark_id: "prefill_throughput_256".to_string(),
            device: DeviceInfo {
                device_name: "Test".into(),
                device_form_factor: DeviceFormFactor::Laptop,
                device_os_name: "Linux".into(),
                device_os_version: "22.04".into(),
                device_os_build: None,
                device_os_security_patch: None,
                device_chip_model: "x86_64".into(),
                device_ram_bytes: 8_000_000_000,
                device_gpu_model: None,
                device_gpu_vram_bytes: None,
                device_npu_model: None,
                device_npu_vram_bytes: None,
            },
            device_battery_level: None,
            device_power_state: None,
            device_power_save_mode: None,
            thermal: ThermalTelemetry::default(),
            memory: MemoryObservation::default(),
            model_descriptor: MODEL_DESC.to_string(),
            runtime_descriptor: RT_DESC.to_string(),
            model_flags: None,
            runtime_flags: None,
            benchmark_flags: None,
            runtime_cpu_variant: None,
            client_version: None,
            submitted_at: "2026-01-01T00:00:00Z".to_string(),
            result: BenchmarkResultData::PrefillThroughput {
                prefill_time_ms: 34.7,
                prefill_time_ms_stddev: None,
            },
        };
        let value = serde_json::to_value(&payload)?;
        // Device fields should be at the top level, not nested
        assert_eq!(value["device_name"], "Test");
        assert_eq!(value["device_form_factor"], "laptop");
        assert_eq!(value["device_os_name"], "Linux");
        assert_eq!(value["device_ram_bytes"], 8_000_000_000u64);
        assert_eq!(value["benchmark_id"], "prefill_throughput_256");
        Ok(())
    }

    #[test]
    fn old_payload_without_device_fields_still_deserializes() -> anyhow::Result<()> {
        // Old payload.json files have `hardware_details` and no `device_*`
        // fields.  Thanks to `#[serde(default)]` on DeviceInfo, they
        // deserialize with zero-valued defaults rather than erroring.
        let old_json = serde_json::json!({
            "benchmark_id": "prefill_throughput_256",
            "hardware_details": "mac-studio-m3-ultra-96gb",
            "model_name": "llama-3.2-1b",
            "model_quant": "q4_0",
            "runtime_name": "llama.cpp",
            "runtime_version": "b5000",
            "submitted_at": "2026-01-01T00:00:00Z",
            "prefill_time_ms": 34.7
        });
        let payload: BenchmarkSubmissionPayload = serde_json::from_value(old_json)?;
        assert_eq!(payload.benchmark_id, "prefill_throughput_256");
        // Device fields fall back to defaults
        assert_eq!(payload.device.device_name.as_ref(), "");
        assert_eq!(payload.device.device_ram_bytes, 0);
        Ok(())
    }

    #[test]
    fn submission_payload_round_trip_with_device_info() -> anyhow::Result<()> {
        let payload = BenchmarkSubmissionPayload {
            benchmark_id: "decode_throughput_256_128".to_string(),
            device: DeviceInfo {
                device_name: "Jetson Orin Nano 8GB".into(),
                device_form_factor: DeviceFormFactor::Embedded,
                device_os_name: "Linux".into(),
                device_os_version: "22.04".into(),
                device_os_build: Some("6.8.0-45-generic".into()),
                device_os_security_patch: Some("2025-06-01".into()),
                device_chip_model: "NVIDIA Jetson Orin Nano".into(),
                device_ram_bytes: 8_589_934_592,
                device_gpu_model: None,
                device_gpu_vram_bytes: None,
                device_npu_model: None,
                device_npu_vram_bytes: None,
            },
            device_battery_level: None,
            device_power_state: None,
            device_power_save_mode: None,
            // Populate a Linux family (Jetson device) across two reps so the
            // round trip exercises the flattened, iteration-tagged
            // `List<Struct>` columns, not just the empty case. `iteration` on
            // the snapshot zones is a placeholder; `from_series` stamps the rep.
            thermal: ThermalTelemetry::from_series(
                &[
                    ThermalReading {
                        linux_thermal_zones: Some(vec![
                            LinuxThermalZone {
                                iteration: 0,
                                zone_type: "cpu-thermal".into(),
                                celsius: 44,
                            },
                            LinuxThermalZone {
                                iteration: 0,
                                zone_type: "gpu-thermal".into(),
                                celsius: 41,
                            },
                        ]),
                        ..Default::default()
                    },
                    ThermalReading {
                        linux_thermal_zones: Some(vec![
                            LinuxThermalZone {
                                iteration: 0,
                                zone_type: "cpu-thermal".into(),
                                celsius: 55,
                            },
                            LinuxThermalZone {
                                iteration: 0,
                                zone_type: "gpu-thermal".into(),
                                celsius: 50,
                            },
                        ]),
                        ..Default::default()
                    },
                ],
                &[ThermalReading {
                    linux_thermal_zones: Some(vec![
                        LinuxThermalZone {
                            iteration: 0,
                            zone_type: "cpu-thermal".into(),
                            celsius: 63,
                        },
                        LinuxThermalZone {
                            iteration: 0,
                            zone_type: "gpu-thermal".into(),
                            celsius: 58,
                        },
                    ]),
                    ..Default::default()
                }],
            ),
            memory: MemoryObservation {
                max_host_bytes: Some(6_594_494_464),
                max_swap_bytes: Some(356_515_840),
            },
            model_descriptor: MODEL_DESC.to_string(),
            runtime_descriptor: RT_DESC.to_string(),
            model_flags: None,
            runtime_flags: None,
            benchmark_flags: None,
            runtime_cpu_variant: None,
            client_version: None,
            submitted_at: "2026-01-01T00:00:00Z".to_string(),
            result: BenchmarkResultData::DecodeThroughput {
                decode_time_ms: 50.0,
                decode_time_ms_stddev: Some(2.1),
            },
        };
        let json = serde_json::to_value(&payload)?;
        // Flattened at the top level (not nested under `thermal`), matching the
        // warehouse columns. The before series flattens 2 reps × 2 zones = 4
        // elements, each tagged with its `iteration`.
        assert_eq!(
            json["device_linux_thermal_zones_before"][0]["type"],
            "cpu-thermal"
        );
        assert_eq!(json["device_linux_thermal_zones_before"][0]["iteration"], 0);
        assert_eq!(json["device_linux_thermal_zones_before"][2]["iteration"], 1);
        assert_eq!(json["device_linux_thermal_zones_before"][2]["celsius"], 55);
        assert!(json.get("thermal").is_none());
        // The memory observation flattens the same way, under the
        // `observation_*` names the warehouse uses for measured workload facts.
        // Note this is a decode-throughput row: the observation rides on every
        // benchmark, not just the memory one.
        assert_eq!(json["observation_max_host_bytes"], 6_594_494_464u64);
        assert_eq!(json["observation_max_swap_bytes"], 356_515_840u64);
        assert!(
            json.get("memory").is_none(),
            "must flatten, not nest: {json}"
        );
        let round_tripped: BenchmarkSubmissionPayload = serde_json::from_value(json)?;
        assert_eq!(round_tripped, payload);
        Ok(())
    }

    #[test]
    fn thermal_headroom_f32_round_trips_under_flatten() -> anyhow::Result<()> {
        // The `Option<f32>` headroom fields are flattened alongside the
        // untagged `BenchmarkResultData` enum; guard that a float value the
        // client emits survives serialize → deserialize through that path.
        let thermal = ThermalTelemetry::from_series(
            &[
                ThermalReading {
                    android_thermal_headroom: Some(0.31),
                    ..Default::default()
                },
                ThermalReading {
                    android_thermal_headroom: Some(0.44),
                    ..Default::default()
                },
            ],
            &[ThermalReading {
                android_thermal_headroom: Some(0.62),
                ..Default::default()
            }],
        );
        assert_eq!(
            thermal.device_android_thermal_headroom_before,
            Some(vec![0.31, 0.44])
        );
        let mut payload = sample_payload();
        payload.thermal = thermal.clone();
        let json = serde_json::to_value(&payload)?;
        assert!(json["device_android_thermal_headroom_before"][0].is_number());
        // The round trip is the real guard: `from_value` must reconstruct the
        // `f32` from the buffered `flatten` content without erroring.
        let round_tripped: BenchmarkSubmissionPayload = serde_json::from_value(json)?;
        assert_eq!(round_tripped.thermal, thermal);
        Ok(())
    }

    #[test]
    fn apple_soc_temp_c_f32_round_trips_under_flatten() -> anyhow::Result<()> {
        // The iOS-only raw SoC die temp is a `Option<Vec<f32>>` scalar family
        // parallel to `device_apple_thermal_state_*`; guard that a fractional
        // value survives serialize → deserialize through the flatten path.
        let thermal = ThermalTelemetry::from_series(
            &[
                ThermalReading {
                    apple_soc_temp_c: Some(38.5),
                    ..Default::default()
                },
                ThermalReading {
                    apple_soc_temp_c: Some(41.25),
                    ..Default::default()
                },
            ],
            &[ThermalReading {
                apple_soc_temp_c: Some(44.0),
                ..Default::default()
            }],
        );
        assert_eq!(
            thermal.device_apple_soc_temp_c_before,
            Some(vec![38.5, 41.25])
        );
        assert_eq!(thermal.device_apple_soc_temp_c_after, Some(vec![44.0]));
        let mut payload = sample_payload();
        payload.thermal = thermal.clone();
        let json = serde_json::to_value(&payload)?;
        assert!(json["device_apple_soc_temp_c_before"][0].is_number());
        let round_tripped: BenchmarkSubmissionPayload = serde_json::from_value(json)?;
        assert_eq!(round_tripped.thermal, thermal);
        Ok(())
    }
    #[test]
    fn submission_payload_carries_the_descriptors() -> anyhow::Result<()> {
        let value = serde_json::to_value(sample_payload())?;
        assert_eq!(value["model_descriptor"], MODEL_DESC);
        assert_eq!(value["runtime_descriptor"], RT_DESC);
        Ok(())
    }

    // mgmt-side ingestion / dashboards key on the literal string here, so the
    // wire field must carry exactly what `canonical_string()` emits (its
    // per-flag spelling is covered in `pipette-plan-types`).
    #[rstest::rstest]
    #[case::unset(None, None)]
    #[case::thinking_on(Some(true), Some("enable_thinking=true"))]
    #[case::thinking_off(Some(false), Some("enable_thinking=false"))]
    fn submission_payload_model_flags_wire_shape(
        #[case] enable_thinking: Option<bool>,
        #[case] canonical: Option<&str>,
    ) -> anyhow::Result<()> {
        let flags = ModelFlags::EvalGgufText { enable_thinking };
        let mut payload = sample_payload();
        payload.model_flags = flags.canonical_string();
        let json = serde_json::to_value(&payload)?;
        let actual = json.get("model_flags").and_then(serde_json::Value::as_str);
        assert_eq!(actual, canonical);
        Ok(())
    }
}
