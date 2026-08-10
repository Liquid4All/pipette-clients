//! Clap surface for the doom-loop pipeline on `pipette benchmarks run`.
//!
//! Flag definitions live with the unified binary shell; plan form
//! ([`pipette_doomloop::plan::DoomloopOverrides`]) and the runnable
//! [`pipette_doomloop::DoomloopPipeline`] live in `pipette-doomloop`.
//! After parse, [`DoomloopCliArgs::into_overrides`] is the only conversion
//! the CLI needs — engines call `pipeline_from_overrides` themselves.

use clap::Args;

use pipette_doomloop::plan::{
    BlockRepeatOverrides, CyclicLineRepeatOverrides, DoomloopOverrides, ExactRepeatOverrides,
    LineRepeatOverrides, NumericEnumerationOverrides, SuffixProbeOverrides,
};

/// Flattened parent carrying every per-detector args group.
///
/// Call [`DoomloopCliArgs::into_overrides`] after parsing to materialize
/// plan-form overrides for the shared [`pipette_plan_types::run::RunRequest`].
#[derive(Args, Debug, Default)]
#[command(next_help_heading = "Doom-loop detector tuning (eval benchmarks only)")]
pub struct DoomloopCliArgs {
    #[command(flatten)]
    pub exact_repeat: DoomloopExactRepeatCliArgs,

    #[command(flatten)]
    pub line_repeat: DoomloopLineRepeatCliArgs,

    #[command(flatten)]
    pub cyclic_line_repeat: DoomloopCyclicLineRepeatCliArgs,

    #[command(flatten)]
    pub block_repeat: DoomloopBlockRepeatCliArgs,

    #[command(flatten)]
    pub numeric_enumeration: DoomloopNumericEnumerationCliArgs,

    #[command(flatten)]
    pub suffix_probe: DoomloopSuffixProbeCliArgs,
}

impl DoomloopCliArgs {
    /// Materialize plan-form [`DoomloopOverrides`]. A detector group with no
    /// flag set maps to `None` (inherit built-in defaults), so a bare
    /// `DoomloopCliArgs` yields `DoomloopOverrides::default()`.
    pub fn into_overrides(self) -> DoomloopOverrides {
        fn non_default<T: Default + PartialEq>(v: T) -> Option<T> {
            (v != T::default()).then_some(v)
        }
        DoomloopOverrides {
            exact_repeat: non_default(ExactRepeatOverrides {
                enabled: self.exact_repeat.enabled,
                min_chars: self.exact_repeat.min_chars,
                window: self.exact_repeat.window,
                min_period: self.exact_repeat.min_period,
                required: self.exact_repeat.required,
            }),
            line_repeat: non_default(LineRepeatOverrides {
                enabled: self.line_repeat.enabled,
                min_chars: self.line_repeat.min_chars,
                window: self.line_repeat.window,
                count: self.line_repeat.count,
                min_len: self.line_repeat.min_len,
                counter_min_len: self.line_repeat.counter_min_len,
                counter_count: self.line_repeat.counter_count,
            }),
            cyclic_line_repeat: non_default(CyclicLineRepeatOverrides {
                enabled: self.cyclic_line_repeat.enabled,
                min_chars: self.cyclic_line_repeat.min_chars,
                window: self.cyclic_line_repeat.window,
                count: self.cyclic_line_repeat.count,
                min_len: self.cyclic_line_repeat.min_len,
                counter_min_len: self.cyclic_line_repeat.counter_min_len,
                counter_count: self.cyclic_line_repeat.counter_count,
                min_period: self.cyclic_line_repeat.min_period,
                max_period: self.cyclic_line_repeat.max_period,
            }),
            block_repeat: non_default(BlockRepeatOverrides {
                enabled: self.block_repeat.enabled,
                min_chars: self.block_repeat.min_chars,
                window: self.block_repeat.window,
                min_period: self.block_repeat.min_period,
                required: self.block_repeat.required,
            }),
            numeric_enumeration: non_default(NumericEnumerationOverrides {
                enabled: self.numeric_enumeration.enabled,
                min_chars: self.numeric_enumeration.min_chars,
                window: self.numeric_enumeration.window,
                count: self.numeric_enumeration.count,
                min_template_len: self.numeric_enumeration.min_template_len,
            }),
            suffix_probe: non_default(SuffixProbeOverrides {
                enabled: self.suffix_probe.enabled,
                min_chars: self.suffix_probe.min_chars,
                window: self.suffix_probe.window,
                probe_len: self.suffix_probe.probe_len,
                required: self.suffix_probe.required,
            }),
        }
    }
}

#[derive(Args, Debug, Default)]
pub struct DoomloopExactRepeatCliArgs {
    /// Enable or disable the exact-repetition detector. The other five keep
    /// running. Omit to leave it active.
    #[arg(
        id = "doomloop-exact-repeat-enabled",
        long = "doomloop-exact-repeat-enabled"
    )]
    pub enabled: Option<bool>,

    /// Minimum bytes of output before exact-repetition will fire
    #[arg(
        id = "doomloop-exact-repeat-min-chars",
        long = "doomloop-exact-repeat-min-chars"
    )]
    pub min_chars: Option<usize>,

    /// Tail size in bytes inspected by exact-repetition; must be > 0.
    /// Use `--doomloop-exact-repeat-enabled false` to switch the detector off
    #[arg(
        id = "doomloop-exact-repeat-window",
        long = "doomloop-exact-repeat-window"
    )]
    pub window: Option<usize>,

    /// Shortest repeating period in bytes considered by exact-repetition
    #[arg(
        id = "doomloop-exact-repeat-min-period",
        long = "doomloop-exact-repeat-min-period"
    )]
    pub min_period: Option<usize>,

    /// Consecutive byte-identical copies required to fire exact-repetition (>= 2)
    #[arg(
        id = "doomloop-exact-repeat-required",
        long = "doomloop-exact-repeat-required"
    )]
    pub required: Option<usize>,
}

#[derive(Args, Debug, Default)]
pub struct DoomloopLineRepeatCliArgs {
    /// Enable or disable line-repetition entirely.
    #[arg(
        id = "doomloop-line-repeat-enabled",
        long = "doomloop-line-repeat-enabled"
    )]
    pub enabled: Option<bool>,

    /// Minimum bytes of output before line-repetition will fire
    #[arg(
        id = "doomloop-line-repeat-min-chars",
        long = "doomloop-line-repeat-min-chars"
    )]
    pub min_chars: Option<usize>,

    /// Tail size in bytes scanned by line-repetition; must be > 0.
    /// Use `--doomloop-line-repeat-enabled false` to switch the detector off
    #[arg(
        id = "doomloop-line-repeat-window",
        long = "doomloop-line-repeat-window"
    )]
    pub window: Option<usize>,

    /// Consecutive normalized-identical trailing lines required to fire line-repetition (>= 2)
    #[arg(id = "doomloop-line-repeat-count", long = "doomloop-line-repeat-count")]
    pub count: Option<usize>,

    /// Skip normalized lines shorter than this length (avoids triggering on JSON/YAML scaffolding)
    #[arg(
        id = "doomloop-line-repeat-min-len",
        long = "doomloop-line-repeat-min-len"
    )]
    pub min_len: Option<usize>,

    /// Relaxed min_len applied when the normalized line contains a digit-run marker (catches counter-scaffolded short lines)
    #[arg(
        id = "doomloop-line-repeat-counter-min-len",
        long = "doomloop-line-repeat-counter-min-len"
    )]
    pub counter_min_len: Option<usize>,

    /// Stricter match count required when the streak used the relaxed counter_min_len gate (>= count)
    #[arg(
        id = "doomloop-line-repeat-counter-count",
        long = "doomloop-line-repeat-counter-count"
    )]
    pub counter_count: Option<usize>,
}

#[derive(Args, Debug, Default)]
pub struct DoomloopCyclicLineRepeatCliArgs {
    /// Enable or disable cyclic-line-repeat entirely.
    #[arg(
        id = "doomloop-cyclic-line-repeat-enabled",
        long = "doomloop-cyclic-line-repeat-enabled"
    )]
    pub enabled: Option<bool>,

    /// Minimum bytes of output before cyclic-line-repeat will fire
    #[arg(
        id = "doomloop-cyclic-line-repeat-min-chars",
        long = "doomloop-cyclic-line-repeat-min-chars"
    )]
    pub min_chars: Option<usize>,

    /// Tail size in bytes scanned by cyclic-line-repeat; must be > 0.
    /// Use `--doomloop-cyclic-line-repeat-enabled false` to switch the detector off
    #[arg(
        id = "doomloop-cyclic-line-repeat-window",
        long = "doomloop-cyclic-line-repeat-window"
    )]
    pub window: Option<usize>,

    /// Cycles of the winning period required to fire cyclic-line-repeat (>= 2)
    #[arg(
        id = "doomloop-cyclic-line-repeat-count",
        long = "doomloop-cyclic-line-repeat-count"
    )]
    pub count: Option<usize>,

    /// Skip normalized lines shorter than this length
    #[arg(
        id = "doomloop-cyclic-line-repeat-min-len",
        long = "doomloop-cyclic-line-repeat-min-len"
    )]
    pub min_len: Option<usize>,

    /// Relaxed min_len applied when the normalized line contains a digit-run marker
    #[arg(
        id = "doomloop-cyclic-line-repeat-counter-min-len",
        long = "doomloop-cyclic-line-repeat-counter-min-len"
    )]
    pub counter_min_len: Option<usize>,

    /// Stricter cycle count required when a cycle used the relaxed counter_min_len gate (>= count)
    #[arg(
        id = "doomloop-cyclic-line-repeat-counter-count",
        long = "doomloop-cyclic-line-repeat-counter-count"
    )]
    pub counter_count: Option<usize>,

    /// Smallest cyclic period (in lines) tried by cyclic-line-repeat (>= 2)
    #[arg(
        id = "doomloop-cyclic-line-repeat-min-period",
        long = "doomloop-cyclic-line-repeat-min-period"
    )]
    pub min_period: Option<usize>,

    /// Largest cyclic period (in lines) tried by cyclic-line-repeat
    #[arg(
        id = "doomloop-cyclic-line-repeat-max-period",
        long = "doomloop-cyclic-line-repeat-max-period"
    )]
    pub max_period: Option<usize>,
}

#[derive(Args, Debug, Default)]
pub struct DoomloopBlockRepeatCliArgs {
    /// Enable or disable block-repetition entirely.
    #[arg(
        id = "doomloop-block-repeat-enabled",
        long = "doomloop-block-repeat-enabled"
    )]
    pub enabled: Option<bool>,

    /// Minimum bytes of output before block-repetition will fire
    #[arg(
        id = "doomloop-block-repeat-min-chars",
        long = "doomloop-block-repeat-min-chars"
    )]
    pub min_chars: Option<usize>,

    /// Scan range in bytes for the block-repetition detector; must be > 0.
    /// Use `--doomloop-block-repeat-enabled false` to switch the detector off
    #[arg(
        id = "doomloop-block-repeat-window",
        long = "doomloop-block-repeat-window"
    )]
    pub window: Option<usize>,

    /// Length in bytes of the trailing probe block used by block-repetition
    #[arg(
        id = "doomloop-block-repeat-min-period",
        long = "doomloop-block-repeat-min-period"
    )]
    pub min_period: Option<usize>,

    /// Minimum non-overlapping probe occurrences in the scan range to fire block-repetition (>= 2)
    #[arg(
        id = "doomloop-block-repeat-required",
        long = "doomloop-block-repeat-required"
    )]
    pub required: Option<usize>,
}

#[derive(Args, Debug, Default)]
pub struct DoomloopNumericEnumerationCliArgs {
    /// Enable or disable numeric-enumeration entirely.
    #[arg(
        id = "doomloop-numeric-enumeration-enabled",
        long = "doomloop-numeric-enumeration-enabled"
    )]
    pub enabled: Option<bool>,

    /// Minimum bytes of output before numeric-enumeration will fire
    #[arg(
        id = "doomloop-numeric-enumeration-min-chars",
        long = "doomloop-numeric-enumeration-min-chars"
    )]
    pub min_chars: Option<usize>,

    /// Tail size in bytes scanned by numeric-enumeration; must be > 0.
    /// Use `--doomloop-numeric-enumeration-enabled false` to switch the detector off
    #[arg(
        id = "doomloop-numeric-enumeration-window",
        long = "doomloop-numeric-enumeration-window"
    )]
    pub window: Option<usize>,

    /// Consecutive trailing lines sharing a non-digit template required to fire numeric-enumeration (>= 2)
    #[arg(
        id = "doomloop-numeric-enumeration-count",
        long = "doomloop-numeric-enumeration-count"
    )]
    pub count: Option<usize>,

    /// Minimum byte length of the non-digit template (>= 1; 0 would permit pure-number streams)
    #[arg(
        id = "doomloop-numeric-enumeration-min-template-len",
        long = "doomloop-numeric-enumeration-min-template-len"
    )]
    pub min_template_len: Option<usize>,
}

#[derive(Args, Debug, Default)]
pub struct DoomloopSuffixProbeCliArgs {
    /// Enable or disable suffix-probe entirely.
    #[arg(
        id = "doomloop-suffix-probe-enabled",
        long = "doomloop-suffix-probe-enabled"
    )]
    pub enabled: Option<bool>,

    /// Minimum bytes of output before suffix-probe will fire
    #[arg(
        id = "doomloop-suffix-probe-min-chars",
        long = "doomloop-suffix-probe-min-chars"
    )]
    pub min_chars: Option<usize>,

    /// Scan range in bytes for the suffix-probe detector; must be > 0.
    /// Use `--doomloop-suffix-probe-enabled false` to switch the detector off
    #[arg(
        id = "doomloop-suffix-probe-window",
        long = "doomloop-suffix-probe-window"
    )]
    pub window: Option<usize>,

    /// Length in bytes of the trailing probe used by suffix-probe
    #[arg(
        id = "doomloop-suffix-probe-probe-len",
        long = "doomloop-suffix-probe-probe-len"
    )]
    pub probe_len: Option<usize>,

    /// Minimum non-overlapping probe occurrences in the scan range to fire suffix-probe (>= 2)
    #[arg(
        id = "doomloop-suffix-probe-required",
        long = "doomloop-suffix-probe-required"
    )]
    pub required: Option<usize>,
}

#[cfg(test)]
mod tests {
    use pipette_doomloop::plan::pipeline_from_overrides;

    use super::*;

    fn pipeline_or_err(
        overrides: &DoomloopOverrides,
    ) -> anyhow::Result<pipette_doomloop::DoomloopPipeline> {
        pipeline_from_overrides(Some(overrides)).map_err(|e| anyhow::anyhow!(e))
    }

    #[test]
    fn default_args_produce_default_pipeline() -> anyhow::Result<()> {
        let pipeline = pipeline_or_err(&DoomloopCliArgs::default().into_overrides())?;
        let default = pipette_doomloop::DoomloopPipeline::default();
        assert_eq!(pipeline.detectors.len(), default.detectors.len());
        assert_eq!(pipeline.detectors[0].name(), "exact_repeat");
        assert_eq!(pipeline.detectors[1].name(), "line_repeat");
        assert_eq!(pipeline.detectors[2].name(), "cyclic_line_repeat");
        assert_eq!(pipeline.detectors[3].name(), "block_repeat");
        assert_eq!(pipeline.detectors[4].name(), "numeric_enumeration");
        assert_eq!(pipeline.detectors[5].name(), "suffix_probe");
        Ok(())
    }

    #[test]
    fn into_overrides_maps_set_flags_and_leaves_rest_default() -> anyhow::Result<()> {
        let args = DoomloopCliArgs {
            exact_repeat: DoomloopExactRepeatCliArgs {
                required: Some(3),
                window: Some(8192),
                ..Default::default()
            },
            ..Default::default()
        };
        let overrides = args.into_overrides();
        let exact = overrides
            .exact_repeat
            .ok_or_else(|| anyhow::anyhow!("exact_repeat set"))?;
        assert_eq!(exact.required, Some(3));
        assert_eq!(exact.window, Some(8192));
        assert_eq!(exact.min_chars, None);
        assert!(overrides.line_repeat.is_none());
        assert!(overrides.suffix_probe.is_none());
        assert_eq!(
            DoomloopCliArgs::default().into_overrides(),
            DoomloopOverrides::default()
        );
        Ok(())
    }

    #[test]
    fn disabled_detector_is_absent_from_pipeline() -> anyhow::Result<()> {
        let args = DoomloopCliArgs {
            suffix_probe: DoomloopSuffixProbeCliArgs {
                enabled: Some(false),
                ..Default::default()
            },
            block_repeat: DoomloopBlockRepeatCliArgs {
                enabled: Some(false),
                ..Default::default()
            },
            ..Default::default()
        };
        let pipeline = pipeline_or_err(&args.into_overrides())?;
        let names: Vec<_> = pipeline.detectors.iter().map(|d| d.name()).collect();
        assert_eq!(
            names,
            vec![
                "exact_repeat",
                "line_repeat",
                "cyclic_line_repeat",
                "numeric_enumeration",
            ]
        );
        Ok(())
    }

    #[test]
    fn enabled_true_keeps_detector_in_pipeline() -> anyhow::Result<()> {
        let args = DoomloopCliArgs {
            suffix_probe: DoomloopSuffixProbeCliArgs {
                enabled: Some(true),
                ..Default::default()
            },
            ..Default::default()
        };
        let pipeline = pipeline_or_err(&args.into_overrides())?;
        assert_eq!(
            pipeline.detectors.len(),
            pipette_doomloop::DoomloopPipeline::default()
                .detectors
                .len()
        );
        Ok(())
    }
}
