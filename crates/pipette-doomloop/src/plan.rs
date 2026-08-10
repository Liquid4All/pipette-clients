//! Plan-level TOML overrides for the doom-loop pipeline, gated behind the
//! `plan` feature.
//!
//! Plan runners deserialize `[doomloop]` from their plan TOML and forward
//! only the fields the operator explicitly set as `--doomloop-*` argv
//! pairs to the remote binary. Missing fields produce no argv, so the
//! remote binary keeps its built-in defaults.

use serde::{Deserialize, Serialize};

/// Parent struct matching `[doomloop]` in plan TOML. Every child is an
/// `Option<T>` so an omitted `[doomloop.foo]` section leaves the remote
/// binary's defaults in charge of `foo`.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct DoomloopOverrides {
    pub exact_repeat: Option<ExactRepeatOverrides>,
    pub line_repeat: Option<LineRepeatOverrides>,
    pub cyclic_line_repeat: Option<CyclicLineRepeatOverrides>,
    pub block_repeat: Option<BlockRepeatOverrides>,
    pub numeric_enumeration: Option<NumericEnumerationOverrides>,
    pub suffix_probe: Option<SuffixProbeOverrides>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ExactRepeatOverrides {
    /// If `Some(false)`, this detector is skipped entirely on the
    /// remote binary. Omit (or `Some(true)`) to keep it active.
    pub enabled: Option<bool>,
    pub min_chars: Option<usize>,
    pub window: Option<usize>,
    pub min_period: Option<usize>,
    pub required: Option<usize>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct LineRepeatOverrides {
    pub enabled: Option<bool>,
    pub min_chars: Option<usize>,
    pub window: Option<usize>,
    pub count: Option<usize>,
    pub min_len: Option<usize>,
    pub counter_min_len: Option<usize>,
    pub counter_count: Option<usize>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CyclicLineRepeatOverrides {
    pub enabled: Option<bool>,
    pub min_chars: Option<usize>,
    pub window: Option<usize>,
    pub count: Option<usize>,
    pub min_len: Option<usize>,
    pub counter_min_len: Option<usize>,
    pub counter_count: Option<usize>,
    pub min_period: Option<usize>,
    pub max_period: Option<usize>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct BlockRepeatOverrides {
    pub enabled: Option<bool>,
    pub min_chars: Option<usize>,
    pub window: Option<usize>,
    pub min_period: Option<usize>,
    pub required: Option<usize>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct NumericEnumerationOverrides {
    pub enabled: Option<bool>,
    pub min_chars: Option<usize>,
    pub window: Option<usize>,
    pub count: Option<usize>,
    pub min_template_len: Option<usize>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SuffixProbeOverrides {
    pub enabled: Option<bool>,
    pub min_chars: Option<usize>,
    pub window: Option<usize>,
    pub probe_len: Option<usize>,
    pub required: Option<usize>,
}

impl DoomloopOverrides {
    /// Materialize a runnable [`crate::DoomloopPipeline`] from plan-form
    /// overrides. Missing detector groups inherit built-in defaults; an
    /// explicit `enabled = false` omits that detector. Order matches
    /// [`crate::DoomloopPipeline::default`].
    pub fn to_pipeline(&self) -> crate::DoomloopPipeline {
        use crate::{
            BlockRepeat, CyclicLineRepeat, Detector, ExactRepeat, LineRepeat, NumericEnumeration,
            SuffixProbe,
        };

        let mut detectors: Vec<Box<dyn Detector>> = Vec::with_capacity(6);

        {
            let o = self.exact_repeat.as_ref();
            if o.and_then(|e| e.enabled).unwrap_or(true) {
                let d = ExactRepeat::default();
                detectors.push(Box::new(ExactRepeat {
                    min_chars: o.and_then(|e| e.min_chars).unwrap_or(d.min_chars),
                    window: o.and_then(|e| e.window).unwrap_or(d.window),
                    min_period: o.and_then(|e| e.min_period).unwrap_or(d.min_period),
                    required: o.and_then(|e| e.required).unwrap_or(d.required),
                }));
            }
        }
        {
            let o = self.line_repeat.as_ref();
            if o.and_then(|e| e.enabled).unwrap_or(true) {
                let d = LineRepeat::default();
                detectors.push(Box::new(LineRepeat {
                    min_chars: o.and_then(|e| e.min_chars).unwrap_or(d.min_chars),
                    window: o.and_then(|e| e.window).unwrap_or(d.window),
                    count: o.and_then(|e| e.count).unwrap_or(d.count),
                    min_len: o.and_then(|e| e.min_len).unwrap_or(d.min_len),
                    counter_min_len: o
                        .and_then(|e| e.counter_min_len)
                        .unwrap_or(d.counter_min_len),
                    counter_count: o.and_then(|e| e.counter_count).unwrap_or(d.counter_count),
                }));
            }
        }
        {
            let o = self.cyclic_line_repeat.as_ref();
            if o.and_then(|e| e.enabled).unwrap_or(true) {
                let d = CyclicLineRepeat::default();
                detectors.push(Box::new(CyclicLineRepeat {
                    min_chars: o.and_then(|e| e.min_chars).unwrap_or(d.min_chars),
                    window: o.and_then(|e| e.window).unwrap_or(d.window),
                    count: o.and_then(|e| e.count).unwrap_or(d.count),
                    min_len: o.and_then(|e| e.min_len).unwrap_or(d.min_len),
                    counter_min_len: o
                        .and_then(|e| e.counter_min_len)
                        .unwrap_or(d.counter_min_len),
                    counter_count: o.and_then(|e| e.counter_count).unwrap_or(d.counter_count),
                    min_period: o.and_then(|e| e.min_period).unwrap_or(d.min_period),
                    max_period: o.and_then(|e| e.max_period).unwrap_or(d.max_period),
                }));
            }
        }
        {
            let o = self.block_repeat.as_ref();
            if o.and_then(|e| e.enabled).unwrap_or(true) {
                let d = BlockRepeat::default();
                detectors.push(Box::new(BlockRepeat {
                    min_chars: o.and_then(|e| e.min_chars).unwrap_or(d.min_chars),
                    window: o.and_then(|e| e.window).unwrap_or(d.window),
                    min_period: o.and_then(|e| e.min_period).unwrap_or(d.min_period),
                    required: o.and_then(|e| e.required).unwrap_or(d.required),
                }));
            }
        }
        {
            let o = self.numeric_enumeration.as_ref();
            if o.and_then(|e| e.enabled).unwrap_or(true) {
                let d = NumericEnumeration::default();
                detectors.push(Box::new(NumericEnumeration {
                    min_chars: o.and_then(|e| e.min_chars).unwrap_or(d.min_chars),
                    window: o.and_then(|e| e.window).unwrap_or(d.window),
                    count: o.and_then(|e| e.count).unwrap_or(d.count),
                    min_template_len: o
                        .and_then(|e| e.min_template_len)
                        .unwrap_or(d.min_template_len),
                }));
            }
        }
        {
            let o = self.suffix_probe.as_ref();
            if o.and_then(|e| e.enabled).unwrap_or(true) {
                let d = SuffixProbe::default();
                detectors.push(Box::new(SuffixProbe {
                    min_chars: o.and_then(|e| e.min_chars).unwrap_or(d.min_chars),
                    window: o.and_then(|e| e.window).unwrap_or(d.window),
                    probe_len: o.and_then(|e| e.probe_len).unwrap_or(d.probe_len),
                    required: o.and_then(|e| e.required).unwrap_or(d.required),
                }));
            }
        }

        crate::DoomloopPipeline { detectors }
    }

    /// Append `--doomloop-*` argv pairs for every overridden field. Fields
    /// left `None` produce nothing, so the remote binary's
    /// `pipette-doomloop` defaults govern everything the plan didn't
    /// explicitly set.
    pub fn append_argv(&self, argv: &mut Vec<String>) {
        if let Some(d) = &self.exact_repeat {
            push_opt_bool(argv, "--doomloop-exact-repeat-enabled", d.enabled);
            push_opt(argv, "--doomloop-exact-repeat-min-chars", d.min_chars);
            push_opt(argv, "--doomloop-exact-repeat-window", d.window);
            push_opt(argv, "--doomloop-exact-repeat-min-period", d.min_period);
            push_opt(argv, "--doomloop-exact-repeat-required", d.required);
        }
        if let Some(d) = &self.line_repeat {
            push_opt_bool(argv, "--doomloop-line-repeat-enabled", d.enabled);
            push_opt(argv, "--doomloop-line-repeat-min-chars", d.min_chars);
            push_opt(argv, "--doomloop-line-repeat-window", d.window);
            push_opt(argv, "--doomloop-line-repeat-count", d.count);
            push_opt(argv, "--doomloop-line-repeat-min-len", d.min_len);
            push_opt(
                argv,
                "--doomloop-line-repeat-counter-min-len",
                d.counter_min_len,
            );
            push_opt(
                argv,
                "--doomloop-line-repeat-counter-count",
                d.counter_count,
            );
        }
        if let Some(d) = &self.cyclic_line_repeat {
            push_opt_bool(argv, "--doomloop-cyclic-line-repeat-enabled", d.enabled);
            push_opt(argv, "--doomloop-cyclic-line-repeat-min-chars", d.min_chars);
            push_opt(argv, "--doomloop-cyclic-line-repeat-window", d.window);
            push_opt(argv, "--doomloop-cyclic-line-repeat-count", d.count);
            push_opt(argv, "--doomloop-cyclic-line-repeat-min-len", d.min_len);
            push_opt(
                argv,
                "--doomloop-cyclic-line-repeat-counter-min-len",
                d.counter_min_len,
            );
            push_opt(
                argv,
                "--doomloop-cyclic-line-repeat-counter-count",
                d.counter_count,
            );
            push_opt(
                argv,
                "--doomloop-cyclic-line-repeat-min-period",
                d.min_period,
            );
            push_opt(
                argv,
                "--doomloop-cyclic-line-repeat-max-period",
                d.max_period,
            );
        }
        if let Some(d) = &self.block_repeat {
            push_opt_bool(argv, "--doomloop-block-repeat-enabled", d.enabled);
            push_opt(argv, "--doomloop-block-repeat-min-chars", d.min_chars);
            push_opt(argv, "--doomloop-block-repeat-window", d.window);
            push_opt(argv, "--doomloop-block-repeat-min-period", d.min_period);
            push_opt(argv, "--doomloop-block-repeat-required", d.required);
        }
        if let Some(d) = &self.numeric_enumeration {
            push_opt_bool(argv, "--doomloop-numeric-enumeration-enabled", d.enabled);
            push_opt(
                argv,
                "--doomloop-numeric-enumeration-min-chars",
                d.min_chars,
            );
            push_opt(argv, "--doomloop-numeric-enumeration-window", d.window);
            push_opt(argv, "--doomloop-numeric-enumeration-count", d.count);
            push_opt(
                argv,
                "--doomloop-numeric-enumeration-min-template-len",
                d.min_template_len,
            );
        }
        if let Some(d) = &self.suffix_probe {
            push_opt_bool(argv, "--doomloop-suffix-probe-enabled", d.enabled);
            push_opt(argv, "--doomloop-suffix-probe-min-chars", d.min_chars);
            push_opt(argv, "--doomloop-suffix-probe-window", d.window);
            push_opt(argv, "--doomloop-suffix-probe-probe-len", d.probe_len);
            push_opt(argv, "--doomloop-suffix-probe-required", d.required);
        }
    }
}

fn push_opt(argv: &mut Vec<String>, flag: &str, value: Option<usize>) {
    if let Some(v) = value {
        argv.push(flag.to_string());
        argv.push(v.to_string());
    }
}

fn push_opt_bool(argv: &mut Vec<String>, flag: &str, value: Option<bool>) {
    if let Some(v) = value {
        argv.push(flag.to_string());
        argv.push(v.to_string());
    }
}

/// Build a validated pipeline from optional plan overrides. `None` yields the
/// default detector set (same as a bare plan cell with no doom-loop knobs).
pub fn pipeline_from_overrides(
    overrides: Option<&DoomloopOverrides>,
) -> Result<crate::DoomloopPipeline, String> {
    let pipeline = match overrides {
        Some(o) => o.to_pipeline(),
        None => crate::DoomloopPipeline::default(),
    };
    pipeline.validate()?;
    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_overrides_to_pipeline_matches_default_detector_count() {
        let from_empty = DoomloopOverrides::default().to_pipeline();
        let default = crate::DoomloopPipeline::default();
        assert_eq!(from_empty.detectors.len(), default.detectors.len());
        assert!(pipeline_from_overrides(None).is_ok());
    }

    #[test]
    fn disabled_detector_is_omitted() {
        let overrides = DoomloopOverrides {
            exact_repeat: Some(ExactRepeatOverrides {
                enabled: Some(false),
                ..ExactRepeatOverrides::default()
            }),
            ..DoomloopOverrides::default()
        };
        let pipeline = overrides.to_pipeline();
        let default = crate::DoomloopPipeline::default();
        assert_eq!(pipeline.detectors.len(), default.detectors.len() - 1);
        assert!(pipeline
            .detectors
            .iter()
            .all(|d| d.name() != "exact_repeat"));
    }

    #[test]
    fn empty_overrides_emit_no_argv() {
        let mut argv = Vec::<String>::new();
        DoomloopOverrides::default().append_argv(&mut argv);
        assert!(argv.is_empty());
    }

    #[test]
    fn set_fields_emit_paired_flag_and_value() {
        let overrides = DoomloopOverrides {
            cyclic_line_repeat: Some(CyclicLineRepeatOverrides {
                count: Some(9),
                max_period: Some(16),
                ..CyclicLineRepeatOverrides::default()
            }),
            ..DoomloopOverrides::default()
        };
        let mut argv = Vec::<String>::new();
        overrides.append_argv(&mut argv);
        assert_eq!(
            argv,
            vec![
                "--doomloop-cyclic-line-repeat-count".to_string(),
                "9".to_string(),
                "--doomloop-cyclic-line-repeat-max-period".to_string(),
                "16".to_string(),
            ]
        );
    }

    #[test]
    fn enabled_emits_paired_flag() {
        let overrides = DoomloopOverrides {
            suffix_probe: Some(SuffixProbeOverrides {
                enabled: Some(false),
                ..SuffixProbeOverrides::default()
            }),
            block_repeat: Some(BlockRepeatOverrides {
                required: Some(4),
                ..BlockRepeatOverrides::default()
            }),
            ..DoomloopOverrides::default()
        };
        let mut argv = Vec::<String>::new();
        overrides.append_argv(&mut argv);
        assert_eq!(
            argv,
            vec![
                "--doomloop-block-repeat-required".to_string(),
                "4".to_string(),
                "--doomloop-suffix-probe-enabled".to_string(),
                "false".to_string(),
            ]
        );
    }
}
