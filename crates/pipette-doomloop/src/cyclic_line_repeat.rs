use crate::{line_repeat::normalize_line_into, tail_of, Detector};

/// Cyclic normalized line repetition: the last `count` cycles of some
/// period `p ∈ min_period..=max_period` non-empty lines are byte-identical
/// after per-line normalization (trim ASCII whitespace, collapse digit runs
/// to `0`, ASCII-lowercase). The companion to `line_repeat`, which handles
/// the `p = 1` case (consecutive identical lines); this detector starts at
/// `p ≥ 2` so the two don't fight for attribution on the same pattern.
///
/// The motivating pattern is a numbered checklist like
/// `10. No code block? Yes.` / `11. No object wrapper? Yes.` repeated for
/// hundreds of lines with the counter drifting — each line is unique in
/// bytes, and consecutive lines differ from each other, so every
/// byte-level detector misses it.
///
/// Worst-case cost per check is `O(window)` — lines are normalized once,
/// then each period scan exits at the first mismatch.
///
/// Pattern catalog, real log examples, and tuning rationale live in
/// `docs/pipette-doomloop/doomloop-detection.md` (the operator-facing doc).
/// This rustdoc is the implementation spec.
#[derive(Debug, Clone)]
pub struct CyclicLineRepeat {
    pub min_chars: usize,
    pub window: usize,
    /// How many consecutive cycles of the winning period must match (e.g.
    /// `count = 6, period = 2` requires 12 lines forming 6 copies of an
    /// A,B pair).
    pub count: usize,
    /// Skip normalized lines shorter than this length (avoids triggering on
    /// JSON/YAML scaffolding such as `},` or `]`).
    pub min_len: usize,
    /// Relaxed gate applied when the normalized line contains a digit-run
    /// marker. Mirror of `LineRepeat::counter_min_len` — catches cyclic
    /// counter-scaffolded loops like the Egypt/Libya/Morocco/Algeria/Tunisia
    /// period-5 run whose normalized line (`0. tunisia`, 10 bytes) falls
    /// below the JSON-scaffold gate. Must be `<= min_len`.
    pub counter_min_len: usize,
    /// Cycle-count threshold when any collected line was admitted via the
    /// relaxed `counter_min_len` gate. Mirror of `LineRepeat::counter_count`
    /// — short digit-containing lines carry less signal so a longer cyclic
    /// streak is required to rule out table-style legit output. Must
    /// satisfy `counter_count >= count`.
    pub counter_count: usize,
    /// Smallest cyclic period (in lines) tried. Default is 2 so this
    /// detector doesn't shadow `line_repeat` on consecutive-identical runs.
    pub min_period: usize,
    /// Largest cyclic period (in lines) tried. Periods are scanned
    /// smallest-first and the first match wins.
    pub max_period: usize,
}

impl Default for CyclicLineRepeat {
    fn default() -> Self {
        Self {
            min_chars: 8192,
            window: 4096,
            count: 6,
            min_len: 12,
            counter_min_len: 8,
            counter_count: 10,
            min_period: 2,
            max_period: 16,
        }
    }
}

impl Detector for CyclicLineRepeat {
    fn name(&self) -> &'static str {
        "cyclic_line_repeat"
    }

    fn validate(&self) -> Result<(), String> {
        if self.window == 0 {
            return Err("window must be > 0".into());
        }
        if self.count < 2 {
            return Err("count must be >= 2".into());
        }
        if self.min_len == 0 {
            return Err("min_len must be > 0".into());
        }
        if self.counter_min_len == 0 {
            return Err("counter_min_len must be > 0".into());
        }
        if self.counter_min_len > self.min_len {
            return Err("counter_min_len must be <= min_len (relaxed gate cannot be stricter than the base gate)".into());
        }
        if self.counter_count < self.count {
            return Err("counter_count must be >= count (relaxed-gate streak needs at least as many cycles)".into());
        }
        if self.min_period < 2 {
            return Err("min_period must be >= 2 (period 1 belongs to line_repeat)".into());
        }
        if self.max_period < self.min_period {
            return Err("max_period must be >= min_period".into());
        }
        // Overflow alone is not enough: any product this side of usize::MAX still
        // passes. A product above `window` cannot buy a detection either, since a
        // line costs at least its newline, so reject it as a misconfiguration.
        //
        // This is a config-sanity gate, NOT an allocation guard: `window` is
        // caller-supplied too, so it cannot bound anything on its own. `check`
        // reserves nothing up front, so there is no budget to guard.
        let Some(needed) = self.max_period.checked_mul(self.counter_count) else {
            return Err("max_period * counter_count overflows usize".into());
        };
        if needed > self.window {
            return Err(
                "max_period * counter_count must be <= window (a window that small cannot hold that many lines)"
                    .into(),
            );
        }
        Ok(())
    }

    fn check(&self, content: &str) -> bool {
        if content.len() < self.min_chars {
            return false;
        }
        if self.window == 0
            || self.count < 2
            || self.min_len == 0
            || self.counter_min_len == 0
            || self.counter_min_len > self.min_len
            || self.counter_count < self.count
            || self.min_period < 2
            || self.max_period < self.min_period
        {
            return false;
        }
        // Collect enough lines to verify the stricter `counter_count`
        // threshold; the base `count` path uses a prefix of the same
        // buffer.
        let Some(needed) = self.max_period.checked_mul(self.counter_count) else {
            return false;
        };
        let tail = tail_of(content.as_bytes(), self.window);
        // `lines_rev[0]` is the final non-empty normalized line;
        // `lines_rev[needed - 1]` is the earliest one still relevant.
        // Walks imperatively so `scratch` retains its capacity across
        // lines that fail the min_len filter — no alloc per discarded line.
        //
        // Not reserved up front: every candidate ceiling (`needed`, `window`, even
        // the tail's length) is caller-supplied or scales with a buffer that grows
        // unbounded while a loop runs, and these grow geometrically anyway. Letting
        // them size themselves keeps the cost proportional to the lines actually
        // admitted, which is what the scan needs.
        let mut lines_rev: Vec<Vec<u8>> = Vec::new();
        let mut relaxed_flags: Vec<bool> = Vec::new();
        let mut scratch: Vec<u8> = Vec::new();
        for line in tail.split(|&b| b == b'\n').rev() {
            scratch.clear();
            normalize_line_into(line, &mut scratch);
            let effective_min_len = if scratch.contains(&b'0') {
                self.counter_min_len
            } else {
                self.min_len
            };
            if scratch.len() < effective_min_len {
                continue;
            }
            let relaxed_line = scratch.len() < self.min_len;
            lines_rev.push(std::mem::take(&mut scratch));
            relaxed_flags.push(relaxed_line);
            if lines_rev.len() == needed {
                break;
            }
        }
        (self.min_period..=self.max_period).any(|period| {
            // A cycle using only `count` repetitions is enough when every
            // line in that slice passed the base `min_len` gate. If any
            // line was admitted via the relaxed gate, require
            // `counter_count` cycles instead — short digit-scaffolded
            // lines carry less signal.
            let base_total = period * self.count;
            let counter_total = period * self.counter_count;
            if lines_rev.len() < base_total {
                return false;
            }
            let anchor = &lines_rev[..period];
            let cycles_match = |cycles: usize| {
                (1..cycles).all(|cycle_idx| {
                    let start = cycle_idx * period;
                    &lines_rev[start..start + period] == anchor
                })
            };
            let any_relaxed_in_window = |end: usize| relaxed_flags[..end].iter().any(|&r| r);
            if any_relaxed_in_window(base_total) {
                // Need the stricter counter_count streak.
                if lines_rev.len() < counter_total {
                    return false;
                }
                cycles_match(self.counter_count)
            } else {
                cycles_match(self.count)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One row of the A,B,A,B checklist fixture from
    /// `2026.04.16-macos-windows.log` sample id=629: the counter drifts
    /// but the body alternates between exactly two phrasings.
    fn checklist_line(i: u32) -> String {
        let body = if i.is_multiple_of(2) {
            "No code block? Yes."
        } else {
            "No object wrapper? Yes."
        };
        format!("            {i}. {body}\n")
    }

    /// Twelve lines: six A,B cycles, the minimum that fires at the default
    /// `count = 6` against `period = 2`.
    fn six_checklist_cycles() -> String {
        (16..=27).map(checklist_line).collect()
    }

    #[test]
    fn catches_checklist_alternation_from_real_log() {
        let d = CyclicLineRepeat {
            min_chars: 0,
            ..CyclicLineRepeat::default()
        };
        // 12 lines = 6 A,B cycles; count=6 × period=2 fires exactly here.
        let content = six_checklist_cycles();
        assert!(d.check(&content));
    }

    #[test]
    fn needs_six_full_cycles_for_period_two() {
        let d = CyclicLineRepeat {
            min_chars: 0,
            ..CyclicLineRepeat::default()
        };
        // Five A,B cycles = 10 lines — below the 12-line threshold
        // (count=6 × period=2).
        let content: String = (16..=25).map(checklist_line).collect();
        assert!(!d.check(&content));
    }

    // 7 cycles × 11 fields = 77 lines, safely above the 66-line minimum
    // (count=6 × period=11) and matches the log shape.
    fn schema_checklist_content() -> String {
        const FIELDS: [&str; 11] = [
            "ticket_type",
            "face_value_total_usd",
            "total_paid_usd",
            "days_until_event",
            "ticket_count",
            "section_label",
            "parking_pass_usd",
            "is_refundable",
            "includes_fast_entry",
            "delivery_method",
            "purchase_channel",
        ];
        (0..7)
            .flat_map(|_| FIELDS.iter())
            .zip(296u32..)
            .map(|(field, n)| format!("        {n}. `{field}`? Yes.\n"))
            .collect()
    }

    /// Fixture mirroring `2026.04.16-macos-windows.log` sample id=506:
    /// a model cycling through an 11-field schema checklist for hundreds
    /// of lines. Exercises periods beyond the common 2–4 range.
    #[test]
    fn catches_period_eleven_schema_checklist_from_real_log() {
        let d = CyclicLineRepeat {
            min_chars: 0,
            ..CyclicLineRepeat::default()
        };
        assert!(d.check(&schema_checklist_content()));
    }

    /// `max_period` bounds the longest cycle the detector will probe:
    /// a configured cap below the true period must not fire.
    #[test]
    fn max_period_below_actual_period_does_not_fire() {
        let d = CyclicLineRepeat {
            max_period: 4,
            ..CyclicLineRepeat::default()
        };
        assert!(!d.check(&schema_checklist_content()));
    }

    /// Synthetic period-13 cycle — exceeds the pre-bump cap of 12 but fits
    /// the current default of 16.
    #[test]
    fn catches_period_thirteen_cycle() {
        let d = CyclicLineRepeat {
            min_chars: 0,
            ..CyclicLineRepeat::default()
        };
        let fields: [&str; 13] = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
            "juliet", "kilo", "lima", "mike",
        ];
        let content: String = (0..6)
            .flat_map(|_| fields.iter())
            .zip(100u32..)
            .map(|(field, n)| format!("        {n}. `{field}`? Yes.\n"))
            .collect();
        assert!(d.check(&content));
    }

    #[test]
    fn catches_three_line_cycle() {
        let d = CyclicLineRepeat {
            min_chars: 0,
            ..CyclicLineRepeat::default()
        };
        let cycle = concat!(
            "    Checking the invariants before the next step.\n",
            "    Reviewing the original constraints once more.\n",
            "    Reconfirming the output format requirements.\n",
        );
        let content = cycle.repeat(6);
        assert!(d.check(&content));
    }

    #[test]
    fn ignores_single_abc_occurrence() {
        let d = CyclicLineRepeat {
            min_chars: 0,
            ..CyclicLineRepeat::default()
        };
        let content = concat!(
            "    The first unique point about the schema constraints.\n",
            "    The second unique point about the validation rules.\n",
            "    The third unique point about the output structure.\n",
        );
        assert!(!d.check(content));
    }

    #[test]
    fn ignores_consecutive_identical_lines() {
        // `line_repeat` owns the period-1 case. `cyclic_line_repeat` with
        // min_period=2 must leave it alone so the pipeline attributes the
        // trigger to the more specific detector.
        let d = CyclicLineRepeat {
            min_chars: 0,
            ..CyclicLineRepeat::default()
        };
        let content: String = (0..10)
            .map(|i| format!("    *   *Recipe {i}: Ethiopian Fried Chicken* (Classic side dish)\n"))
            .collect();
        assert!(!d.check(&content));
    }

    #[test]
    fn ignores_non_repeating_prose() {
        let d = CyclicLineRepeat {
            min_chars: 0,
            ..CyclicLineRepeat::default()
        };
        let content = concat!(
            "    * Step 1: check the raw input value and move to the parser.\n",
            "    * Step 2: tokenize the payload and record the boundaries.\n",
            "    * Step 3: validate the schema against the registered types.\n",
            "    * Step 4: emit a serialized record to the downstream buffer.\n",
            "    * Step 5: flush the batch once the watermark is exceeded.\n",
            "    * Step 6: log a completion event with the request identifier.\n",
            "    * Step 7: close the writer and return the accumulated totals.\n",
            "    * Step 8: schedule the follow-up task for the idle worker.\n",
        );
        assert!(!d.check(content));
    }

    /// A legitimate 12-row JSON-like table with per-row-distinct values —
    /// no period-p cycle forms because every normalized row differs.
    /// Guards against false-positives on structured tabular output.
    #[test]
    fn does_not_fire_on_json_table_with_distinct_values() {
        let d = CyclicLineRepeat {
            min_chars: 0,
            ..CyclicLineRepeat::default()
        };
        const NAMES: [&str; 12] = [
            "Alice Johnson",
            "Bob Smith",
            "Carol Davis",
            "David Brown",
            "Emma Wilson",
            "Frank Miller",
            "Grace Lee",
            "Henry Moore",
            "Iris Taylor",
            "Jack Anderson",
            "Karen Martin",
            "Larry Clark",
        ];
        const ROLES: [&str; 12] = [
            "engineer",
            "designer",
            "manager",
            "analyst",
            "architect",
            "consultant",
            "developer",
            "operator",
            "researcher",
            "trainer",
            "auditor",
            "strategist",
        ];
        let content: String = (0..12)
            .map(|i| {
                format!(
                    "  {{\"id\": {id}, \"name\": \"{name}\", \"role\": \"{role}\", \"team\": \"Alpha-{team}\", \"since\": {year}}},\n",
                    id = i + 100,
                    name = NAMES[i],
                    role = ROLES[i],
                    team = i,
                    year = 2020 + i,
                )
            })
            .collect();
        assert!(!d.check(&content));
    }

    /// Real-log Pattern C (macos-windows, sample d00dce92cf82): 1331 lines
    /// cycling through a 5-country set with a drifting counter. Normalized
    /// form `"0. tunisia"` is 10 bytes — below `min_len = 12` but above
    /// `counter_min_len = 8`, contains a digit-run marker, so the relaxed
    /// gate lets the period-5 cycle through once `counter_count = 10`
    /// cycles (50 lines) accumulate.
    #[test]
    fn catches_counter_plus_cyclic_short_body() {
        let d = CyclicLineRepeat {
            min_chars: 0,
            ..CyclicLineRepeat::default()
        };
        const COUNTRIES: [&str; 5] = ["Tunisia", "Egypt", "Libya", "Morocco", "Algeria"];
        // 10 cycles × period 5 = 50 lines — the counter_count threshold.
        let content: String = (0..10)
            .flat_map(|_| COUNTRIES.iter())
            .zip(700u32..)
            .map(|(country, n)| format!("{n}. {country}\n"))
            .collect();
        assert!(d.check(&content));
    }

    /// A plausible period-2 JSON-like cycle: alternating `"year"` / `"month"`
    /// rows where only the numeric field varies. Each line normalizes to
    /// 10-byte `"year": 0,` / `"month": 0,` — passes `counter_min_len = 8`
    /// but a mere 6-cycle streak (12 lines) should NOT fire. The stricter
    /// `counter_count = 10` (20 lines at period 2) is the threshold.
    #[test]
    fn relaxed_gate_does_not_fire_on_six_cycle_json_table() {
        let d = CyclicLineRepeat {
            min_chars: 0,
            ..CyclicLineRepeat::default()
        };
        let content: String = (0..6)
            .flat_map(|i| {
                [
                    format!("\"year\": {year},\n", year = 2020 + i),
                    format!("\"month\": {month},\n", month = i + 1),
                ]
            })
            .collect();
        assert!(!d.check(&content));
    }

    /// And the existing period-2 alternating checklist from the real log —
    /// long lines (normalized >= min_len = 12) — still fires at the
    /// original `count = 6` threshold, since no line needed the relaxed
    /// gate.
    #[test]
    fn long_line_cycles_still_fire_at_count_six() {
        let d = CyclicLineRepeat {
            min_chars: 0,
            ..CyclicLineRepeat::default()
        };
        let content = six_checklist_cycles();
        assert!(d.check(&content));
    }

    #[test]
    fn validate_rejects_counter_min_len_above_min_len() {
        let d = CyclicLineRepeat {
            counter_min_len: 20,
            min_len: 12,
            ..CyclicLineRepeat::default()
        };
        assert!(d.validate().is_err());
    }

    #[test]
    fn validate_rejects_line_budget_larger_than_the_window() {
        // The pre-cap gate only caught arithmetic overflow, so a product this
        // far below usize::MAX passed and `check` sized two Vecs from it.
        let d = CyclicLineRepeat {
            max_period: 1 << 30,
            counter_count: 1 << 30,
            ..CyclicLineRepeat::default()
        };
        assert!(d.validate().is_err());
    }

    #[test]
    fn oversized_line_budget_does_not_reserve_it() {
        // `check` runs on configs that never saw `validate`, so an absurd line
        // budget has to stay harmless on its own. Reserving `needed` up front
        // aborted the process here on the capacity request.
        let d = CyclicLineRepeat {
            min_chars: 0,
            window: 64,
            max_period: 1 << 30,
            counter_count: 1 << 30,
            ..CyclicLineRepeat::default()
        };
        let content = six_checklist_cycles();
        // Far more lines are demanded than the window can hold, so no cycle is
        // ever confirmed; the point is that it returns instead of aborting.
        assert!(!d.check(&content));
    }

    #[test]
    fn oversized_line_budget_survives_a_window_raised_to_match_it() {
        // The bypass any config-derived ceiling misses: raising `window` alongside
        // the period fields keeps `needed <= window`, so `validate` passes and a
        // `window`-bounded reserve is right back to 2^60. Bounding by the tail's
        // length instead only traded that for a reserve scaling with the buffer.
        // Reaching the assertion at all is the point.
        let d = CyclicLineRepeat {
            min_chars: 0,
            window: usize::MAX,
            max_period: 1 << 40,
            counter_count: 1 << 20,
            // Pinned rather than inherited: the verdict below depends on these.
            count: 6,
            min_period: 2,
            ..CyclicLineRepeat::default()
        };
        assert!(
            d.validate().is_ok(),
            "the config-sanity gate cannot catch this"
        );
        // Six cycles of a period-2 alternation, scanned smallest-period first, so
        // firing is the correct verdict; the abort is what regressed before.
        let content = six_checklist_cycles();
        assert!(d.check(&content));
    }

    #[test]
    fn validate_accepts_the_default_line_budget() {
        // Guards the new bound against being tightened past the shipped config.
        assert!(CyclicLineRepeat::default().validate().is_ok());
    }

    #[test]
    fn validate_rejects_counter_count_below_count() {
        let d = CyclicLineRepeat {
            counter_count: 4,
            count: 6,
            ..CyclicLineRepeat::default()
        };
        assert!(d.validate().is_err());
    }

    #[test]
    fn zeroed_window_returns_false() {
        let d = CyclicLineRepeat {
            window: 0,
            ..CyclicLineRepeat::default()
        };
        let content = "    A line that normalizes consistently and is long enough.\n".repeat(20);
        assert!(!d.check(&content));
    }

    #[test]
    fn min_period_below_two_is_misconfigured() {
        // A configured min_period of 1 would overlap with line_repeat —
        // the detector refuses to run rather than producing ambiguous
        // attribution.
        let d = CyclicLineRepeat {
            min_period: 1,
            ..CyclicLineRepeat::default()
        };
        let content: String = (0..10)
            .map(|i| format!("    *   *Recipe {i}: Ethiopian Fried Chicken*\n"))
            .collect();
        assert!(!d.check(&content));
    }
}
