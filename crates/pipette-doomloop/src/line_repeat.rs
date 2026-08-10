use crate::{tail_of, Detector};

/// Normalize a single line for the line-repeat detector: trim leading/trailing
/// ASCII whitespace, collapse runs of ASCII digits to a single `0`, and
/// ASCII-lowercase letters. Non-ASCII bytes pass through unchanged. Writes
/// into the caller's scratch buffer to avoid per-line allocations on the hot
/// path.
pub(crate) fn normalize_line_into(line: &[u8], out: &mut Vec<u8>) {
    let trimmed = line.trim_ascii();
    out.reserve(trimmed.len());
    let mut prev_digit = false;
    for &b in trimmed {
        if b.is_ascii_digit() {
            if !prev_digit {
                out.push(b'0');
                prev_digit = true;
            }
        } else {
            prev_digit = false;
            out.push(b.to_ascii_lowercase());
        }
    }
}

/// Normalized line repetition: the last `count` non-empty lines of the
/// window are byte-identical after normalization (trim ASCII whitespace,
/// collapse digit runs to `0`, ASCII-lowercase). Worst-case cost per check
/// is `O(window)` — the scan exits at the first mismatch.
///
/// Pattern catalog, real log examples, and tuning rationale live in
/// `docs/pipette-doomloop/doomloop-detection.md` (the operator-facing doc).
/// This rustdoc is the implementation spec.
#[derive(Debug, Clone)]
pub struct LineRepeat {
    pub min_chars: usize,
    pub window: usize,
    /// How many trailing non-empty normalized lines must be byte-identical.
    pub count: usize,
    /// Skip normalized lines shorter than this length (avoids triggering on
    /// JSON/YAML scaffolding such as `},` or `]`).
    pub min_len: usize,
    /// Relaxed gate applied when the normalized line contains a digit-run
    /// marker (`b'0'` left behind by the digit-collapse step). Counter-
    /// scaffolded loops like `"740. paradox\n741. paradox\n…"` normalize to
    /// a 10-byte line that would be filtered by `min_len = 12` but is an
    /// obvious pathology. Digits-in-line distinguish counter scaffolding
    /// from JSON fragments (which never contain digits), so the relaxed
    /// gate doesn't regress on the cases `min_len` was designed to skip.
    /// Must be `<= min_len` — a stricter counter gate would be nonsensical.
    pub counter_min_len: usize,
    /// Consecutive-match threshold when any matched line was admitted via
    /// the relaxed `counter_min_len` gate (`< min_len`). Raised above
    /// `count` because short digit-containing lines carry less signal — a
    /// JSON-like stream `"year": 2024,\n"year": 2025,\n…` normalizes to an
    /// identical 10-byte form and would false-fire at `count = 6`, but
    /// requiring more repetitions pushes the threshold past the plausible
    /// legit-enumeration band. Must satisfy `counter_count >= count`.
    pub counter_count: usize,
}

impl Default for LineRepeat {
    fn default() -> Self {
        Self {
            min_chars: 8192,
            window: 4096,
            count: 6,
            min_len: 12,
            counter_min_len: 8,
            counter_count: 10,
        }
    }
}

impl Detector for LineRepeat {
    fn name(&self) -> &'static str {
        "line_repeat"
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
            return Err("counter_count must be >= count (relaxed-gate streak needs at least as many matches)".into());
        }
        Ok(())
    }

    fn check(&self, content: &str) -> bool {
        if content.len() < self.min_chars {
            return false;
        }
        // Validity guards. `min_len == 0` would let blank/whitespace-only
        // lines (which normalize to an empty byte string) count as matches,
        // so any two consecutive blank lines would trip the detector.
        if self.window == 0
            || self.count < 2
            || self.min_len == 0
            || self.counter_min_len == 0
            || self.counter_min_len > self.min_len
            || self.counter_count < self.count
        {
            return false;
        }
        let tail = tail_of(content.as_bytes(), self.window);
        let mut anchor: Option<Vec<u8>> = None;
        let mut matched = 0usize;
        // Tracks whether any matched line was admitted via the relaxed
        // gate (shorter than `min_len`). If so the streak needs
        // `counter_count` matches instead of `count` to fire.
        let mut any_relaxed = false;
        let mut scratch = Vec::new();
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
            match anchor.as_deref() {
                None => {
                    anchor = Some(scratch.clone());
                    matched = 1;
                    any_relaxed = relaxed_line;
                }
                Some(a) if a == scratch.as_slice() => {
                    matched += 1;
                    any_relaxed |= relaxed_line;
                }
                Some(_) => return false,
            }
            let required = if any_relaxed {
                self.counter_count
            } else {
                self.count
            };
            if matched >= required {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalize(line: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        normalize_line_into(line, &mut out);
        out
    }

    #[test]
    fn normalize_trims_collapses_digits_and_lowercases() {
        assert_eq!(
            normalize(b"  Step 42: Retry 1003 Times  "),
            b"step 0: retry 0 times"
        );
        assert_eq!(normalize(b"Item 1000000 price"), b"item 0 price");
        let cn = "现在，我需要确保".as_bytes();
        assert_eq!(normalize(cn), cn);
    }

    #[test]
    fn catches_numbered_property_drift() {
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        let content: String = (0..8)
            .map(|i| {
                format!(
                    "        *   Property {i}: 1 story, 1 bedroom, {price} price, {year} year built\n",
                    price = 1_000_000 + i,
                    year = 1880 + i,
                )
            })
            .collect();
        assert!(d.check(&content));
    }

    #[test]
    fn catches_recipe_counter_drift() {
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        let content: String = (200..220)
            .map(|i| format!("    *   *Recipe {i}: Ethiopian Fried Chicken* (Classic side dish)\n"))
            .collect();
        assert!(d.check(&content));
    }

    #[test]
    fn catches_case_and_space_drift() {
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        let lines = [
            "  I will ensure the values are consistent with a venue access summary sheet.",
            "I will Ensure the values are consistent with a venue access summary sheet.  ",
            "I WILL ensure the values are consistent with a venue access summary sheet.",
            "I will ensure the values are consistent with a venue access summary sheet.",
            "I will ensure the values are consistent with a venue access summary sheet.",
            "I will ensure the values are consistent with a venue access summary sheet.",
            "  I will ensure the values are consistent with a venue access summary sheet.  ",
        ];
        let content = lines.join("\n");
        assert!(d.check(&content));
    }

    #[test]
    fn catches_iso_timestamp_drift() {
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        let content: String = (0..7)
            .map(|i| {
                format!(
                    "[step 42{i}] retrying request id={id}, backoff=1234ms\n",
                    id = 100000 + i,
                )
            })
            .collect();
        assert!(d.check(&content));
    }

    #[test]
    fn ignores_short_lines() {
        // min_chars: 0 so the test exercises the line-length-skip logic
        // rather than being short-circuited by the length gate.
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        let content = "},\n]\n  -\n},\n]\n  -\n},\n]\n";
        assert!(!d.check(content));
    }

    #[test]
    fn skips_interleaved_blank_lines() {
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        let content = "I will ensure the values are consistent.\n\n".repeat(7);
        assert!(d.check(&content));
    }

    #[test]
    fn ignores_similar_but_varying_lines() {
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        let content = concat!(
            "    * Step 1: check the raw input value and move to the parser.\n",
            "    * Step 2: tokenize the payload and record the boundaries.\n",
            "    * Step 3: validate the schema against the registered types.\n",
            "    * Step 4: emit a serialized record to the downstream buffer.\n",
            "    * Step 5: flush the batch once the watermark is exceeded.\n",
            "    * Step 6: log a completion event with the request identifier.\n",
        );
        assert!(!d.check(content));
    }

    #[test]
    fn ignores_json_schema_scaffolding() {
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        let content = concat!(
            "{\n",
            "  \"type\": \"array\",\n",
            "  \"items\": {\n",
            "    \"type\": \"object\",\n",
            "    \"properties\": {\n",
            "      \"title\": { \"type\": \"string\" },\n",
            "      \"body\":  { \"type\": \"string\" },\n",
            "      \"done\":  { \"type\": \"boolean\" },\n",
            "      \"count\": { \"type\": \"integer\" }\n",
            "    }\n",
            "  }\n",
            "}\n",
        );
        assert!(!d.check(content));
    }

    /// Real-log Pattern A (macos-llamacpp, sample 30ad83827a3e): hundreds of
    /// `"<N>. paradox"` lines. Normalized form `"0. paradox"` is 10 bytes —
    /// below the JSON-scaffold `min_len = 12` gate but above
    /// `counter_min_len = 8`, and contains a digit-run marker, so the
    /// relaxed gate lets it through.
    #[test]
    fn catches_counter_plus_fixed_word() {
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        let content: String = (700..=720u32)
            .map(|n| format!("        {n}. paradox\n"))
            .collect();
        assert!(d.check(&content));
    }

    /// Real-log Pattern E (macos-windows, sample ed47baed382d): scaffolded
    /// counter with the digits parenthesized mid-line. Normalized `"*   . (0)"`
    /// is 9 bytes, contains a digit-run marker, passes the relaxed gate.
    #[test]
    fn catches_parenthesized_counter() {
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        let content: String = (195..=215u32)
            .map(|n| format!("        *   . ({n})\n"))
            .collect();
        assert!(d.check(&content));
    }

    /// Version strings like `v1.0.0` × 6 normalize to `"v0.0.0"` — 6 bytes,
    /// still below `counter_min_len = 8`, so the relaxed gate does not
    /// misfire on this legitimate-looking short pattern.
    #[test]
    fn still_gates_version_strings_below_counter_min_len() {
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        let content = "v1.0.0\nv1.0.0\nv1.0.0\nv1.0.0\nv1.0.0\nv1.0.0\nv1.0.0\n";
        assert!(!d.check(content));
    }

    /// JSON-scaffold lines contain no digits, so the relaxed gate is
    /// irrelevant and `min_len = 12` continues to guard them. SweepPins the
    /// non-regression property.
    #[test]
    fn counter_gate_does_not_affect_json_scaffolding() {
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        let content = "},\n]\n},\n]\n},\n]\n},\n]\n},\n]\n".repeat(3);
        assert!(!d.check(&content));
    }

    /// A JSON row with one varying numeric field normalizes to a 10-byte
    /// form that passes `counter_min_len = 8` (so `min_len = 12` no longer
    /// protects it). `counter_count = 10` pushes the required streak past
    /// `count = 6`, so a 6-row legit table does NOT false-fire.
    #[test]
    fn relaxed_gate_does_not_fire_on_six_json_rows() {
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        let content: String = (2020..=2025)
            .map(|year| format!("\"year\": {year},\n"))
            .collect();
        assert!(!d.check(&content));
    }

    /// But 10+ consecutive identical normalized rows still fire — runaway
    /// output pins this as pathological.
    #[test]
    fn relaxed_gate_fires_at_counter_count() {
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        // 20 rows × ~15 bytes = 300 bytes, above min_chars = 256.
        let content: String = (2020..=2039)
            .map(|year| format!("\"year\": {year},\n"))
            .collect();
        assert!(d.check(&content));
    }

    /// When every matched line is >= min_len (normalized ≥ 12 bytes), the
    /// streak uses `count = 6`, not `counter_count` — the existing
    /// behavior is preserved. 6 identical long lines fire as before.
    #[test]
    fn long_lines_still_fire_at_count_six() {
        let d = LineRepeat {
            min_chars: 0,
            ..LineRepeat::default()
        };
        let content = "I will ensure the values are consistent with the schema.\n".repeat(6);
        assert!(d.check(&content));
    }

    #[test]
    fn validate_rejects_counter_min_len_above_min_len() {
        let d = LineRepeat {
            counter_min_len: 20,
            min_len: 12,
            ..LineRepeat::default()
        };
        assert!(d.validate().is_err());
    }

    #[test]
    fn validate_rejects_counter_count_below_count() {
        let d = LineRepeat {
            counter_count: 4,
            count: 6,
            ..LineRepeat::default()
        };
        assert!(d.validate().is_err());
    }

    #[test]
    fn min_len_zero_is_treated_as_misconfigured() {
        // Without the min_len guard, any two consecutive blank lines would
        // match the anchor since `normalize_line_into` on whitespace yields
        // "". min_chars: 0 so we exercise the min_len guard, not the length
        // gate.
        let d = LineRepeat {
            min_chars: 0,
            min_len: 0,
            count: 2,
            ..LineRepeat::default()
        };
        let content = "\n\n\n\n\n\n\n".to_string();
        assert!(!d.check(&content));
    }
}
