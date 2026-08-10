use crate::{tail_of, Detector};

/// Numeric enumeration: ≥ `count` consecutive non-empty trailing lines
/// differ only in their trailing ASCII-digit run. Catches the runaway
/// "enumerate every integer from 2 to 2500" pattern that eludes
/// `line_repeat` because the line, once digit-collapsed and trimmed, falls
/// below `line_repeat.min_len = 12`.
///
/// False-positive surface is kept narrow by three rules:
/// - the non-digit template (bytes before the trailing digits, post-trim)
///   must be at least `min_template_len` bytes, ruling out pure number
///   streams like `1\n2\n3\n`;
/// - at least two distinct digit values must appear in the matched streak,
///   so same-number runs stay the responsibility of `exact_repeat` /
///   `suffix_probe` and this detector specializes in digit drift;
/// - `count` defaults to 40 so short legitimate numbered lists pass
///   through while runaway enumerations (hundreds of items) still fire.
///
/// Pattern catalog and tuning rationale live in
/// `docs/pipette-doomloop/doomloop-detection.md`.
#[derive(Debug, Clone)]
pub struct NumericEnumeration {
    pub min_chars: usize,
    pub window: usize,
    /// Consecutive non-empty lines with identical non-digit template needed
    /// to fire. Short legitimate numbered lists rarely exceed a few dozen
    /// items; runaway enumerations run to hundreds or thousands.
    pub count: usize,
    /// Minimum byte length of the non-digit template. Zero would let pure
    /// number streams (`1\n2\n3\n`) fire.
    pub min_template_len: usize,
}

impl Default for NumericEnumeration {
    fn default() -> Self {
        Self {
            min_chars: 8192,
            window: 8192,
            count: 40,
            // 3 bytes (not 2) to tighten the FP surface: 2 would admit
            // templates like `. `, `# `, `= ` that give the detector too
            // little structural signal to distinguish loop from legit
            // enumeration. The real-log target template `*   ` is 4 bytes,
            // comfortably above 3.
            min_template_len: 3,
        }
    }
}

impl Detector for NumericEnumeration {
    fn name(&self) -> &'static str {
        "numeric_enumeration"
    }

    fn validate(&self) -> Result<(), String> {
        if self.window == 0 {
            return Err("window must be > 0".into());
        }
        if self.count < 2 {
            return Err("count must be >= 2".into());
        }
        if self.min_template_len == 0 {
            return Err("min_template_len must be > 0 (zero permits pure-number streams)".into());
        }
        Ok(())
    }

    fn check(&self, content: &str) -> bool {
        if content.len() < self.min_chars {
            return false;
        }
        if self.window == 0 || self.count < 2 || self.min_template_len == 0 {
            return false;
        }
        let tail = tail_of(content.as_bytes(), self.window);
        let mut anchor: Option<Vec<u8>> = None;
        let mut first_digits: Option<Vec<u8>> = None;
        let mut saw_distinct = false;
        let mut matched = 0usize;
        for line in tail.split(|&b| b == b'\n').rev() {
            let trimmed = line.trim_ascii();
            if trimmed.is_empty() {
                continue;
            }
            let digit_start = trimmed
                .iter()
                .rposition(|b| !b.is_ascii_digit())
                .map(|i| i + 1)
                .unwrap_or(0);
            if digit_start == trimmed.len() {
                return false;
            }
            let template = &trimmed[..digit_start];
            let digits = &trimmed[digit_start..];
            if template.len() < self.min_template_len {
                return false;
            }
            match anchor.as_deref() {
                None => {
                    anchor = Some(template.to_vec());
                    first_digits = Some(digits.to_vec());
                    matched = 1;
                }
                Some(a) if a == template => {
                    matched += 1;
                    if first_digits.as_deref() != Some(digits) {
                        saw_distinct = true;
                    }
                }
                Some(_) => return false,
            }
            if matched >= self.count && saw_distinct {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catches_ssh_prime_enumeration() {
        // Bare bullet list with trailing integers — the real-log pattern
        // that slipped past `line_repeat` (normalized line `*   0` is 5
        // bytes, below line_repeat.min_len = 12).
        let d = NumericEnumeration {
            min_chars: 0,
            ..NumericEnumeration::default()
        };
        let content: String = (2..=2500u32)
            .map(|n| format!("        *   {n}\n"))
            .collect();
        assert!(d.check(&content));
    }

    #[test]
    fn ignores_short_legitimate_enumeration() {
        // 20 items is plausibly a user-requested list; below count=40.
        // Template `*   ` is 4 bytes so the count gate — not the
        // min_template_len gate — is what blocks detection.
        let d = NumericEnumeration {
            min_chars: 0,
            ..NumericEnumeration::default()
        };
        let content: String = (1..=20u32).map(|n| format!("*   {n}\n")).collect();
        assert!(!d.check(&content));
    }

    /// A 2-byte template (`* `) is now below `min_template_len = 3` and
    /// must not fire even with 100+ matching lines. Tightens the FP
    /// surface on sparse enumerations like `* 1`, `- 2`, `. 3`.
    #[test]
    fn ignores_two_byte_templates() {
        let d = NumericEnumeration {
            min_chars: 0,
            ..NumericEnumeration::default()
        };
        let content: String = (1..=100u32).map(|n| format!("* {n}\n")).collect();
        assert!(!d.check(&content));
    }

    #[test]
    fn ignores_pure_number_stream() {
        // Template is empty — min_template_len gate blocks it.
        let d = NumericEnumeration {
            min_chars: 0,
            ..NumericEnumeration::default()
        };
        let content: String = (1..=200u32).map(|n| format!("{n}\n")).collect();
        assert!(!d.check(&content));
    }

    #[test]
    fn ignores_varying_template() {
        // Different non-digit prefixes break the streak after one line.
        let d = NumericEnumeration {
            min_chars: 0,
            ..NumericEnumeration::default()
        };
        let mut content = String::new();
        for n in 1..=200u32 {
            let verb = match n % 4 {
                0 => "Step",
                1 => "Item",
                2 => "Entry",
                _ => "Row",
            };
            content.push_str(&format!("* {verb} {n}\n"));
        }
        assert!(!d.check(&content));
    }

    #[test]
    fn ignores_identical_digit_runs() {
        // Same template AND same digits across the streak — this is
        // `exact_repeat`/`suffix_probe` territory. `saw_distinct` guard
        // prevents us from fighting for attribution.
        let d = NumericEnumeration {
            min_chars: 0,
            ..NumericEnumeration::default()
        };
        let content: String = "* 5\n".repeat(200);
        assert!(!d.check(&content));
    }

    #[test]
    fn skips_interleaved_blank_lines() {
        // Enough lines to clear both count=40 and min_chars=512.
        // Template `*   ` is 4 bytes — clears min_template_len=3.
        let d = NumericEnumeration {
            min_chars: 0,
            ..NumericEnumeration::default()
        };
        let content: String = (1..=200u32).map(|n| format!("*   {n}\n\n")).collect();
        assert!(d.check(&content));
    }

    #[test]
    fn respects_min_chars() {
        // Pattern is present but total bytes < min_chars → no fire. Use a
        // 4-byte template so min_template_len isn't what blocks it.
        let d = NumericEnumeration {
            min_chars: 10_000,
            ..NumericEnumeration::default()
        };
        let content: String = (1..=50u32).map(|n| format!("*   {n}\n")).collect();
        assert!(!d.check(&content));
    }

    #[test]
    fn stops_on_non_digit_terminated_line() {
        // A prose line breaks the tail streak and we bail immediately.
        // Template `*   ` is 4 bytes so the prose tail — not min_template_len —
        // is what stops detection.
        let d = NumericEnumeration {
            min_chars: 0,
            ..NumericEnumeration::default()
        };
        let mut content: String = (1..=200u32).map(|n| format!("*   {n}\n")).collect();
        content.push_str("and finally, the conclusion follows here.\n");
        assert!(!d.check(&content));
    }

    #[test]
    fn validate_rejects_bad_knobs() {
        let bad = NumericEnumeration {
            window: 0,
            ..NumericEnumeration::default()
        };
        assert!(bad.validate().is_err());
        let bad = NumericEnumeration {
            count: 1,
            ..NumericEnumeration::default()
        };
        assert!(bad.validate().is_err());
        let bad = NumericEnumeration {
            min_template_len: 0,
            ..NumericEnumeration::default()
        };
        assert!(bad.validate().is_err());
        assert!(NumericEnumeration::default().validate().is_ok());
    }
}
