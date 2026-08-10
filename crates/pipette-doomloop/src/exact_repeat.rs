use crate::{tail_of, Detector};

/// Exact byte-for-byte tail repetition: the suffix of the window consists of
/// `>= required` consecutive byte-identical copies of a block whose length is
/// `>= min_period`. Worst-case cost per check is `O(window² / required)`
/// before short-circuit.
///
/// Pattern catalog, real log examples, and tuning rationale live in
/// `docs/pipette-doomloop/doomloop-detection.md` (the operator-facing doc).
/// This rustdoc is the implementation spec.
#[derive(Debug, Clone)]
pub struct ExactRepeat {
    pub min_chars: usize,
    pub window: usize,
    /// Shortest repeating unit (period) considered, in bytes.
    pub min_period: usize,
    /// Number of consecutive byte-identical copies required to fire.
    pub required: usize,
}

impl Default for ExactRepeat {
    fn default() -> Self {
        Self {
            min_chars: 8192,
            window: 4096,
            min_period: 32,
            required: 3,
        }
    }
}

impl Detector for ExactRepeat {
    fn name(&self) -> &'static str {
        "exact_repeat"
    }

    fn validate(&self) -> Result<(), String> {
        if self.window == 0 {
            return Err("window must be > 0".into());
        }
        if self.min_period == 0 {
            return Err("min_period must be > 0".into());
        }
        if self.required < 2 {
            return Err("required must be >= 2".into());
        }
        Ok(())
    }

    fn check(&self, content: &str) -> bool {
        if content.len() < self.min_chars {
            return false;
        }
        // Validity guards — degenerate config can't produce a meaningful match.
        if self.window == 0 || self.min_period == 0 || self.required < 2 {
            return false;
        }
        let tail = tail_of(content.as_bytes(), self.window);
        let min_needed = match self.min_period.checked_mul(self.required) {
            Some(n) => n,
            None => return false,
        };
        if tail.len() < min_needed {
            return false;
        }
        let max_period = tail.len() / self.required;
        let mut period = self.min_period;
        while period <= max_period {
            if exact_suffix_repeats(tail, period, self.required) {
                return true;
            }
            period += 1;
        }
        false
    }
}

fn exact_suffix_repeats(tail: &[u8], period: usize, required: usize) -> bool {
    let needed = match period.checked_mul(required) {
        Some(n) => n,
        None => return false,
    };
    if tail.len() < needed {
        return false;
    }
    let region = &tail[tail.len() - needed..];
    let pattern = &region[..period];
    (1..required).all(|i| &region[i * period..(i + 1) * period] == pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catches_single_newline_token_loop() {
        let d = ExactRepeat {
            min_chars: 0,
            ..ExactRepeat::default()
        };
        let content = "\n".repeat(500);
        assert!(d.check(&content));
    }

    #[test]
    fn catches_single_short_word_token_loop() {
        let d = ExactRepeat {
            min_chars: 0,
            ..ExactRepeat::default()
        };
        let content = "the ".repeat(300);
        assert!(d.check(&content));
    }

    #[test]
    fn catches_single_3char_token_loop() {
        let d = ExactRepeat {
            min_chars: 0,
            ..ExactRepeat::default()
        };
        let content = "ab\n".repeat(200);
        assert!(d.check(&content));
    }

    #[test]
    fn log_pattern_1_wait_actually_abab_loop() {
        let d = ExactRepeat {
            min_chars: 0,
            ..ExactRepeat::default()
        };
        let block = concat!(
            "        *   Wait, there is a specific historical figure named ",
            "John of Brienne who was a knight in the 12th century, but he ",
            "was not the King of Jerusalem.\n",
            "        *   Actually, there is a specific historical figure named ",
            "John of Brienne who was a knight in the 12th century, but he ",
            "was not the King of Jerusalem.\n",
        );
        let content = block.repeat(5);
        assert!(d.check(&content));
    }

    #[test]
    fn log_pattern_2_short_phrase_filler_loop() {
        let d = ExactRepeat {
            min_chars: 0,
            ..ExactRepeat::default()
        };
        let preamble = concat!(
            "This project aims to study how words change over time. ",
            "We will analyze the data from different languages. ",
            "We will look at the patterns of change. ",
            "We will compare the structures. ",
            "We will find the roots. We will see the impact. ",
        );
        let loop_block = "We will see the future. We will see the past. ";
        let content = format!("{}{}", preamble, loop_block.repeat(50));
        assert!(d.check(&content));
    }

    #[test]
    fn log_pattern_3_large_block_outline_loop() {
        let d = ExactRepeat {
            min_chars: 0,
            ..ExactRepeat::default()
        };
        let block = concat!(
            "    *Let's try to make the outline more detailed.*\n",
            "    *   1. Introduction\n",
            "    *   2. Ancient Origins\n",
            "    *   3. The 19th Century Trade\n",
            "    *   4. The 1950s Industrial Boom\n",
            "    *   5. The 1970s Coffee Revolution\n",
            "    *   6. The 1990s Market Expansion\n",
            "    *   7. The 2000s Global Influence\n",
            "    *   8. The 2010s Sustainability\n",
            "    *   9. The 2020s Future\n",
            "    *   10. The Yemeni Coffee Environment\n",
            "    *   11. The Yemeni Coffee Society\n",
            "    *   12. The Yemeni Coffee Technology\n",
            "    *   13. The Yemeni Coffee Policy\n",
            "    *   14. The Yemeni Coffee Education\n",
            "    *   15. The Yemeni Coffee Conclusion\n",
            "\n",
        );
        let content = block.repeat(4);
        assert!(d.check(&content));
    }

    #[test]
    fn ignores_non_repeating_text() {
        // Use min_chars: 0 so the test exercises the period-scan logic
        // rather than being short-circuited by the length gate.
        let d = ExactRepeat {
            min_chars: 0,
            ..ExactRepeat::default()
        };
        let content = "The quick brown fox jumps over the lazy dog. \
                        Pack my box with five dozen liquor jugs. \
                        How vexingly quick daft zebras jump.";
        assert!(!d.check(content));
    }

    #[test]
    fn respects_min_period() {
        let d = ExactRepeat {
            min_chars: 0,
            min_period: 200,
            ..ExactRepeat::default()
        };
        let content = "short block! ".repeat(20);
        assert!(!d.check(&content));
    }

    #[test]
    fn too_few_copies_does_not_trigger() {
        let d = ExactRepeat {
            min_chars: 0,
            min_period: 10,
            required: 10,
            ..ExactRepeat::default()
        };
        let block = "abcdefghij0123456789ABCDEFGHIJ!@#$%^&*()abcdefghij9876543210";
        assert_eq!(block.len(), 60);
        let content = block.repeat(3);
        assert!(!d.check(&content));
    }

    #[test]
    fn zeroed_required_returns_false() {
        let d = ExactRepeat {
            required: 1,
            ..ExactRepeat::default()
        };
        let content = "the ".repeat(300);
        assert!(!d.check(&content));
    }
}
