use crate::{tail_of, Detector};

/// Count non-overlapping occurrences of `needle` in `hay`. Used by the
/// suffix-probe density check.
fn count_non_overlapping(hay: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() || hay.len() < needle.len() {
        return 0;
    }
    let mut count = 0usize;
    let mut i = 0usize;
    while i + needle.len() <= hay.len() {
        if &hay[i..i + needle.len()] == needle {
            count += 1;
            i += needle.len();
        } else {
            i += 1;
        }
    }
    count
}

/// Suffix-probe density: the trailing `probe_len` bytes of the window
/// appear `>= required` times non-overlapping within `window`. Worst-case
/// cost per check is `O(window * probe_len)` (naive substring scan).
///
/// Pattern catalog, real log examples, and tuning rationale live in
/// `docs/pipette-doomloop/doomloop-detection.md` (the operator-facing doc).
/// This rustdoc is the implementation spec.
#[derive(Debug, Clone)]
pub struct SuffixProbe {
    pub min_chars: usize,
    /// Scan range in bytes. Independent from the exact-repeat window so that
    /// long-period loops (period > exact_repeat.window/required) can still be
    /// caught — the probe is short and its scan cost stays linear.
    pub window: usize,
    /// Length in bytes of the probe taken from the tail.
    pub probe_len: usize,
    /// Minimum non-overlapping occurrences of the probe in `window` to fire.
    pub required: usize,
}

impl Default for SuffixProbe {
    fn default() -> Self {
        Self {
            min_chars: 8192,
            window: 16384,
            probe_len: 64,
            required: 4,
        }
    }
}

impl Detector for SuffixProbe {
    fn name(&self) -> &'static str {
        "suffix_probe"
    }

    fn validate(&self) -> Result<(), String> {
        if self.window == 0 {
            return Err("window must be > 0".into());
        }
        if self.probe_len == 0 {
            return Err("probe_len must be > 0".into());
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
        if self.window == 0 || self.probe_len == 0 || self.required < 2 {
            return false;
        }
        let tail = tail_of(content.as_bytes(), self.window);
        if tail.len() < self.probe_len {
            return false;
        }
        let probe = &tail[tail.len() - self.probe_len..];
        count_non_overlapping(tail, probe) >= self.required
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_non_overlapping_basic() {
        assert_eq!(count_non_overlapping(b"abcabcabc", b"abc"), 3);
        assert_eq!(count_non_overlapping(b"aaaaaa", b"aa"), 3);
        assert_eq!(count_non_overlapping(b"hello world", b"zz"), 0);
        assert_eq!(count_non_overlapping(b"", b"abc"), 0);
        assert_eq!(count_non_overlapping(b"abc", b""), 0);
    }

    #[test]
    fn catches_prefix_drift_chinese() {
        let d = SuffixProbe {
            min_chars: 0,
            ..SuffixProbe::default()
        };
        let body = "每个讨论的review_phase是有效的，并且每个conversation_turns的列表中的每个条目都符合schema的要求。同时，确保每个讨论的review_phase是有效的。";
        let prefixes = [
            "现在，我需要确保",
            "好的，现在需要确保",
            "好的，现在需要",
            "现在，我需要确保",
            "好的，现在需要确保",
        ];
        let mut content = String::new();
        prefixes.iter().for_each(|p| {
            content.push_str(p);
            content.push_str(body);
            content.push('\n');
        });
        assert!(d.check(&content));
    }

    #[test]
    fn catches_prefix_drift_english() {
        let d = SuffixProbe {
            min_chars: 0,
            ..SuffixProbe::default()
        };
        let body = " — and the values must be consistent with the schema requirements defined in the prompt.";
        let mut content = String::new();
        [
            "Wait, I need to check",
            "Actually, I will ensure",
            "Okay, I need to verify",
            "Now, I need to confirm",
            "Again, I will ensure",
        ]
        .iter()
        .for_each(|prefix| {
            content.push_str(prefix);
            content.push_str(body);
            content.push('\n');
        });
        assert!(d.check(&content));
    }

    #[test]
    fn catches_long_period_block_beyond_exact_repeat_reach() {
        let d = SuffixProbe {
            min_chars: 0,
            ..SuffixProbe::default()
        };
        let mut block = String::new();
        block.push_str("```json\n{\n  \"type\": \"object\",\n  \"properties\": {\n");
        (0..22).for_each(|i| {
            block.push_str(&format!(
                "    \"field_{i:02}\": {{ \"type\": \"string\", \"description\": \"Schema field {i} with filler text padding.\" }},\n",
            ));
        });
        block.push_str("  }\n}\n```\n*Wait, looking at the schema again.*\n");
        assert!(
            block.len() > 1500 && block.len() < 4096,
            "block size ({}) must exceed exact-repeat reach (4096/3 = 1365) but fit 4 copies in 16 KB",
            block.len()
        );
        let content = block.repeat(8);
        assert!(d.check(&content));
    }

    #[test]
    fn ignores_unique_tail() {
        let d = SuffixProbe {
            min_chars: 0,
            ..SuffixProbe::default()
        };
        let content = concat!(
            "The quick brown fox jumps over the lazy dog. ",
            "Pack my box with five dozen liquor jugs. ",
            "How vexingly quick daft zebras jump. ",
            "Sphinx of black quartz, judge my vow. ",
            "Mr Jock, TV quiz PhD, bags few lynx. ",
            "Amazingly few discotheques provide jukeboxes. ",
            "Grumpy wizards make toxic brew for the evil queen and jack. ",
        );
        assert!(!d.check(content));
    }
}
