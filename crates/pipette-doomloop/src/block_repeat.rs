use crate::{tail_of, Detector};

/// Count non-overlapping occurrences of `needle` in `hay`.
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

/// Large-block repetition: the trailing `min_period` bytes of the window
/// appear `>= required` times non-overlapping within `window`. Shares the
/// suffix-probe shape but tuned for long, byte-identical blocks — catches
/// cases like a JSON-schema block that is repeated verbatim two or three
/// times in a `<think>` block before the output settles.
///
/// Pattern catalog, real log examples, and tuning rationale live in
/// `docs/pipette-doomloop/doomloop-detection.md` (the operator-facing doc).
/// This rustdoc is the implementation spec.
#[derive(Debug, Clone)]
pub struct BlockRepeat {
    pub min_chars: usize,
    /// Scan range in bytes. Chosen to fit two copies of a ~1 KB block plus
    /// filler without relying on the 16 KB `suffix_probe` window.
    pub window: usize,
    /// Length in bytes of the trailing probe block. Long enough that two
    /// byte-identical occurrences in normal prose are implausible.
    pub min_period: usize,
    /// Minimum non-overlapping probe occurrences in `window` to fire.
    pub required: usize,
}

impl Default for BlockRepeat {
    fn default() -> Self {
        Self {
            min_chars: 8192,
            window: 8192,
            min_period: 512,
            required: 2,
        }
    }
}

impl Detector for BlockRepeat {
    fn name(&self) -> &'static str {
        "block_repeat"
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
        if self.window == 0 || self.min_period == 0 || self.required < 2 {
            return false;
        }
        let tail = tail_of(content.as_bytes(), self.window);
        if tail.len() < self.min_period {
            return false;
        }
        let probe = &tail[tail.len() - self.min_period..];
        count_non_overlapping(tail, probe) >= self.required
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_non_overlapping_basic() {
        assert_eq!(count_non_overlapping(b"abcabcabc", b"abc"), 3);
        assert_eq!(count_non_overlapping(b"aaaa", b"aa"), 2);
        assert_eq!(count_non_overlapping(b"", b"abc"), 0);
        assert_eq!(count_non_overlapping(b"abc", b""), 0);
    }

    /// Motivating log pattern: a ~1 KB JSON-schema block written out
    /// verbatim twice inside a `<think>` block, separated by drifting
    /// commentary. Fires at the second copy — no byte-identical third
    /// copy required, which is what distinguishes this from
    /// `exact_repeat` and `suffix_probe`.
    #[test]
    fn catches_large_json_schema_block_at_second_copy() {
        let d = BlockRepeat {
            min_chars: 0,
            ..BlockRepeat::default()
        };
        let mut block = String::new();
        block.push_str("```json\n{\n  \"type\": \"object\",\n  \"properties\": {\n");
        (0..22).for_each(|i| {
            block.push_str(&format!(
                "    \"field_{i:02}\": {{ \"type\": \"string\", \"description\": \"Schema field {i} with filler text padding.\" }},\n",
            ));
        });
        block.push_str("  }\n}\n```\n");
        assert!(
            block.len() >= 512,
            "fixture block must exceed min_period ({} bytes)",
            block.len()
        );
        let gap = "\n*Wait, looking at the schema again.*\nThe schema provided in the prompt is:\n";
        let content = format!("{block}{gap}{block}");
        assert!(d.check(&content));
    }

    #[test]
    fn ignores_single_copy() {
        let d = BlockRepeat {
            min_chars: 0,
            ..BlockRepeat::default()
        };
        // One long byte-identical block — no second occurrence to match.
        let block: String = (0..60)
            .map(|i| format!("    \"field_{i:02}\": \"value {i} with padding bytes\",\n"))
            .collect();
        assert!(block.len() >= 1024);
        assert!(!d.check(&block));
    }

    #[test]
    fn ignores_clean_prose_under_min_period() {
        // min_chars: 0 so the short-circuit doesn't mask the min_period gate.
        let d = BlockRepeat {
            min_chars: 0,
            ..BlockRepeat::default()
        };
        let content = "The quick brown fox jumps over the lazy dog. ".repeat(4);
        assert!(content.len() < 512);
        assert!(!d.check(&content));
    }

    #[test]
    fn ignores_clean_prose_at_full_length() {
        let d = BlockRepeat {
            min_chars: 0,
            ..BlockRepeat::default()
        };
        // Long but non-repeating prose — no 512-byte tail occurs twice.
        let content: String = (0..40)
            .map(|i| {
                format!(
                    "Paragraph {i}: the narrator walked down the {place} and heard the {sound} from a {distance} before reaching the {landmark}.\n",
                    place = ["alley", "boulevard", "lane", "causeway", "quay", "viaduct", "esplanade"][i % 7],
                    sound = ["whistle", "clang", "hum", "chime", "rumble", "whirr", "echo"][i % 7],
                    distance = ["stone's throw", "long walk", "short hop", "marathon", "quick jaunt", "half league", "mile"][i % 7],
                    landmark = ["clock tower", "iron gate", "old library", "harbor wall", "market square", "willow tree", "dry fountain"][i % 7],
                )
            })
            .collect();
        assert!(content.len() > 2048);
        assert!(!d.check(&content));
    }

    /// A legitimate 12-row JSON array with per-row-distinct values — no
    /// 512-byte trailing block recurs earlier in the window. Guards
    /// against false-positives on structured tabular output.
    #[test]
    fn does_not_fire_on_json_table_with_distinct_values() {
        let d = BlockRepeat {
            min_chars: 0,
            ..BlockRepeat::default()
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
        let rows: String = (0..12)
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
        let content = format!("[\n{rows}]\n");
        // `d` uses `min_chars: 0`, so the gate can't trivially suppress
        // this check; assertion below tests detection logic directly.
        assert!(!d.check(&content));
    }

    #[test]
    fn zeroed_required_returns_false() {
        let d = BlockRepeat {
            required: 1,
            ..BlockRepeat::default()
        };
        let block = "x".repeat(600);
        let content = format!("{block}{block}");
        assert!(!d.check(&content));
    }

    #[test]
    fn zeroed_window_returns_false() {
        let d = BlockRepeat {
            window: 0,
            ..BlockRepeat::default()
        };
        let block = "x".repeat(600);
        let content = format!("{block}{block}");
        assert!(!d.check(&content));
    }
}
