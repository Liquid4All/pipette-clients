//! Shared natural-language passage used by every benchmark client to
//! build synthetic prompts of an exact token count.
//!
//! Each runtime feeds a tokenizer callback into [`build_prompt_text`],
//! which returns a string that tokenizes to exactly `target_tokens`.
//! The string is sent to the inference API as a prompt, so server-side
//! tokenization remains inside the timed window for end-to-end latency.

/// The seed passage. Embedded at compile time so binaries are
/// self-contained and benchmark runs reproduce regardless of the
/// surrounding filesystem.
pub const PROMPT_SEED_TEXT: &str = include_str!("prompt_seed.txt");

/// Cap on `count` callback invocations per build. The expected shape is
/// one calibration call, a handful of pool-extension calls, logarithmic
/// bisection, and a short tail-growth pass.
const TOKENIZE_BUDGET: u32 = 64;
const PREFIX_SCAN_WINDOW_CHARS: usize = 32;

/// Candidate suffixes for tail growth. Each candidate is re-tokenized;
/// the builder only accepts one that moves the count toward the target.
const TAIL_EXTENSIONS: &[&str] = &[
    " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ",", ".",
];

/// Build a string that tokenizes to exactly `target_tokens` under the
/// caller's `count` callback. The caller's tokenize settings must match
/// the inference path's settings for string prompts.
pub fn build_prompt_text<F>(target_tokens: u32, count: F) -> anyhow::Result<String>
where
    F: FnMut(&str) -> anyhow::Result<usize>,
{
    build_prompt_text_from(PROMPT_SEED_TEXT, target_tokens, count)
}

fn build_prompt_text_from<F>(seed: &str, target_tokens: u32, mut count: F) -> anyhow::Result<String>
where
    F: FnMut(&str) -> anyhow::Result<usize>,
{
    if target_tokens == 0 {
        return Ok(String::new());
    }
    if seed.is_empty() {
        anyhow::bail!("seed text is empty");
    }
    let target = target_tokens as usize;
    let mut budget = Budget::new(TOKENIZE_BUDGET);

    let seed_count = budget.count(&mut count, seed)?;
    if seed_count == 0 {
        anyhow::bail!("seed text tokenizes to zero tokens");
    }
    let chars_per_token = seed.len() as f64 / seed_count as f64;

    let initial_chars = ((target as f64) * chars_per_token * 1.20) as usize;
    let repeats = (initial_chars / seed.len()) + 2;
    let mut pool = seed.repeat(repeats);

    let mut full_count = budget.count(&mut count, &pool)?;
    while full_count < target {
        pool.push_str(seed);
        full_count = budget.count(&mut count, &pool)?;
    }
    if full_count == target {
        return Ok(pool);
    }

    if target <= PREFIX_SCAN_WINDOW_CHARS {
        if let PrefixScan::Done(text) =
            scan_prefix_window(&pool, 0, target, &mut budget, &mut count)?
        {
            return Ok(text);
        }
    }

    let mut lo = 0usize;
    let mut hi = pool.len();
    while hi - lo > 1 {
        let mid = clamp_char_boundary(&pool, lo + (hi - lo) / 2);
        if mid == lo || mid == hi {
            break;
        }
        let count_at_mid = budget.count(&mut count, &pool[..mid])?;
        match count_at_mid.cmp(&target) {
            std::cmp::Ordering::Less => lo = mid,
            std::cmp::Ordering::Equal => return Ok(pool[..mid].to_string()),
            std::cmp::Ordering::Greater => hi = mid,
        }
    }

    let PrefixCandidate {
        mut text,
        count: mut current,
    } = match scan_prefix_window(&pool, lo, target, &mut budget, &mut count)? {
        PrefixScan::Done(text) => return Ok(text),
        PrefixScan::Under(candidate) => candidate,
    };

    while current < target {
        match tail_step(&text, current, target, &mut budget, &mut count)? {
            Some(TailStep::Done(candidate)) => return Ok(candidate),
            Some(TailStep::Progress {
                text: candidate,
                count,
            }) => {
                text = candidate;
                current = count;
            }
            None => anyhow::bail!(
                "tail-grow stuck at {current} tokens, target {target}: \
                 no extension advanced without overshooting"
            ),
        }
    }

    anyhow::bail!(
        "builder ended at {current} tokens, target was {target}: \
         bisection invariant broke"
    )
}

fn clamp_char_boundary(s: &str, mut pos: usize) -> usize {
    if pos >= s.len() {
        return s.len();
    }
    while pos > 0 && !s.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

struct PrefixCandidate {
    text: String,
    count: usize,
}

enum PrefixScan {
    Done(String),
    Under(PrefixCandidate),
}

fn scan_prefix_window<F>(
    pool: &str,
    start: usize,
    target: usize,
    budget: &mut Budget,
    count: &mut F,
) -> anyhow::Result<PrefixScan>
where
    F: FnMut(&str) -> anyhow::Result<usize>,
{
    let mut pos = start.min(pool.len());
    let mut best: Option<PrefixCandidate> = None;
    let mut steps = 0usize;
    loop {
        let text = &pool[..pos];
        let token_count = budget.count(count, text)?;
        if token_count == target {
            return Ok(PrefixScan::Done(text.to_string()));
        }
        if token_count < target {
            let should_replace = match &best {
                None => true,
                Some(best) => {
                    token_count > best.count
                        || (token_count == best.count && text.len() > best.text.len())
                }
            };
            if should_replace {
                best = Some(PrefixCandidate {
                    text: text.to_string(),
                    count: token_count,
                });
            }
        }

        if steps >= PREFIX_SCAN_WINDOW_CHARS || pos >= pool.len() {
            break;
        }
        pos = next_char_boundary(pool, pos);
        steps += 1;
    }

    match best {
        Some(best) => Ok(PrefixScan::Under(best)),
        None => anyhow::bail!("no prefix below target near bisection boundary"),
    }
}

fn next_char_boundary(s: &str, pos: usize) -> usize {
    let mut next = (pos + 1).min(s.len());
    while next < s.len() && !s.is_char_boundary(next) {
        next += 1;
    }
    next
}

enum TailStep {
    Done(String),
    Progress { text: String, count: usize },
}

fn tail_step<F>(
    text: &str,
    current: usize,
    target: usize,
    budget: &mut Budget,
    count: &mut F,
) -> anyhow::Result<Option<TailStep>>
where
    F: FnMut(&str) -> anyhow::Result<usize>,
{
    TAIL_EXTENSIONS
        .iter()
        .copied()
        .find_map(|ext| {
            let candidate = format!("{text}{ext}");
            let candidate_count = match budget.count(count, &candidate) {
                Ok(count) => count,
                Err(err) => return Some(Err(err)),
            };
            if candidate_count == target {
                Some(Ok(TailStep::Done(candidate)))
            } else if candidate_count > current && candidate_count < target {
                Some(Ok(TailStep::Progress {
                    text: candidate,
                    count: candidate_count,
                }))
            } else {
                None
            }
        })
        .transpose()
}

struct Budget(u32);

impl Budget {
    fn new(limit: u32) -> Self {
        Self(limit)
    }

    fn count<F>(&mut self, count: &mut F, text: &str) -> anyhow::Result<usize>
    where
        F: FnMut(&str) -> anyhow::Result<usize>,
    {
        if self.0 == 0 {
            anyhow::bail!("prompt builder exceeded tokenize budget of {TOKENIZE_BUDGET} calls");
        }
        self.0 -= 1;
        count(text)
    }
}

/// Repeat `seed` until the output reaches exactly `target_len` tokens,
/// truncating the final repetition as needed.
pub fn repeat_token_sequence(seed: &[u32], target_len: usize) -> anyhow::Result<Vec<u32>> {
    if target_len == 0 {
        return Ok(Vec::new());
    }
    if seed.is_empty() {
        anyhow::bail!("seed token list must not be empty");
    }
    Ok((0..target_len).map(|idx| seed[idx % seed.len()]).collect())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn seed_text_is_substantial_natural_language() {
        assert!(!PROMPT_SEED_TEXT.is_empty());
        assert!(
            PROMPT_SEED_TEXT.len() > 20_000,
            "seed shrank below the 4096-token coverage threshold ({} chars)",
            PROMPT_SEED_TEXT.len()
        );
    }

    #[rstest]
    #[case(&[1, 2, 3], 7, vec![1, 2, 3, 1, 2, 3, 1])]
    #[case(&[1, 2], 0, vec![])]
    #[case(&[1, 2, 3, 4, 5], 3, vec![1, 2, 3])]
    fn repeat_token_sequence_cases(
        #[case] seed: &[u32],
        #[case] target_len: usize,
        #[case] expected: Vec<u32>,
    ) -> anyhow::Result<()> {
        assert_eq!(repeat_token_sequence(seed, target_len)?, expected);
        Ok(())
    }

    fn count_bytes(calls: &std::cell::Cell<u32>) -> impl FnMut(&str) -> anyhow::Result<usize> + '_ {
        move |text: &str| {
            calls.set(calls.get() + 1);
            Ok(text.len())
        }
    }

    #[test]
    fn target_zero_returns_empty_without_tokenize() -> anyhow::Result<()> {
        let calls = std::cell::Cell::new(0);
        let text = build_prompt_text_from("seed", 0, count_bytes(&calls))?;
        assert_eq!(text, "");
        assert_eq!(calls.get(), 0);
        Ok(())
    }

    #[rstest]
    #[case(1)]
    #[case(5)]
    #[case(17)]
    #[case(128)]
    #[case(1024)]
    #[case(4096)]
    #[case(8192)]
    fn hits_target_on_byte_tokenizer_across_sizes(#[case] target: u32) -> anyhow::Result<()> {
        let calls = std::cell::Cell::new(0);
        let text =
            build_prompt_text_from("abcdefghijklmnopqrstuvwxyz", target, count_bytes(&calls))?;
        assert_eq!(text.len(), target as usize);
        assert!(
            calls.get() < 32,
            "byte tokenizer should converge in <32 calls, got {} for target {target}",
            calls.get()
        );
        Ok(())
    }

    #[rstest]
    #[case(1)]
    #[case(5)]
    #[case(7)]
    #[case(10)]
    #[case(100)]
    fn hits_target_on_lumpy_tokenizer(#[case] target: u32) -> anyhow::Result<()> {
        let calls = std::cell::Cell::new(0);
        let text = build_prompt_text_from("abcdefghi", target, |s: &str| {
            calls.set(calls.get() + 1);
            Ok(s.len() / 3)
        })?;
        assert_eq!(text.len() / 3, target as usize);
        assert!(calls.get() < TOKENIZE_BUDGET);
        Ok(())
    }

    #[test]
    fn repeat_nonzero_empty_seed_errors() -> anyhow::Result<()> {
        let Err(err) = repeat_token_sequence(&[], 1) else {
            anyhow::bail!("expected empty seed token error");
        };
        assert!(err.to_string().contains("seed token list"));
        Ok(())
    }

    #[test]
    fn finds_target_when_token_count_drops_after_an_early_exact_prefix() -> anyhow::Result<()> {
        let calls = std::cell::Cell::new(0);
        let text = build_prompt_text_from("abcdefghijkl", 2, |s: &str| {
            calls.set(calls.get() + 1);
            Ok(match s.len() {
                0 => 0,
                1 => 1,
                2 => 2,
                3..=47 => 1,
                _ => 3,
            })
        })?;
        assert_eq!(text.len(), 2);
        Ok(())
    }

    #[test]
    fn errors_on_empty_seed() -> anyhow::Result<()> {
        let calls = std::cell::Cell::new(0);
        let Err(err) = build_prompt_text_from("", 10, count_bytes(&calls)) else {
            anyhow::bail!("expected empty seed error");
        };
        assert!(err.to_string().contains("seed"));
        assert_eq!(calls.get(), 0);
        Ok(())
    }

    #[test]
    fn public_entry_uses_embedded_seed() -> anyhow::Result<()> {
        let calls = std::cell::Cell::new(0);
        let text = build_prompt_text(50, count_bytes(&calls))?;
        assert_eq!(text.len(), 50);
        assert!(PROMPT_SEED_TEXT.contains(&text[..text.len().min(20)]));
        Ok(())
    }
}
