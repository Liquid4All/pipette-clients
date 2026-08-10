//! Argv helpers for llama-bench / llama-server extra flags.
//!
//! Callers build tokens via the execute args builders, then use
//! [`has_flag`], [`reject_reserved_flags`], and [`canonicalize_flag_order`].
//! Reserved *lists* live in `pipette_plan_types::reserved_flags`.

/// Matches a flag token, accepting both the bare form (`--mmap`) and the
/// equals-glued form (`--mmap=0`).
pub fn has_flag(flags: &[String], name: &str) -> bool {
    flags
        .iter()
        .any(|f| f == name || f.starts_with(&format!("{name}=")))
}

/// Canonicalize argv-style flag order so the persisted record and
/// the spawned argv compare byte-identically when the underlying flag
/// set is the same. Groups adjacent tokens into `(flag, values...)`
/// clusters keyed on the `--name` (or `-x` short form) token, sorts
/// the clusters by name (case-sensitive, with `--flag=value` keyed on
/// the pre-`=` half), and flattens. The sort is **stable** —
/// repeated flags (`--mmap 0 --mmap 1`) keep their insertion order so
/// last-wins semantics are preserved.
///
/// Limitation: a value token that itself starts with `-` (e.g. a
/// negative number) would be treated as a new flag boundary. Llama
/// flags don't take negative arguments today; if that changes, switch
/// to a per-flag arity table.
pub fn canonicalize_flag_order(flags: &[String]) -> Vec<String> {
    // Open a new group on flag-shaped tokens, or for a bare value at
    // the head (no preceding flag — shouldn't happen in practice but
    // pass through verbatim so we don't lose data). Otherwise append
    // to the current group's value list.
    let mut groups = flags
        .iter()
        .fold(Vec::<Vec<String>>::new(), |mut groups, token| {
            let opens_group = token.starts_with('-') || groups.is_empty();
            if opens_group {
                groups.push(vec![token.clone()]);
            } else if let Some(g) = groups.last_mut() {
                g.push(token.clone());
            }
            groups
        });
    groups.sort_by(|a, b| flag_key(&a[0]).cmp(flag_key(&b[0])));
    groups.into_iter().flatten().collect()
}

/// Sort key for [`canonicalize_flag_order`]: the part of a flag
/// token before any `=`, so `--ctx-size` and `--ctx-size=8192` sort
/// together.
fn flag_key(token: &str) -> &str {
    token.split_once('=').map(|(name, _)| name).unwrap_or(token)
}

/// Refuse overrides of any flag the benchmark sets internally.
pub fn reject_reserved_flags(
    extra_flags: &[String],
    reserved: &[&str],
    benchmark_label: &str,
) -> anyhow::Result<()> {
    if let Some(name) = reserved.iter().find(|name| has_flag(extra_flags, name)) {
        log::warn!(
            "{benchmark_label}: refusing user-supplied {name} in --runtime-flags; \
             this flag is fixed by the benchmark for cross-run comparability"
        );
        anyhow::bail!(
            "{benchmark_label} benchmark does not accept {name} via --runtime-flags; \
             this flag is fixed by the benchmark; remove the override and re-run"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;

    const RESERVED: &[&str] = &["--ctx-size", "-c", "--mmap"];

    #[rstest]
    #[case::unrelated_passes(&["--n-gpu-layers", "0"], None)]
    #[case::long_form(&["--ctx-size", "8192"], Some("--ctx-size"))]
    #[case::equals_form(&["--ctx-size=8192"], Some("--ctx-size"))]
    #[case::short_form(&["-c", "4096"], Some("-c"))]
    #[case::mmap(&["--mmap", "1"], Some("--mmap"))]
    fn reject_reserved_flags_cases(
        #[case] input: &[&str],
        #[case] expected_marker: Option<&str>,
    ) -> anyhow::Result<()> {
        let owned: Vec<String> = input.iter().map(|s| s.to_string()).collect();
        let result = reject_reserved_flags(&owned, RESERVED, "my_bench");
        match expected_marker {
            None => assert!(result.is_ok(), "expected Ok, got {result:?}"),
            Some(needle) => {
                let err = result.err().context("expected Err")?.to_string();
                assert!(err.contains(needle), "missing {needle:?} in {err}");
                assert!(err.contains("my_bench"), "missing label in {err}");
            }
        }
        Ok(())
    }

    #[rstest]
    #[case::empty(&[], &[])]
    #[case::single_bare(&["--no-warmup"], &["--no-warmup"])]
    #[case::single_with_value(&["--threads", "8"], &["--threads", "8"])]
    #[case::sorts_two_pairs(
        &["--threads", "8", "--mmap", "0"],
        &["--mmap", "0", "--threads", "8"],
    )]
    #[case::mixes_bare_and_valued(
        &["--no-warmup", "--ctx-size", "8448", "--no-mmap"],
        &["--ctx-size", "8448", "--no-mmap", "--no-warmup"],
    )]
    #[case::equals_form_groups_with_long_form(
        &["--ctx-size=8192", "--mmap", "0"],
        &["--ctx-size=8192", "--mmap", "0"],
    )]
    #[case::short_form_sorts_after_long_double_dash(
        &["-c", "4096", "--no-warmup"],
        &["--no-warmup", "-c", "4096"],
    )]
    #[case::repeated_flag_keeps_insertion_order(
        &["--mmap", "0", "--threads", "4", "--mmap", "1"],
        &["--mmap", "0", "--mmap", "1", "--threads", "4"],
    )]
    #[case::gnu_double_dash_sorts_as_flag(
        // `--` is the GNU "end of options" sentinel; we treat it as
        // a flag-shaped token (it starts with `-`). Sort puts it
        // ahead of `--mmap` because `--` < `--m...` byte-for-byte.
        // Llamacpp doesn't honor `--`; this case pins behavior so a
        // future reader knows what to expect.
        &["--mmap", "0", "--", "foo"],
        &["--", "foo", "--mmap", "0"],
    )]
    fn canonicalize_flag_order_cases(#[case] input: &[&str], #[case] expected: &[&str]) {
        let input: Vec<String> = input.iter().map(|s| s.to_string()).collect();
        let expected: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
        assert_eq!(canonicalize_flag_order(&input), expected);
    }
}
