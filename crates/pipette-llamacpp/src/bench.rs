use std::process::{Command, Stdio};

use anyhow::Context;
use serde::Deserialize;
use serde_json::Value;

use pipette_ops::measurement;
use pipette_ops::readiness::{ReadinessGate, RepObserver};
use pipette_plan_types::{LlamacppFlashAttention, RuntimeFlagRef, RuntimeFlags};
use pipette_subprocess::{argv, echo_info};

use crate::common::{deadline_error_message, spawn_timeout_killer, LLAMA_BENCH_TIMING_TIMEOUT};
use crate::flags::{canonicalize_flag_order, has_flag, reject_reserved_flags};

/// One row of `llama-bench` JSON stdout.
///
/// Extra numeric fields are part of the upstream JSON shape and kept for
/// round-trip fidelity even when a given path only reads `avg_ns` / dims.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct LlamaBenchRow {
    pub avg_ns: f64,
    pub avg_ts: f64,
    pub stddev_ns: f64,
    pub stddev_ts: f64,
    pub n_prompt: u32,
    pub n_gen: u32,
    pub n_depth: u32,
}

/// llama-bench extra argv for a bench cell's resolved flags — the plan's entry
/// with the benchmark's defaults already overlaid
/// ([`crate::runtime_flags::for_bench`]). Reads the flat [`RuntimeFlagRef`]
/// form rather than per-variant patterns, so the bench cells share one renderer.
pub fn args_for(flags: &RuntimeFlags) -> LlamaBenchArgsBuilder {
    let r = RuntimeFlagRef::from(flags.clone());
    LlamaBenchArgsBuilder::new()
        .threads(r.threads)
        .gpu_layers(r.number_gpu_layers)
        .flash_attention(r.flash_attention)
        .mmap(r.mmap)
        .raw(&r.raw)
}

/// Builds llama-bench extra argv from common knobs; finalizes on [`Self::build`].
///
/// Reached through [`args_for`]; `build(reserved, label)` then adds the flags
/// the benchmark fixes for every run.
pub struct LlamaBenchArgsBuilder {
    tokens: Vec<String>,
}

impl LlamaBenchArgsBuilder {
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    pub fn threads(mut self, threads: Option<u32>) -> Self {
        if let Some(t) = threads {
            self.push_pair("-t", t.to_string());
        }
        self
    }

    pub fn gpu_layers(mut self, gpu_layers: Option<u32>) -> Self {
        if let Some(n) = gpu_layers {
            self.push_pair("-ngl", n.to_string());
        }
        self
    }

    pub fn flash_attention(mut self, flash_attention: Option<LlamacppFlashAttention>) -> Self {
        if let Some(fa) = flash_attention {
            self.push_pair("-fa", fa.as_str().to_string());
        }
        self
    }

    /// llama-bench: `--mmap 0|1` (no bare `--no-mmap`).
    pub fn mmap(mut self, mmap: Option<bool>) -> Self {
        if let Some(on) = mmap {
            self.push_pair("--mmap", if on { "1" } else { "0" }.to_string());
        }
        self
    }

    pub fn raw(mut self, raw: &[String]) -> Self {
        self.tokens.extend(raw.iter().cloned());
        self
    }

    /// Reject reserved flags, add the benchmark's fixed `-r 1`, canonicalize
    /// order. Cell-level defaults (mmap) arrive typed, from `runtime_flags`.
    pub fn build(mut self, reserved_list: &[&str], label: &str) -> anyhow::Result<Vec<String>> {
        reject_reserved_flags(&self.tokens, reserved_list, label)?;
        // One outer rep per invoke; multi-rep aggregation is `execute_reps`.
        if !has_flag(&self.tokens, "-r") && !has_flag(&self.tokens, "--repetitions") {
            self.tokens.push("-r".to_string());
            self.tokens.push("1".to_string());
        }
        Ok(canonicalize_flag_order(&self.tokens))
    }

    fn push_pair(&mut self, flag: &str, val: String) {
        self.tokens.push(flag.to_string());
        self.tokens.push(val);
    }
}

struct LlamaBenchExecution {
    preview: Vec<String>,
    stdout: String,
    stderr: String,
    rows: Vec<LlamaBenchRow>,
}

fn execute(mut command: Command) -> anyhow::Result<LlamaBenchExecution> {
    let program = command.get_program().to_string_lossy().into_owned();
    let preview = argv(&command);
    echo_info(&command);
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    // Process-group leader on Unix so a deadline SIGKILL takes any
    // child llama-bench might exec (none today, but cheap insurance
    // and matches the Android max_memory_usage path). No-op on
    // Windows where TerminateProcess kills the whole subtree.
    //
    // Propagate the errno from `setpgid` rather than swallowing it.
    // If setpgid fails (rare: session-leader, race) and we then send
    // `kill(-pid, SIGKILL)`, we'd target whatever pgid happens to
    // equal the child's pid — possibly an unrelated process group.
    // Returning Err from `pre_exec` aborts the exec, which is the
    // safe failure mode.
    #[cfg(unix)]
    unsafe {
        use std::os::unix::process::CommandExt;
        command.pre_exec(|| {
            if libc::setpgid(0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }

    let child = command
        .spawn()
        .with_context(|| format!("failed to spawn {program}"))?;
    let pid = child.id();
    // Register the pgid leader with the SIGINT/SIGTERM handler. The guard
    // deregisters on Drop so every exit path (success, wait-error, the
    // timeout-killer firing) leaves the registry empty for this pid.
    let _cleanup_guard = pipette_subprocess::cleanup::Guard::for_process_group(pid);
    let killer = spawn_timeout_killer(LLAMA_BENCH_TIMING_TIMEOUT, move || kill_pid(pid));
    let output = child
        .wait_with_output()
        .with_context(|| format!("failed to wait for {program}"))?;
    let killer_fired = killer.fired();
    drop(killer);

    if killer_fired {
        anyhow::bail!(
            "{}",
            deadline_error_message(
                LLAMA_BENCH_TIMING_TIMEOUT,
                &String::from_utf8_lossy(&output.stderr)
            )
        );
    }
    if !output.status.success() {
        anyhow::bail!(
            "llama-bench failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8(output.stdout).context("llama-bench returned non-utf8 JSON")?;
    let rows = parse_rows(&stdout)?;
    Ok(LlamaBenchExecution {
        preview,
        stdout,
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        rows,
    })
}

/// One llama-bench invocation per measurement rep, with a caller-supplied
/// `before_rep` gate (typically readiness) between reps so each starts from
/// a known thermal baseline. Aggregates the per-rep `avg_ns` from the
/// selected row into mean + sample stddev — replacing llama-bench's
/// internal `-r N` aggregation (which runs the reps back-to-back and bakes
/// thermal drift into the reported numbers).
///
/// `before_rep` runs at the start of each rep. `build_cmd` returns a fresh
/// [`Command`] each rep (including framework defaults such as `-r 1` from
/// [`LlamaBenchArgsBuilder::build`]). `select` picks the rep's row from the
/// JSON output (decode benchmarks return two rows; the caller filters to the
/// one they're measuring).
///
/// Captures `preview` from the first invocation (argv is representative of
/// the whole sequence) and `stdout` / `stderr` from the last (most relevant
/// if anything went wrong at the end of the run).
pub struct BenchRepSummary {
    pub preview: Vec<String>,
    pub stdout: String,
    pub stderr: String,
    pub mean_ms: f64,
    pub stddev_ms: f64,
}

pub fn execute_reps(
    label: &str,
    readiness_gate: ReadinessGate,
    observer: &RepObserver,
    mut build_cmd: impl FnMut() -> anyhow::Result<Command>,
    select: impl Fn(&[LlamaBenchRow]) -> anyhow::Result<&LlamaBenchRow>,
) -> anyhow::Result<BenchRepSummary> {
    let measured = measurement::run(
        label,
        readiness_gate,
        observer,
        // No untimed per-rep setup: the server holds no state a rep resets.
        |_| Ok(()),
        |_| {
            let cmd = build_cmd()?;
            execute(cmd)
        },
        |_, rep| Ok(select(&rep.value.rows)?.avg_ns / 1_000_000.0),
    )?;
    let stats = measured.stats();

    // The preview is the first rep's argv — every rep runs the same command —
    // and the logs are the last rep's, the one a failure would be about.
    let preview = measured
        .first()
        .map(|execution| execution.preview.clone())
        .unwrap_or_default();
    let (stdout, stderr) = measured
        .into_iter()
        .next_back()
        .map(|rep| (rep.value.stdout, rep.value.stderr))
        .unwrap_or_default();

    Ok(BenchRepSummary {
        preview,
        stdout,
        stderr,
        mean_ms: stats.mean_ms,
        stddev_ms: stats.stddev_ms,
    })
}

/// Best-effort kill of `pid`. Process-group SIGKILL on Unix
/// (negative pid takes the whole group set up by `pre_exec`);
/// `TerminateProcess` via a freshly-opened handle on Windows.
fn kill_pid(pid: u32) {
    #[cfg(unix)]
    // SAFETY: `kill(2)` is signal-safe; targeting a non-existent
    // pgid is benign (sets errno = ESRCH).
    unsafe {
        libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
    }
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::{
            Foundation::CloseHandle,
            System::Threading::{OpenProcess, TerminateProcess, PROCESS_TERMINATE},
        };
        let h = OpenProcess(PROCESS_TERMINATE, 0, pid);
        if !h.is_null() {
            TerminateProcess(h, 1);
            CloseHandle(h);
        }
    }
}

/// Parse llama-bench's JSON output into row records. Public to the
/// per-OS measured runners under `max_memory_usage/`, which call
/// `Command::output()` (or equivalent) directly and need the same
/// row shape.
fn parse_rows(stdout: &str) -> anyhow::Result<Vec<LlamaBenchRow>> {
    let raw: Value = serde_json::from_str(stdout).context("failed to parse llama-bench JSON")?;
    let rows = raw
        .as_array()
        .cloned()
        .context("llama-bench returned a non-array JSON payload")?;
    if rows.is_empty() {
        anyhow::bail!("llama-bench returned no rows");
    }
    rows.into_iter()
        .map(|row| {
            serde_json::from_value(row).context("failed to deserialize llama-bench result row")
        })
        .collect()
}

pub fn select_row<'a>(
    rows: &'a [LlamaBenchRow],
    description: &str,
    predicate: impl Fn(&LlamaBenchRow) -> bool,
) -> anyhow::Result<&'a LlamaBenchRow> {
    let mut matches = rows.iter().filter(|row| predicate(row));
    let first = matches
        .next()
        .with_context(|| format!("llama-bench returned no row matching {description}"))?;
    if matches.next().is_some() {
        anyhow::bail!("llama-bench returned multiple rows matching {description}");
    }
    Ok(first)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn bench_flags(
        threads: Option<u32>,
        mmap: Option<bool>,
        raw: &[&str],
    ) -> pipette_plan_types::RuntimeFlags {
        pipette_plan_types::RuntimeFlags::PrefillLlamacppCliStockToolsGgufText {
            threads,
            number_gpu_layers: None,
            mmap,
            flash_attention: Some(LlamacppFlashAttention::On),
            raw: raw.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    /// Every typed field reaches argv in canonical order, and `build` adds the
    /// `-r 1` the benchmark fixes (cell defaults arrive already resolved).
    #[rstest]
    #[case::mmap_off(bench_flags(None, Some(false), &[]), &["--mmap", "0", "-fa", "on", "-r", "1"])]
    #[case::mmap_on(bench_flags(None, Some(true), &[]), &["--mmap", "1", "-fa", "on", "-r", "1"])]
    #[case::unset_mmap_emits_nothing(bench_flags(None, None, &[]), &["-fa", "on", "-r", "1"])]
    #[case::threads_and_raw(
        bench_flags(Some(8), Some(false), &["--poll", "0"]),
        &["--mmap", "0", "--poll", "0", "-fa", "on", "-r", "1", "-t", "8"]
    )]
    fn args_for_renders_the_typed_fields(
        #[case] flags: pipette_plan_types::RuntimeFlags,
        #[case] expected: &[&str],
    ) -> anyhow::Result<()> {
        let got = args_for(&flags).build(&[], "test")?;
        assert_eq!(got, expected);
        Ok(())
    }

    /// `-r` is reserved from `raw`, so the builder's default is its only source.
    #[test]
    fn build_adds_one_repetition() -> anyhow::Result<()> {
        let got = args_for(&bench_flags(None, None, &[])).build(&[], "test")?;
        assert_eq!(got.windows(2).filter(|w| w[0] == "-r").count(), 1);
        Ok(())
    }

    /// The repetition count is the benchmark's, not the cell's: it reaches argv
    /// but has no field to occupy in the submitted record.
    #[test]
    fn the_fixed_repetition_count_stays_out_of_the_record() -> anyhow::Result<()> {
        let flags = bench_flags(None, Some(false), &[]);
        assert!(args_for(&flags).build(&[], "test")?.contains(&"-r".into()));

        let record = flags.submission_value();
        assert!(
            !record.to_string().contains("\"-r\""),
            "reserved flag reached the record: {record}"
        );
        Ok(())
    }
}
