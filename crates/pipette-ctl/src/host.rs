//! What this machine calls itself, for the labels a human reads later.
//!
//! Both values are best-effort. A host that will not name its user or itself
//! is not a reason to fail a command, so each answers `None` and the caller
//! decides what to put in its place.

/// The operator running this process, as the shell exported it.
pub(crate) fn current_user() -> Option<String> {
    env_value("USER").or_else(|| env_value("USERNAME"))
}

/// This machine's name.
///
/// Asked of the host before the environment: `HOSTNAME` is a bash-ism that
/// fish, cron and most service managers never export, so the command is the
/// value that is actually there. `COMPUTERNAME` is the Windows spelling and is
/// set by the system.
pub(crate) fn hostname() -> Option<String> {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| non_empty(&String::from_utf8_lossy(&output.stdout)))
        .or_else(|| env_value("HOSTNAME"))
        .or_else(|| env_value("COMPUTERNAME"))
}

fn env_value(key: &str) -> Option<String> {
    std::env::var(key).ok().as_deref().and_then(non_empty)
}

fn non_empty(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_value_that_is_present_is_trimmed_and_not_blank() {
        [current_user(), hostname()]
            .into_iter()
            .flatten()
            .for_each(|value| {
                assert_eq!(value.trim(), value);
                assert!(!value.is_empty());
            });
    }

    #[test]
    fn a_blank_value_reads_as_absent() {
        assert_eq!(non_empty("  \n"), None);
        assert_eq!(non_empty(" box \n"), Some("box".to_string()));
    }
}
