//! `tracing` to stderr (stdout stays parseable for `probe`/`status`), text or
//! JSON. Each phone's thread runs inside a `phone{device, client_id}` span, so
//! every line it logs names the phone.

use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

/// Text lines or one JSON object per line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, strum::Display, strum::EnumString)]
#[strum(serialize_all = "lowercase", ascii_case_insensitive)]
pub(crate) enum LogFormat {
    #[default]
    Text,
    Json,
}

pub(crate) fn init(level: LevelFilter, format: LogFormat) {
    let spec = filter_spec(level, std::env::var("RUST_LOG").ok().as_deref());
    let filter =
        EnvFilter::try_new(&spec).unwrap_or_else(|_| EnvFilter::new(filter_spec(level, None)));
    let builder = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        // Colour only on a terminal: a journal would store the escape codes.
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_target(false)
        .with_env_filter(filter);
    // Ignored if a subscriber is already set (tests, or a repeat call).
    let _ = match format {
        LogFormat::Text => builder.try_init(),
        LogFormat::Json => builder
            .json()
            .flatten_event(true)
            // Every enclosing span: `phone` and, inside a job, `job`.
            .with_current_span(false)
            .with_span_list(true)
            .try_init(),
    };
}

/// This crate at `level`, dependencies at `warn`; a set `RUST_LOG` replaces both.
fn filter_spec(level: LevelFilter, rust_log: Option<&str>) -> String {
    match rust_log.map(str::trim).filter(|spec| !spec.is_empty()) {
        Some(spec) => spec.to_owned(),
        None => format!("warn,{}={level}", env!("CARGO_CRATE_NAME")),
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use std::io::{self, Write};
    use std::sync::{Arc, Mutex, OnceLock};

    use tracing::Dispatch;
    use tracing::level_filters::LevelFilter;
    use tracing::subscriber::NoSubscriber;

    /// What `f` returns, and what it logs on this thread at `info`, as text
    /// lines like `init`'s.
    pub(crate) fn captured<T>(f: impl FnOnce() -> T) -> (T, String) {
        // With one live dispatcher, `tracing` sets a callsite's interest from
        // the default of the thread that first reaches it. A test thread with
        // none would switch that span or event off in the capture running
        // beside it; a second dispatcher makes it ask every live one.
        static SECOND: OnceLock<Dispatch> = OnceLock::new();
        SECOND.get_or_init(|| Dispatch::new(NoSubscriber::default()));
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&buffer);
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || Sink(Arc::clone(&sink)))
            .with_ansi(false)
            .with_target(false)
            .with_max_level(LevelFilter::INFO)
            .finish();
        let returned = tracing::subscriber::with_default(subscriber, f);
        let log = buffer
            .lock()
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .unwrap_or_default();
        (returned, log)
    }

    /// Each line of `log` past its timestamp: the level, then what was said.
    pub(crate) fn said(log: &str) -> Vec<&str> {
        log.lines()
            .filter_map(|line| line.split_once(' '))
            .map(|(_, said)| said.trim_start())
            .collect()
    }

    struct Sink(Arc<Mutex<Vec<u8>>>);

    impl Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .map_err(|_| io::Error::other("the log buffer is poisoned"))?
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    #[test]
    fn the_level_applies_to_this_crate_and_dependencies_stay_at_warn() {
        assert_eq!(
            filter_spec(LevelFilter::DEBUG, None),
            "warn,pipette_ios_proxy=debug"
        );
        assert_eq!(
            filter_spec(LevelFilter::INFO, Some("  ")),
            "warn,pipette_ios_proxy=info"
        );
    }

    #[test]
    fn rust_log_replaces_the_default_filter() {
        assert_eq!(
            filter_spec(LevelFilter::INFO, Some("reqwest=debug")),
            "reqwest=debug"
        );
    }

    #[test]
    fn the_format_parses_from_its_own_rendering() {
        [LogFormat::Text, LogFormat::Json]
            .into_iter()
            .for_each(|format| assert_eq!(LogFormat::from_str(&format.to_string()), Ok(format)));
        assert_eq!(LogFormat::from_str("JSON"), Ok(LogFormat::Json));
        assert!(LogFormat::from_str("yaml").is_err());
    }
}
