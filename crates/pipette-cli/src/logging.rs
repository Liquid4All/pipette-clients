//! `tracing` to stderr (stdout stays parseable for `results show`, `auth
//! export` and every `list`), text or JSON. `log` records from dependencies are
//! bridged into the same stream.
//!
//! Free text (a prompt, a command's output) goes in as a `&str` field, so a text
//! line quotes and escapes it; names, paths, URLs, command lines and errors go
//! in with `%`, unquoted, so an error quoting a child's output keeps its lines
//! on a terminal. Messages are constant, with values in fields so JSON carries
//! each under its own key; the one exception is a child's own line, forwarded by
//! `pipette_subprocess::log_line`.

use std::io::IsTerminal;

use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

use crate::progress::WaitLogWriter;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum LogFormat {
    #[default]
    Text,
    Json,
}

/// Install the subscriber. Once per process; a second call is ignored.
pub fn init(level: LevelFilter, format: LogFormat) {
    let spec = filter_spec(level, std::env::var("RUST_LOG").ok().as_deref());
    let filter =
        EnvFilter::try_new(&spec).unwrap_or_else(|_| EnvFilter::new(filter_spec(level, None)));
    let tty = std::io::stderr().is_terminal();
    let builder = tracing_subscriber::fmt()
        // A closed stderr stops a worker that asked for it.
        .with_writer(move || {
            WaitLogWriter::new(
                pipette_subprocess::log_stderr(),
                &mut std::io::stderr(),
                tty,
            )
        })
        // The subscriber reports a failed write on the same stderr. With
        // `SIGPIPE` ignored, that report panics.
        .log_internal_errors(false)
        // Colour only on a terminal: a journal would store the escape codes.
        .with_ansi(tty)
        .with_env_filter(filter);
    // Ignored if a subscriber is already set (tests, or a repeat call).
    let _ = match format {
        LogFormat::Text => builder.try_init(),
        LogFormat::Json => builder
            .json()
            .flatten_event(true)
            // The enclosing `job` span, when there is one.
            .with_current_span(false)
            .with_span_list(true)
            .try_init(),
    };
}

/// Every `pipette*` crate at `level`, dependencies at `warn`; a set `RUST_LOG`
/// replaces both. The prefix match is deliberate: the runtime and store crates
/// narrate a run as much as the CLI does, and `-v` has to reach them.
fn filter_spec(level: LevelFilter, rust_log: Option<&str>) -> String {
    match rust_log.map(str::trim).filter(|spec| !spec.is_empty()) {
        Some(spec) => spec.to_owned(),
        None => format!("warn,pipette={level}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_level_applies_to_every_pipette_crate_and_dependencies_stay_at_warn() {
        assert_eq!(filter_spec(LevelFilter::DEBUG, None), "warn,pipette=debug");
        assert_eq!(
            filter_spec(LevelFilter::INFO, Some("  ")),
            "warn,pipette=info"
        );
    }

    #[test]
    fn rust_log_replaces_the_default_filter() {
        assert_eq!(
            filter_spec(LevelFilter::INFO, Some("reqwest=debug")),
            "reqwest=debug"
        );
    }
}
