//! `SourcedBenchmarkId`: a catalog address, `local/<id>` or `remote/<id>`.

use std::fmt;

use thiserror::Error;

use pipette_plan_types::{BenchmarkId, BenchmarkIdError, BenchmarkSource};

/// A reference that names no benchmark. The offending reference is echoed whole,
/// escaped.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SourcedBenchmarkIdError {
    /// An unknown catalog side or a nested address (`local/remote/foo`), where
    /// the accepted forms are the useful part of the message.
    #[error(
        "\"{}\" is not a benchmark reference: expected `<id>`, `local/<id>` or `remote/<id>`",
        .0.escape_default()
    )]
    Shape(String),
    /// A well-shaped reference whose id the catalog grammar refuses.
    #[error("\"{}\" is not a benchmark reference: {reason}", .reference.escape_default())]
    Id {
        reference: String,
        reason: BenchmarkIdError,
    },
}

/// A location-qualified benchmark id (`local/<id>` or `remote/<id>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourcedBenchmarkId {
    /// The on-disk local catalog.
    Local(BenchmarkId),
    /// The synced server catalog.
    Remote(BenchmarkId),
}

impl SourcedBenchmarkId {
    pub fn new(source: BenchmarkSource, id: BenchmarkId) -> Self {
        match source {
            BenchmarkSource::Local => Self::Local(id),
            BenchmarkSource::Remote => Self::Remote(id),
        }
    }

    pub fn id(&self) -> &BenchmarkId {
        match self {
            Self::Local(id) | Self::Remote(id) => id,
        }
    }

    pub fn source(&self) -> BenchmarkSource {
        match self {
            Self::Local(_) => BenchmarkSource::Local,
            Self::Remote(_) => BenchmarkSource::Remote,
        }
    }
}

impl std::str::FromStr for SourcedBenchmarkId {
    type Err = SourcedBenchmarkIdError;

    /// A bare id means the synced catalog: that is the form plans and claims
    /// carry. `local/` is the opt-in for a definition only this machine has.
    ///
    /// clap takes `--benchmark` as this type, so a bad reference is a usage
    /// error rather than a failure inside `execute`.
    fn from_str(reference: &str) -> Result<Self, Self::Err> {
        let shape = || SourcedBenchmarkIdError::Shape(reference.to_string());
        let (source, raw) = match reference.split_once('/') {
            Some(("local", raw)) => (BenchmarkSource::Local, raw),
            Some(("remote", raw)) => (BenchmarkSource::Remote, raw),
            Some(_) => return Err(shape()),
            None => (BenchmarkSource::Remote, reference),
        };
        if raw.contains('/') {
            return Err(shape());
        }
        let id = BenchmarkId::try_new(raw.to_string()).map_err(|reason| {
            SourcedBenchmarkIdError::Id {
                reference: reference.to_string(),
                reason,
            }
        })?;
        Ok(Self::new(source, id))
    }
}

impl fmt::Display for SourcedBenchmarkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Local(id) => write!(f, "local/{id}"),
            Self::Remote(id) => write!(f, "remote/{id}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn bid(s: &str) -> anyhow::Result<BenchmarkId> {
        Ok(BenchmarkId::try_new(s.to_string())?)
    }

    #[rstest]
    #[case::bare("foo", BenchmarkSource::Remote)]
    #[case::local("local/foo", BenchmarkSource::Local)]
    #[case::remote("remote/foo", BenchmarkSource::Remote)]
    fn parse_accepts_bare_ids_and_explicit_sides(
        #[case] reference: &str,
        #[case] source: BenchmarkSource,
    ) -> anyhow::Result<()> {
        assert_eq!(
            reference.parse(),
            Ok(SourcedBenchmarkId::new(source, bid("foo")?))
        );
        Ok(())
    }

    #[rstest]
    #[case::nested_address("local/remote/foo", None)]
    #[case::unknown_side("elsewhere/foo", None)]
    #[case::empty_after_local("local/", Some(BenchmarkIdError::Empty))]
    #[case::empty_after_remote("remote/", Some(BenchmarkIdError::Empty))]
    #[case::empty("", Some(BenchmarkIdError::Empty))]
    #[case::whitespace("foo bar", Some(BenchmarkIdError::InvalidChar(' ')))]
    #[case::colon("local/foo:bar", Some(BenchmarkIdError::InvalidChar(':')))]
    #[case::leading_dot("remote/.hidden", Some(BenchmarkIdError::InvalidFirst('.')))]
    fn a_refusal_says_whether_the_shape_or_the_id_is_wrong(
        #[case] reference: &str,
        #[case] reason: Option<BenchmarkIdError>,
    ) {
        let expected = match reason {
            None => SourcedBenchmarkIdError::Shape(reference.to_string()),
            Some(reason) => SourcedBenchmarkIdError::Id {
                reference: reference.to_string(),
                reason,
            },
        };
        assert_eq!(reference.parse::<SourcedBenchmarkId>(), Err(expected));
    }

    /// The reference is escaped so an invisible or look-alike character shows.
    #[rstest]
    #[case::shape(
        "elsewhere/a\nb\u{200B}\u{430}",
        r#""elsewhere/a\nb\u{200b}\u{430}" is not a benchmark reference: expected `<id>`, `local/<id>` or `remote/<id>`"#
    )]
    #[case::id(
        "local/a\nb\u{200B}\u{430}",
        r#""local/a\nb\u{200b}\u{430}" is not a benchmark reference: benchmark id contains '\n'; allowed: [A-Za-z0-9][A-Za-z0-9_.-]*"#
    )]
    fn a_refusal_escapes_the_reference(#[case] reference: &str, #[case] expected: &str) {
        assert_eq!(
            reference
                .parse::<SourcedBenchmarkId>()
                .err()
                .map(|e| e.to_string())
                .as_deref(),
            Some(expected)
        );
    }

    #[test]
    fn display_round_trips_through_parse() -> anyhow::Result<()> {
        for reference in [
            SourcedBenchmarkId::Local(bid("foo")?),
            SourcedBenchmarkId::Remote(bid("foo")?),
        ] {
            let wire = reference.to_string();
            assert_eq!(wire.parse(), Ok(reference));
        }
        Ok(())
    }

    #[test]
    fn new_matches_source() -> anyhow::Result<()> {
        let id = bid("x")?;
        assert_eq!(
            SourcedBenchmarkId::new(BenchmarkSource::Local, id.clone()),
            SourcedBenchmarkId::Local(id.clone())
        );
        assert_eq!(
            SourcedBenchmarkId::new(BenchmarkSource::Remote, id.clone()),
            SourcedBenchmarkId::Remote(id)
        );
        Ok(())
    }

    #[test]
    fn source_matches_variant() -> anyhow::Result<()> {
        assert!(matches!(
            SourcedBenchmarkId::Local(bid("x")?).source(),
            BenchmarkSource::Local
        ));
        assert!(matches!(
            SourcedBenchmarkId::Remote(bid("x")?).source(),
            BenchmarkSource::Remote
        ));
        Ok(())
    }
}
