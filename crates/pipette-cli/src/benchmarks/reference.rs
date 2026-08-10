//! `SourcedBenchmarkId` — location-qualified catalog address: `local/<id>` or `remote/<id>`.

use std::fmt;

use thiserror::Error;

use pipette_plan_types::BenchmarkId;

use pipette_plan_types::benchmark::BenchmarkSource;

/// A reference that names no benchmark: an unknown catalog side, or an id that
/// is empty, holds whitespace, or is itself addressed (`local/remote/foo`).
///
/// One case rather than one per prefix — the accepted forms are the useful part
/// of the message, and the offending reference is echoed whole.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("`{0}` is not a benchmark reference: expected `<id>`, `local/<id>` or `remote/<id>`")]
pub struct SourcedBenchmarkIdError(String);

/// A location-qualified benchmark id (`local/<id>` or `remote/<id>`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourcedBenchmarkId {
    /// `local/<id>` — the on-disk local catalog.
    Local(BenchmarkId),
    /// `remote/<id>` — the synced server catalog.
    Remote(BenchmarkId),
}

impl SourcedBenchmarkId {
    /// Build from catalog source + validated id.
    pub fn new(source: BenchmarkSource, id: BenchmarkId) -> Self {
        match source {
            BenchmarkSource::Local => Self::Local(id),
            BenchmarkSource::Remote => Self::Remote(id),
        }
    }

    /// The id without the location prefix.
    pub fn id(&self) -> &BenchmarkId {
        match self {
            Self::Local(id) | Self::Remote(id) => id,
        }
    }

    /// Catalog side implied by this address.
    pub fn source(&self) -> BenchmarkSource {
        match self {
            Self::Local(_) => BenchmarkSource::Local,
            Self::Remote(_) => BenchmarkSource::Remote,
        }
    }
}

impl std::str::FromStr for SourcedBenchmarkId {
    type Err = SourcedBenchmarkIdError;

    /// Parse a bare id, `local/<id>`, or `remote/<id>`.
    ///
    /// A bare id means the synced catalog: that is the form plans and claims
    /// carry, so the distributed case needs no prefix. `local/` stays as the
    /// explicit opt-in for a definition only this machine has.
    ///
    /// The one entry point, so clap takes `--benchmark` as this type and a bad
    /// reference is a usage error rather than a failure inside `execute`.
    fn from_str(reference: &str) -> Result<Self, Self::Err> {
        let invalid = || SourcedBenchmarkIdError(reference.to_string());
        let (source, raw) = match reference.split_once('/') {
            Some(("local", raw)) => (BenchmarkSource::Local, raw),
            Some(("remote", raw)) => (BenchmarkSource::Remote, raw),
            // `BenchmarkId` rejects a `/`, so a nested address or an unknown
            // side lands here rather than becoming part of the id.
            Some(_) => return Err(invalid()),
            None => (BenchmarkSource::Remote, reference),
        };
        let id = BenchmarkId::try_new(raw.to_string()).map_err(|_| invalid())?;
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
#[allow(clippy::expect_used)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn bid(s: &str) -> BenchmarkId {
        BenchmarkId::try_new(s.to_string()).expect("test id")
    }

    /// A bare id is the distributed form and means the synced catalog; a prefix
    /// is an explicit override; anything else names no benchmark.
    #[rstest]
    #[case::bare("foo", Some(SourcedBenchmarkId::Remote(bid("foo"))))]
    #[case::local("local/foo", Some(SourcedBenchmarkId::Local(bid("foo"))))]
    #[case::remote("remote/foo", Some(SourcedBenchmarkId::Remote(bid("foo"))))]
    #[case::nested_address("local/remote/foo", None)]
    #[case::unknown_side("elsewhere/foo", None)]
    #[case::empty_after_local("local/", None)]
    #[case::empty_after_remote("remote/", None)]
    #[case::empty("", None)]
    #[case::whitespace("foo bar", None)]
    fn parse_accepts_bare_ids_and_explicit_sides(
        #[case] reference: &str,
        #[case] expected: Option<SourcedBenchmarkId>,
    ) {
        match expected {
            Some(parsed) => assert_eq!(reference.parse(), Ok(parsed)),
            None => assert_eq!(
                reference.parse::<SourcedBenchmarkId>(),
                Err(SourcedBenchmarkIdError(reference.to_string()))
            ),
        }
    }

    #[test]
    fn display_round_trips_through_parse() {
        for reference in [
            SourcedBenchmarkId::Local(bid("foo")),
            SourcedBenchmarkId::Remote(bid("foo")),
        ] {
            let wire = reference.to_string();
            assert_eq!(wire.parse(), Ok(reference));
        }
    }

    #[test]
    fn new_matches_source() {
        let id = bid("x");
        assert_eq!(
            SourcedBenchmarkId::new(BenchmarkSource::Local, id.clone()),
            SourcedBenchmarkId::Local(id.clone())
        );
        assert_eq!(
            SourcedBenchmarkId::new(BenchmarkSource::Remote, id.clone()),
            SourcedBenchmarkId::Remote(id)
        );
    }

    #[test]
    fn source_matches_variant() {
        assert!(matches!(
            SourcedBenchmarkId::Local(bid("x")).source(),
            BenchmarkSource::Local
        ));
        assert!(matches!(
            SourcedBenchmarkId::Remote(bid("x")).source(),
            BenchmarkSource::Remote
        ));
    }
}
