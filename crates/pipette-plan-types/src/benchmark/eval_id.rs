//! `EvalId` — the eval a benchmark names, loose on the wire and strict
//! inside.
//!
//! The management server may list evals this client doesn't know yet, so
//! the parse is total: recognized ids become [`KnownEvalId`], anything
//! else is preserved verbatim.

use serde::{Deserialize, Serialize};
use strum::{AsRefStr, Display, EnumString, VariantArray};

/// The evals this client knows how to run end to end, identified by the stable
/// `parameter_eval_id`.
///
/// This mirrors `pipette-scores`' `EvalId` — the authoritative set of evals the
/// `POST /score` backend can actually score (it 404s any other id). Running an
/// eval the scorer doesn't recognize can't complete the loop, so this is the
/// client's source of truth for the supported evals.
///
/// `parameter_eval_id` stays a free-form `String` on the wire and in storage:
/// the mgmt server may list evals this client doesn't know yet, and that must
/// not break sync or listing (the same loose-in / strict-internal split as the
/// benchmark boundary). `KnownEvalId` is the strict, internal view used by the
/// code paths that decide something per eval. An unrecognized id simply has no
/// `KnownEvalId` — [`KnownEvalId::from_eval_id`] returns `None` and callers
/// apply a documented default. Variant spellings (and `FromStr`) come from the
/// strum `serialize_all = "snake_case"`, so this enum is the single source of
/// truth — no parallel string list to drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display, AsRefStr, EnumString, VariantArray)]
#[strum(serialize_all = "snake_case")]
pub enum KnownEvalId {
    Ifbench,
    Ifstruct,
    GpqaDiamond,
    // strum's snake_case inserts no `_` before digits ("Math500" -> "math500"),
    // so spell this explicitly to match the scoring backend's `math_500`.
    #[strum(serialize = "math_500")]
    Math500,
}

impl KnownEvalId {
    /// All variants, in declaration order. Auto-generated via
    /// `#[derive(VariantArray)]`; re-projected as an inherent const so callers
    /// don't need to import the `strum::VariantArray` trait.
    pub const ALL: &'static [Self] = <Self as VariantArray>::VARIANTS;

    /// Resolve a (loose) `parameter_eval_id` to a known eval, or `None` if this
    /// client doesn't recognize it.
    pub fn from_eval_id(eval_id: &str) -> Option<Self> {
        eval_id.parse().ok()
    }

    /// Client-side sampling temperature for this eval's `/completion` requests.
    ///
    /// The server sends no temperature; the client assigns one from the eval.
    /// Every eval the scorer currently supports is generative pass@k, so all
    /// sample at `0.6` — their repeated `#k` draws must differ, pass@1 is only
    /// meaningful with independent draws, and no fixed seed is sent alongside,
    /// so the repeats stay independent. The match is exhaustive on purpose:
    /// adding a variant forces a temperature decision here.
    pub fn sampling_temperature(self) -> f64 {
        match self {
            Self::Ifbench | Self::Ifstruct | Self::GpqaDiamond | Self::Math500 => 0.6,
        }
    }
}

/// An eval identifier as it appears in a strict [`super::BenchmarkDefinition::Eval`],
/// parsed from the loose `parameter_eval_id` at the upstream boundary.
///
/// Total by construction: a recognized id becomes [`EvalId::Known`]; anything
/// else is preserved verbatim as [`EvalId::Unknown`]. The mgmt server may list
/// evals this client doesn't know yet, so parsing must never fail or rewrite an
/// id — the loose → strict → loose round-trip is lossless. The wire form stays
/// a plain string (`#[serde(from/into = "String")]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum EvalId {
    Known(KnownEvalId),
    Unknown(String),
}

impl EvalId {
    /// The wire spelling: a known eval's canonical id, or the preserved string.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Known(known) => known.as_ref(),
            Self::Unknown(raw) => raw,
        }
    }

    /// Client-side sampling temperature for this eval's `/completion` requests
    /// (see [`KnownEvalId::sampling_temperature`]); an unknown eval is greedy.
    pub fn sampling_temperature(&self) -> f64 {
        match self {
            Self::Known(known) => known.sampling_temperature(),
            Self::Unknown(_) => 0.0,
        }
    }
}

impl From<String> for EvalId {
    fn from(raw: String) -> Self {
        // Reuse the allocation for the unknown case rather than re-copying.
        KnownEvalId::from_eval_id(&raw).map_or(Self::Unknown(raw), Self::Known)
    }
}

impl From<&str> for EvalId {
    fn from(raw: &str) -> Self {
        KnownEvalId::from_eval_id(raw).map_or_else(|| Self::Unknown(raw.to_string()), Self::Known)
    }
}

impl From<EvalId> for String {
    fn from(id: EvalId) -> Self {
        match id {
            EvalId::Known(known) => known.as_ref().to_string(),
            EvalId::Unknown(raw) => raw,
        }
    }
}

impl std::fmt::Display for EvalId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    #[test]
    fn known_eval_id_round_trips_every_variant() {
        // Each variant's Display (its snake_case eval id) resolves back to the
        // same variant — keeps the eval id <-> variant mapping honest.
        KnownEvalId::ALL.iter().for_each(|&eval| {
            assert_eq!(KnownEvalId::from_eval_id(eval.as_ref()), Some(eval));
        });
    }

    #[test]
    fn eval_id_parses_known_and_preserves_unknown() {
        assert_eq!(EvalId::from("ifbench"), EvalId::Known(KnownEvalId::Ifbench));
        assert_eq!(
            EvalId::from("math_500"),
            EvalId::Known(KnownEvalId::Math500)
        );
        // An id the client doesn't recognize is kept verbatim, not dropped.
        assert_eq!(
            EvalId::from("some_future_eval"),
            EvalId::Unknown("some_future_eval".to_string())
        );
    }

    // Known and unknown ids both survive serialize -> deserialize unchanged
    // (the loose<->strict bridge): known -> canonical wire spelling, unknown
    // -> preserved exactly.
    #[rstest]
    #[case(EvalId::Known(KnownEvalId::Math500))]
    #[case(EvalId::Unknown("some_future_eval".to_string()))]
    fn eval_id_round_trips_through_string_losslessly(#[case] id: EvalId) -> anyhow::Result<()> {
        let json = serde_json::to_value(&id)?;
        assert!(json.is_string());
        assert_eq!(serde_json::from_value::<EvalId>(json)?, id);
        Ok(())
    }

    #[test]
    fn eval_id_known_serializes_to_canonical_wire_spelling() -> anyhow::Result<()> {
        assert_eq!(
            serde_json::to_value(EvalId::Known(KnownEvalId::Math500))?,
            serde_json::json!("math_500")
        );
        Ok(())
    }

    #[test]
    fn eval_id_unknown_is_greedy() {
        assert_eq!(
            EvalId::Unknown("anything".to_string()).sampling_temperature(),
            0.0
        );
        assert_eq!(
            EvalId::Known(KnownEvalId::Ifbench).sampling_temperature(),
            0.6
        );
    }

    #[test]
    fn known_eval_id_spelling_matches_scoring_backend() {
        // Wire spellings must match pipette-scores' `EvalId` exactly (POST /score
        // 404s otherwise). Digit-adjacent ids are the trap: "Math500" snake_cases
        // to "math500", so Math500 carries an explicit serialize.
        assert_eq!(KnownEvalId::Ifbench.to_string(), "ifbench");
        assert_eq!(KnownEvalId::Ifstruct.to_string(), "ifstruct");
        assert_eq!(KnownEvalId::GpqaDiamond.to_string(), "gpqa_diamond");
        assert_eq!(KnownEvalId::Math500.to_string(), "math_500");
    }

    #[test]
    fn known_eval_id_unrecognized_is_none() {
        // Calibration-only ids (not scoreable) and anything unknown are unrecognized.
        assert_eq!(KnownEvalId::from_eval_id("mmlu_pro"), None);
        assert_eq!(KnownEvalId::from_eval_id("totally_unknown_eval"), None);
    }
}
