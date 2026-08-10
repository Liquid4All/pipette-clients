//! The bundled-catalog machinery every venv-backed backend shares: parsing a
//! `[[<table>]]` file of version-keyed rows, rejecting duplicates, looking one
//! up, and reporting a miss.
//!
//! [`VersionCatalog`] is the whole lookup surface, so a backend keeps only what
//! is genuinely its own: the `OnceLock` (a static belongs to the crate that owns
//! the file it caches), the typed row, and the row → `Runtime` mapping.
//!
//! `pipette-torch-oai` keeps its own: slug-keyed and `build.rs`-validated, so
//! it shares the idea but not the shape.

use std::collections::BTreeMap;

use anyhow::Context;
use serde::de::DeserializeOwned;

/// A parsed bundled catalog: version-keyed rows plus the two words a miss has
/// to name to be actionable.
///
/// Holds those words rather than taking them per call because they are
/// properties of the catalog, not of the lookup — and because a miss reported
/// with the wrong runtime subword sends the reader to a listing of something
/// else.
pub struct VersionCatalog<R> {
    rows: BTreeMap<String, R>,
    /// The `runtimes catalog <subword>` that lists this catalog.
    catalog_subword: &'static str,
    /// The URI scheme a `runtimes pull` of one of these rows takes.
    uri_scheme: &'static str,
}

impl<R: DeserializeOwned> VersionCatalog<R> {
    /// Parse `toml_str`'s `[[table]]` rows into a catalog.
    pub fn parse<K: Fn(&R) -> String>(
        toml_str: &str,
        table: &str,
        version_of: K,
        catalog_subword: &'static str,
        uri_scheme: &'static str,
    ) -> anyhow::Result<Self> {
        Ok(Self::from_rows(
            parse_catalog_rows(toml_str, table, version_of)?,
            catalog_subword,
            uri_scheme,
        ))
    }
}

impl<R> VersionCatalog<R> {
    /// Wrap rows a backend parsed itself.
    ///
    /// For a catalog whose row is not the deserialized shape: mlx types its
    /// `flavor` column after parsing.
    pub fn from_rows(
        rows: BTreeMap<String, R>,
        catalog_subword: &'static str,
        uri_scheme: &'static str,
    ) -> Self {
        Self {
            rows,
            catalog_subword,
            uri_scheme,
        }
    }

    /// The row for `version`, or the miss error naming what is available and
    /// both ways past it.
    pub fn get(&self, version: &str) -> anyhow::Result<&R> {
        self.rows.get(version).ok_or_else(|| {
            unknown_catalog_entry(
                version,
                &self.versions(),
                self.catalog_subword,
                self.uri_scheme,
            )
        })
    }

    /// Every bundled version, sorted.
    pub fn versions(&self) -> Vec<&str> {
        self.rows.keys().map(String::as_str).collect()
    }

    /// Every row, in version order.
    pub fn rows(&self) -> impl Iterator<Item = &R> {
        self.rows.values()
    }
}

/// Parse a catalog body into rows keyed by version.
///
/// `table` is the array-of-tables name (`mlx`, `openvino`); `version_of` reads
/// the key out of a parsed row, since which column is the version is the
/// backend's business.
///
/// Duplicate versions are rejected rather than last-one-wins: two rows under
/// one key means the file disagrees with itself about what that version *is*,
/// and silently picking one would install something the author did not choose.
pub fn parse_catalog_rows<R, K>(
    toml_str: &str,
    table: &str,
    version_of: K,
) -> anyhow::Result<BTreeMap<String, R>>
where
    R: DeserializeOwned,
    K: Fn(&R) -> String,
{
    // Through `toml::Value` because the table name is a parameter: a derived
    // `Deserialize` would have to name the field at compile time.
    let doc: toml::Value = toml::from_str(toml_str).context("malformed bundled catalog.toml")?;
    // An absent table is a mistake, not an empty catalog: `table` is a string
    // literal at the call site, so nothing else catches a typo in it, and the
    // symptom would otherwise be every lookup failing with an empty
    // available-list. TOML cannot express an empty array-of-tables anyway, so
    // there is no legitimate caller this refuses.
    let rows = doc
        .get(table)
        .with_context(|| format!("bundled catalog.toml has no `[[{table}]]` entries"))?;
    let rows = rows
        .as_array()
        .with_context(|| format!("bundled catalog.toml `{table}` is not an array of tables"))?;

    rows.iter().try_fold(BTreeMap::new(), |mut map, raw| {
        let row: R = raw
            .clone()
            .try_into()
            .with_context(|| format!("malformed `{table}` entry in bundled catalog.toml"))?;
        let version = version_of(&row);
        if map.insert(version.clone(), row).is_some() {
            anyhow::bail!("bundled catalog.toml has duplicate version '{version}'");
        }
        Ok(map)
    })
}

/// The error for a catalog lookup that found nothing.
///
/// Lists what *is* available and both ways to get past it, because the usual
/// cause is a version that was never bundled rather than a typo, and the reader
/// needs to know which of the two they are looking at.
pub fn unknown_catalog_entry(
    name: &str,
    available: &[&str],
    catalog_subword: &str,
    uri_scheme: &str,
) -> anyhow::Error {
    anyhow::anyhow!(
        "runtime '{name}' not found; no matching bundled catalog entry\n\
         auto-install only works for bundled catalog entries ({});\n\
         list them with `pipette runtimes catalog {catalog_subword}`\n\
         or pull an explicit runtime with `pipette runtimes pull --runtime '{uri_scheme}://…'`",
        available.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde::Deserialize;

    use super::*;

    #[derive(Debug, Deserialize, PartialEq, Eq)]
    struct Row {
        version: String,
        requirements: String,
    }

    fn rows(toml_str: &str) -> anyhow::Result<BTreeMap<String, Row>> {
        parse_catalog_rows(toml_str, "thing", |r: &Row| r.version.clone())
    }

    #[test]
    fn parses_rows_keyed_by_version() -> anyhow::Result<()> {
        let parsed = rows(
            "[[thing]]\nversion = \"1.0\"\nrequirements = \"a\"\n\
             [[thing]]\nversion = \"2.0\"\nrequirements = \"b\"\n",
        )?;
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed["2.0"].requirements, "b");
        // BTreeMap, so listing order is stable rather than file order.
        assert_eq!(parsed.keys().collect::<Vec<_>>(), vec!["1.0", "2.0"]);
        Ok(())
    }

    /// Silently keeping one of two rows would install something the author did
    /// not pick, and the file would still look fine.
    #[test]
    fn duplicate_versions_are_rejected() -> anyhow::Result<()> {
        let Err(err) = rows(
            "[[thing]]\nversion = \"1.0\"\nrequirements = \"a\"\n\
             [[thing]]\nversion = \"1.0\"\nrequirements = \"b\"\n",
        ) else {
            anyhow::bail!("expected duplicate versions to be rejected");
        };
        assert!(err.to_string().contains("duplicate version"), "got {err}");
        Ok(())
    }

    /// The table name is a literal at the call site, so a typo in it has no
    /// other check; failing loudly beats every lookup reporting an empty
    /// available-list.
    #[rstest]
    #[case::wrong_table("[[other]]\nversion = \"1.0\"\nrequirements = \"a\"\n")]
    #[case::empty_file("")]
    fn an_absent_table_is_rejected(#[case] toml_str: &str) -> anyhow::Result<()> {
        let Err(err) = rows(toml_str) else {
            anyhow::bail!("expected a missing table to be rejected");
        };
        assert!(format!("{err:#}").contains("[[thing]]"), "got {err:#}");
        Ok(())
    }

    #[test]
    fn a_malformed_row_names_the_table() -> anyhow::Result<()> {
        let Err(err) = rows("[[thing]]\nversion = \"1.0\"\n") else {
            anyhow::bail!("expected a row missing `requirements` to be rejected");
        };
        assert!(format!("{err:#}").contains("thing"), "got {err:#}");
        Ok(())
    }

    #[test]
    fn a_non_array_table_is_rejected() -> anyhow::Result<()> {
        let Err(err) = rows("thing = \"not an array\"\n") else {
            anyhow::bail!("expected a scalar `thing` to be rejected");
        };
        assert!(err.to_string().contains("array of tables"), "got {err}");
        Ok(())
    }

    #[test]
    fn the_unknown_entry_error_shows_the_way_out() {
        let err = unknown_catalog_entry("9.9.9", &["1.0", "2.0"], "uv_openvino", "uv-openvino");
        let msg = err.to_string();
        assert!(msg.contains("1.0, 2.0"), "should list what exists: {msg}");
        assert!(msg.contains("runtimes catalog uv_openvino"), "got {msg}");
        assert!(msg.contains("uv-openvino://"), "got {msg}");
    }
}
