//! Resolving the artifact-store disk cap (`docs/storage-quota.md`), and which
//! source set it, so `storage status` can name it.

use anyhow::Context;

use crate::identity::{ClientSettings, ClientSettingsStore};

/// Built-in cap when nothing configures one. Bench boxes have room; phones do
/// not, and iOS picks its own default.
pub const DEFAULT_STORAGE_QUOTA_BYTES: u64 = 200 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuotaSource {
    Arg,
    Settings,
    Default,
}

impl QuotaSource {
    /// How the source reads in `storage status`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Arg => "--storage-quota / PIPETTE_STORAGE_QUOTA",
            Self::Settings => "identity/settings.json",
            Self::Default => "built-in default",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageQuota {
    pub bytes: u64,
    pub source: QuotaSource,
}

/// `--storage-quota` (or its env var), then `identity/settings.json`, then
/// [`DEFAULT_STORAGE_QUOTA_BYTES`]. `override_bytes` arrives parsed, so a
/// malformed value fails before a workspace is opened.
pub fn resolve_storage_quota(
    settings: &ClientSettingsStore,
    override_bytes: Option<u64>,
) -> anyhow::Result<StorageQuota> {
    if let Some(bytes) = override_bytes {
        return Ok(StorageQuota {
            bytes,
            source: QuotaSource::Arg,
        });
    }
    if let Some(ClientSettings {
        storage_quota_bytes,
    }) = settings.get()?
    {
        anyhow::ensure!(
            storage_quota_bytes > 0,
            "identity/settings.json sets storage_quota_bytes to 0; \
             a zero quota can hold no artifact"
        );
        return Ok(StorageQuota {
            bytes: storage_quota_bytes,
            source: QuotaSource::Settings,
        });
    }
    Ok(StorageQuota {
        bytes: DEFAULT_STORAGE_QUOTA_BYTES,
        source: QuotaSource::Default,
    })
}

/// Plain bytes (`214748364800`) or an IEC suffix (`200GiB`, `512 MiB`, `4kib`).
/// Case-insensitive, optional space. Rejects zero (a typo, not a configuration)
/// and anything overflowing `u64`.
pub fn parse_quota_bytes(raw: &str) -> anyhow::Result<u64> {
    const UNITS: [(&str, u64); 5] = [
        ("b", 1),
        ("kib", 1 << 10),
        ("mib", 1 << 20),
        ("gib", 1 << 30),
        ("tib", 1 << 40),
    ];

    let trimmed = raw.trim();
    let digits_end = trimmed
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(trimmed.len());
    let (digits, suffix) = trimmed.split_at(digits_end);
    let value: u64 = digits
        .parse()
        .with_context(|| format!("storage quota `{raw}` is not a byte count"))?;

    let suffix = suffix.trim().to_ascii_lowercase();
    let unit = if suffix.is_empty() {
        1
    } else {
        UNITS
            .iter()
            .find(|(name, _)| *name == suffix)
            .map(|(_, multiplier)| *multiplier)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "storage quota `{raw}` has an unknown unit `{suffix}`; \
                     use B, KiB, MiB, GiB, or TiB"
                )
            })?
    };

    let bytes = value
        .checked_mul(unit)
        .ok_or_else(|| anyhow::anyhow!("storage quota `{raw}` overflows a byte count"))?;
    anyhow::ensure!(bytes > 0, "storage quota must be greater than zero");
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(QuotaSource::Arg, "PIPETTE_STORAGE_QUOTA")]
    #[case(QuotaSource::Settings, "identity/settings.json")]
    #[case(QuotaSource::Default, "default")]
    fn each_source_names_where_to_change_it(#[case] source: QuotaSource, #[case] needle: &str) {
        assert!(source.label().contains(needle), "{}", source.label());
    }

    fn settings() -> anyhow::Result<(tempfile::TempDir, ClientSettingsStore)> {
        let tmp = tempfile::tempdir()?;
        let store = ClientSettingsStore::new(tmp.path().join("identity"));
        Ok((tmp, store))
    }

    #[rstest]
    #[case::plain_bytes("1024", 1024)]
    #[case::gibibytes("200GiB", 200 * 1024 * 1024 * 1024)]
    #[case::spaced("512 MiB", 512 * 1024 * 1024)]
    #[case::lowercase("4kib", 4096)]
    #[case::bare_b("100B", 100)]
    fn parse_quota_bytes_accepts_plain_and_iec(
        #[case] raw: &str,
        #[case] expected: u64,
    ) -> anyhow::Result<()> {
        assert_eq!(parse_quota_bytes(raw)?, expected);
        Ok(())
    }

    #[rstest]
    #[case::empty("")]
    #[case::zero("0")]
    #[case::negative("-1")]
    #[case::prose("200 gigs")]
    #[case::unit_only("GiB")]
    #[case::overflow("18446744073709551615TiB")]
    fn parse_quota_bytes_rejects_anything_it_cannot_enforce(#[case] raw: &str) {
        assert!(parse_quota_bytes(raw).is_err(), "`{raw}` should be refused");
    }

    #[test]
    fn the_arg_wins_over_the_settings_file() -> anyhow::Result<()> {
        let (_tmp, store) = settings()?;
        store.put(&ClientSettings {
            storage_quota_bytes: 4096,
        })?;
        let resolved = resolve_storage_quota(&store, Some(parse_quota_bytes("1GiB")?))?;
        assert_eq!(resolved.bytes, 1024 * 1024 * 1024);
        assert_eq!(resolved.source, QuotaSource::Arg);
        Ok(())
    }

    #[test]
    fn the_settings_file_wins_over_the_default() -> anyhow::Result<()> {
        let (_tmp, store) = settings()?;
        store.put(&ClientSettings {
            storage_quota_bytes: 4096,
        })?;
        let resolved = resolve_storage_quota(&store, None)?;
        assert_eq!(resolved.bytes, 4096);
        assert_eq!(resolved.source, QuotaSource::Settings);
        Ok(())
    }

    #[test]
    fn an_unconfigured_client_gets_the_built_in_default() -> anyhow::Result<()> {
        let (_tmp, store) = settings()?;
        let resolved = resolve_storage_quota(&store, None)?;
        assert_eq!(resolved.bytes, DEFAULT_STORAGE_QUOTA_BYTES);
        assert_eq!(resolved.source, QuotaSource::Default);
        Ok(())
    }

    #[test]
    fn a_zero_quota_in_the_settings_file_is_refused() -> anyhow::Result<()> {
        let (_tmp, store) = settings()?;
        store.put(&ClientSettings {
            storage_quota_bytes: 0,
        })?;
        let err = resolve_storage_quota(&store, None)
            .err()
            .ok_or_else(|| anyhow::anyhow!("a zero quota should be refused"))?;
        assert!(format!("{err:#}").contains("storage_quota_bytes"));
        Ok(())
    }
}
