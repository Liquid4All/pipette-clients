//! Unpacking the archives a runtime or a vault model arrives as.
//!
//! A public archive is named by its URL, so its kind is read off the
//! extension; a vault object is addressed by an id, so its kind is read off
//! its first bytes.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::Context;
use flate2::read::GzDecoder;
use tar::Archive;
use zip::ZipArchive;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArchiveKind {
    Zip,
    Tar,
    TarGz,
}

/// The archive kind from a filename/URL: `.zip` is zip, anything else tar.gz.
pub(crate) fn infer_archive_kind(name: &str) -> ArchiveKind {
    if Path::new(name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("zip"))
    {
        ArchiveKind::Zip
    } else {
        ArchiveKind::TarGz
    }
}

/// Offset of the `ustar` magic in a POSIX tar header.
const TAR_MAGIC_OFFSET: usize = 257;

/// The archive kind from its leading bytes, for an artifact with no name to
/// read one off. `None` is not an archive this crate unpacks.
pub(crate) fn sniff_archive_kind(head: &[u8]) -> Option<ArchiveKind> {
    if head.starts_with(&[0x1f, 0x8b]) {
        return Some(ArchiveKind::TarGz);
    }
    if head.starts_with(b"PK\x03\x04") {
        return Some(ArchiveKind::Zip);
    }
    if head
        .get(TAR_MAGIC_OFFSET..TAR_MAGIC_OFFSET + 5)
        .is_some_and(|magic| magic == b"ustar")
    {
        return Some(ArchiveKind::Tar);
    }
    None
}

/// [`sniff_archive_kind`] over a reader, leaving it where it started.
pub(crate) fn sniff_archive<R: Read + Seek>(reader: &mut R) -> anyhow::Result<ArchiveKind> {
    let mut head = Vec::with_capacity(TAR_MAGIC_OFFSET + 5);
    reader
        .take((TAR_MAGIC_OFFSET + 5) as u64)
        .read_to_end(&mut head)?;
    reader.seek(SeekFrom::Start(0))?;
    sniff_archive_kind(&head).context("the artifact is not a tar, tar.gz or zip archive")
}

/// Unpack `archive` into `destination`, restoring unix exec bits for zip
/// entries so a binary inside stays runnable.
///
/// Path-traversal safety: the zip path normalizes each entry via
/// `mangled_name()`; the tar path relies on `tar::Archive::unpack`, which
/// refuses to write outside `destination` (rejecting `..` components).
pub(crate) fn extract_archive<R: Read + Seek>(
    archive: R,
    kind: ArchiveKind,
    destination: &Path,
) -> anyhow::Result<()> {
    match kind {
        ArchiveKind::Zip => {
            let mut archive = ZipArchive::new(archive).context("failed to read zip archive")?;
            (0..archive.len()).try_for_each(|index| {
                let entry = archive
                    .by_index(index)
                    .with_context(|| format!("failed to read zip entry #{index}"))?;
                extract_zip_entry(entry, destination)
            })?;
        }
        ArchiveKind::Tar => unpack_tar(Archive::new(archive), destination)?,
        ArchiveKind::TarGz => unpack_tar(Archive::new(GzDecoder::new(archive)), destination)?,
    }
    Ok(())
}

fn extract_zip_entry<R: Read + Seek>(
    mut entry: zip::read::ZipFile<'_, R>,
    destination: &Path,
) -> anyhow::Result<()> {
    let out_path = destination.join(
        entry
            .mangled_name()
            .context("failed to read a zip entry's name")?,
    );
    if entry.is_dir() {
        return std::fs::create_dir_all(&out_path)
            .with_context(|| format!("failed to create {}", out_path.display()));
    }
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut out_file = std::fs::File::create(&out_path)
        .with_context(|| format!("failed to create {}", out_path.display()))?;
    std::io::copy(&mut entry, &mut out_file)
        .with_context(|| format!("failed to extract {}", out_path.display()))?;
    #[cfg(unix)]
    if let Some(mode) = entry.unix_mode() {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&out_path, std::fs::Permissions::from_mode(mode))
            .with_context(|| format!("failed to set permissions on {}", out_path.display()))?;
    }
    Ok(())
}

fn unpack_tar<R: Read>(mut archive: Archive<R>, destination: &Path) -> anyhow::Result<()> {
    archive
        .unpack(destination)
        .with_context(|| format!("failed to unpack into {}", destination.display()))
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Write};

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case("llama-b1-bin-macos-arm64.zip", ArchiveKind::Zip)]
    #[case("llama-b1-bin-ubuntu-x64.tar.gz", ArchiveKind::TarGz)]
    #[case("weird.ZIP", ArchiveKind::Zip)]
    #[case("noext", ArchiveKind::TarGz)]
    fn infer_archive_kind_cases(#[case] name: &str, #[case] want: ArchiveKind) {
        assert_eq!(infer_archive_kind(name), want);
    }

    fn tar_bytes() -> anyhow::Result<Vec<u8>> {
        crate::vault::test_support::tar_with("config.json", b"{}")
    }

    fn tar_gz_bytes() -> anyhow::Result<Vec<u8>> {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&tar_bytes()?)?;
        Ok(encoder.finish()?)
    }

    fn zip_bytes() -> anyhow::Result<Vec<u8>> {
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        zip.start_file("config.json", zip::write::SimpleFileOptions::default())?;
        zip.write_all(b"{}")?;
        Ok(zip.finish()?.into_inner())
    }

    /// Each kind is told from its own leading bytes, and each unpacks to the
    /// same tree, so a vault archive needs no name to say what it is.
    #[rstest]
    #[case::tar(tar_bytes()?, ArchiveKind::Tar)]
    #[case::tar_gz(tar_gz_bytes()?, ArchiveKind::TarGz)]
    #[case::zip(zip_bytes()?, ArchiveKind::Zip)]
    fn a_sniffed_archive_unpacks(
        #[case] bytes: Vec<u8>,
        #[case] want: ArchiveKind,
    ) -> anyhow::Result<()> {
        let mut cursor = Cursor::new(bytes);
        assert_eq!(sniff_archive(&mut cursor)?, want);
        assert_eq!(
            cursor.position(),
            0,
            "the sniff leaves the reader at the start"
        );
        let dir = tempfile::tempdir()?;
        extract_archive(cursor, want, dir.path())?;
        assert_eq!(std::fs::read(dir.path().join("config.json"))?, b"{}");
        Ok(())
    }

    #[test]
    fn bytes_that_are_no_archive_are_refused() {
        assert!(sniff_archive(&mut Cursor::new(b"GGUF....".to_vec())).is_err());
        assert!(sniff_archive_kind(b"").is_none());
    }
}
