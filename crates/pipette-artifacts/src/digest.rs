//! Inline SHA-256 over a stream, so a multi-GB download is verified without a
//! second pass over the file.

use std::io::Write;

use sha2::{Digest, Sha256 as Sha256Hasher};

/// A [`Write`] that hashes every byte it forwards.
pub(crate) struct HashingWriter<W> {
    inner: W,
    hasher: Sha256Hasher,
    written: u64,
}

impl<W: Write> HashingWriter<W> {
    pub(crate) fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256Hasher::new(),
            written: 0,
        }
    }

    /// The bytes the inner writer accepted so far.
    pub(crate) fn written(&self) -> u64 {
        self.written
    }

    pub(crate) fn hex_digest(self) -> String {
        hex::encode(self.hasher.finalize())
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // Hash only the bytes the inner writer accepted, so a short write can't
        // desync the digest from what landed on disk.
        let written = self.inner.write(buf)?;
        let accepted = buf.get(..written).ok_or_else(|| {
            std::io::Error::other("inner writer reported more bytes than it was given")
        })?;
        self.hasher.update(accepted);
        self.written = self.written.saturating_add(accepted.len() as u64);
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashing_writer_matches_sha256_of_the_bytes() -> anyhow::Result<()> {
        let mut writer = HashingWriter::new(Vec::new());
        writer.write_all(b"abc")?;
        writer.flush()?;
        assert_eq!(writer.written(), 3);
        assert_eq!(
            writer.hex_digest(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        Ok(())
    }
}
