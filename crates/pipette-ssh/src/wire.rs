//! The SSH wire vocabulary: a `string` is a big-endian `u32` length followed by
//! that many bytes, and every message in the agent protocol and every field in
//! the OpenSSH key formats is a sequence of those.
//!
//! Shared so `pipette-ctl`'s key generator and the agent client emit and
//! parse identical framing.

use crate::error::{Error, Result};

/// Append `bytes` as an SSH `string`.
pub fn push_string(out: &mut Vec<u8>, bytes: &[u8]) -> Result<()> {
    let len = u32::try_from(bytes.len()).map_err(|_| Error::StringTooLong(bytes.len()))?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

/// A cursor over one SSH message, handing back borrowed fields as it advances.
///
/// Every read is length-checked and returns [`Error::Malformed`]: the bytes
/// come from another process, so a truncated message must not panic.
pub struct Reader<'a> {
    rest: &'a [u8],
}

impl<'a> Reader<'a> {
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { rest: bytes }
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let (head, tail) = self
            .rest
            .split_at_checked(n)
            .ok_or(Error::Malformed("message ends mid-field"))?;
        self.rest = tail;
        Ok(head)
    }

    pub fn u8(&mut self) -> Result<u8> {
        let byte: [u8; 1] = self
            .take(1)?
            .try_into()
            .map_err(|_| Error::Malformed("message ends before a byte"))?;
        Ok(u8::from_be_bytes(byte))
    }

    pub fn u32(&mut self) -> Result<u32> {
        let word: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| Error::Malformed("message ends before a u32"))?;
        Ok(u32::from_be_bytes(word))
    }

    /// The next length-prefixed `string`.
    pub fn string(&mut self) -> Result<&'a [u8]> {
        let len = self.u32()?;
        let len =
            usize::try_from(len).map_err(|_| Error::Malformed("string longer than memory"))?;
        self.take(len)
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::empty(b"")]
    #[case::short(b"ssh-ed25519")]
    fn a_pushed_string_reads_back_whole(#[case] payload: &[u8]) -> anyhow::Result<()> {
        let mut out = Vec::new();
        push_string(&mut out, payload)?;
        push_string(&mut out, b"trailer")?;

        assert_eq!(
            out.first().copied(),
            Some(0),
            "length is big-endian, so the high byte leads"
        );

        let mut reader = Reader::new(&out);
        assert_eq!(reader.string()?, payload);
        assert_eq!(reader.string()?, b"trailer");
        Ok(())
    }

    #[test]
    fn a_length_past_the_end_is_an_error() -> anyhow::Result<()> {
        let mut reader = Reader::new(&[0, 0, 0, 9, b'x']);
        let err = reader
            .string()
            .err()
            .context("a 9-byte string in a 1-byte buffer must fail")?;
        assert!(matches!(err, Error::Malformed(_)), "got {err}");
        Ok(())
    }

    #[test]
    fn reads_advance_past_what_they_consumed() -> anyhow::Result<()> {
        let mut message = vec![12u8];
        message.extend_from_slice(&2u32.to_be_bytes());
        push_string(&mut message, b"k")?;

        let mut reader = Reader::new(&message);
        assert_eq!(reader.u8()?, 12);
        assert_eq!(reader.u32()?, 2);
        assert_eq!(reader.string()?, b"k");
        assert!(reader.u8().is_err(), "the message is fully consumed");
        Ok(())
    }
}
