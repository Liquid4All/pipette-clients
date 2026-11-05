//! The image a VL cell sends. Its content changes neither the token count nor
//! the encode's cost, so every engine draws the same solid gray one.

use std::io::{self, Write};
use std::num::TryFromIntError;

use flate2::{Compression, write::ZlibEncoder};

/// Past any encoder's input, and before the pixels can exhaust memory.
pub const MAX_PIXELS: u64 = 64 << 20;

#[derive(Debug, thiserror::Error)]
pub enum GrayError {
    #[error("image dimensions {width}x{height} out of range (max {MAX_PIXELS} pixels)")]
    OutOfRange { width: u32, height: u32 },
    #[error(transparent)]
    Encode(#[from] io::Error),
    #[error(transparent)]
    Length(#[from] TryFromIntError),
}

pub fn gray(width: u32, height: u32) -> Result<Vec<u8>, GrayError> {
    if !(1..=MAX_PIXELS).contains(&(u64::from(width) * u64::from(height))) {
        return Err(GrayError::OutOfRange { width, height });
    }
    // A row is its filter byte (none), then three bytes a pixel.
    let row: Vec<u8> = std::iter::once(0)
        .chain(std::iter::repeat_n(128, usize::try_from(width)? * 3))
        .collect();
    let mut pixels = ZlibEncoder::new(Vec::new(), Compression::fast());
    (0..height).try_for_each(|_| pixels.write_all(&row))?;
    // 8-bit RGB, deflate, standard filtering, no interlace.
    let header = [
        &width.to_be_bytes()[..],
        &height.to_be_bytes(),
        &[8, 2, 0, 0, 0],
    ]
    .concat();
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    chunk(&mut png, b"IHDR", &header)?;
    chunk(&mut png, b"IDAT", &pixels.finish()?)?;
    chunk(&mut png, b"IEND", &[])?;
    Ok(png)
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) -> Result<(), TryFromIntError> {
    let mut crc = crc32fast::Hasher::new();
    crc.update(kind);
    crc.update(data);
    png.extend(u32::try_from(data.len())?.to_be_bytes());
    png.extend(kind);
    png.extend(data);
    png.extend(crc.finalize().to_be_bytes());
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use anyhow::Context;
    use flate2::read::ZlibDecoder;
    use rstest::rstest;

    use super::*;

    /// Every chunk's CRC holds, the header names the size as 8-bit RGB, and
    /// the pixels inflate to gray rows of that size.
    #[rstest]
    #[case::one_pixel(1, 1)]
    #[case::wide(3, 2)]
    #[case::smoke(224, 224)]
    fn a_gray_png_is_gray_rows_of_its_size(
        #[case] width: u32,
        #[case] height: u32,
    ) -> anyhow::Result<()> {
        let png = gray(width, height)?;
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        let mut chunks = Vec::new();
        let mut rest = &png[8..];
        while !rest.is_empty() {
            let len = usize::try_from(u32::from_be_bytes(rest[..4].try_into()?))?;
            let (kind, data) = (&rest[4..8], &rest[8..8 + len]);
            let mut crc = crc32fast::Hasher::new();
            crc.update(kind);
            crc.update(data);
            assert_eq!(rest[8 + len..12 + len], crc.finalize().to_be_bytes());
            chunks.push((kind.to_vec(), data.to_vec()));
            rest = &rest[12 + len..];
        }
        let kinds: Vec<_> = chunks.iter().map(|(kind, _)| kind.as_slice()).collect();
        assert_eq!(kinds, [b"IHDR", b"IDAT", b"IEND"]);
        let header = [
            &width.to_be_bytes()[..],
            &height.to_be_bytes(),
            &[8, 2, 0, 0, 0],
        ]
        .concat();
        assert_eq!(chunks[0].1, header);
        let mut pixels = Vec::new();
        ZlibDecoder::new(chunks[1].1.as_slice()).read_to_end(&mut pixels)?;
        let row: Vec<u8> = std::iter::once(0)
            .chain(std::iter::repeat_n(128, usize::try_from(width)? * 3))
            .collect();
        assert_eq!(pixels, row.repeat(usize::try_from(height)?));
        Ok(())
    }

    #[rstest]
    #[case::no_width(0, 512)]
    #[case::no_height(512, 0)]
    #[case::past_64_megapixels(8192, 8193)]
    fn a_size_outside_one_pixel_to_64_megapixels_is_refused(
        #[case] width: u32,
        #[case] height: u32,
    ) -> anyhow::Result<()> {
        let err = gray(width, height)
            .err()
            .context("an out-of-range size was drawn")?;
        assert!(
            matches!(
                err,
                GrayError::OutOfRange { width: w, height: h } if (w, h) == (width, height)
            ),
            "{err:?}"
        );
        Ok(())
    }
}
