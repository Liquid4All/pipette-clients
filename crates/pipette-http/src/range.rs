//! The headers of a `206` that answers `Range: bytes=N-`. A resumed download
//! reads them to check that the response continues its file.

use reqwest::header::{CONTENT_RANGE, ETAG, HeaderMap};

/// The `ETag` in `headers`, quotes included, if it is strong. A weak `ETag`
/// does not promise the same bytes, so it cannot prove that a `206` continues
/// a file.
pub fn strong_etag(headers: &HeaderMap) -> Option<String> {
    let value = headers.get(ETAG)?.to_str().ok()?.trim();
    (!value.starts_with("W/")).then(|| value.to_owned())
}

/// A `Content-Range` of the form `bytes first-last/total`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContentRange {
    pub first: u64,
    pub last: u64,
    /// `None` when the server sent `*`.
    pub total: Option<u64>,
}

impl ContentRange {
    /// The `Content-Range` in `headers`, if it names a range of one byte or
    /// more.
    pub fn from_headers(headers: &HeaderMap) -> Option<Self> {
        let value = headers.get(CONTENT_RANGE)?.to_str().ok()?;
        let (range, total) = value.strip_prefix("bytes ")?.split_once('/')?;
        let (first, last) = range.split_once('-')?;
        let (first, last) = (first.trim().parse().ok()?, last.trim().parse().ok()?);
        (first <= last).then(|| Self {
            first,
            last,
            total: total.trim().parse().ok(),
        })
    }
}

#[cfg(test)]
mod tests {
    use reqwest::header::HeaderValue;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::strong(Some("\"abc\""), Some("\"abc\""))]
    #[case::weak(Some("W/\"abc\""), None)]
    #[case::absent(None, None)]
    fn only_a_strong_etag_is_returned(
        #[case] value: Option<&'static str>,
        #[case] expected: Option<&str>,
    ) {
        let mut headers = HeaderMap::new();
        if let Some(value) = value {
            headers.insert(ETAG, HeaderValue::from_static(value));
        }
        assert_eq!(strong_etag(&headers).as_deref(), expected);
    }

    #[rstest]
    #[case::known_total(Some("bytes 4-9/10"), Some((4, 9, Some(10))))]
    #[case::unknown_total(Some("bytes 4-9/*"), Some((4, 9, None)))]
    #[case::unsatisfied(Some("bytes */10"), None)]
    #[case::backwards(Some("bytes 9-4/10"), None)]
    #[case::other_unit(Some("items 4-9/10"), None)]
    #[case::absent(None, None)]
    fn a_content_range_names_one_byte_or_more(
        #[case] value: Option<&'static str>,
        #[case] expected: Option<(u64, u64, Option<u64>)>,
    ) {
        let mut headers = HeaderMap::new();
        if let Some(value) = value {
            headers.insert(CONTENT_RANGE, HeaderValue::from_static(value));
        }
        assert_eq!(
            ContentRange::from_headers(&headers),
            expected.map(|(first, last, total)| ContentRange { first, last, total })
        );
    }
}
