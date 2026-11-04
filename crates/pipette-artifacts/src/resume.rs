//! A download that continues a broken body: the next attempt asks for the
//! rest with `Range` and an `If-Range` of the strong `ETag`. The model path
//! and the vault path both download through [`attempt`].

use std::fs;
use std::io::{Read, Write};
use std::path::Path;

use reqwest::blocking::Response;

use pipette_http::range::{ContentRange, strong_etag};
use pipette_mgmt_client::{ByteStream, EntityTag};

use crate::digest::HashingWriter;
use crate::progress::{Reporter, copy_reporting};

/// A response body that can continue a file.
pub(crate) trait RangeBody: Read {
    fn status(&self) -> u16;
    /// The `ETag` of the response, quotes included, if it is strong.
    fn strong_etag(&self) -> Option<String>;
    /// The `Content-Range` of the response, if it names one byte or more.
    fn content_range(&self) -> Option<ContentRange>;
}

impl RangeBody for Response {
    fn status(&self) -> u16 {
        Response::status(self).as_u16()
    }

    fn strong_etag(&self) -> Option<String> {
        strong_etag(self.headers())
    }

    fn content_range(&self) -> Option<ContentRange> {
        ContentRange::from_headers(self.headers())
    }
}

impl RangeBody for ByteStream {
    fn status(&self) -> u16 {
        ByteStream::status(self)
    }

    fn strong_etag(&self) -> Option<String> {
        ByteStream::strong_etag(self)
    }

    fn content_range(&self) -> Option<ContentRange> {
        ByteStream::content_range(self)
    }
}

/// The server side of a download.
pub(crate) trait Source {
    type Body: RangeBody;
    type Error;

    /// Send the request. With `resume`, ask for the bytes from its offset to
    /// the end, with an `If-Range` of its `ETag`.
    fn get(&self, resume: Option<&ResumePoint>) -> Result<Self::Body, Self::Error>;

    /// Whether `error` is a `416`: the range starts at or past the end of the
    /// object.
    fn range_refused(&self, error: &Self::Error) -> bool;

    /// What a later range must match to continue a file that `body` starts.
    fn expect(&self, body: &Self::Body) -> Expect;

    /// The error for a body that broke, or a disk that refused the bytes.
    fn io_error(&self, source: std::io::Error) -> Self::Error;
}

/// What a range must match to continue a file: a strong `ETag`, and the size
/// of the object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Expect {
    /// `None` if the server gave no strong `ETag`. A file without one cannot
    /// continue.
    pub(crate) etag: Option<EntityTag>,
    /// `None` if the size is not known.
    pub(crate) total: Option<u64>,
}

/// A download that an attempt did not finish: the bytes on disk, their hash,
/// and what a range must match to continue them.
pub(crate) struct Partial {
    pub(crate) writer: HashingWriter<fs::File>,
    pub(crate) expect: Expect,
    /// The byte where the first resumed attempt continued the file.
    pub(crate) resumed_at: Option<u64>,
}

impl Partial {
    fn start(dest: &Path, expect: Expect) -> std::io::Result<Self> {
        Ok(Self {
            writer: HashingWriter::new(fs::File::create(dest)?),
            expect,
            resumed_at: None,
        })
    }

    /// If the size is not known, take it from the `Content-Range` of the `206`.
    fn resume(&mut self, offset: u64, body: &impl RangeBody) {
        self.resumed_at.get_or_insert(offset);
        self.expect.total = self
            .expect
            .total
            .or_else(|| body.content_range().and_then(|range| range.total));
    }

    /// `None` if the file holds no bytes, or if the server gave no strong `ETag`.
    fn resume_point(&self) -> Option<ResumePoint> {
        let offset = self.writer.written();
        if offset == 0 {
            return None;
        }
        Some(ResumePoint {
            offset,
            etag: self.expect.etag.clone()?,
            total: self.expect.total,
        })
    }
}

/// The byte where a resumed download continues, and the `ETag` and size that
/// the next response must match.
pub(crate) struct ResumePoint {
    pub(crate) offset: u64,
    pub(crate) etag: EntityTag,
    total: Option<u64>,
}

impl ResumePoint {
    /// Compares the `ETag` itself, because the HuggingFace CDN answers a stale
    /// `If-Range` with a `206`.
    fn continued_by(&self, body: &impl RangeBody) -> bool {
        body.status() == 206
            && body.strong_etag().as_deref() == Some(self.etag.as_str())
            && body.content_range().is_some_and(|range| {
                range.first == self.offset && (self.total.is_none() || range.total == self.total)
            })
    }
}

/// Why a download that could resume starts again from zero.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Restart {
    NoStrongEtag,
    /// The server answered the range with a `200` and the whole file.
    WholeFile,
    RangeRefused,
    /// The server answered with a `206` for another file, offset or length.
    NotAContinuation,
    /// The vault record pins a different object from the object of the file.
    RecordChanged,
}

/// Send the request for an attempt. The returned `partial` is `None` if the
/// body starts the file from zero; a `partial` the body does not continue is
/// dropped, so a later attempt does not ask for its range again.
fn open<S: Source>(
    source: &S,
    partial: &mut Option<Partial>,
) -> Result<(S::Body, Option<Partial>), S::Error> {
    let Some(point) = partial.as_ref().and_then(Partial::resume_point) else {
        if partial.take().is_some_and(|file| file.writer.written() > 0) {
            tracing::info!(reason = ?Restart::NoStrongEtag, "restarting download from zero");
        }
        return Ok((source.get(None)?, None));
    };
    let (whole_file, restart) = match source.get(Some(&point)) {
        Ok(body) if point.continued_by(&body) => {
            tracing::info!(offset = point.offset, "resuming download");
            let continued = partial.take().map(|mut file| {
                file.resume(point.offset, &body);
                file
            });
            return Ok((body, continued));
        }
        Ok(body) if body.status() == 200 => (Some(body), Restart::WholeFile),
        Ok(_) => (None, Restart::NotAContinuation),
        Err(error) if source.range_refused(&error) => (None, Restart::RangeRefused),
        // The server never judged the range, so a later attempt asks again.
        Err(error) => return Err(error),
    };
    *partial = None;
    tracing::info!(reason = ?restart, "restarting download from zero");
    let body = match whole_file {
        Some(body) => body,
        None => source.get(None)?,
    };
    Ok((body, None))
}

/// One attempt to stream the object from `source` into `dest`. It continues
/// `partial` if the server agrees the object is the same, else starts from
/// zero and rewinds progress to `start`. A body that breaks part-way stays in
/// `partial`; the caller checks the returned file and puts it back in
/// `partial` if a later attempt can continue it.
pub(crate) fn attempt<S: Source>(
    source: &S,
    dest: &Path,
    partial: &mut Option<Partial>,
    start: u64,
    reporter: &mut Reporter,
) -> Result<Partial, S::Error> {
    // Named for the reader, not the store: the staged destination is a
    // content-addressed directory nobody asked about.
    let name = dest
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    loop {
        let (mut body, continued) = open(source, partial)?;
        let piece_end = continued
            .as_ref()
            .and_then(|_| body.content_range())
            .map(|range| range.last);
        let mut file = match continued {
            Some(file) => file,
            None => {
                if let Some(parent) = dest.parent() {
                    fs::create_dir_all(parent).map_err(|e| source.io_error(e))?;
                }
                reporter.rewind(start);
                Partial::start(dest, source.expect(&body)).map_err(|e| source.io_error(e))?
            }
        };
        // Hashed as it is written, so a multi-GB object never sits in memory.
        let copied = copy_reporting(&mut body, &mut file.writer, &name, reporter)
            .and_then(|()| file.writer.flush());
        if let Err(error) = copied {
            *partial = Some(file);
            return Err(source.io_error(error));
        }
        // A `206` can send less than the rest of the file. If it sent all of
        // its range, ask for the bytes after it.
        let written = file.writer.written();
        let stopped_short = piece_end.is_some_and(|last| written == last.saturating_add(1))
            && file.expect.total.is_some_and(|total| written < total);
        if !stopped_short {
            return Ok(file);
        }
        tracing::info!(offset = written, "range answer stopped short of the end");
        *partial = Some(file);
    }
}
