//! `lloom-serve` runtime install: a prebuilt archive, fetched and unpacked
//! into the store entry's blobs the way the llama.cpp CLI archive is — the
//! same reader, extractor and binary check, one binary (`lloom-serve`)
//! instead of two.

use std::path::Path;

use pipette_http::HttpClient;
use pipette_plan_types::{LloomServeSource, Runtime};

use super::llamacpp::{
    binary_name, extract_archive, find_binary, infer_archive_kind, read_archive,
};
use crate::progress::Reporter;

/// The binary the archive must carry, anywhere under its root.
pub(crate) const LLOOM_SERVE_BINARY: &str = "lloom-serve";

pub(crate) fn install_lloom_archive(
    http: &HttpClient,
    declared: &Runtime,
    blobs_dir: &Path,
    reporter: &mut Reporter,
) -> anyhow::Result<()> {
    let Runtime::LloomServeMacos(rt) = declared else {
        anyhow::bail!("not a lloom-serve runtime: `{}`", declared.headless_token());
    };
    let url = match &rt.source {
        LloomServeSource::RemoteArchive { url } => url.download_url(),
        LloomServeSource::RelativeDir { .. } | LloomServeSource::AbsoluteDir { .. } => {
            anyhow::bail!(
                "cannot fetch an installed form ({}); pass the declared remote_archive \
                 coordinate instead",
                rt.source
            );
        }
    };
    let kind = infer_archive_kind(&url);
    let bytes = read_archive(http, &url, reporter)?;
    extract_archive(&bytes, kind, blobs_dir)?;
    find_binary(blobs_dir, &binary_name(LLOOM_SERVE_BINARY))?;
    Ok(())
}
