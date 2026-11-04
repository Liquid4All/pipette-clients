//! Whose fault an `io::Error` is: this host's, or the transfer's.

use std::io::{self, ErrorKind};

/// Whether `error`'s kind names a disk or a permission that gave out on this
/// host. Such a failure meets every retry the same way; a broken read of a
/// response body does not.
pub(crate) fn names_the_host(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        ErrorKind::StorageFull
            | ErrorKind::QuotaExceeded
            | ErrorKind::PermissionDenied
            | ErrorKind::ReadOnlyFilesystem
            | ErrorKind::OutOfMemory
            | ErrorKind::ResourceBusy
    )
}
