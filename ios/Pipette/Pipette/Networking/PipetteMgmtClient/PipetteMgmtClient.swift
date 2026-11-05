import Foundation

/// The management-server HTTP surface: the Swift half of `pipette-mgmt-client`,
/// file for file with the crate (this file is its `lib.rs`).
///
/// Nothing here logs: every call returns its outcome and the caller decides what
/// to say about it, as `pipette-cli` does around the crate.
///
/// The crate's `user_api` and `httpsig` (`UserIdentity`, its RFC 9421 signature,
/// the plan and user routes) are not mirrored: this app only authenticates as a device.
nonisolated enum PipetteMgmtClient {}
