//! Benchmark catalog: generated `local/` + synced `remote/`.

mod reference;
mod remote;
mod standard;
mod store;

pub use reference::SourcedBenchmarkId;
pub(crate) use remote::ConvertedRemote;
pub use remote::RemoteSyncState;
pub use standard::seed_standard_local;
pub use store::BenchmarkStore;
