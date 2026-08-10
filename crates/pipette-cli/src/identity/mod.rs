//! Workspace identity — the [`IdentityStore`] capability plus the local
//! identity files it persists (`registration.json`, `identity/device.json`).

mod store;
mod types;

pub use store::IdentityStore;
pub use types::{ClientSettings, DeviceLabels, IdentityRegistration};
