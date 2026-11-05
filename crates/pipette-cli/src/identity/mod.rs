//! Workspace identity: one store per file under `identity/`, because the files
//! do not share a lifetime, plus the [`IdentityBundle`] that `auth export`
//! moves between machines.
//!
//! - [`ClientRegistrationStore`]: `registration.json` and the keypair, created
//!   and torn down as a unit.
//! - [`ClientDeviceStore`]: `device.json`, needed by a local `benchmarks run`
//!   that never registers, and kept across a failed or reset registration.
//! - [`ClientSettingsStore`]: `settings.json`, local policy read before any
//!   command runs.

mod bundle;
mod device;
mod files;
mod registration;
mod settings;

pub use bundle::IdentityBundle;
pub use device::{ClientDeviceStore, DeviceLabels};
pub use registration::{ClientRegistrationStore, IdentityRegistration, StoredRegistration};
pub use settings::{ClientSettings, ClientSettingsStore};
