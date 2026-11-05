//! Putting a build on a phone without losing the phone: the build's signature
//! before the install, the phone's client after it. The proxy installs
//! unattended, so a build that would replace the client must not slip through.

use std::path::Path;
use std::time::Duration;

use anyhow::Context;

use crate::app;
use crate::config::DeviceConfig;
use crate::device::DeviceControl;

/// The phone's client is not the one configured after an install: a build signed
/// by another team keeps the app but not its Keychain, so the app minted a new
/// identity. It cannot be re-adopted, and every plan pinning the old id has lost
/// this phone.
#[derive(Debug, thiserror::Error)]
#[error(
    "{checked}, {device} reports client {found} where {expected} was configured: the build \
     replaced the phone's identity (signed by another team?). Stop driving this phone and \
     re-register it before anything claims as {found}"
)]
pub(crate) struct IdentityReplacedError {
    /// When the client was read: `after installing <path>`, or `on build
    /// <version> found running`.
    pub(crate) checked: String,
    pub(crate) device: String,
    pub(crate) expected: String,
    pub(crate) found: String,
}

/// Whether `error` says the phone's identity was replaced, at any depth.
pub(crate) fn replaced_identity(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<IdentityReplacedError>().is_some())
}

/// Before `install_app`: the bundle is signed and carries a profile.
pub(crate) fn check_before(control: &dyn DeviceControl, build: &Path) -> anyhow::Result<()> {
    control
        .verify_installable(build, None)
        .with_context(|| format!("refusing to install {}", build.display()))
}

/// After `install_app`, or on a build this proxy did not see installed: the
/// app still holds the configured client. `checked` says when, as
/// [`IdentityReplacedError::checked`] does.
pub(crate) fn check_after(
    control: &dyn DeviceControl,
    entry: &DeviceConfig,
    checked: &str,
    timeout: Duration,
    first_line: Duration,
) -> anyhow::Result<()> {
    let found = app::local_client_id(control, &entry.bundle_id, timeout, first_line)
        .with_context(|| format!("reading the phone's client {checked}"))?;
    let expected = entry.credentials.client_id.as_ref();
    if found != expected {
        return Err(IdentityReplacedError {
            checked: checked.to_owned(),
            device: control.udid().to_owned(),
            expected: expected.to_owned(),
            found,
        }
        .into());
    }
    Ok(())
}
