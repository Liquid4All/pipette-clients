//! Whether a build may go onto a phone, checked on the Mac before `install`: a
//! bundle that fails `codesign --verify --deep --strict` fails `install` with
//! only `CoreDeviceError 3002` to show for it. A build signed here for the
//! fleet must also carry the fleet's team everywhere the phone's identity
//! depends on it (DESIGN.md section 9). The phones are trusted: nothing here asks
//! whether a profile lists the phone or whether the phone runs the iOS a build
//! needs.

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime};

use anyhow::Context;
use serde::Deserialize;

use super::{APP_EXTENSION, IPA_EXTENSION};
use crate::releases::{BuildVersion, Fault, UnreadyError};

/// A cached fleet build's profile must outlive the install by this much, so it
/// is signed again before it stops launching mid-campaign.
pub(crate) const EXPIRY_MARGIN: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// The flavour suffix the app appends to `PipetteBuildVersion`
/// (`BuildFlavor.versionSuffix`).
const INTERNAL_SUFFIX: &str = "-internal";

/// What a build this Mac signed for the fleet must carry.
#[derive(Debug, Clone)]
pub(crate) struct FleetSigned<'a> {
    pub(crate) team_id: &'a str,
    pub(crate) bundle_id: &'a str,
    pub(crate) now: SystemTime,
    /// How long past `now` the profile must stay valid.
    pub(crate) expiry_margin: Duration,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum BundleError {
    #[error("{path} is not validly signed (`codesign --verify --deep --strict`): {detail}")]
    Unsigned { path: String, detail: String },
    #[error("{path} carries no embedded.mobileprovision, so no phone can install it")]
    NoProfile { path: String },
    #[error("{step} {path}: {detail}")]
    Undecodable {
        path: String,
        step: &'static str,
        detail: String,
    },
    #[error("{path} is not an .app, which is all a fleet build is installed as")]
    NotAnApp { path: String },
    /// Another team's build installs cleanly and destroys the phone's identity.
    #[error("{path} carries team {found} in its {place}, where {expected} is required")]
    WrongTeam {
        path: String,
        place: &'static str,
        found: String,
        expected: String,
    },
    #[error("{path}'s profile expires {expires}, too soon to install it")]
    ProfileExpired { path: String, expires: String },
    #[error(
        "{path} lacks com.apple.developer.kernel.increased-memory-limit, so its memory cells \
         would run under iOS's default ceiling (an archive built without the ad-hoc signing \
         of ios/adhoc-sign-archive.sh)"
    )]
    NoMemoryEntitlement { path: String },
    #[error("{path} is bundle {found}, where the phone's bundle_id is {expected}")]
    WrongBundleId {
        path: String,
        found: String,
        expected: String,
    },
    #[error("{path} reports PipetteBuildVersion {found}, published as {expected}")]
    WrongVersion {
        path: String,
        found: String,
        expected: String,
    },
}

impl BundleError {
    /// Who can clear this refusal of a build this Mac just signed (DESIGN.md
    /// section 9).
    pub(crate) fn fault(&self) -> Fault {
        match self {
            Self::WrongTeam { .. } | Self::ProfileExpired { .. } => Fault::Operator,
            Self::NoMemoryEntitlement { .. }
            | Self::WrongBundleId { .. }
            | Self::WrongVersion { .. } => Fault::Version,
            Self::Unsigned { .. }
            | Self::NoProfile { .. }
            | Self::Undecodable { .. }
            | Self::NotAnApp { .. } => Fault::Transient,
        }
    }
}

pub(crate) fn fault_of(error: &anyhow::Error) -> Fault {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<BundleError>())
        .map_or(Fault::Transient, BundleError::fault)
}

/// The fields of a decoded provisioning profile this check reads.
#[derive(Debug, Deserialize, PartialEq)]
pub(crate) struct Profile {
    #[serde(rename = "TeamIdentifier", default)]
    pub(crate) team_identifier: Vec<String>,
    #[serde(rename = "ExpirationDate", default)]
    pub(crate) expiration_date: Option<plist::Date>,
}

impl Profile {
    pub(crate) fn team(&self) -> String {
        self.team_identifier.join(",")
    }
}

/// Refuse a build that `devicectl install` would reject or that would silently
/// mint the phone a new identity. `fleet` is `None` for a pool build.
pub(crate) fn verify_installable(
    build: &Path,
    fleet: Option<&FleetSigned<'_>>,
) -> anyhow::Result<()> {
    match fleet {
        None => verify_pool_build(build),
        Some(fleet) => verify_fleet_build(build, fleet),
    }
}

/// Signed, with a profile. An `.ipa` cannot be checked without unpacking and
/// is let through with a warning; the post-install identity check still
/// applies.
fn verify_pool_build(build: &Path) -> anyhow::Result<()> {
    match build.extension().and_then(|ext| ext.to_str()) {
        Some(APP_EXTENSION) => {}
        Some(IPA_EXTENSION) => {
            tracing::warn!(
                build = %build.display(),
                "an .ipa cannot be verified before install; unpack it to a .app to have its \
                 signature and profile checked"
            );
            return Ok(());
        }
        _ => {}
    }
    codesign_verify(build)?;
    read_profile(build).map(drop)
}

/// Every gate of DESIGN.md section 9, in its order: the cheap ones first.
fn verify_fleet_build(build: &Path, fleet: &FleetSigned<'_>) -> anyhow::Result<()> {
    if build.extension().and_then(|ext| ext.to_str()) != Some(APP_EXTENSION) {
        return Err(BundleError::NotAnApp {
            path: build.display().to_string(),
        }
        .into());
    }
    codesign_verify(build)?;
    check_fleet(
        build,
        &read_profile(build)?,
        &read_signature(build)?,
        &read_info(build)?,
        fleet,
    )
}

fn check_fleet(
    build: &Path,
    profile: &Profile,
    signature: &Signature,
    info: &AppInfo,
    fleet: &FleetSigned<'_>,
) -> anyhow::Result<()> {
    check_profile(build, profile, fleet)?;
    check_signature(build, signature, fleet)?;
    check_info(build, info, fleet.bundle_id)
}

/// The profile's team, and an expiry past the margin; a fresh export's profile
/// inside the cached margin is warned about.
pub(crate) fn check_profile(
    build: &Path,
    profile: &Profile,
    fleet: &FleetSigned<'_>,
) -> anyhow::Result<()> {
    if profile.team_identifier != [fleet.team_id] {
        return Err(BundleError::WrongTeam {
            path: build.display().to_string(),
            place: "profile",
            found: profile.team(),
            expected: fleet.team_id.to_owned(),
        }
        .into());
    }
    let expires = profile.expiration_date.map(SystemTime::from);
    let stated = || {
        profile.expiration_date.map_or_else(
            || "at no stated date".to_owned(),
            |date| date.to_xml_format(),
        )
    };
    if expires.is_none_or(|expires| expires <= fleet.now + fleet.expiry_margin) {
        return Err(BundleError::ProfileExpired {
            path: build.display().to_string(),
            expires: stated(),
        }
        .into());
    }
    if expires.is_some_and(|expires| expires <= fleet.now + EXPIRY_MARGIN) {
        tracing::warn!(
            expires = %stated(),
            "the fleet profile expires within a week; renew the signing certificate"
        );
    }
    Ok(())
}

/// What `codesign -d -v --entitlements - --xml` says about a bundle.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Signature {
    /// `TeamIdentifier=` on stderr; `None` for `not set`, as an ad-hoc
    /// signature has.
    pub(crate) team: Option<String>,
    pub(crate) entitlements: Entitlements,
}

/// The entitlements the gates read, from the XML on stdout.
#[derive(Debug, Default, PartialEq, Deserialize)]
pub(crate) struct Entitlements {
    #[serde(rename = "application-identifier")]
    pub(crate) application_identifier: Option<String>,
    #[serde(rename = "com.apple.developer.team-identifier")]
    pub(crate) team_identifier: Option<String>,
    #[serde(rename = "com.apple.developer.kernel.increased-memory-limit")]
    pub(crate) increased_memory_limit: Option<bool>,
}

/// An unsigned bundle has neither a team nor entitlements, which the gates
/// then refuse by name.
fn read_signature(build: &Path) -> anyhow::Result<Signature> {
    let output = Command::new("codesign")
        .args(["-d", "-v", "--entitlements", "-", "--xml"])
        .arg(build)
        .stdin(Stdio::null())
        .output()
        .context("running `codesign -d`")?;
    if !output.status.success() {
        return Ok(Signature::default());
    }
    parse_signature(&String::from_utf8_lossy(&output.stderr), &output.stdout)
        .with_context(|| format!("reading the signature of {}", build.display()))
}

pub(crate) fn parse_signature(stderr: &str, entitlements: &[u8]) -> anyhow::Result<Signature> {
    let team = stderr
        .lines()
        .find_map(|line| line.trim().strip_prefix("TeamIdentifier="))
        .map(str::trim)
        .filter(|team| !team.is_empty() && *team != "not set")
        .map(str::to_owned);
    let entitlements = if entitlements.iter().all(u8::is_ascii_whitespace) {
        Entitlements::default()
    } else {
        plist::from_bytes(entitlements).context("the entitlements are not the expected plist")?
    };
    Ok(Signature { team, entitlements })
}

/// The team in the signature and in the entitlements the phone's keychain
/// item is scoped by, and the memory entitlement.
pub(crate) fn check_signature(
    build: &Path,
    signature: &Signature,
    fleet: &FleetSigned<'_>,
) -> anyhow::Result<()> {
    let wrong = |place: &'static str, found: Option<&str>| BundleError::WrongTeam {
        path: build.display().to_string(),
        place,
        found: found.unwrap_or("none").to_owned(),
        expected: fleet.team_id.to_owned(),
    };
    if signature.team.as_deref() != Some(fleet.team_id) {
        return Err(wrong("signature", signature.team.as_deref()).into());
    }
    let entitlements = &signature.entitlements;
    let application_identifier = format!("{}.{}", fleet.team_id, fleet.bundle_id);
    if entitlements.application_identifier.as_deref() != Some(application_identifier.as_str()) {
        return Err(wrong(
            "application-identifier entitlement",
            entitlements.application_identifier.as_deref(),
        )
        .into());
    }
    if entitlements.team_identifier.as_deref() != Some(fleet.team_id) {
        return Err(wrong(
            "team-identifier entitlement",
            entitlements.team_identifier.as_deref(),
        )
        .into());
    }
    check_memory_entitlement(build, entitlements)
}

fn check_memory_entitlement(build: &Path, entitlements: &Entitlements) -> anyhow::Result<()> {
    if entitlements.increased_memory_limit == Some(true) {
        return Ok(());
    }
    Err(BundleError::NoMemoryEntitlement {
        path: build.display().to_string(),
    }
    .into())
}

/// The `Info.plist` keys the gates read.
#[derive(Debug, Default, PartialEq, Deserialize)]
pub(crate) struct AppInfo {
    #[serde(rename = "CFBundleIdentifier")]
    pub(crate) bundle_id: Option<String>,
    /// What the app reports as its version, before the flavour suffix.
    #[serde(rename = "PipetteBuildVersion")]
    pub(crate) build_version: Option<String>,
}

fn read_info(build: &Path) -> anyhow::Result<AppInfo> {
    let path = build.join("Info.plist");
    plist::from_file(&path).map_err(|error| {
        BundleError::Undecodable {
            path: path.display().to_string(),
            step: "reading",
            detail: error.to_string(),
        }
        .into()
    })
}

/// The bundle id. `MinimumOSVersion` is not judged: a plan that needs an iOS
/// requires `os_version:<v>`, which the server matches.
pub(crate) fn check_info(build: &Path, info: &AppInfo, bundle_id: &str) -> anyhow::Result<()> {
    if info.bundle_id.as_deref() == Some(bundle_id) {
        return Ok(());
    }
    Err(BundleError::WrongBundleId {
        path: build.display().to_string(),
        found: info.bundle_id.clone().unwrap_or_else(|| "none".to_owned()),
        expected: bundle_id.to_owned(),
    }
    .into())
}

/// The checks that need no export, on the archived app (DESIGN.md section 9): the
/// bundle id, the version it reports, and the memory entitlement its ad-hoc
/// signature carries. Every refusal is the version's.
pub(crate) fn check_archived(
    app: &Path,
    version: &BuildVersion,
    bundle_id: &str,
) -> Result<(), UnreadyError> {
    let refused = |error: anyhow::Error| {
        let fault = if error.chain().any(|cause| cause.is::<BundleError>()) {
            Fault::Version
        } else {
            Fault::Transient
        };
        UnreadyError::new(
            fault,
            error.context("checking the archived app before signing it"),
        )
    };
    let info = read_info(app).map_err(refused)?;
    check_info(app, &info, bundle_id).map_err(refused)?;
    check_build_version(app, &info, version).map_err(refused)?;
    let signature = read_signature(app).map_err(refused)?;
    check_memory_entitlement(app, &signature.entitlements).map_err(refused)
}

/// `PipetteBuildVersion`, with or without the flavour suffix, is `version`.
pub(crate) fn check_build_version(
    app: &Path,
    info: &AppInfo,
    version: &BuildVersion,
) -> anyhow::Result<()> {
    // Trimmed, as the app reads it (`Bundle.normalizedInfoString`).
    let found = info.build_version.as_deref().map_or("none", str::trim);
    if found == version.as_str() || format!("{found}{INTERNAL_SUFFIX}") == version.as_str() {
        return Ok(());
    }
    Err(BundleError::WrongVersion {
        path: app.display().to_string(),
        found: found.to_owned(),
        expected: version.to_string(),
    }
    .into())
}

fn codesign_verify(build: &Path) -> anyhow::Result<()> {
    let output = Command::new("codesign")
        .args(["--verify", "--deep", "--strict"])
        .arg(build)
        .stdin(Stdio::null())
        .output()
        .context("running `codesign`")?;
    if output.status.success() {
        return Ok(());
    }
    Err(BundleError::Unsigned {
        path: build.display().to_string(),
        detail: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
    }
    .into())
}

/// `security cms -D` decodes the CMS envelope to an XML plist, read as is: a
/// profile's dates and certificates have no JSON form, so `plutil -convert json`
/// refuses every real one.
fn read_profile(build: &Path) -> anyhow::Result<Profile> {
    let profile_path = build.join("embedded.mobileprovision");
    if !profile_path.is_file() {
        return Err(BundleError::NoProfile {
            path: build.display().to_string(),
        }
        .into());
    }
    let decoded = Command::new("security")
        .args(["cms", "-D", "-i"])
        .arg(&profile_path)
        .stdin(Stdio::null())
        .output()
        .context("running `security cms -D`")?;
    if !decoded.status.success() {
        return Err(BundleError::Undecodable {
            path: profile_path.display().to_string(),
            step: "decoding",
            detail: String::from_utf8_lossy(&decoded.stderr).trim().to_owned(),
        }
        .into());
    }
    parse_profile(&decoded.stdout).with_context(|| format!("reading {}", profile_path.display()))
}

pub(crate) fn parse_profile(plist: &[u8]) -> anyhow::Result<Profile> {
    plist::from_bytes(plist).context("the profile is not the expected plist")
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;

    /// `got` is `want`: a pass, or a refusal naming the needle under the fault.
    fn check(got: anyhow::Result<()>, want: Result<(), (&str, Fault)>) -> anyhow::Result<()> {
        match (got, want) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Err((needle, fault))) => {
                assert!(format!("{error:#}").contains(needle), "{error:#}");
                assert_eq!(fault_of(&error), fault);
                Ok(())
            }
            (got, expected) => anyhow::bail!("got {got:?}, expected {expected:?}"),
        }
    }

    /// Shaped like `security cms -D` output for a real profile. The date and data
    /// values are what `plutil -convert json` refused.
    const PROFILE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>AppIDName</key>
	<string>Pipette</string>
	<key>CreationDate</key>
	<date>2026-09-25T23:10:56Z</date>
	<key>DeveloperCertificates</key>
	<array>
		<data>MIIFwzCCBKugAwIBAgIQ</data>
	</array>
	<key>ExpirationDate</key>
	<date>2027-09-25T23:10:56Z</date>
	<key>ProvisionedDevices</key>
	<array>
		<string>00008150-0014155C3600C01C</string>
		<string>00008140-000A2B3C4D5E6F70</string>
	</array>
	<key>TeamIdentifier</key>
	<array>
		<string>353N5JSDGZ</string>
	</array>
	<key>Version</key>
	<integer>1</integer>
</dict>
</plist>"#;

    #[rstest::rstest]
    #[case::fleet_profile(PROFILE, TEAM)]
    #[case::bare(
        r#"<plist version="1.0"><dict><key>TeamIdentifier</key><array><string>T</string></array></dict></plist>"#,
        "T"
    )]
    fn a_profile_names_its_team(#[case] profile: &str, #[case] team: &str) -> anyhow::Result<()> {
        assert_eq!(parse_profile(profile.as_bytes())?.team(), team);
        Ok(())
    }

    /// `codesign` runs before the profile is looked for, and refuses an unsigned
    /// directory on any Mac.
    #[test]
    fn an_unsigned_app_is_refused_by_codesign() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let app = dir.path().join("Pipette.app");
        std::fs::create_dir(&app)?;
        let error = verify_installable(&app, None)
            .err()
            .ok_or_else(|| anyhow::anyhow!("an unsigned bundle must be refused"))?;
        assert!(
            matches!(
                error.downcast_ref::<BundleError>(),
                Some(BundleError::Unsigned { .. })
            ),
            "{error:#}"
        );
        Ok(())
    }

    #[test]
    fn an_ipa_is_not_checked() -> anyhow::Result<()> {
        verify_installable(Path::new("/pool/Pipette.ipa"), None)
    }

    const TEAM: &str = "353N5JSDGZ";
    const BUNDLE: &str = "ai.liquid.liquid-pipette";

    fn fleet(now_secs: u64, expiry_margin: Duration) -> FleetSigned<'static> {
        FleetSigned {
            team_id: TEAM,
            bundle_id: BUNDLE,
            now: SystemTime::UNIX_EPOCH + Duration::from_secs(now_secs),
            expiry_margin,
        }
    }

    /// 2026-10-03, a year before [`PROFILE`] expires.
    const OCT_2026: u64 = 1_791_000_000;
    /// 2027-09-21, four days before [`PROFILE`] expires.
    const SEP_2027: u64 = 1_821_484_800;
    /// 2027-10-03, a week after [`PROFILE`] expired.
    const OCT_2027: u64 = 1_822_536_000;

    /// The profile's team and its expiry: a cached copy must outlive the
    /// margin, a fresh one only must not have expired.
    #[rstest::rstest]
    #[case::fleet_profile(PROFILE.to_owned(), OCT_2026, EXPIRY_MARGIN, Ok(()))]
    #[case::other_team(
        PROFILE.replace(TEAM, "ABCDE12345"),
        OCT_2026,
        Duration::ZERO,
        Err("team ABCDE12345 in its profile")
    )]
    #[case::cached_within_a_week(
        PROFILE.to_owned(),
        SEP_2027,
        EXPIRY_MARGIN,
        Err("expires 2027-09-25T23:10:56Z")
    )]
    #[case::fresh_within_a_week(PROFILE.to_owned(), SEP_2027, Duration::ZERO, Ok(()))]
    #[case::fresh_expired(
        PROFILE.to_owned(),
        OCT_2027,
        Duration::ZERO,
        Err("expires 2027-09-25T23:10:56Z")
    )]
    #[case::no_expiry(
        PROFILE.replace("<key>ExpirationDate</key>\n\t<date>2027-09-25T23:10:56Z</date>", ""),
        OCT_2026,
        Duration::ZERO,
        Err("at no stated date")
    )]
    fn a_fleet_profile_needs_the_team_and_time_to_live(
        #[case] text: String,
        #[case] now: u64,
        #[case] margin: Duration,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        let profile = parse_profile(text.as_bytes())?;
        check(
            check_profile(Path::new("/s/Pipette.app"), &profile, &fleet(now, margin)),
            expected.map_err(|needle| (needle, Fault::Operator)),
        )
    }

    /// What `codesign -d -v --entitlements - --xml` prints for an export: the
    /// team on stderr, the entitlements on stdout.
    const EXPORT_STDERR: &str = "Executable=/s/Pipette.app/Pipette\n\
         Identifier=ai.liquid.liquid-pipette\n\
         Format=app bundle with Mach-O thin (arm64)\n\
         CodeDirectory v=20500 size=1 flags=0x0(none) hashes=1+7 location=embedded\n\
         Signature size=4797\n\
         Info.plist entries=40\n\
         TeamIdentifier=353N5JSDGZ\n\
         Sealed Resources version=2 rules=13 files=10\n";

    fn entitlements_xml(entries: &str) -> String {
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><!DOCTYPE plist PUBLIC \
             \"-//Apple//DTD PLIST 1.0//EN\" \"https://www.apple.com/DTDs/PropertyList-1.0.dtd\">\
             <plist version=\"1.0\"><dict>{entries}</dict></plist>"
        )
    }

    const FLEET_ENTITLEMENTS: &str = "<key>application-identifier</key>\
         <string>353N5JSDGZ.ai.liquid.liquid-pipette</string>\
         <key>com.apple.developer.kernel.increased-memory-limit</key><true/>\
         <key>com.apple.developer.team-identifier</key><string>353N5JSDGZ</string>\
         <key>get-task-allow</key><true/>";

    const UDID: &str = "00008150-0014155C3600C01C";

    /// Every gate over what a fleet export carries, each case changing one
    /// thing, with the fault a fresh export's refusal is filed under. Every
    /// place the team shows is checked, so another team's build fails closed.
    /// Which phones the profile lists is not: the phones are trusted.
    #[rstest::rstest]
    #[case::fleet_export(
        PROFILE.to_owned(),
        EXPORT_STDERR.to_owned(),
        FLEET_ENTITLEMENTS.to_owned(),
        BUNDLE,
        Ok(())
    )]
    #[case::profile_listing_other_phones(
        PROFILE.replace(UDID, "00008150-DEADBEEF00000000"),
        EXPORT_STDERR.to_owned(),
        FLEET_ENTITLEMENTS.to_owned(),
        BUNDLE,
        Ok(())
    )]
    #[case::profile_of_another_team(
        PROFILE.replace(TEAM, "ABCDE12345"),
        EXPORT_STDERR.to_owned(),
        FLEET_ENTITLEMENTS.to_owned(),
        BUNDLE,
        Err(("team ABCDE12345 in its profile", Fault::Operator))
    )]
    #[case::signed_by_another_team(
        PROFILE.to_owned(),
        EXPORT_STDERR.replace("TeamIdentifier=353N5JSDGZ", "TeamIdentifier=ABCDE12345"),
        FLEET_ENTITLEMENTS.to_owned(),
        BUNDLE,
        Err(("team ABCDE12345 in its signature", Fault::Operator))
    )]
    #[case::ad_hoc(
        PROFILE.to_owned(),
        EXPORT_STDERR.replace("TeamIdentifier=353N5JSDGZ", "TeamIdentifier=not set"),
        FLEET_ENTITLEMENTS.to_owned(),
        BUNDLE,
        Err(("team none in its signature", Fault::Operator))
    )]
    #[case::other_application_identifier(
        PROFILE.to_owned(),
        EXPORT_STDERR.to_owned(),
        FLEET_ENTITLEMENTS.replace("353N5JSDGZ.ai", "ABCDE12345.ai"),
        BUNDLE,
        Err(("application-identifier entitlement", Fault::Operator))
    )]
    #[case::other_team_identifier(
        PROFILE.to_owned(),
        EXPORT_STDERR.to_owned(),
        FLEET_ENTITLEMENTS.replace(
            "<string>353N5JSDGZ</string>",
            "<string>ABCDE12345</string>"
        ),
        BUNDLE,
        Err(("team-identifier entitlement", Fault::Operator))
    )]
    #[case::no_memory_limit(
        PROFILE.to_owned(),
        EXPORT_STDERR.to_owned(),
        FLEET_ENTITLEMENTS.replace(
            "<key>com.apple.developer.kernel.increased-memory-limit</key><true/>",
            ""
        ),
        BUNDLE,
        Err(("increased-memory-limit", Fault::Version))
    )]
    #[case::memory_limit_off(
        PROFILE.to_owned(),
        EXPORT_STDERR.to_owned(),
        FLEET_ENTITLEMENTS.replace("increased-memory-limit</key><true/>", "increased-memory-limit</key><false/>"),
        BUNDLE,
        Err(("increased-memory-limit", Fault::Version))
    )]
    #[case::other_bundle(
        PROFILE.to_owned(),
        EXPORT_STDERR.to_owned(),
        FLEET_ENTITLEMENTS.to_owned(),
        "ai.liquid.other",
        Err(("bundle ai.liquid.other", Fault::Version))
    )]
    fn the_fleet_gates_fail_closed(
        #[case] profile: String,
        #[case] stderr: String,
        #[case] entitlements: String,
        #[case] bundle_id: &str,
        #[case] expected: Result<(), (&str, Fault)>,
    ) -> anyhow::Result<()> {
        let info = AppInfo {
            bundle_id: Some(bundle_id.to_owned()),
            build_version: None,
        };
        let outcome = check_fleet(
            Path::new("/s/Pipette.app"),
            &parse_profile(profile.as_bytes())?,
            &parse_signature(&stderr, entitlements_xml(&entitlements).as_bytes())?,
            &info,
            &fleet(OCT_2026, EXPIRY_MARGIN),
        );
        check(outcome, expected)
    }

    #[test]
    fn an_unsigned_bundle_has_no_team_and_no_entitlements() -> anyhow::Result<()> {
        assert_eq!(parse_signature("", b"")?, Signature::default());
        Ok(())
    }

    /// Whatever `MinimumOSVersion` says, only the bundle id is judged.
    #[rstest::rstest]
    #[case::the_phones_bundle(Some(BUNDLE), Ok(()))]
    #[case::other_bundle(Some("ai.liquid.other"), Err("bundle ai.liquid.other"))]
    #[case::no_bundle(None, Err("bundle none"))]
    fn the_app_must_be_the_phones_bundle(
        #[case] bundle_id: Option<&str>,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let app = dir.path().join("Pipette.app");
        std::fs::create_dir(&app)?;
        let bundle_key = bundle_id.map_or_else(String::new, |id| {
            format!("<key>CFBundleIdentifier</key><string>{id}</string>")
        });
        std::fs::write(
            app.join("Info.plist"),
            format!(
                "<plist version=\"1.0\"><dict>{bundle_key}\
                 <key>MinimumOSVersion</key><string>99.0</string></dict></plist>"
            ),
        )?;
        check(
            check_info(&app, &read_info(&app)?, BUNDLE),
            expected.map_err(|needle| (needle, Fault::Version)),
        )
    }

    #[rstest::rstest]
    #[case::internal(Some("2026.10.02-e6a9434"), "2026.10.02-e6a9434-internal", Ok(()))]
    #[case::no_suffix(Some("feature-x.1"), "feature-x.1", Ok(()))]
    #[case::padded(Some(" 2026.10.02-e6a9434\n"), "2026.10.02-e6a9434-internal", Ok(()))]
    #[case::other(
        Some("2026.10.01-0000000"),
        "2026.10.02-e6a9434-internal",
        Err("reports PipetteBuildVersion 2026.10.01-0000000")
    )]
    #[case::local_build(
        None,
        "2026.10.02-e6a9434-internal",
        Err("reports PipetteBuildVersion none")
    )]
    fn the_archive_reports_the_published_version(
        #[case] build_version: Option<&str>,
        #[case] published: &str,
        #[case] expected: Result<(), &str>,
    ) -> anyhow::Result<()> {
        let info = AppInfo {
            build_version: build_version.map(str::to_owned),
            ..AppInfo::default()
        };
        check(
            check_build_version(Path::new("/a/Pipette.app"), &info, &published.parse()?),
            expected.map_err(|needle| (needle, Fault::Version)),
        )
    }

    #[test]
    fn a_fleet_build_must_be_an_app() -> anyhow::Result<()> {
        let error = verify_installable(
            Path::new("/s/Pipette.ipa"),
            Some(&fleet(OCT_2026, EXPIRY_MARGIN)),
        )
        .err()
        .context("the build must be refused")?;
        assert!(format!("{error:#}").contains("is not an .app"), "{error:#}");
        Ok(())
    }
}
