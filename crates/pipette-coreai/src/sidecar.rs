//! Resolve / build the bundled Swift sidecar.

use std::{
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use anyhow::Context;

use pipette_plan_types::{AppleCoreAiMacosPipette, RepositoryUrl};

/// Env override for the sidecar binary path (bypasses building).
const SIDECAR_BIN_ENV: &str = "PIPETTE_COREAI_SIDECAR";
/// Env override for the Swift package directory (when the binary is off the
/// build machine and `CARGO_MANIFEST_DIR` no longer points at sources).
const SIDECAR_SWIFT_ENV: &str = "PIPETTE_COREAI_SWIFT";
/// Env override for the cache root (tests; operators who want a non-default
/// location). Default: `~/Library/Caches/pipette-coreai`.
const SIDECAR_CACHE_ENV: &str = "PIPETTE_COREAI_CACHE";
/// The Swift package's executable product, and the file name in the cache.
const SIDECAR_PRODUCT: &str = "pipette-coreai-sidecar";
/// The oldest macOS major version that ships the Core AI framework.
const MIN_MACOS_MAJOR: u32 = 27;
/// Deadline for a host probe (`sw_vers`, `xcrun`). Without developer tools,
/// `xcrun` can open an install dialog; the deadline stops a wait on it.
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// Resolve the Swift sidecar binary.
///
/// Honors `PIPETTE_COREAI_SIDECAR`, a prebuilt path for development that
/// skips the Swift build. Pipette does not check that binary against the
/// bundled stack. Otherwise builds the crate's bundled
/// `swift/` package into a **pin-and-source-keyed** cache directory and
/// copies the product there. A pin bump or a sidecar source change therefore
/// cannot keep executing a stale binary.
pub fn require_coreai_sidecar() -> anyhow::Result<PathBuf> {
    if !macos_supported() {
        anyhow::bail!("Core AI requires macOS {MIN_MACOS_MAJOR} or later");
    }
    if let Some(path) = std::env::var_os(SIDECAR_BIN_ENV) {
        let path = PathBuf::from(path);
        if !path.is_file() {
            anyhow::bail!(
                "{SIDECAR_BIN_ENV}={path} is not a file",
                path = path.display()
            );
        }
        tracing::warn!(
            path = %path.display(),
            "using a prebuilt Core AI sidecar; its stack is not checked"
        );
        return Ok(path);
    }
    let cache = sidecar_cache_dir()?;
    let bin = cache.join(SIDECAR_PRODUCT);
    if bin.is_file() {
        return Ok(bin);
    }
    build_sidecar(&bin)
}

/// The published sidecar for the bundled Swift stack, if the cache has one.
pub fn cached_sidecar() -> Option<PathBuf> {
    let bin = sidecar_cache_root()
        .ok()?
        .join(bundled_stack_key())
        .join(SIDECAR_PRODUCT);
    bin.is_file().then_some(bin)
}

/// Delete the sidecar cache root (every pin). Returns whether anything was
/// removed.
pub fn clear_sidecar_cache() -> anyhow::Result<bool> {
    clear_sidecar_cache_in(&sidecar_cache_root()?)
}

fn clear_sidecar_cache_in(root: &Path) -> anyhow::Result<bool> {
    if !root.exists() {
        return Ok(false);
    }
    std::fs::remove_dir_all(root)
        .with_context(|| format!("removing the Core AI sidecar cache at {}", root.display()))?;
    Ok(true)
}

/// True when this host can run Core AI and get a sidecar. The host needs
/// macOS 27 or later, and one of these:
///
/// - a prebuilt override
/// - a published cache entry
/// - a Swift toolchain plus the bundled Swift package, for a first-use build
///
/// The worker advertises `runtime:core_ai` from this.
pub fn sidecar_obtainable() -> bool {
    if !macos_supported() {
        return false;
    }
    if let Some(path) = std::env::var_os(SIDECAR_BIN_ENV) {
        return PathBuf::from(path).is_file();
    }
    cached_sidecar().is_some() || swift_toolchain_installed() && swift_package_dir().is_ok()
}

/// True when `xcrun` finds a Swift toolchain. `/usr/bin/swift` exists on every
/// macOS install as an `xcrun` shim, so a file on `PATH` proves nothing.
fn swift_toolchain_installed() -> bool {
    let mut command = Command::new("xcrun");
    command.args(["--find", "swift"]);
    pipette_subprocess::output_within(command, PROBE_TIMEOUT)
        .is_ok_and(|output| output.status.success())
}

fn macos_supported() -> bool {
    let mut command = Command::new("sw_vers");
    command.arg("-productVersion");
    pipette_subprocess::output_within(command, PROBE_TIMEOUT)
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|version| macos_major(&version))
        .is_some_and(|major| major >= MIN_MACOS_MAJOR)
}

/// The major version in a `sw_vers -productVersion` string (`27.0.1` → 27).
fn macos_major(product_version: &str) -> Option<u32> {
    product_version.trim().split('.').next()?.parse().ok()
}

/// Cache key for the sidecar: a short digest of two inputs, the bundled Swift
/// stack and the sidecar sources. A change to either input changes the
/// directory name, so the sidecar never runs a stale binary under the same
/// recorded identity.
///
/// Hashed with SHA-256 (not `DefaultHasher`) so the key is stable across
/// Rust releases; a toolchain bump must not silently strand the cache.
/// The sidecar sources this crate was compiled with, by path under the Swift
/// package directory. A source edit rebuilds this crate and moves the key.
const SIDECAR_SOURCES: [(&str, &str); 3] = [
    (
        "Sources/pipette-coreai-sidecar/main.swift",
        include_str!("../swift/Sources/pipette-coreai-sidecar/main.swift"),
    ),
    ("Package.swift", include_str!("../swift/Package.swift")),
    (
        "Package.resolved",
        include_str!("../swift/Package.resolved"),
    ),
];

fn bundled_stack_key() -> String {
    use std::fmt::Write;

    use sha2::{Digest, Sha256};

    let stack = AppleCoreAiMacosPipette::bundled().to_string();
    let mut hasher = Sha256::new();
    hasher.update(stack.as_bytes());
    SIDECAR_SOURCES
        .iter()
        .for_each(|(_, embedded)| hasher.update(embedded.as_bytes()));
    let digest = hasher.finalize();
    let hex = digest
        .iter()
        .take(8)
        .fold(String::with_capacity(16), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        });
    let coreai_models = AppleCoreAiMacosPipette::bundled()
        .packages
        .coreai_models
        .repository_version
        .to_string();
    format!("{coreai_models}-{hex}")
}

fn sidecar_cache_root() -> anyhow::Result<PathBuf> {
    if let Some(path) = std::env::var_os(SIDECAR_CACHE_ENV) {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME").context("cannot resolve $HOME for the sidecar cache")?;
    Ok(PathBuf::from(home).join("Library/Caches/pipette-coreai"))
}

/// `cache_root/<pin>/` — one compiled sidecar per bundled pin.
fn sidecar_cache_dir() -> anyhow::Result<PathBuf> {
    sidecar_cache_dir_in(&sidecar_cache_root()?)
}

fn sidecar_cache_dir_in(root: &Path) -> anyhow::Result<PathBuf> {
    let cache = root.join(bundled_stack_key());
    std::fs::create_dir_all(&cache)
        .with_context(|| format!("failed to create {}", cache.display()))?;
    Ok(cache)
}

fn swift_package_dir() -> anyhow::Result<PathBuf> {
    if let Some(path) = std::env::var_os(SIDECAR_SWIFT_ENV) {
        let path = PathBuf::from(path);
        if path.join("Package.swift").is_file() {
            return Ok(path);
        }
        anyhow::bail!(
            "{SIDECAR_SWIFT_ENV}={} has no Package.swift",
            path.display()
        );
    }
    let bundled = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("swift");
    if bundled.join("Package.swift").is_file() {
        return Ok(bundled);
    }
    anyhow::bail!(
        "the bundled Swift package is missing: this pipette binary runs without \
         its swift/ directory. Set {SIDECAR_BIN_ENV} to a prebuilt \
         `pipette-coreai-sidecar`, or set {SIDECAR_SWIFT_ENV} to the swift/ \
         directory of the crate."
    )
}

#[expect(
    clippy::disallowed_methods,
    reason = "Swift build may take an hour; stderr reports progress and stdout is discarded"
)]
fn build_sidecar(bin: &Path) -> anyhow::Result<PathBuf> {
    let src = swift_package_dir()?;
    verify_sources_match_embedded(&src)?;
    tracing::info!(
        source = %src.display(),
        cache = %bin.display(),
        "building the Core AI sidecar"
    );
    let mut cmd = Command::new("swift");
    cmd.current_dir(&src)
        .arg("build")
        .arg("-c")
        .arg("release")
        .stdout(std::process::Stdio::null());
    pipette_subprocess::echo_info(&cmd);
    let status = cmd
        .status()
        .context("failed to run `swift build` for the pipette-coreai sidecar")?;
    if !status.success() {
        anyhow::bail!("pipette-coreai sidecar `swift build` exited with {status}");
    }

    let show = Command::new("swift")
        .current_dir(&src)
        .args(["build", "--show-bin-path", "-c", "release"])
        .output()
        .context("failed to run `swift build --show-bin-path`")?;
    if !show.status.success() {
        anyhow::bail!(
            "`swift build --show-bin-path -c release` failed: {}",
            String::from_utf8_lossy(&show.stderr)
        );
    }
    let bin_dir = String::from_utf8(show.stdout)
        .context("swift --show-bin-path stdout was not UTF-8")?
        .trim()
        .to_owned();
    let built = PathBuf::from(bin_dir).join(SIDECAR_PRODUCT);
    if !built.is_file() {
        anyhow::bail!("swift build produced no binary at {}", built.display());
    }
    verify_resolved_matches_bundled(&src)?;
    publish_sidecar_atomically(&built, bin)?;
    Ok(bin.to_path_buf())
}

/// Refuse to build from a package directory whose sources differ from the
/// copies this crate embeds: the cache key hashes the embedded copies, so a
/// sidecar built from other sources would run under the bundled identity.
fn verify_sources_match_embedded(src: &Path) -> anyhow::Result<()> {
    SIDECAR_SOURCES.iter().try_for_each(|(relative, embedded)| {
        let path = src.join(relative);
        let on_disk = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        if on_disk != *embedded {
            anyhow::bail!(
                "{} differs from the sidecar source this pipette was built with; \
                 rebuild pipette from the same checkout",
                path.display()
            );
        }
        Ok(())
    })
}

/// Refuse to cache a binary whose `Package.resolved` is not the stack
/// `bundled()` records. A drifted checkout behind `CARGO_MANIFEST_DIR` (or
/// `PIPETTE_COREAI_SWIFT`) would otherwise build under the bundled identity.
fn verify_resolved_matches_bundled(src: &Path) -> anyhow::Result<()> {
    let path = src.join("Package.resolved");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).context("Package.resolved is not JSON")?;
    let pins = value
        .get("pins")
        .context("Package.resolved missing pins")?
        .as_array()
        .context("Package.resolved missing pins")?;
    let bundled = AppleCoreAiMacosPipette::bundled().packages;
    let expected = [
        (
            "coreai-models",
            bundled.coreai_models.repository_version.as_ref(),
            &bundled.coreai_models.repository_url,
        ),
        (
            "swift-transformers",
            bundled.swift_transformers.repository_version.as_ref(),
            &bundled.swift_transformers.repository_url,
        ),
        (
            "xgrammar",
            bundled.xgrammar.repository_version.as_ref(),
            &bundled.xgrammar.repository_url,
        ),
        (
            "swift-jinja",
            bundled.swift_jinja.repository_version.as_ref(),
            &bundled.swift_jinja.repository_url,
        ),
    ];
    expected
        .into_iter()
        .try_for_each(|(identity, version, url)| {
            let pin = pins
                .iter()
                .find(|p| p.get("identity").and_then(serde_json::Value::as_str) == Some(identity))
                .with_context(|| format!("Package.resolved missing pin {identity}"))?;
            let state = pin
                .get("state")
                .with_context(|| format!("Package.resolved pin {identity} missing state"))?;
            let resolved = state
                .get("version")
                .and_then(serde_json::Value::as_str)
                .or_else(|| state.get("revision").and_then(serde_json::Value::as_str))
                .with_context(|| {
                    format!("Package.resolved pin {identity} has no version or revision")
                })?;
            if resolved != version {
                anyhow::bail!(
                    "built sidecar Package.resolved {identity} is {resolved}, \
                 bundled pin is {version}"
                );
            }
            let location = pin
                .get("location")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            if RepositoryUrl::new(location) != *url {
                anyhow::bail!(
                    "built sidecar Package.resolved {identity} location {location} \
                 does not match bundled {url}"
                );
            }
            Ok(())
        })
}

/// Stage beside the live path and rename over it. A partial `fs::copy` onto the
/// live path would leave a truncated binary, and every later run would start it.
fn publish_sidecar_atomically(built: &Path, dest: &Path) -> anyhow::Result<()> {
    let tmp = dest.with_file_name(format!("{SIDECAR_PRODUCT}.{}.tmp", std::process::id()));
    if let Err(err) = std::fs::copy(built, &tmp) {
        let _ = std::fs::remove_file(&tmp);
        return Err(anyhow::Error::new(err).context(format!(
            "failed to stage sidecar from {} to {}",
            built.display(),
            tmp.display()
        )));
    }
    if let Err(err) = std::fs::rename(&tmp, dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(anyhow::Error::new(err)
            .context(format!("failed to publish sidecar into {}", dest.display())));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_is_keyed_on_the_full_stack() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let dir = sidecar_cache_dir_in(tmp.path())?;
        assert!(dir.is_dir());
        assert_eq!(dir, tmp.path().join(bundled_stack_key()));
        // Key changes when any pin in the stack changes: the hash suffix is over
        // the whole `Display`, not only `coreai_models`.
        let key = bundled_stack_key();
        let coreai_only = AppleCoreAiMacosPipette::bundled()
            .packages
            .coreai_models
            .repository_version
            .to_string();
        let prefix = format!("{coreai_only}-");
        assert!(key.starts_with(&prefix));
        let suffix = key
            .get(prefix.len()..)
            .context("invalid stack key boundary")?;
        assert_eq!(suffix.len(), 16, "sha256 prefix is 8 bytes / 16 hex chars");
        assert!(
            suffix.chars().all(|c| c.is_ascii_hexdigit()),
            "hash suffix must be hex, got {suffix}"
        );
        assert_eq!(key, bundled_stack_key(), "key must be stable");
        Ok(())
    }

    #[test]
    fn clearing_the_cache_removes_every_pin() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let root = tmp.path().join("pipette-coreai");
        assert!(
            !clear_sidecar_cache_in(&root)?,
            "a missing cache is a no-op"
        );
        sidecar_cache_dir_in(&root)?;
        assert!(clear_sidecar_cache_in(&root)?);
        assert!(!root.exists());
        Ok(())
    }

    #[rstest::rstest]
    #[case::patch_release("27.0.1\n", Some(27))]
    #[case::minor_release("26.4", Some(26))]
    #[case::major_only("27", Some(27))]
    #[case::empty("", None)]
    #[case::not_a_number("beta", None)]
    fn macos_major_reads_the_first_field(
        #[case] product_version: &str,
        #[case] expected: Option<u32>,
    ) {
        assert_eq!(macos_major(product_version), expected);
    }

    #[test]
    fn the_bundled_package_matches_the_embedded_sources() -> anyhow::Result<()> {
        verify_sources_match_embedded(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("swift"))
    }

    #[test]
    fn an_edited_main_swift_is_refused() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        SIDECAR_SOURCES
            .iter()
            .try_for_each(|(relative, embedded)| -> anyhow::Result<()> {
                let path = tmp.path().join(relative);
                std::fs::create_dir_all(path.parent().context("source path has no parent")?)?;
                std::fs::write(path, embedded)?;
                Ok(())
            })?;
        let [(main_relative, main_embedded), ..] = SIDECAR_SOURCES;
        std::fs::write(
            tmp.path().join(main_relative),
            format!("{main_embedded}\n// edited\n"),
        )?;
        let Err(err) = verify_sources_match_embedded(tmp.path()) else {
            anyhow::bail!("an edited main.swift passed");
        };
        anyhow::ensure!(err.to_string().contains("main.swift"), "{err:#}");
        Ok(())
    }

    #[test]
    fn bundled_package_resolved_matches_identity() -> anyhow::Result<()> {
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("swift");
        verify_resolved_matches_bundled(&src)
    }
}
