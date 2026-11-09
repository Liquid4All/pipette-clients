//! Build the embedded Core AI sidecar into a runtime-store entry.
use std::{path::Path, process::Command};

use anyhow::Context;
use sha2::{Digest, Sha256};

use pipette_plan_types::{AppleCoreAiMacosPipette, RepositoryUrl};

const PRODUCT: &str = "pipette-coreai-sidecar";
/// Deadline for a host probe (`sw_vers`, `xcrun`).
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
const SIDECAR_SOURCES: [(&str, &str); 3] = [
    (
        "Sources/pipette-coreai-sidecar/main.swift",
        include_str!("../../swift/Sources/pipette-coreai-sidecar/main.swift"),
    ),
    ("Package.swift", include_str!("../../swift/Package.swift")),
    (
        "Package.resolved",
        include_str!("../../swift/Package.resolved"),
    ),
];

pub(super) fn source_digest(rt: &AppleCoreAiMacosPipette) -> String {
    digest_sources(rt, &SIDECAR_SOURCES)
}

fn digest_sources(rt: &AppleCoreAiMacosPipette, sources: &[(&str, &str)]) -> String {
    let hash = sources.iter().fold(
        Sha256::new().chain_update(rt.to_string()),
        |hash, (name, contents)| {
            hash.chain_update(name)
                .chain_update([0])
                .chain_update(contents)
                .chain_update([0])
        },
    );
    hex::encode(hash.finalize())
}

pub(crate) fn install(blobs: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        cfg!(all(target_os = "macos", target_arch = "aarch64")),
        "Core AI requires Apple Silicon and macOS 27 or later"
    );
    let mut probe = Command::new("sw_vers");
    probe.arg("-productVersion");
    let version = pipette_subprocess::output_within(probe, PROBE_TIMEOUT)?;
    anyhow::ensure!(
        version.status.success()
            && macos_major(&String::from_utf8_lossy(&version.stdout)).is_some_and(|v| v >= 27),
        "Core AI requires macOS 27 or later"
    );
    // `/usr/bin/swift` exists on every macOS install as an `xcrun` shim, so ask
    // `xcrun` for the toolchain. Without developer tools, `xcrun` can open an
    // install dialog; the deadline stops a wait on it.
    let mut toolchain = Command::new("xcrun");
    toolchain.args(["--find", "swift"]);
    anyhow::ensure!(
        pipette_subprocess::output_within(toolchain, PROBE_TIMEOUT)
            .is_ok_and(|output| output.status.success()),
        "Core AI needs a Swift toolchain (Xcode or the Command Line Tools)"
    );
    build(blobs)
}

/// The major version in `sw_vers -productVersion` output, for example 27 in
/// `27.0.1`.
fn macos_major(product_version: &str) -> Option<u32> {
    product_version.trim().split('.').next()?.parse().ok()
}

#[expect(
    clippy::disallowed_methods,
    reason = "Swift compilation streams progress and may take an hour"
)]
fn build(blobs: &Path) -> anyhow::Result<()> {
    let src = blobs.join("swift-build");
    SIDECAR_SOURCES
        .into_iter()
        .try_for_each(|(relative, contents)| -> anyhow::Result<()> {
            let path = src.join(relative);
            std::fs::create_dir_all(path.parent().context("source has no parent")?)?;
            std::fs::write(&path, contents).with_context(|| format!("writing {}", path.display()))
        })?;
    verify_resolved_matches_bundled(&src)?;
    tracing::info!(destination = %blobs.display(), "building Core AI sidecar");
    let status = Command::new("swift")
        .current_dir(&src)
        .args(["build", "-c", "release", "--disable-automatic-resolution"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .status()
        .context("failed to run swift build for Core AI")?;
    anyhow::ensure!(status.success(), "Core AI swift build failed: {status}");
    let output = Command::new("swift")
        .current_dir(&src)
        .args(["build", "--show-bin-path", "-c", "release"])
        .output()?;
    anyhow::ensure!(output.status.success(), "swift --show-bin-path failed");
    let dir = String::from_utf8(output.stdout)?;
    let built = Path::new(dir.trim()).join(PRODUCT);
    std::fs::copy(&built, blobs.join(PRODUCT))
        .with_context(|| format!("copying {} into the runtime store", built.display()))?;
    std::fs::remove_dir_all(src)?;
    tracing::info!(destination = %blobs.display(), "Core AI sidecar built");
    Ok(())
}

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

#[cfg(test)]
mod tests {
    use pipette_plan_types::{AppleCoreAiMacosPipetteSource, Runtime};

    use super::*;
    use crate::runtime::{RuntimeArtifactStore, RuntimeStorageKey};

    /// An empty store under a temporary directory, and the bundled runtime.
    fn store_with_runtime() -> anyhow::Result<(tempfile::TempDir, RuntimeArtifactStore, Runtime)> {
        let tmp = tempfile::tempdir()?;
        let store = RuntimeArtifactStore::new(tmp.path().join("runtimes"));
        let runtime = Runtime::AppleCoreAiMacosPipette(AppleCoreAiMacosPipette::bundled());
        Ok((tmp, store, runtime))
    }

    /// Stands in for the Swift build: writes a placeholder product.
    fn fake_install(_: &Runtime, blobs: &Path) -> anyhow::Result<()> {
        std::fs::write(blobs.join(PRODUCT), b"binary")?;
        Ok(())
    }

    #[rstest::rstest]
    #[case::patch_release("27.0.1\n", Some(27))]
    #[case::older_release("26.4", Some(26))]
    #[case::major_only("27", Some(27))]
    #[case::empty("", None)]
    #[case::not_a_number("beta", None)]
    fn reads_the_macos_major_version(#[case] output: &str, #[case] expected: Option<u32>) {
        assert_eq!(macos_major(output), expected);
    }

    #[test]
    fn store_lifecycle_is_independent_of_the_host() -> anyhow::Result<()> {
        let (_tmp, store, runtime) = store_with_runtime()?;
        assert!(
            store
                .ensure(&runtime, |_, _| anyhow::bail!("build failed"))
                .is_err()
        );
        assert!(store.find(&runtime)?.is_none());
        let manifest = store.ensure(&runtime, fake_install)?;
        store.ensure(&runtime, |_, _| anyhow::bail!("must reuse installation"))?;
        assert!(manifest.blobs_bytes.is_some_and(|bytes| bytes >= 6));
        assert!(
            manifest
                .resolve_tool(store.runtimes_dir(), PRODUCT)?
                .is_file()
        );
        assert!(
            matches!(manifest.bind_under(store.runtimes_dir())?, Runtime::AppleCoreAiMacosPipette(rt)
            if matches!(rt.source, AppleCoreAiMacosPipetteSource::AbsoluteDir { .. }))
        );
        assert!(store.remove(&runtime)?);
        assert!(store.list()?.is_empty());
        Ok(())
    }

    #[test]
    fn an_older_source_build_is_not_advertised_or_reused() -> anyhow::Result<()> {
        let (_tmp, store, runtime) = store_with_runtime()?;
        let manifest = store.ensure(&runtime, fake_install)?;
        let old_entry = store.runtimes_dir().join("core-ai__older-source-digest");
        std::fs::rename(manifest.entry_dir(store.runtimes_dir())?, &old_entry)?;
        assert!(store.list()?.is_empty());
        assert!(store.list_intact()?.is_empty());
        assert!(store.find(&runtime)?.is_none());
        assert!(matches!(
            super::super::store::read_manifest(&old_entry),
            Err(super::super::store::RuntimeStoreError::Obsolete(_))
        ));
        Ok(())
    }

    #[test]
    fn an_entry_with_a_foreign_swift_stack_is_obsolete() -> anyhow::Result<()> {
        let (_tmp, store, runtime) = store_with_runtime()?;
        let mut manifest = store.ensure(&runtime, fake_install)?;
        let entry = manifest.entry_dir(store.runtimes_dir())?;
        let Runtime::AppleCoreAiMacosPipette(rt) = &mut manifest.declared else {
            anyhow::bail!("expected a Core AI manifest");
        };
        rt.packages.coreai_models.repository_version =
            pipette_plan_types::NonEmptyString::try_new("older-commit".to_owned())?;
        std::fs::write(entry.join("manifest.toml"), toml::to_string(&manifest)?)?;
        assert!(matches!(
            super::super::store::read_manifest(&entry),
            Err(super::super::store::RuntimeStoreError::Obsolete(_))
        ));
        assert!(store.list_intact()?.is_empty());
        Ok(())
    }

    #[test]
    fn concurrent_installers_publish_once() -> anyhow::Result<()> {
        use std::sync::{
            Barrier,
            atomic::{AtomicUsize, Ordering},
        };

        let (_tmp, store, runtime) = store_with_runtime()?;
        let starts = Barrier::new(2);
        let builds = AtomicUsize::new(0);
        let install = || -> anyhow::Result<()> {
            starts.wait();
            store.ensure(&runtime, |rt, blobs| {
                builds.fetch_add(1, Ordering::SeqCst);
                fake_install(rt, blobs)
            })?;
            Ok(())
        };
        std::thread::scope(|scope| -> anyhow::Result<()> {
            let first = scope.spawn(install);
            install()?;
            first
                .join()
                .map_err(|_| anyhow::anyhow!("installer panicked"))??;
            Ok(())
        })?;
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn every_sidecar_source_determines_identity() {
        let rt = AppleCoreAiMacosPipette::bundled();
        let original = source_digest(&rt);
        assert_eq!(original, source_digest(&rt));
        (0..SIDECAR_SOURCES.len()).for_each(|index| {
            let mut changed = SIDECAR_SOURCES;
            changed[index].1 = "changed source";
            assert_ne!(original, digest_sources(&rt, &changed), "source {index}");
        });
    }

    #[rstest::rstest]
    #[case::coreai_models(0)]
    #[case::swift_transformers(1)]
    #[case::xgrammar(2)]
    #[case::swift_jinja(3)]
    fn every_stack_pin_determines_identity(#[case] index: usize) -> anyhow::Result<()> {
        let rt = AppleCoreAiMacosPipette::bundled();
        let mut changed = rt.clone();
        let packages = &mut changed.packages;
        [
            &mut packages.coreai_models,
            &mut packages.swift_transformers,
            &mut packages.xgrammar,
            &mut packages.swift_jinja,
        ]
        .into_iter()
        .nth(index)
        .context("missing package")?
        .repository_version = pipette_plan_types::NonEmptyString::try_new("changed".to_owned())?;
        assert_ne!(source_digest(&rt), source_digest(&changed));
        assert!(RuntimeStorageKey::of(&Runtime::AppleCoreAiMacosPipette(changed)).is_err());
        Ok(())
    }

    #[test]
    fn the_embedded_package_resolved_matches_the_bundled_pins() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        std::fs::write(
            tmp.path().join("Package.resolved"),
            include_str!("../../swift/Package.resolved"),
        )?;
        verify_resolved_matches_bundled(tmp.path())
    }
}
