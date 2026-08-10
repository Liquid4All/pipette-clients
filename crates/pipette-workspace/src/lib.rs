use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use serde::{Deserialize, Serialize};

const JSON_MANIFEST_FILE: &str = "manifest.json";
const TOML_MANIFEST_FILE: &str = "manifest.toml";

#[derive(Debug)]
pub enum InitResult {
    /// Fresh init — manifest was created.
    Created(PathBuf),
    /// Manifest already existed — subdirs were ensured but manifest untouched.
    AlreadyExists(PathBuf),
}

/// On-disk workspace layout version. Bump when the directory structure, file
/// formats, or key schemes change so `open` can detect (and reject or migrate)
/// incompatible workspaces. Distinct from the binary/release version.
const WORKSPACE_VERSION: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    /// Layout schema version (see [`WORKSPACE_VERSION`]). Manifests written
    /// before versioning omit it — they are the v1 layout, so default to 1.
    #[serde(default = "default_manifest_version")]
    version: u32,
    created_at: String,
}

fn default_manifest_version() -> u32 {
    1
}

/// A validated handle to an initialized workspace.
///
/// The base workspace is generic — it knows nothing about benchmarks,
/// identity, runtimes, or plans. It provides lifecycle (`init`/`open`)
/// and path resolution (`root`). Domain-specific paths and data access
/// belong on per-CLI wrapper types that encapsulate this.
#[derive(Debug)]
pub struct Workspace {
    root: PathBuf,
}

impl Workspace {
    /// Create the workspace under `work_dir/.<marker_name>` (idempotent,
    /// repair-safe). Ensures the dir tree exists and writes `manifest.toml`
    /// (layout v1) only if no manifest is present. A legacy `manifest.json`
    /// is migrated to TOML in place rather than duplicated.
    pub fn init(
        work_dir: &Path,
        marker_name: &str,
        dirs: impl IntoIterator<Item = PathBuf>,
    ) -> anyhow::Result<InitResult> {
        let root = storage_root(work_dir, marker_name);

        fs::create_dir_all(&root)
            .with_context(|| format!("failed to create {}", root.display()))?;
        dirs.into_iter().try_for_each(|dir| {
            fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))
        })?;

        let manifest_path = root.join(TOML_MANIFEST_FILE);
        if manifest_path.exists() {
            read_manifest(&manifest_path).with_context(|| {
                format!(
                    "manifest is corrupt at {}; delete it and re-run init",
                    manifest_path.display()
                )
            })?;
            return Ok(InitResult::AlreadyExists(root));
        }
        if migrate_legacy_manifest(&root)? {
            return Ok(InitResult::AlreadyExists(root));
        }

        write_manifest(&manifest_path)?;
        Ok(InitResult::Created(root))
    }

    /// Open an existing workspace, migrating a legacy `manifest.json` to
    /// `manifest.toml` first. Validates the manifest is readable and that its
    /// layout version is one this build understands.
    ///
    /// Migrating a legacy workspace *writes* `manifest.toml`, so opening one
    /// requires a writable workspace directory; a read-only legacy workspace
    /// fails here rather than opening read-only.
    pub fn open(work_dir: &Path, marker_name: &str) -> anyhow::Result<Self> {
        let root = storage_root(work_dir, marker_name);
        if !work_dir.exists() {
            anyhow::bail!("working directory does not exist: {}", work_dir.display());
        }
        migrate_legacy_manifest(&root)?;

        let manifest_path = root.join(TOML_MANIFEST_FILE);
        if !manifest_path.exists() {
            anyhow::bail!(
                "not initialized; run '{marker_name} init' first\n\
                 expected manifest at {}",
                manifest_path.display()
            );
        }
        let manifest = read_manifest(&manifest_path).with_context(|| {
            format!(
                "manifest is corrupt at {}; run '{marker_name} init' to repair",
                manifest_path.display()
            )
        })?;
        if manifest.version > WORKSPACE_VERSION {
            anyhow::bail!(
                "workspace at {path} needs a newer client \
                 (layout v{ver}, this build supports v{supported})",
                path = manifest_path.display(),
                ver = manifest.version,
                supported = WORKSPACE_VERSION,
            );
        }
        Ok(Self { root })
    }

    /// The storage root path (the `.{marker_name}/` directory).
    pub fn root(&self) -> &Path {
        &self.root
    }
}

/// Write a fresh `manifest.toml` at the current layout version.
fn write_manifest(manifest_path: &Path) -> anyhow::Result<()> {
    let manifest = Manifest {
        version: WORKSPACE_VERSION,
        created_at: time::OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)
            .context("failed to format timestamp")?,
    };
    let encoded = toml::to_string_pretty(&manifest).context("failed to serialize manifest")?;
    fs::write(manifest_path, encoded)
        .with_context(|| format!("failed to write {}", manifest_path.display()))
}

/// Read and parse the TOML `manifest.toml`.
fn read_manifest(manifest_path: &Path) -> anyhow::Result<Manifest> {
    let raw = fs::read_to_string(manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    toml::from_str(&raw).context("failed to parse TOML")
}

/// Migrate a legacy `manifest.json` to `manifest.toml` (layout v1), preserving
/// `created_at`. No-op when the TOML manifest already exists or no legacy
/// manifest is present. Returns whether a migration was performed.
fn migrate_legacy_manifest(root: &Path) -> anyhow::Result<bool> {
    let toml_path = root.join(TOML_MANIFEST_FILE);
    let json_path = root.join(JSON_MANIFEST_FILE);
    if toml_path.exists() || !json_path.exists() {
        return Ok(false);
    }
    let raw = fs::read_to_string(&json_path)
        .with_context(|| format!("failed to read {}", json_path.display()))?;
    let legacy: Manifest = serde_json::from_str(&raw).with_context(|| {
        format!(
            "legacy JSON manifest is corrupt at {}; delete it and re-run init",
            json_path.display()
        )
    })?;
    let migrated = Manifest {
        version: WORKSPACE_VERSION,
        created_at: legacy.created_at,
    };
    let encoded = toml::to_string_pretty(&migrated).context("failed to serialize manifest")?;
    fs::write(&toml_path, encoded)
        .with_context(|| format!("failed to write {}", toml_path.display()))?;
    let _ = fs::remove_file(&json_path);
    Ok(true)
}

/// Return `<work_dir>/.<marker_name>`.
pub fn storage_root(work_dir: &Path, marker_name: &str) -> PathBuf {
    work_dir.join(format!(".{marker_name}"))
}

/// Whether an initialized workspace exists under `<work_dir>/.<marker_name>` —
/// a `manifest.toml`, or a legacy `manifest.json` that [`Workspace::open`]
/// migrates. A cheap existence check; it neither reads nor parses the manifest.
pub fn is_initialized(work_dir: &Path, marker_name: &str) -> bool {
    let root = storage_root(work_dir, marker_name);
    root.join(TOML_MANIFEST_FILE).exists() || root.join(JSON_MANIFEST_FILE).exists()
}

/// Resolve the working directory: the caller's `--work-dir` value (which clap
/// fills from `PIPETTE_WORK_DIR` when the flag is absent) if set, else the
/// current directory. The crate reads no environment itself.
pub fn resolve_work_dir(work_dir_arg: Option<&Path>) -> anyhow::Result<PathBuf> {
    match work_dir_arg {
        Some(path) => Ok(path.to_path_buf()),
        None => std::env::current_dir().context("failed to determine current directory"),
    }
}

/// Ensure `work_dir` exists and holds an initialized `.<name>` workspace, or
/// return an actionable error. A missing directory keeps the clear "does not
/// exist" diagnostic (so a typo isn't mistaken for a missing workspace); a
/// present-but-uninitialized directory yields a hint showing how to `init` at
/// each place the resolver honors (`--work-dir` > `PIPETTE_WORK_DIR` > current
/// dir).
///
/// `name` is the client's marker/binary name (e.g. `"pipette-llamacpp"`) — used
/// both to locate the workspace and to render the suggested commands.
pub fn require_workspace(
    work_dir: &Path,
    work_dir_arg: Option<&Path>,
    name: &str,
) -> anyhow::Result<()> {
    if !work_dir.exists() {
        anyhow::bail!("working directory does not exist: {}", work_dir.display());
    }
    if is_initialized(work_dir, name) {
        return Ok(());
    }
    let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    anyhow::bail!(
        "{}",
        not_initialized_hint(
            &storage_root(work_dir, name),
            work_dir_arg,
            &current_dir,
            name
        )
    )
}

/// Message for a workspace-dependent command when no workspace exists at the
/// resolved location. States where it looked, then how to create one.
///
/// `work_dir_arg` is the resolved `--work-dir` / `PIPETTE_WORK_DIR` value (or
/// `None` when neither is set). When set, only the explicit `--work-dir` form
/// is offered — a bare `init` could resolve through the env var — plus the
/// current directory as a second option (collapsed to one line when the target
/// already is the current directory). When unset, a bare `<name> init` targets
/// the current directory and another dir is described in prose so a `<path>`
/// placeholder is never mistaken for a real path.
fn not_initialized_hint(
    resolved_root: &Path,
    work_dir_arg: Option<&Path>,
    current_dir: &Path,
    name: &str,
) -> String {
    let header = format!("no {name} workspace at {}", resolved_root.display());
    let sp = "  ";
    match work_dir_arg {
        None => format!(
            "{header}\n\
             initialize one with:\n\
             {sp}{name} init  (the current directory: {cwd})\n\
             {sp}or: {name} --work-dir <path> init  (another directory)",
            cwd = current_dir.display(),
        ),
        Some(dir) if dir == current_dir => format!(
            "{header}\n\
             initialize one with:\n\
             {sp}{name} --work-dir {arg} init  (the current directory: {cwd})",
            arg = shell_quote(dir),
            cwd = current_dir.display(),
        ),
        Some(dir) => format!(
            "{header}\n\
             initialize one with either:\n\
             {sp}{name} --work-dir {arg} init  (a specific work dir)\n\
             {sp}{name} --work-dir {cwd_arg} init  (the current directory: {cwd})",
            arg = shell_quote(dir),
            cwd_arg = shell_quote(current_dir),
            cwd = current_dir.display(),
        ),
    }
}

/// Single-quote a path for safe shell copy-paste (POSIX-style).
fn shell_quote(path: &Path) -> String {
    format!("'{}'", path.to_string_lossy().replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "pipette-store-workspace-{label}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    }

    #[test]
    fn storage_root_joins_dot_prefix() {
        let root = storage_root(Path::new("/tmp/work"), "pipette-llamacpp");
        assert_eq!(root, PathBuf::from("/tmp/work/.pipette-llamacpp"));
    }

    #[test]
    fn init_creates_dirs_and_manifest() -> anyhow::Result<()> {
        let work = temp_dir("create");
        let root = storage_root(&work, "test-tool");
        let dirs = [root.join("identity"), root.join("results").join("local")];
        let result = Workspace::init(&work, "test-tool", dirs)?;
        let root = match &result {
            InitResult::Created(p) => p.clone(),
            InitResult::AlreadyExists(_) => anyhow::bail!("expected Created"),
        };
        let manifest = root.join("manifest.toml");
        assert!(manifest.exists());
        assert!(!root.join("manifest.json").exists());
        let raw = fs::read_to_string(manifest)?;
        assert!(raw.contains("created_at"));
        assert!(raw.contains("version"));
        assert!(root.join("identity").is_dir());
        assert!(root.join("results/local").is_dir());
        Workspace::open(&work, "test-tool")?;

        let _ = fs::remove_dir_all(&work);
        Ok(())
    }

    #[test]
    fn init_is_idempotent() -> anyhow::Result<()> {
        let work = temp_dir("idempotent");
        let root = storage_root(&work, "test-tool");
        Workspace::init(&work, "test-tool", [root.join("plans")])?;

        let manifest_before = fs::read_to_string(root.join("manifest.toml"))?;

        let result = Workspace::init(&work, "test-tool", [root.join("plans")])?;
        assert!(matches!(result, InitResult::AlreadyExists(_)));

        let manifest_after = fs::read_to_string(root.join("manifest.toml"))?;
        assert_eq!(manifest_before, manifest_after);

        let _ = fs::remove_dir_all(&work);
        Ok(())
    }

    #[test]
    fn init_repairs_missing_subdirs() -> anyhow::Result<()> {
        let work = temp_dir("repair");
        let root = storage_root(&work, "test-tool");
        Workspace::init(&work, "test-tool", [root.join("a"), root.join("b")])?;

        fs::remove_dir_all(root.join("b"))?;
        assert!(!root.join("b").exists());

        let result = Workspace::init(&work, "test-tool", [root.join("a"), root.join("b")])?;
        assert!(matches!(result, InitResult::AlreadyExists(_)));
        assert!(root.join("b").is_dir());

        let _ = fs::remove_dir_all(&work);
        Ok(())
    }

    #[test]
    fn init_detects_corrupt_manifest() -> anyhow::Result<()> {
        let work = temp_dir("corrupt-init");
        let root = storage_root(&work, "test-tool");
        fs::create_dir_all(&root)?;
        fs::write(root.join("manifest.toml"), "not valid toml")?;

        let err = Workspace::init(&work, "test-tool", [root.join("a")])
            .err()
            .context("expected init to fail on corrupt manifest")?;
        assert!(format!("{err:#}").contains("corrupt"));

        let _ = fs::remove_dir_all(&work);
        Ok(())
    }

    #[test]
    fn open_fails_when_not_initialized() -> anyhow::Result<()> {
        let work = temp_dir("open-missing");
        fs::create_dir_all(&work)?;
        let err = Workspace::open(&work, "pipette-llamacpp")
            .err()
            .context("expected open to fail when not initialized")?;
        let msg = format!("{err:#}");
        assert!(msg.contains("not initialized"));
        assert!(msg.contains("pipette-llamacpp init"));

        let _ = fs::remove_dir_all(&work);
        Ok(())
    }

    #[test]
    fn open_fails_when_work_dir_missing() -> anyhow::Result<()> {
        let work = temp_dir("open-no-workdir");
        let err = Workspace::open(&work, "test-tool")
            .err()
            .context("expected open to fail when work dir missing")?;
        assert!(format!("{err:#}").contains("does not exist"));
        Ok(())
    }

    #[test]
    fn open_fails_on_corrupt_manifest() -> anyhow::Result<()> {
        let work = temp_dir("open-corrupt");
        let root = storage_root(&work, "test-tool");
        fs::create_dir_all(&root)?;
        fs::write(root.join("manifest.toml"), "")?;

        let err = Workspace::open(&work, "test-tool")
            .err()
            .context("expected open to fail on corrupt manifest")?;
        assert!(format!("{err:#}").contains("corrupt"));

        let _ = fs::remove_dir_all(&work);
        Ok(())
    }

    #[test]
    fn open_returns_workspace() -> anyhow::Result<()> {
        let work = temp_dir("open-present");
        let root = storage_root(&work, "test-tool");
        Workspace::init(&work, "test-tool", [root.join("plans")])?;

        let ws = Workspace::open(&work, "test-tool")?;
        assert_eq!(ws.root(), root);

        let _ = fs::remove_dir_all(&work);
        Ok(())
    }

    #[test]
    fn resolve_work_dir_uses_arg() -> anyhow::Result<()> {
        let result = resolve_work_dir(Some(Path::new("/explicit")))?;
        assert_eq!(result, PathBuf::from("/explicit"));
        Ok(())
    }

    #[test]
    fn resolve_work_dir_falls_back_to_cwd() -> anyhow::Result<()> {
        let result = resolve_work_dir(None)?;
        assert_eq!(result, std::env::current_dir()?);
        Ok(())
    }

    #[test]
    fn is_initialized_detects_a_manifest() -> anyhow::Result<()> {
        let work = temp_dir("is-init");
        let _ = fs::remove_dir_all(&work);
        assert!(!is_initialized(&work, "test-tool"));
        Workspace::init(&work, "test-tool", Vec::<PathBuf>::new())?;
        assert!(is_initialized(&work, "test-tool"));
        let _ = fs::remove_dir_all(&work);
        Ok(())
    }

    #[test]
    fn open_migrates_legacy_json_manifest() -> anyhow::Result<()> {
        let work = temp_dir("migrate-json");
        let root = storage_root(&work, "test-tool");
        fs::create_dir_all(&root)?;
        fs::write(
            root.join("manifest.json"),
            r#"{"created_at":"2020-01-01T00:00:00Z"}"#,
        )?;
        // The legacy marker still counts as initialized.
        assert!(is_initialized(&work, "test-tool"));

        let ws = Workspace::open(&work, "test-tool")?;
        assert_eq!(ws.root(), root);
        // JSON replaced by TOML in place; created_at preserved.
        assert!(root.join("manifest.toml").exists());
        assert!(!root.join("manifest.json").exists());
        let raw = fs::read_to_string(root.join("manifest.toml"))?;
        assert!(raw.contains("2020-01-01T00:00:00Z"));
        assert!(raw.contains("version"));

        let _ = fs::remove_dir_all(&work);
        Ok(())
    }

    #[test]
    fn open_rejects_newer_layout_version() -> anyhow::Result<()> {
        let work = temp_dir("newer-version");
        let root = storage_root(&work, "test-tool");
        fs::create_dir_all(&root)?;
        fs::write(
            root.join("manifest.toml"),
            "version = 9999\ncreated_at = \"2020-01-01T00:00:00Z\"\n",
        )?;
        let err = Workspace::open(&work, "test-tool")
            .err()
            .context("expected open to reject a newer layout version")?;
        assert!(format!("{err:#}").contains("needs a newer client"));
        let _ = fs::remove_dir_all(&work);
        Ok(())
    }

    #[test]
    fn open_accepts_versionless_toml_manifest() -> anyhow::Result<()> {
        let work = temp_dir("versionless");
        let root = storage_root(&work, "test-tool");
        fs::create_dir_all(&root)?;
        fs::write(
            root.join("manifest.toml"),
            "created_at = \"2020-01-01T00:00:00Z\"\n",
        )?;
        Workspace::open(&work, "test-tool")?;
        let _ = fs::remove_dir_all(&work);
        Ok(())
    }

    #[test]
    fn shell_quote_escapes_embedded_single_quotes() {
        assert_eq!(shell_quote(Path::new("a'b")), "'a'\\''b'");
    }

    #[rstest]
    #[case::none_at_cwd(
        "pipette",
        "/home/u/proj/.pipette",
        None,
        "/home/u/proj",
        &[
            "pipette init  (the current directory: /home/u/proj)",
            "or: pipette --work-dir <path> init  (another directory)",
        ],
        &["--work-dir '", "a specific work dir"],
    )]
    #[case::explicit_other_dir(
        "pipette",
        "/foo/.pipette",
        Some("/foo"),
        "/home/u/proj",
        &[
            "/foo/.pipette",
            "pipette --work-dir '/foo' init  (a specific work dir)",
            "pipette --work-dir '/home/u/proj' init  (the current directory: /home/u/proj)",
        ],
        &["--work-dir <path>", "--work-dir <DIR>"],
    )]
    #[case::explicit_cwd(
        "pipette",
        "/home/u/proj/.pipette",
        Some("/home/u/proj"),
        "/home/u/proj",
        &["pipette --work-dir '/home/u/proj' init  (the current directory: /home/u/proj)"],
        &["either:", "a specific work dir"],
    )]
    #[case::path_with_spaces(
        "pipette",
        "/home/u/My Projects/.pipette",
        Some("/home/u/My Projects"),
        "/tmp",
        &["pipette --work-dir '/home/u/My Projects' init"],
        &[],
    )]
    #[case::named_client(
        "pipette-llamacpp",
        "/w/.pipette-llamacpp",
        Some("/w"),
        "/home/u/proj",
        &[
            "no pipette-llamacpp workspace at /w/.pipette-llamacpp",
            "pipette-llamacpp --work-dir '/w' init  (a specific work dir)",
            "pipette-llamacpp --work-dir '/home/u/proj' init  (the current directory: /home/u/proj)",
        ],
        &["pipette --work-dir", "pipette init"],
    )]
    fn not_initialized_hint_covers_all_resolution_sources(
        #[case] name: &str,
        #[case] resolved_root: &str,
        #[case] work_dir_arg: Option<&str>,
        #[case] cwd: &str,
        #[case] contains: &[&str],
        #[case] absent: &[&str],
    ) -> anyhow::Result<()> {
        let msg = not_initialized_hint(
            Path::new(resolved_root),
            work_dir_arg.map(Path::new),
            Path::new(cwd),
            name,
        );
        contains.iter().try_for_each(|needle| {
            anyhow::ensure!(msg.contains(needle), "expected {needle:?} in:\n{msg}");
            anyhow::Ok(())
        })?;
        absent.iter().try_for_each(|needle| {
            anyhow::ensure!(!msg.contains(needle), "unexpected {needle:?} in:\n{msg}");
            anyhow::Ok(())
        })?;
        Ok(())
    }

    #[test]
    fn require_workspace_missing_dir() -> anyhow::Result<()> {
        let missing = PathBuf::from("/no/such/pipette-workspace-guard-test");
        let Err(err) = require_workspace(&missing, None, "test-tool") else {
            anyhow::bail!("expected an error for a missing work dir");
        };
        let msg = format!("{err:#}");
        anyhow::ensure!(
            msg.contains("working directory does not exist"),
            "unexpected: {msg}"
        );
        anyhow::ensure!(!msg.contains("no test-tool workspace"), "unexpected: {msg}");
        Ok(())
    }

    #[test]
    fn require_workspace_uninitialized_dir() -> anyhow::Result<()> {
        let work = temp_dir("require-uninit");
        let _ = fs::remove_dir_all(&work);
        fs::create_dir_all(&work)?;
        let result = require_workspace(&work, None, "test-tool");
        let _ = fs::remove_dir_all(&work);
        let Err(err) = result else {
            anyhow::bail!("expected the not-initialized hint");
        };
        let msg = format!("{err:#}");
        anyhow::ensure!(
            msg.contains("no test-tool workspace"),
            "expected hint, got:\n{msg}"
        );
        anyhow::ensure!(
            !msg.contains("working directory does not exist"),
            "unexpected: {msg}"
        );
        Ok(())
    }

    #[test]
    fn require_workspace_accepts_initialized() -> anyhow::Result<()> {
        let work = temp_dir("require-ok");
        let _ = fs::remove_dir_all(&work);
        Workspace::init(&work, "test-tool", Vec::<PathBuf>::new())?;
        require_workspace(&work, None, "test-tool")?;
        let _ = fs::remove_dir_all(&work);
        Ok(())
    }
}
