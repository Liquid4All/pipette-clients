//! Configuration in the registry, and the service's log under ProgramData.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use known_folders::{KnownFolder, get_known_folder_path};
use winreg::RegKey;
use winreg::enums::{HKEY_LOCAL_MACHINE, KEY_READ, KEY_SET_VALUE};
use winreg::types::FromRegValue;

use crate::adapter::AdapterChoice;
use crate::{DEFAULT_INTERVAL, interval_from_millis};

/// Machine configuration. The MSI deliberately never writes this key, so an
/// operator's `disable` survives upgrades and uninstall.
pub const CONFIG_KEY: &str = r"SOFTWARE\Pipette\GpuKeepalive";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub enabled: bool,
    pub interval: Duration,
    pub adapter: AdapterChoice,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: DEFAULT_INTERVAL,
            adapter: AdapterChoice::default(),
        }
    }
}

fn config_error(operation: &str, error: &io::Error) -> anyhow::Error {
    if error.kind() == io::ErrorKind::PermissionDenied {
        return anyhow::anyhow!(
            "Changing the configuration needs an elevated (Administrator) shell"
        );
    }
    anyhow::anyhow!("{operation} failed: {error}")
}

fn value<T: FromRegValue>(key: &RegKey, name: &str) -> anyhow::Result<Option<T>> {
    match key.get_value::<T, _>(name) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(config_error(&format!("Reading {name}"), &error)),
    }
}

/// An open settings key with enabled policy read once. The service checks that
/// policy before reading pulse settings; `status` reads them even when disabled.
pub struct Settings {
    key: Option<RegKey>,
    pub enabled: bool,
}

impl Settings {
    pub fn open() -> anyhow::Result<Self> {
        let key = match RegKey::predef(HKEY_LOCAL_MACHINE)
            .open_subkey_with_flags(CONFIG_KEY, KEY_READ)
        {
            Ok(key) => Some(key),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(config_error(&format!("Opening HKLM\\{CONFIG_KEY}"), &error)),
        };
        let enabled = match &key {
            Some(key) => value::<u32>(key, "Enabled")?.is_none_or(|value| value != 0),
            None => Config::default().enabled,
        };
        Ok(Self { key, enabled })
    }

    pub fn read_config(&self) -> anyhow::Result<Config> {
        let mut config = Config {
            enabled: self.enabled,
            ..Config::default()
        };
        let Some(key) = &self.key else {
            return Ok(config);
        };
        if let Some(value) = value::<u32>(key, "IntervalMs")? {
            config.interval = interval_from_millis(u64::from(value))
                .map_err(|error| anyhow::anyhow!("IntervalMs {error}"))?;
        }
        if let Some(value) = value::<String>(key, "Adapter")? {
            config.adapter = value
                .parse()
                .map_err(|error| anyhow::anyhow!("Adapter is invalid: {error}"))?;
        }
        Ok(config)
    }
}

/// Writes `Enabled`, creating the key. Needs an elevated caller.
pub fn save_enabled(enabled: bool) -> anyhow::Result<()> {
    let (key, _) = RegKey::predef(HKEY_LOCAL_MACHINE)
        .create_subkey_with_flags(CONFIG_KEY, KEY_SET_VALUE)
        .map_err(|error| config_error(&format!("Creating HKLM\\{CONFIG_KEY}"), &error))?;
    key.set_value("Enabled", &u32::from(enabled))
        .map_err(|error| config_error("Writing Enabled", &error))
}

/// `%ProgramData%\Pipette\gpu-keepalive\gpu-keepalive.log`, where the service
/// records why it started, stopped or failed. Program Files is read-only by
/// convention, so this cannot live beside the binary.
pub fn log_path() -> anyhow::Result<PathBuf> {
    let base = get_known_folder_path(KnownFolder::ProgramData)
        .ok_or_else(|| anyhow::anyhow!("Locating ProgramData failed"))?;
    Ok(base
        .join("Pipette")
        .join("gpu-keepalive")
        .join("gpu-keepalive.log"))
}

/// Past this the log rotates, keeping one previous generation. The MSI's failure
/// actions restart a service that cannot start every 60 seconds, so the log
/// would otherwise grow without bound.
const LOG_MAX_BYTES: u64 = 1 << 20;

/// Best effort, like everything else on the logging path.
fn rotate(path: &Path) {
    let Ok(metadata) = fs::metadata(path) else {
        return;
    };
    if metadata.len() >= LOG_MAX_BYTES {
        let _ = fs::rename(path, path.with_extension("log.1"));
    }
}

/// Appends one line. Never fails the caller: logging must not be able to take
/// the service down.
pub fn log_line(level: &str, message: &str) {
    let Ok(path) = log_path() else { return };
    if let Some(directory) = path.parent() {
        let _ = fs::create_dir_all(directory);
    }
    rotate(&path);
    let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(&path) else {
        return;
    };
    let _ = writeln!(
        file,
        "{} {level} {}",
        chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ"),
        message.replace(['\r', '\n'], " ")
    );
}
