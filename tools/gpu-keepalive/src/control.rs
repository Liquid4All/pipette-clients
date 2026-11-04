//! CLI verbs that talk to the installed service through SCM.

use std::time::{Duration, Instant};

use windows::Win32::Foundation::{
    ERROR_ACCESS_DENIED, ERROR_SERVICE_ALREADY_RUNNING, ERROR_SERVICE_DISABLED,
    ERROR_SERVICE_DOES_NOT_EXIST, ERROR_SERVICE_NOT_ACTIVE,
};
use windows_service::service::{
    Service, ServiceAccess, ServiceExitCode, ServiceStartType, ServiceState, ServiceStatus,
};
use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};

use crate::SERVICE_NAME;
use crate::state::{self, CONFIG_KEY};

/// Covers the 60-second Vulkan bound plus device creation.
const START_WAIT: Duration = Duration::from_secs(120);
const STOP_WAIT: Duration = Duration::from_secs(30);
const NOT_INSTALLED: &str = "The PipetteGpuKeepalive service is not installed; install the MSI";

fn is_winapi_error(error: &windows_service::Error, code: u32) -> bool {
    matches!(error, windows_service::Error::Winapi(io) if io.raw_os_error() == Some(code as i32))
}

fn service_error(operation: &str, error: windows_service::Error) -> anyhow::Error {
    if is_winapi_error(&error, ERROR_ACCESS_DENIED.0) {
        return anyhow::anyhow!("This needs an elevated (Administrator) shell");
    }
    anyhow::anyhow!("{operation} failed: {error}")
}

/// `None` means the service is not installed.
fn open_service(access: ServiceAccess) -> anyhow::Result<Option<Service>> {
    let manager = ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)
        .map_err(|error| service_error("Connecting to the service manager", error))?;
    match manager.open_service(SERVICE_NAME, access) {
        Ok(service) => Ok(Some(service)),
        Err(error) if is_winapi_error(&error, ERROR_SERVICE_DOES_NOT_EXIST.0) => Ok(None),
        Err(error) => Err(service_error("Opening the service", error)),
    }
}

fn query(service: &Service) -> anyhow::Result<ServiceStatus> {
    service
        .query_status()
        .map_err(|error| service_error("Querying the service", error))
}

fn wait_while(
    service: &Service,
    pending: ServiceState,
    timeout: Duration,
) -> anyhow::Result<ServiceStatus> {
    let deadline = Instant::now() + timeout;
    let mut status = query(service)?;
    while status.current_state == pending && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(250));
        status = query(service)?;
    }
    Ok(status)
}

fn state_name(state: ServiceState) -> &'static str {
    match state {
        ServiceState::Stopped => "stopped",
        ServiceState::StartPending => "start_pending",
        ServiceState::StopPending => "stop_pending",
        ServiceState::Running => "running",
        ServiceState::ContinuePending => "continue_pending",
        ServiceState::PausePending => "pause_pending",
        ServiceState::Paused => "paused",
    }
}

fn start_type_name(start_type: ServiceStartType) -> &'static str {
    match start_type {
        ServiceStartType::AutoStart => "auto",
        ServiceStartType::OnDemand => "manual",
        ServiceStartType::Disabled => "disabled",
        ServiceStartType::BootStart => "boot",
        ServiceStartType::SystemStart => "system",
    }
}

/// The raw code, Win32 and service-specific alike, for the operator to read
/// beside the log line that says what actually went wrong.
fn exit_code(status: &ServiceStatus) -> u32 {
    match status.exit_code {
        ServiceExitCode::Win32(code) => code,
        ServiceExitCode::ServiceSpecific(code) => code,
    }
}

fn start_and_wait(missing_ok: bool) -> anyhow::Result<()> {
    let Some(service) = open_service(ServiceAccess::START | ServiceAccess::QUERY_STATUS)? else {
        if missing_ok {
            println!("saved; the service is not installed");
            return Ok(());
        }
        return Err(anyhow::anyhow!(NOT_INSTALLED));
    };
    match service.start::<&str>(&[]) {
        Ok(()) => {}
        Err(error) if is_winapi_error(&error, ERROR_SERVICE_ALREADY_RUNNING.0) => {}
        Err(error) if is_winapi_error(&error, ERROR_SERVICE_DISABLED.0) => {
            return Err(anyhow::anyhow!(
                "The service start type is Disabled, which the MSI never sets; repair the install or run: \
                 sc config PipetteGpuKeepalive start= delayed-auto"
            ));
        }
        Err(error) => return Err(service_error("Starting the service", error)),
    }
    let status = wait_while(&service, ServiceState::StartPending, START_WAIT)?;
    if status.current_state != ServiceState::Running {
        return Err(anyhow::anyhow!(
            "The service did not start ({}, exit code {}); see {}",
            state_name(status.current_state),
            exit_code(&status),
            state::log_path()?.display()
        ));
    }
    println!("running (pid {})", status.process_id.unwrap_or_default());
    Ok(())
}

fn stop_and_wait(missing_ok: bool) -> anyhow::Result<()> {
    let Some(service) = open_service(ServiceAccess::STOP | ServiceAccess::QUERY_STATUS)? else {
        if missing_ok {
            println!("saved; the service is not installed");
            return Ok(());
        }
        return Err(anyhow::anyhow!(NOT_INSTALLED));
    };
    // A service still starting cannot accept a stop; let it get there first.
    let mut status = wait_while(&service, ServiceState::StartPending, START_WAIT)?;
    if status.current_state != ServiceState::Stopped {
        match service.stop() {
            Ok(_) => {}
            Err(error) if is_winapi_error(&error, ERROR_SERVICE_NOT_ACTIVE.0) => {}
            Err(error) => return Err(service_error("Stopping the service", error)),
        }
        status = wait_while(&service, ServiceState::StopPending, STOP_WAIT)?;
    }
    if status.current_state != ServiceState::Stopped {
        return Err(anyhow::anyhow!(
            "The service did not stop within {} seconds ({})",
            STOP_WAIT.as_secs(),
            state_name(status.current_state)
        ));
    }
    println!("stopped");
    Ok(())
}

pub fn status() -> anyhow::Result<()> {
    match open_service(ServiceAccess::QUERY_STATUS | ServiceAccess::QUERY_CONFIG)? {
        Some(service) => {
            let status = query(&service)?;
            let config = service
                .query_config()
                .map_err(|error| service_error("Reading the service configuration", error))?;
            print!("service:   {}", state_name(status.current_state));
            if status.current_state == ServiceState::Running {
                print!(" (pid {})", status.process_id.unwrap_or_default());
            } else if exit_code(&status) != 0 {
                print!(" (exit code {})", exit_code(&status));
            }
            println!(", start type {}", start_type_name(config.start_type));
        }
        None => println!("service:   not installed"),
    }
    match state::Settings::open().and_then(|settings| settings.read_config()) {
        Ok(config) => println!(
            "config:    {}, interval {} ms, adapter {}  [HKLM\\{CONFIG_KEY}]",
            if config.enabled {
                "enabled"
            } else {
                "disabled"
            },
            config.interval.as_millis(),
            config.adapter
        ),
        Err(error) => println!("config:    invalid: {error:#}  [HKLM\\{CONFIG_KEY}]"),
    }
    println!("log:       {}", state::log_path()?.display());
    Ok(())
}

pub fn enable() -> anyhow::Result<()> {
    state::save_enabled(true)?;
    start_and_wait(true)
}

pub fn disable() -> anyhow::Result<()> {
    state::save_enabled(false)?;
    stop_and_wait(true)
}

pub fn start() -> anyhow::Result<()> {
    if !state::Settings::open()?.enabled {
        return Err(anyhow::anyhow!(
            "Disabled in the configuration (Enabled=0); use 'gpu-keepalive enable'"
        ));
    }
    start_and_wait(false)
}

pub fn stop() -> anyhow::Result<()> {
    stop_and_wait(false)
}

/// Never fails: a service that cannot be opened or queried counts as not running.
pub fn service_running() -> bool {
    open_service(ServiceAccess::QUERY_STATUS)
        .ok()
        .flatten()
        .and_then(|service| service.query_status().ok())
        .is_some_and(|status| status.current_state == ServiceState::Running)
}
