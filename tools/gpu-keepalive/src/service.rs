//! The Windows service: what SCM runs as `gpu-keepalive service`.

use std::ffi::OsString;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use windows::Win32::Foundation::ERROR_FAILED_SERVICE_CONTROLLER_CONNECT;
use windows_service::service::{
    ServiceControl, ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus, ServiceType,
};
use windows_service::service_control_handler::{
    self, ServiceControlHandlerResult, ServiceStatusHandle,
};
use windows_service::{define_windows_service, service_dispatcher};

use crate::SERVICE_NAME;
use crate::adapter::{VULKAN_TIMEOUT, select_adapter};
use crate::keepalive::{Pulse, Stop, millis, pulse_loop};
use crate::state;

/// Any nonzero code reported with Stopped trips the MSI's FailureActionsWhen and
/// gets the service restarted. Nothing reads the number; the reason is in the log.
const FAILED: u32 = 1;

/// Cover Vulkan enumeration plus device creation and the initial GPU readback.
const START_HINT: Duration = VULKAN_TIMEOUT.saturating_add(Duration::from_secs(30));
/// Covers the final readback, which waits at most five seconds for the GPU.
const STOP_HINT: Duration = Duration::from_secs(10);

define_windows_service!(ffi_service_main, service_main);

/// Connects to SCM and runs until the service stops.
pub fn run_as_service() -> anyhow::Result<()> {
    service_dispatcher::start(SERVICE_NAME, ffi_service_main).map_err(|error| {
        if matches!(&error, windows_service::Error::Winapi(io)
            if io.raw_os_error() == Some(ERROR_FAILED_SERVICE_CONTROLLER_CONNECT.0 as i32))
        {
            anyhow::anyhow!("'service' is started by Windows; use 'gpu-keepalive start' or 'run'")
        } else {
            anyhow::anyhow!("StartServiceCtrlDispatcher failed: {error}")
        }
    })
}

fn service_status(
    state: ServiceState,
    wait_hint: Duration,
    exit_code: ServiceExitCode,
) -> ServiceStatus {
    ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: state,
        controls_accepted: if state == ServiceState::Running {
            ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN
        } else {
            ServiceControlAccept::empty()
        },
        exit_code,
        checkpoint: 0,
        wait_hint,
        process_id: None,
    }
}

fn service_main(_arguments: Vec<OsString>) {
    let stop = Stop::default();
    // The handler needs the status handle that `register` only returns once the
    // handler exists, so it reads it from a slot filled immediately after.
    let slot: Arc<OnceLock<ServiceStatusHandle>> = Arc::new(OnceLock::new());
    let handler_stop = stop.clone();
    let handler_slot = Arc::clone(&slot);
    let handler = move |control| match control {
        ServiceControl::Stop | ServiceControl::Shutdown => {
            if let Some(handle) = handler_slot.get() {
                let _ = handle.set_service_status(service_status(
                    ServiceState::StopPending,
                    STOP_HINT,
                    ServiceExitCode::Win32(0),
                ));
            }
            handler_stop.signal();
            ServiceControlHandlerResult::NoError
        }
        ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
        _ => ServiceControlHandlerResult::NotImplemented,
    };
    let Ok(handle) = service_control_handler::register(SERVICE_NAME, handler) else {
        return;
    };
    let _ = slot.set(handle);
    report_start_pending(&handle, 1);

    let exit_code = match pulse_until_stopped(&handle, &stop) {
        Ok(()) => ServiceExitCode::Win32(0),
        Err(error) => {
            state::log_line("ERROR", &format!("{error:#}"));
            ServiceExitCode::ServiceSpecific(FAILED)
        }
    };
    let _ = handle.set_service_status(service_status(
        ServiceState::Stopped,
        Duration::default(),
        exit_code,
    ));
}

fn report_start_pending(handle: &ServiceStatusHandle, checkpoint: u32) {
    let mut status = service_status(
        ServiceState::StartPending,
        START_HINT,
        ServiceExitCode::Win32(0),
    );
    status.checkpoint = checkpoint;
    let _ = handle.set_service_status(status);
}

fn pulse_until_stopped(handle: &ServiceStatusHandle, stop: &Stop) -> anyhow::Result<()> {
    // Disabled policy must win even when settings for pulsing are malformed.
    let settings = state::Settings::open()?;
    if !settings.enabled {
        state::log_line("INFO", "disabled by configuration (Enabled=0); not pulsing");
        return Ok(());
    }
    let config = settings.read_config()?;
    state::log_line(
        "INFO",
        &format!(
            "starting gpu-keepalive {} interval_ms={} adapter={}",
            env!("GPU_KEEPALIVE_BUILD"),
            config.interval.as_millis(),
            config.adapter
        ),
    );
    let adapter = select_adapter(config.adapter)?;
    report_start_pending(handle, 2);
    let pulse = Pulse::new(&adapter.handle)?;
    pulse.self_check()?;
    state::log_line(
        "INFO",
        &format!(
            "READY adapter={} name=\"{}\" luid={} selection={} interval_ms={}",
            adapter.index,
            adapter.name,
            adapter.luid,
            config.adapter,
            config.interval.as_millis()
        ),
    );
    let _ = handle.set_service_status(service_status(
        ServiceState::Running,
        Duration::default(),
        ServiceExitCode::Win32(0),
    ));

    let stats = pulse_loop(&pulse, config.interval, None, stop)?;
    state::log_line(
        "INFO",
        &format!(
            "DONE pulses={} verified_counter={} missed_deadlines={} process_cpu_ms={:.1} wall_ms={:.1}",
            stats.pulses,
            stats.expected_counter(),
            stats.missed_deadlines,
            millis(stats.process_cpu),
            millis(stats.wall)
        ),
    );
    Ok(())
}
