//! gpu-keepalive: keeps a GPU from idling by dispatching one compute thread
//! against a four-byte buffer per pulse. One binary is the CLI and, when SCM
//! launches it as `gpu-keepalive service`, the PipetteGpuKeepalive service.

#[cfg(not(windows))]
compile_error!("gpu-keepalive requires Windows and Direct3D 11");

#[cfg(all(target_env = "msvc", not(target_feature = "crt-static")))]
compile_error!(
    "gpu-keepalive requires a static C runtime; run Cargo from tools/gpu-keepalive to load .cargo/config.toml"
);

mod adapter;
mod control;
mod keepalive;
mod service;
mod state;

use std::io::Write;
use std::ops::RangeInclusive;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};

use crate::adapter::{AdapterChoice, enumerate_adapters, enumerate_vulkan, select_adapter};
use crate::keepalive::{Pulse, Stop, millis, pulse_loop};
use crate::state::CONFIG_KEY;

/// The service name SCM knows it by, and the one the MSI registers.
pub const SERVICE_NAME: &str = "PipetteGpuKeepalive";

/// The registry's `IntervalMs` is validated against this too, so the bounds
/// live in one place.
pub const INTERVAL: RangeInclusive<Duration> =
    Duration::from_millis(1)..=Duration::from_millis(60_000);
/// The interval when neither the flag nor the registry says otherwise.
pub const DEFAULT_INTERVAL: Duration = Duration::from_millis(100);

/// The same default `--interval-ms` shows in its help and parses; a test ties it
/// to `DEFAULT_INTERVAL`.
const DEFAULT_INTERVAL_MS: &str = "100";

/// Shared by `--interval-ms` and the `IntervalMs` DWORD, so both reject a bad
/// value with the same message.
pub fn interval_from_millis(millis: u64) -> Result<Duration, String> {
    let interval = Duration::from_millis(millis);
    if !INTERVAL.contains(&interval) {
        return Err(format!(
            "must be {} to {} ms, found {millis}",
            INTERVAL.start().as_millis(),
            INTERVAL.end().as_millis()
        ));
    }
    Ok(interval)
}

fn interval_arg(text: &str) -> Result<Duration, String> {
    let millis = text
        .parse::<u64>()
        .map_err(|_| format!("expected whole milliseconds, got '{text}'"))?;
    interval_from_millis(millis)
}

fn config_help() -> String {
    format!(
        "\
Configuration lives in HKLM\\{CONFIG_KEY}, which the MSI never
writes, so `disable` survives upgrades:
  Enabled     DWORD   default 1
  IntervalMs  DWORD   default {}, {}..{}
  Adapter     string  default vulkan; vulkan|software|INDEX
The service reads it at start; stop and start it to apply a change.",
        DEFAULT_INTERVAL.as_millis(),
        INTERVAL.start().as_millis(),
        INTERVAL.end().as_millis()
    )
}

#[derive(Parser)]
#[command(
    name = "gpu-keepalive",
    version = concat!(env!("GPU_KEEPALIVE_VERSION"), " (", env!("GPU_KEEPALIVE_BUILD"), ")"),
    about = "Keeps a GPU from idling by dispatching one compute thread per pulse",
    after_help = config_help()
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Service state and configuration
    Status,
    /// Set Enabled=1 and start the service (elevated)
    Enable,
    /// Set Enabled=0 and stop the service (elevated)
    Disable,
    /// Start the service (elevated)
    Start,
    /// Stop the service until the next start or reboot (elevated)
    Stop,
    /// Pulse in the foreground, independent of the service
    Run {
        /// vulkan, software, or a DXGI adapter index
        #[arg(long, default_value = "vulkan")]
        adapter: AdapterChoice,
        /// Milliseconds between pulses
        #[arg(long = "interval-ms", default_value = DEFAULT_INTERVAL_MS, value_parser = interval_arg)]
        interval: Duration,
        /// Seconds to run for; 0 runs until Ctrl+C
        #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=86_400))]
        duration_secs: u32,
    },
    /// DXGI adapters with LUIDs, and the GPUs Vulkan reports
    List,
    /// SCM entry point, started by Windows rather than by hand
    #[command(hide = true)]
    Service,
}

fn main() -> ExitCode {
    match dispatch(Cli::parse().command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("gpu-keepalive: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn dispatch(command: Command) -> anyhow::Result<()> {
    match command {
        Command::Service => service::run_as_service(),
        Command::Status => control::status(),
        Command::Enable => control::enable(),
        Command::Disable => control::disable(),
        Command::Start => control::start(),
        Command::Stop => control::stop(),
        Command::Run {
            adapter,
            interval,
            duration_secs,
        } => run(adapter, interval, duration_secs),
        Command::List => list(),
    }
}

fn run(choice: AdapterChoice, interval: Duration, duration_secs: u32) -> anyhow::Result<()> {
    if control::service_running() {
        eprintln!(
            "gpu-keepalive: note: the {SERVICE_NAME} service is pulsing too; stop it for a clean comparison"
        );
    }
    let adapter = select_adapter(choice)?;
    let pulse = Pulse::new(&adapter.handle)?;
    pulse.self_check()?;
    println!(
        "READY adapter={} name=\"{}\" luid={} selection={choice} interval_ms={} \
         duration_secs={duration_secs} buffer_bytes=4 threads_per_pulse=1",
        adapter.index,
        adapter.name,
        adapter.luid,
        interval.as_millis()
    );
    std::io::stdout().flush().ok();

    let stop = Stop::default();
    let interrupted = stop.clone();
    ctrlc::set_handler(move || interrupted.signal())
        .map_err(|error| anyhow::anyhow!("Installing the Ctrl+C handler failed: {error}"))?;
    let limit = (duration_secs > 0).then(|| Duration::from_secs(u64::from(duration_secs)));
    let stats = pulse_loop(&pulse, interval, limit, &stop)?;
    println!(
        "DONE pulses={} wall_ms={:.3} process_cpu_ms={:.3} pulse_completion_mean_ms={:.3} \
         pulse_completion_max_ms={:.3} missed_deadlines={} verified_counter={}",
        stats.pulses,
        millis(stats.wall),
        millis(stats.process_cpu),
        millis(stats.mean_completion()),
        millis(stats.completion_max),
        stats.missed_deadlines,
        stats.expected_counter()
    );
    Ok(())
}

fn list() -> anyhow::Result<()> {
    let adapters = enumerate_adapters()?;
    adapters.iter().for_each(|adapter| {
        println!(
            "{}: {} luid={} vendor={:x} device={:x} dedicated_bytes={}{}",
            adapter.index,
            adapter.name,
            adapter.luid,
            adapter.vendor_id,
            adapter.device_id,
            adapter.dedicated_bytes,
            if adapter.software { " [software]" } else { "" }
        );
    });
    match enumerate_vulkan() {
        Ok(gpus) if gpus.is_empty() => println!("vulkan: no devices"),
        Ok(gpus) => gpus.into_iter().for_each(|gpu| {
            print!("vulkan: {} luid=", gpu.name);
            match gpu.luid {
                Some(luid) => print!("{luid}"),
                None => print!("invalid"),
            }
            if gpu.cpu {
                print!(" [cpu]");
            }
            if let Some(luid) = gpu.luid {
                adapters
                    .iter()
                    .filter(|adapter| adapter.luid == luid)
                    .for_each(|adapter| print!(" -> adapter {}", adapter.index));
            }
            println!();
        }),
        Err(error) => println!("vulkan: unavailable: {error:#}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_INTERVAL, DEFAULT_INTERVAL_MS, INTERVAL, interval_arg, interval_from_millis,
    };

    #[test]
    fn the_interval_flag_bounds_what_the_registry_also_has_to_obey() {
        for millis in [1, 100, 60_000] {
            let parsed = interval_arg(&millis.to_string()).expect("within the bound");
            assert_eq!(
                parsed,
                interval_from_millis(millis).expect("within the bound")
            );
            assert_eq!(parsed.as_millis(), u128::from(millis));
        }
        for millis in [0, 60_001, u64::MAX] {
            assert!(
                interval_from_millis(millis).is_err(),
                "{millis} should be rejected"
            );
        }
        for text in ["", "-1", "1.5", "100ms"] {
            assert!(interval_arg(text).is_err(), "{text} should not parse");
        }
        assert!(INTERVAL.contains(&DEFAULT_INTERVAL));
        assert_eq!(interval_arg(DEFAULT_INTERVAL_MS), Ok(DEFAULT_INTERVAL));
    }
}
