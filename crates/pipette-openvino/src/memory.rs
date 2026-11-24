//! What the memory cells share: the device's memory watch, and one server
//! watched from its launch through the cell's workload, which each cell runs
//! twice, a warm-up server and then the measured one.

use std::process::Command;

use anyhow::Context;

use pipette_memprobe::{Peak, Sampler, SupportedWatch, Watch};
use pipette_ops::measurement::MeasurementObserver;
use pipette_plan_types::{MemoryObservation, OpenvinoDevice};
use pipette_subprocess::{OnStop, Process, start_process};

use crate::server;

/// One server from its launch through `workload`, its requests to the server's
/// URL, left in `served` for the entry to check its peak and the cell to
/// record or quote it. One an earlier attempt left there stops first; the
/// observation covers both.
pub(crate) fn serve(
    served: &mut Option<(Process, MemoryObservation)>,
    command: impl Fn(u16) -> anyhow::Result<Command>,
    watch: SupportedWatch,
    observer: MeasurementObserver<'_>,
    workload: impl FnOnce(&str) -> anyhow::Result<()>,
) -> anyhow::Result<Peak> {
    let earlier = match served.take() {
        Some((mut openvino_server, memory)) => {
            openvino_server.stop();
            memory
        }
        None => MemoryObservation::default(),
    };
    let port = pipette_subprocess::pick_server_port()?;
    let url = server::url(port);
    let mut openvino_server =
        start_process(server::SCRIPT_FILENAME, command(port)?, OnStop::SIGKILL)?;
    // Watched from the launch, so the load counts.
    let sampler = Sampler::start(openvino_server.pid(), watch)?;
    let ready = server::ready(&url)?;
    observer.outside(|| openvino_server.wait_ready(server::READY_TIMEOUT, ready))?;
    let measured = workload(&url);
    // Stopped while the server lives: Linux reads its `/proc` entry.
    let (observed, peak) = sampler.stop();
    // Before the checks, so the cell quotes it on any failure from here.
    *served = Some((openvino_server, earlier.merge_max(observed)));
    measured?;
    Ok(peak?)
}

/// `device`'s watch, refused where this OS reads none of its memory, naming
/// the devices it does read.
pub(crate) fn watch(device: &OpenvinoDevice) -> anyhow::Result<SupportedWatch> {
    device_watch(device).supported().with_context(|| {
        let read: Vec<String> = OpenvinoDevice::known()
            .filter(|known| device_watch(known).supported().is_ok())
            .map(|known| known.as_str().to_owned())
            .collect();
        format!(
            "the {} device's memory cannot be read on this OS; run this memory cell on {}",
            crate::runtimes::device_property(device),
            read.join(" or ")
        )
    })
}

/// A custom device (`AUTO`, `HETERO:GPU,CPU`, `GPU.1`) may resolve anywhere,
/// so it requires none. `pipette-ovms` keeps the same rule.
fn device_watch(device: &OpenvinoDevice) -> Watch {
    match device {
        OpenvinoDevice::Cpu => Watch::Peak,
        OpenvinoDevice::Gpu => Watch::PeakWithGpu,
        OpenvinoDevice::Npu => Watch::PeakWithNpu,
        OpenvinoDevice::Custom(_) => Watch::PeakWithAccelerators,
    }
}

#[cfg(test)]
mod tests {
    use pipette_memprobe::{UnsupportedError, WatchError};

    use super::*;

    #[rstest::rstest]
    #[case::cpu(OpenvinoDevice::Cpu, Watch::Peak)]
    #[case::gpu(OpenvinoDevice::Gpu, Watch::PeakWithGpu)]
    #[case::npu(OpenvinoDevice::Npu, Watch::PeakWithNpu)]
    #[case::custom(OpenvinoDevice::Custom("GPU.1".to_owned()), Watch::PeakWithAccelerators)]
    fn each_device_has_its_watch(#[case] device: OpenvinoDevice, #[case] expected: Watch) {
        assert_eq!(device_watch(&device), expected);
    }

    /// Windows reads every device, so there it has nothing to refuse.
    #[test]
    fn a_refusal_names_the_device_and_the_devices_this_os_reads() {
        let Err(err) = watch(&OpenvinoDevice::Npu) else {
            return;
        };
        let refused = UnsupportedError {
            watch: Watch::PeakWithNpu,
        };
        assert_eq!(
            err.downcast_ref::<WatchError>(),
            Some(&refused.into()),
            "{err:#}"
        );
        let read = if cfg!(target_os = "linux") {
            "cpu or gpu"
        } else {
            "cpu"
        };
        let expected = format!(
            "the {} device's memory cannot be read on this OS; run this memory cell on {read}",
            crate::runtimes::device_property(&OpenvinoDevice::Npu)
        );
        assert_eq!(err.to_string(), expected);
    }
}
