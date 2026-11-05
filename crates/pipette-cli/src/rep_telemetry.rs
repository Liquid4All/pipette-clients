//! What the client records beside a cell about each rep, in the shape of mgmt's
//! `docs/design/device-telemetry.md`: the device's and the OS's readings at the
//! rep's two ends, and the readiness gate's report on the wait before it.

use std::cell::Cell;

use serde::Serialize;
use serde_json::{Map, Value};

use pipette_device::DeviceReadings;
use pipette_plan_types::ThermalReading;
use pipette_readiness::ReadinessStatus;

/// What this side samples at one end of a measured rep.
#[derive(Debug)]
pub(crate) struct RepSample {
    pub thermal: ThermalReading,
    pub device: DeviceReadings,
    /// The gate's report on the wait that released the rep; on a start only.
    pub gate: Option<GateWait>,
}

#[derive(Debug, Clone)]
pub(crate) struct GateWait {
    poll_count: usize,
    last: ReadinessStatus,
}

impl GateWait {
    fn element(&self, iteration: usize) -> GateWaitElement {
        GateWaitElement {
            iteration,
            poll_count: self.poll_count,
            platform: self.last.platform.as_str(),
            phase: self.last.phase.as_str(),
            elapsed_ms: u64::try_from(self.last.elapsed.as_millis()).unwrap_or(u64::MAX),
            observed: self.last.observed.clone(),
        }
    }
}

/// Follows the gate's polls, and holds each passed wait until the rep it
/// released starts.
///
/// `Cell`, not a lock: the gate and the observer are `&dyn Fn` without `Sync`,
/// so the engines call them on this thread only.
#[derive(Default)]
pub(crate) struct GateWaitRecorder {
    polls: Cell<usize>,
    last: Cell<Option<ReadinessStatus>>,
    passed: Cell<Option<GateWait>>,
}

impl GateWaitRecorder {
    pub fn begin(&self) {
        self.polls.set(0);
        self.last.set(None);
    }

    pub fn observe(&self, status: &ReadinessStatus) {
        self.last.set(Some(status.clone()));
    }

    pub fn polled(&self) {
        self.polls.set(self.polls.get().saturating_add(1));
    }

    pub fn pass(&self) {
        let poll_count = self.polls.get();
        self.passed
            .set(self.last.take().map(|last| GateWait { poll_count, last }));
    }

    /// The wait that released the rep now starting. A gate inside a rep (between
    /// `vl_spec_decode`'s batches) releases none, so the next wait replaces it.
    pub fn take(&self) -> Option<GateWait> {
        self.passed.take()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SamplePoint {
    /// When the gate released the rep.
    BeforeRep,
    /// When the rep's timed work ended.
    AfterRep,
}

/// One element of `device_hardware_readings` or `device_os_throttling_readings`:
/// each facility read at one sample point, under its own key, as the OS
/// returned it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ReadingsElement {
    pub iteration: usize,
    pub sample_point: SamplePoint,
    #[serde(flatten)]
    pub facilities: Map<String, Value>,
}

/// One element of `readiness_gate_waits`: the gate's last report on the wait
/// that released a rep.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct GateWaitElement {
    pub iteration: usize,
    pub poll_count: usize,
    pub platform: &'static str,
    pub phase: &'static str,
    pub elapsed_ms: u64,
    pub observed: String,
}

#[derive(Debug, Default)]
pub struct RepTelemetry {
    /// `device_hardware_readings`: one element per rep and sample point.
    pub(crate) hardware_readings: Vec<ReadingsElement>,
    /// `device_os_throttling_readings`, likewise.
    pub(crate) os_throttling_readings: Vec<ReadingsElement>,
    /// `readiness_gate_waits`: one element per rep the gate released.
    pub(crate) gate_waits: Vec<GateWaitElement>,
}

impl RepTelemetry {
    /// From each rep's `(start, end)` samples, in rep order.
    pub(crate) fn from_reps(reps: &[(RepSample, RepSample)]) -> Self {
        let points = || {
            reps.iter()
                .enumerate()
                .flat_map(|(iteration, (start, end))| {
                    [
                        (SamplePoint::BeforeRep, start),
                        (SamplePoint::AfterRep, end),
                    ]
                    .map(|(point, sample)| (iteration, point, &sample.device))
                })
        };
        Self {
            hardware_readings: points()
                .filter_map(|(iteration, point, device)| {
                    element(iteration, point, &device.hardware)
                })
                .collect(),
            os_throttling_readings: points()
                .filter_map(|(iteration, point, device)| {
                    element(iteration, point, &device.os_throttling)
                })
                .collect(),
            gate_waits: reps
                .iter()
                .enumerate()
                .filter_map(|(iteration, (start, _))| {
                    start.gate.as_ref().map(|gate| gate.element(iteration))
                })
                .collect(),
        }
    }
}

/// One sample point's facilities, or none where it read nothing.
fn element(
    iteration: usize,
    sample_point: SamplePoint,
    facilities: &Map<String, Value>,
) -> Option<ReadingsElement> {
    (!facilities.is_empty()).then(|| ReadingsElement {
        iteration,
        sample_point,
        facilities: facilities.clone(),
    })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use anyhow::Context;
    use serde_json::json;

    use pipette_readiness::{ReadinessPhase, ReadinessPlatform};

    use super::*;

    fn status(phase: ReadinessPhase, observed: &str, elapsed_ms: u64) -> ReadinessStatus {
        ReadinessStatus {
            phase,
            platform: ReadinessPlatform::Windows,
            observed: observed.to_string(),
            elapsed: Duration::from_millis(elapsed_ms),
            max_wait: Duration::from_secs(300),
        }
    }

    fn waiting() -> ReadinessPhase {
        ReadinessPhase::Waiting {
            poll_in: Duration::from_secs(5),
        }
    }

    fn sample(hardware: Value, gate: Option<GateWait>) -> RepSample {
        RepSample {
            thermal: ThermalReading::default(),
            device: DeviceReadings {
                hardware: hardware.as_object().cloned().unwrap_or_default(),
                os_throttling: Map::new(),
            },
            gate,
        }
    }

    #[test]
    fn a_passed_wait_is_held_for_its_rep() -> anyhow::Result<()> {
        let recorder = GateWaitRecorder::default();
        recorder.begin();
        recorder.polled();
        recorder.observe(&status(waiting(), "THZ0=47", 1_200));
        recorder.polled();
        recorder.observe(&status(ReadinessPhase::Proceeding, "THZ0=43", 10_700));
        recorder.pass();
        let wait = recorder
            .take()
            .ok_or_else(|| anyhow::anyhow!("a wait passed"))?;
        assert_eq!(
            serde_json::to_value(wait.element(0))?,
            json!({
                "iteration": 0, "poll_count": 2, "platform": "windows",
                "phase": "proceeding", "elapsed_ms": 10_700, "observed": "THZ0=43",
            })
        );
        assert!(recorder.take().is_none(), "taken once");
        Ok(())
    }

    /// The next wait counts from its own first poll.
    #[test]
    fn a_wait_inside_a_rep_is_replaced_by_the_next() -> anyhow::Result<()> {
        let recorder = GateWaitRecorder::default();
        [3, 1].into_iter().for_each(|polls| {
            recorder.begin();
            (0..polls).for_each(|_| {
                recorder.polled();
                recorder.observe(&status(ReadinessPhase::Proceeding, "ok", 5));
            });
            recorder.pass();
        });
        let wait = recorder
            .take()
            .ok_or_else(|| anyhow::anyhow!("a wait passed"))?;
        assert_eq!(wait.poll_count, 1);
        Ok(())
    }

    #[test]
    fn a_failed_poll_counts_without_replacing_the_last_status() -> anyhow::Result<()> {
        let recorder = GateWaitRecorder::default();
        recorder.begin();
        recorder.polled(); // A failed probe has no status.
        recorder.polled();
        recorder.observe(&status(ReadinessPhase::Proceeding, "ok", 10));
        recorder.pass();
        let wait = recorder.take().context("missing passed wait")?;
        assert_eq!(wait.poll_count, 2);
        assert_eq!(wait.last.observed, "ok");
        Ok(())
    }

    /// Only sample points that read something; the gate's report once per rep.
    #[test]
    fn reps_become_one_element_per_sample_point() -> anyhow::Result<()> {
        let zones = json!({ "windows_thermal_zone_information_counters": [
            { "InstanceName": r"\_TZ.THZ0", "Temperature": 316 }
        ]});
        let gate = GateWait {
            poll_count: 1,
            last: status(ReadinessPhase::Proceeding, "ok", 40),
        };
        let telemetry = RepTelemetry::from_reps(&[
            (
                sample(zones.clone(), Some(gate)),
                sample(zones.clone(), None),
            ),
            // A rep whose end read nothing posts its start alone.
            (sample(zones.clone(), None), sample(json!({}), None)),
        ]);
        let points: Vec<(usize, SamplePoint)> = telemetry
            .hardware_readings
            .iter()
            .map(|e| (e.iteration, e.sample_point))
            .collect();
        assert_eq!(
            points,
            vec![
                (0, SamplePoint::BeforeRep),
                (0, SamplePoint::AfterRep),
                (1, SamplePoint::BeforeRep)
            ]
        );
        // On the wire the facilities sit beside the rep and point, not under
        // a key of their own.
        assert_eq!(
            serde_json::to_value(&telemetry.hardware_readings[0])?,
            json!({
                "iteration": 0, "sample_point": "before_rep",
                "windows_thermal_zone_information_counters":
                    zones["windows_thermal_zone_information_counters"],
            })
        );
        assert!(telemetry.os_throttling_readings.is_empty());
        assert_eq!(telemetry.gate_waits.len(), 1);
        assert_eq!(telemetry.gate_waits[0].iteration, 0);
        Ok(())
    }
}
