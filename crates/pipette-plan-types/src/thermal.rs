//! Per-platform thermal telemetry types, verbatim to each vendor's API — no
//! cross-platform enum mapping (collapsing Android's seven levels onto Apple's
//! four would lose information). Every type here mirrors the management
//! server's warehouse schema (pipette-mgmt `warehouse.rs`) 1:1 in its
//! snake_case wire form, so a client constructs a compile-checked value the
//! server's enum will accept. Namespaced rather than re-exported flat, so
//! consumers reference these as `pipette_plan_types::thermal::AppleThermalState`.
//!
//! The enums are declared coolest→hottest and derive `Ord` on that order, so
//! severity comparisons are available to consumers.

use serde::{Deserialize, Serialize};

/// Run-environment power state at benchmark time, reported by clients
/// alongside each result.
///
/// Mirrors the management server's `DevicePowerState` (identical snake_case
/// wire form) so the desktop clients construct a *compile-checked* value
/// instead of hand-spelling the string at each per-OS detection site — a
/// typo can no longer ship a value the server's enum would reject. Replaces
/// the earlier boolean `device_is_charging`, which couldn't distinguish
/// "plugged in and topping up" from "plugged in but holding" (battery full
/// or charge-limited): both remove the battery current-limiting that can
/// throttle the SoC, and both differ from running on battery.
///
/// `None` / absent on the wire means the client didn't report it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DevicePowerState {
    /// On external power and the battery is charging.
    Charging,
    /// Running on battery (unplugged), discharging.
    NotCharging,
    /// On external power but not adding charge (battery full or
    /// charge-limited / maintenance), or a battery-less desktop on AC.
    PluggedInNotCharging,
}

impl DevicePowerState {
    /// The snake_case wire string — kept identical to the serde
    /// representation and the management server's `strum` serialization.
    pub fn as_wire_str(self) -> &'static str {
        match self {
            Self::Charging => "charging",
            Self::NotCharging => "not_charging",
            Self::PluggedInNotCharging => "plugged_in_not_charging",
        }
    }
}

impl std::fmt::Display for DevicePowerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_wire_str())
    }
}

/// Apple `ProcessInfo.thermalState` (iOS / iPadOS / macOS) — one device-level
/// enum. No temperature, no headroom. `None` / absent on the wire means the
/// client didn't report it (or the device isn't an Apple platform).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppleThermalState {
    /// No thermal pressure; the device is operating normally.
    Nominal,
    /// Mild thermal pressure; fans may spin up but performance is unaffected.
    Fair,
    /// The system is actively shedding heat and may be throttling.
    Serious,
    /// Severe thermal pressure; aggressive throttling to prevent damage.
    Critical,
}

/// Android thermal status — the OS `PowerManager.getCurrentThermalStatus()`
/// `THERMAL_STATUS_*` levels. Mirrors the upstream Android type 1:1. Distinct
/// from [`AndroidThrottlingSeverity`] (the thermal-HAL per-sensor severity):
/// same seven level names, separate Android APIs, no mapping between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AndroidThermalStatus {
    /// No throttling. Serializes to `"none"`.
    None,
    Light,
    Moderate,
    Severe,
    Critical,
    Emergency,
    Shutdown,
}

/// Android thermal-HAL `ThrottlingSeverity` — the per-sensor throttling
/// severity reported by `android.hardware.thermal`. Mirrors the upstream type
/// 1:1. Same seven levels as [`AndroidThermalStatus`] but a distinct upstream
/// API (per-sensor HAL vs. device-level `PowerManager`); kept separate to
/// mirror upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AndroidThrottlingSeverity {
    /// No throttling. Serializes to `"none"`.
    None,
    Light,
    Moderate,
    Severe,
    Critical,
    Emergency,
    Shutdown,
}

/// One Android thermal-HAL `Temperature` reading for a single sensor at a
/// single measured repetition (an element of the flattened per-run list).
/// `iteration` tags which rep it was sampled at; `celsius` is a plain `i32` °C
/// (the client rounds the HAL `Temperature.value` float), stored as reported.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AndroidTemperatureSensor {
    /// Zero-based index of the measured repetition this reading was sampled at.
    pub iteration: i32,
    /// The sensor's `type` on the wire (renamed because `type` is a Rust
    /// keyword), a snake_case `TemperatureType` name, e.g. `cpu` / `skin`.
    #[serde(rename = "type")]
    pub sensor_type: String,
    pub name: String,
    pub celsius: i32,
    pub throttling_status: AndroidThrottlingSeverity,
}

/// One Linux thermal-zone reading for a single zone at a single measured
/// repetition (an element of the flattened per-run list). `iteration` tags the
/// rep; `celsius` is a plain `i32` °C (client converts + rounds sysfs milli-°C).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinuxThermalZone {
    /// Zero-based index of the measured repetition this reading was sampled at.
    pub iteration: i32,
    /// The zone's `type` on the wire (renamed because `type` is a Rust
    /// keyword), the sysfs zone `type`, e.g. `x86_pkg_temp` / `cpu-thermal`.
    #[serde(rename = "type")]
    pub zone_type: String,
    pub celsius: i32,
}

// ---------------------------------------------------------------------------
// Per-run series
//
// The sampling is per-platform and lives with the prober (`pipette_device`);
// only the shapes are here, with the leaf types they are built from.
// ---------------------------------------------------------------------------

/// One thermal snapshot for the current platform. Every field is `Option`;
/// only the running OS's thermal family is populated.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ThermalReading {
    pub apple_thermal_state: Option<AppleThermalState>,
    /// Apple SoC die temperature in whole °C, from the private IOHID
    /// `PMU tdie*` sensors — hottest reading, rounded at the capture site.
    /// Populated on iOS by the app layer
    /// (`socTemp()`, gated on a `PIPETTE_PRIVATE_THERMAL` build) and on macOS by
    /// `pipette_device::die_temp_max_c`; the same sensors and the same
    /// selection rule on both, so the series is comparable across them.
    /// `None` on Android/Linux, which keep their own families, and on any Apple
    /// host where the private symbols don't resolve.
    pub apple_soc_temp_c: Option<f32>,
    pub android_thermal_status: Option<AndroidThermalStatus>,
    pub android_thermal_headroom: Option<f32>,
    pub android_thermal_sensors: Option<Vec<AndroidTemperatureSensor>>,
    pub linux_thermal_zones: Option<Vec<LinuxThermalZone>>,
}

/// Per-iteration thermal snapshots bracketing each measured repetition:
/// `before[i]` at rep `i`'s gate-pass, `after[i]` once its timed work
/// completes. Carried out of the execute layer on [`RunResponse`](crate::run::RunResponse); the
/// submission builder flattens these into the per-iteration wire lists.
///
/// Empty for runners that don't gate — eval and max-memory — and deliberately
/// so. The series exists to say whether two *timing* results are comparable,
/// and neither an eval score nor a peak byte count moves with temperature: a
/// throttled device emits the same tokens and allocates the same weights, just
/// more slowly. An empty series on those rows is the intended reading rather
/// than a gap to fill.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunThermal {
    pub before: Vec<ThermalReading>,
    pub after: Vec<ThermalReading>,
}

impl RunThermal {
    /// Build from per-repetition `(before, after)` snapshot pairs, in rep order.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (ThermalReading, ThermalReading)>) -> Self {
        let (before, after) = pairs.into_iter().unzip();
        RunThermal { before, after }
    }
}

/// Per-platform per-iteration thermal telemetry — the reduction of a
/// [`ThermalReading`] series into the flattened wire fields, flattened in turn
/// into `BenchmarkSubmissionPayload`. Field names + snake_case enum values
/// mirror the management warehouse schema exactly. Each family is populated
/// only by the platform that exposes it (Apple state, plus SoC die temp on
/// iOS — `PIPETTE_PRIVATE_THERMAL` builds — and macOS; Android status +
/// headroom + thermal-HAL sensors; Linux sysfs zones); the rest stay `None`.
///
/// `_before` is sampled at each measured repetition's gate-pass, `_after` once
/// its timed work completes. Scalar families carry one value per repetition;
/// the sensor/zone families flatten every (iteration, sensor) pair into one
/// list, each element tagged with its `iteration`.
///
/// Lives here rather than with the payload because every platform's measurement
/// kernel builds it — including the Android JNI bridge, which never links the
/// submission path.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ThermalTelemetry {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_apple_thermal_state_before: Option<Vec<AppleThermalState>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_apple_thermal_state_after: Option<Vec<AppleThermalState>>,
    /// Apple SoC die temperature in whole °C, one reading per rep, in rep order
    /// (the array position *is* the iteration index). Sampled at the
    /// same points as the `device_apple_thermal_state_*` series, but *not*
    /// guaranteed element-aligned with it: the state series drops a rep whose
    /// token is unknown, so a shared index may not name the same iteration.
    /// Reported by iOS (gated on the `PIPETTE_PRIVATE_THERMAL` build) and by
    /// macOS, both from the private IOHID `PMU tdie*` sensors, each rounding at
    /// its own capture site. `None` when unavailable.
    ///
    /// Whole degrees rather than the sensor's fractional reading because this
    /// stays an `f32` column: only integers survive a reader widening it to
    /// 64-bit (46.8 comes back out as 46.79999923706055), and consecutive reps
    /// on a resting host span ~1 °C, so the lost digits described jitter. The
    /// type is unchanged — rows written before the rounding stay readable in the
    /// same column.
    ///
    /// On macOS this is recorded but *not* gated on: idle die temperature spans
    /// ~38–53 °C across the Mac fleet and the reading is a max over a per-host
    /// sensor count, so no constant threshold fits (see
    /// `pipette-readiness/src/macos.rs` and
    /// `docs/methodology/device-conditions.md`). It makes "did these two runs
    /// start from comparable temperatures?" answerable from the data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_apple_soc_temp_c_before: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_apple_soc_temp_c_after: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_android_thermal_status_before: Option<Vec<AndroidThermalStatus>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_android_thermal_status_after: Option<Vec<AndroidThermalStatus>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_android_thermal_headroom_before: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_android_thermal_headroom_after: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_android_thermal_sensors_before: Option<Vec<AndroidTemperatureSensor>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_android_thermal_sensors_after: Option<Vec<AndroidTemperatureSensor>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_linux_thermal_zones_before: Option<Vec<LinuxThermalZone>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_linux_thermal_zones_after: Option<Vec<LinuxThermalZone>>,
}

impl ThermalTelemetry {
    /// Build the wire fields from the per-repetition snapshot series (`before[i]`
    /// at rep `i`'s gate-pass, `after[i]` at its completion).
    pub fn from_series(before: &[ThermalReading], after: &[ThermalReading]) -> Self {
        ThermalTelemetry {
            device_apple_thermal_state_before: scalar_series(before, |r| r.apple_thermal_state),
            device_apple_thermal_state_after: scalar_series(after, |r| r.apple_thermal_state),
            device_apple_soc_temp_c_before: scalar_series(before, |r| r.apple_soc_temp_c),
            device_apple_soc_temp_c_after: scalar_series(after, |r| r.apple_soc_temp_c),
            device_android_thermal_status_before: scalar_series(before, |r| {
                r.android_thermal_status
            }),
            device_android_thermal_status_after: scalar_series(after, |r| r.android_thermal_status),
            device_android_thermal_headroom_before: scalar_series(before, |r| {
                r.android_thermal_headroom
            }),
            device_android_thermal_headroom_after: scalar_series(after, |r| {
                r.android_thermal_headroom
            }),
            device_android_thermal_sensors_before: sensor_series(before),
            device_android_thermal_sensors_after: sensor_series(after),
            device_linux_thermal_zones_before: zone_series(before),
            device_linux_thermal_zones_after: zone_series(after),
        }
    }
}

/// Collect a scalar thermal family across the per-rep readings into one value
/// per rep, in rep order. Scalars carry no explicit `iteration` (unlike the
/// sensor/zone lists) — the array index *is* the iteration — so this is
/// all-or-nothing: it emits the series only when every rep reported the family,
/// and `None` otherwise. That avoids silently shifting later values to the
/// wrong iteration if a family is intermittently absent (e.g. an Android
/// `getThermalHeadroom` `NaN` on one rep); an honestly-absent series beats a
/// misaligned one for telemetry.
fn scalar_series<T>(
    reps: &[ThermalReading],
    get: impl Fn(&ThermalReading) -> Option<T>,
) -> Option<Vec<T>> {
    let values: Vec<T> = reps.iter().filter_map(get).collect();
    (!values.is_empty() && values.len() == reps.len()).then_some(values)
}

/// Flatten the per-rep sensor lists into one list, stamping each element's
/// `iteration` with the rep it came from. `None` when no rep reported sensors.
fn sensor_series(reps: &[ThermalReading]) -> Option<Vec<AndroidTemperatureSensor>> {
    let flat: Vec<AndroidTemperatureSensor> = reps
        .iter()
        .enumerate()
        .flat_map(|(i, r)| {
            r.android_thermal_sensors
                .iter()
                .flatten()
                .map(move |s| AndroidTemperatureSensor {
                    iteration: i as i32,
                    ..s.clone()
                })
        })
        .collect();
    (!flat.is_empty()).then_some(flat)
}

/// Flatten the per-rep zone lists into one list, stamping each element's
/// `iteration`. See [`sensor_series`].
fn zone_series(reps: &[ThermalReading]) -> Option<Vec<LinuxThermalZone>> {
    let flat: Vec<LinuxThermalZone> = reps
        .iter()
        .enumerate()
        .flat_map(|(i, r)| {
            r.linux_thermal_zones
                .iter()
                .flatten()
                .map(move |z| LinuxThermalZone {
                    iteration: i as i32,
                    ..z.clone()
                })
        })
        .collect();
    (!flat.is_empty()).then_some(flat)
}

/// Volatile run-environment power state captured per benchmark submission.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PowerState {
    /// Battery charge percent (0–100); `None` on a device with no battery.
    pub battery_level: Option<i32>,
    /// Charging / discharging / on-AC-not-charging; `None` only when the
    /// state is genuinely unknown (detection failed). A battery-less desktop
    /// reports `PluggedInNotCharging` (definitively on external power), which
    /// the warehouse can distinguish from `None`.
    pub power_state: Option<DevicePowerState>,
    /// OS low-power / battery-saver mode active; `None` where undetectable.
    pub power_save_mode: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermal_enum_wire_forms_match_server() -> anyhow::Result<()> {
        // Exact snake_case strings the management server's warehouse enums
        // accept. `none` (not `null`) and the coolest→hottest ordering matter.
        assert_eq!(
            serde_json::to_string(&AppleThermalState::Serious)?,
            "\"serious\""
        );
        assert_eq!(
            serde_json::to_string(&AndroidThermalStatus::None)?,
            "\"none\""
        );
        assert_eq!(
            serde_json::to_string(&AndroidThrottlingSeverity::Shutdown)?,
            "\"shutdown\""
        );
        // `Ord` follows declaration order (coolest→hottest).
        assert!(AppleThermalState::Critical > AppleThermalState::Nominal);
        assert!(AndroidThermalStatus::Severe > AndroidThermalStatus::Light);
        Ok(())
    }

    #[test]
    fn thermal_sensor_element_wire_shape() -> anyhow::Result<()> {
        // The struct-element field names are the Parquet `List<Struct>` child
        // names: `iteration`, `type` (not `sensor_type`), `name`, `celsius`,
        // `throttling_status`.
        let sensor = AndroidTemperatureSensor {
            iteration: 2,
            sensor_type: "cpu".to_string(),
            name: "cpu-0-0-usr".to_string(),
            celsius: 61,
            throttling_status: AndroidThrottlingSeverity::Light,
        };
        let value = serde_json::to_value(&sensor)?;
        assert_eq!(value["iteration"], 2);
        assert_eq!(value["type"], "cpu");
        assert_eq!(value["celsius"], 61);
        assert_eq!(value["throttling_status"], "light");
        assert!(value.get("sensor_type").is_none());
        assert_eq!(
            serde_json::from_value::<AndroidTemperatureSensor>(value)?,
            sensor
        );

        let zone = LinuxThermalZone {
            iteration: 0,
            zone_type: "x86_pkg_temp".to_string(),
            celsius: 44,
        };
        let value = serde_json::to_value(&zone)?;
        assert_eq!(value["type"], "x86_pkg_temp");
        assert_eq!(value["celsius"], 44);
        assert_eq!(serde_json::from_value::<LinuxThermalZone>(value)?, zone);
        Ok(())
    }

    #[test]
    fn device_power_state_wire_form_matches_server() -> anyhow::Result<()> {
        // These exact strings are the management server's `DevicePowerState`
        // serde/strum representation — the serde output, `as_wire_str`, and
        // `Display` must all agree, or submissions get silently dropped.
        for (state, wire) in [
            (DevicePowerState::Charging, "charging"),
            (DevicePowerState::NotCharging, "not_charging"),
            (
                DevicePowerState::PluggedInNotCharging,
                "plugged_in_not_charging",
            ),
        ] {
            assert_eq!(serde_json::to_string(&state)?, format!("\"{wire}\""));
            assert_eq!(state.as_wire_str(), wire);
            assert_eq!(state.to_string(), wire);
        }
        Ok(())
    }

    #[test]
    fn scalar_series_is_all_or_nothing() {
        // Scalars encode iteration by array position, so a family that's
        // present for some reps but not others must not emit a shifted,
        // misaligned series — it drops to `None` instead. (Sensor/zone lists
        // are immune: they carry an explicit `iteration`.)
        let all_present = ThermalTelemetry::from_series(
            &[
                ThermalReading {
                    android_thermal_headroom: Some(0.3),
                    ..Default::default()
                },
                ThermalReading {
                    android_thermal_headroom: Some(0.5),
                    ..Default::default()
                },
            ],
            &[],
        );
        assert_eq!(
            all_present.device_android_thermal_headroom_before,
            Some(vec![0.3, 0.5])
        );

        let one_missing = ThermalTelemetry::from_series(
            &[
                ThermalReading {
                    android_thermal_headroom: Some(0.3),
                    ..Default::default()
                },
                // e.g. this rep's `getThermalHeadroom` returned NaN → None
                ThermalReading::default(),
            ],
            &[],
        );
        assert_eq!(one_missing.device_android_thermal_headroom_before, None);
    }

    /// The macOS die reading has to flow all the way into the wire field, not
    /// just onto `ThermalReading`. Uses synthetic readings so it doesn't shell
    /// out, and pins the per-rep ordering the array index encodes.
    #[test]
    fn apple_soc_temp_series_reaches_the_wire_fields() -> anyhow::Result<()> {
        let rep = |celsius: f32| ThermalReading {
            apple_thermal_state: Some(AppleThermalState::Nominal),
            apple_soc_temp_c: Some(celsius),
            ..Default::default()
        };
        let telemetry =
            ThermalTelemetry::from_series(&[rep(41.5), rep(39.0)], &[rep(58.25), rep(60.5)]);
        assert_eq!(
            telemetry.device_apple_soc_temp_c_before,
            Some(vec![41.5, 39.0])
        );
        assert_eq!(
            telemetry.device_apple_soc_temp_c_after,
            Some(vec![58.25, 60.5])
        );
        // Flattened at the top level under the warehouse's column name.
        let json = serde_json::to_value(&telemetry)?;
        assert_eq!(json["device_apple_soc_temp_c_before"][0], 41.5);
        assert_eq!(json["device_apple_soc_temp_c_after"][1], 60.5);
        Ok(())
    }

    /// A rep that reported no die temperature must suppress the whole series
    /// rather than shift later readings onto the wrong iteration — the array
    /// index *is* the rep number, so a gap would silently mislabel data.
    #[test]
    fn partial_apple_soc_temp_suppresses_the_series() {
        let with_temp = ThermalReading {
            apple_soc_temp_c: Some(44.0),
            ..Default::default()
        };
        let telemetry =
            ThermalTelemetry::from_series(&[with_temp.clone(), ThermalReading::default()], &[]);
        assert_eq!(telemetry.device_apple_soc_temp_c_before, None);
    }
}
