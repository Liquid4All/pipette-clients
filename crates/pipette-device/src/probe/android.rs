//! Android: the thermal status, headroom and HAL sensors `dumpsys` and `cmd
//! thermalservice` report. The parsers build in every test build, so they are
//! tested on every host.

#[cfg(target_os = "android")]
pub(super) fn android_thermal() -> pipette_plan_types::ThermalReading {
    // `dumpsys` needs `android.permission.DUMP` and `cmd thermalservice` a
    // shell-privileged context (adb, the CLI); an app without them gets `None`
    // and reads PowerManager instead.
    let dumpsys = super::cmd_output("dumpsys", &["thermalservice"]);
    pipette_plan_types::ThermalReading {
        android_thermal_status: dumpsys.as_deref().and_then(parse_android_thermal_status),
        android_thermal_headroom: super::cmd_output("cmd", &["thermalservice", "headroom", "0"])
            .as_deref()
            .and_then(parse_android_headroom),
        android_thermal_sensors: dumpsys.as_deref().and_then(parse_android_hal_sensors),
        ..Default::default()
    }
}

/// The thermal-HAL sensors alone, for the in-app sampler, which reads status
/// and headroom from PowerManager. Needs `android.permission.DUMP` (granted on
/// lab devices via `adb shell pm grant`); `None` without it.
#[cfg(target_os = "android")]
pub fn android_hal_sensors() -> Option<Vec<pipette_plan_types::AndroidTemperatureSensor>> {
    super::cmd_output("dumpsys", &["thermalservice"])
        .as_deref()
        .and_then(parse_android_hal_sensors)
}

/// The `Thermal Status: N` line of `dumpsys thermalservice`.
fn parse_android_thermal_status(stdout: &str) -> Option<pipette_plan_types::AndroidThermalStatus> {
    let rest = stdout
        .lines()
        .find_map(|line| line.trim().strip_prefix("Thermal Status:").map(str::trim))?;
    super::android_thermal_status_from_code(rest.parse::<i32>().ok()?)
}

/// `cmd thermalservice headroom N` prints `NaN` when rate-limited (more than
/// about one call a second), which is dropped.
fn parse_android_headroom(stdout: &str) -> Option<f32> {
    let line = stdout.lines().find(|line| line.contains("Headroom"))?;
    let value: f32 = line.rsplit(':').next()?.trim().parse().ok()?;
    value.is_finite().then_some(value)
}

/// The live `Current temperatures from HAL:` block of `dumpsys thermalservice`;
/// a stale `Cached temperatures:` block also appears. A sensor with no reading
/// reports `-FLT_MAX`, which the band check drops.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the band check bounds `value` to -50..=200"
)]
fn parse_android_hal_sensors(
    stdout: &str,
) -> Option<Vec<pipette_plan_types::AndroidTemperatureSensor>> {
    let mut in_current = false;
    let sensors: Vec<pipette_plan_types::AndroidTemperatureSensor> = stdout
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            if line == "Current temperatures from HAL:" {
                in_current = true;
                return None;
            }
            if !in_current {
                return None;
            }
            if !line.starts_with("Temperature{") {
                in_current = false;
                return None;
            }
            let m_type: i32 = dumpsys_field(line, "mType=")?.parse().ok()?;
            let sensor_type = android_temperature_type(m_type)?;
            let value: f64 = dumpsys_field(line, "mValue=")?.parse().ok()?;
            if !value.is_finite() || !(-50.0..=200.0).contains(&value) {
                return None;
            }
            let m_status: i32 = dumpsys_field(line, "mStatus=")?.parse().ok()?;
            Some(pipette_plan_types::AndroidTemperatureSensor {
                // Stamped later by the per-run builder.
                iteration: 0,
                sensor_type: sensor_type.to_string(),
                name: dumpsys_field(line, "mName=")?.to_string(),
                celsius: value.round() as i32,
                throttling_status: android_throttling_severity(m_status),
            })
        })
        .collect();
    (!sensors.is_empty()).then_some(sensors)
}

/// An `android.hardware.thermal` `TemperatureType` ordinal's wire name. `None`
/// for `UNKNOWN` and the `BCL_*` sensors (6 to 8), which are not in °C.
fn android_temperature_type(m_type: i32) -> Option<&'static str> {
    Some(match m_type {
        0 => "cpu",
        1 => "gpu",
        2 => "battery",
        3 => "skin",
        4 => "usb_port",
        5 => "power_amplifier",
        9 => "npu",
        10 => "tpu",
        11 => "display",
        12 => "modem",
        13 => "soc",
        14 => "wifi",
        15 => "camera",
        16 => "flashlight",
        17 => "speaker",
        18 => "ambient",
        19 => "pogo",
        _ => return None,
    })
}

/// An `android.hardware.thermal` `ThrottlingSeverity` ordinal; unknown is `None`.
fn android_throttling_severity(m_status: i32) -> pipette_plan_types::AndroidThrottlingSeverity {
    use pipette_plan_types::AndroidThrottlingSeverity as S;
    match m_status {
        1 => S::Light,
        2 => S::Moderate,
        3 => S::Severe,
        4 => S::Critical,
        5 => S::Emergency,
        6 => S::Shutdown,
        _ => S::None,
    }
}

/// The value of `key` (e.g. `"mType="`) in a `Temperature{..}` line.
fn dumpsys_field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let (_, rest) = line.split_once(key)?;
    let value = rest.split_once([',', '}']).map_or(rest, |(value, _)| value);
    Some(value.trim())
}

#[cfg(test)]
mod tests {
    use anyhow::Context;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(
        "IsStatusOverride: false\nThermal Status: 2\nHAL Ready: true\n",
        Some(pipette_plan_types::AndroidThermalStatus::Moderate)
    )]
    #[case(
        "Thermal Status: 0\n",
        Some(pipette_plan_types::AndroidThermalStatus::None)
    )]
    #[case("no status line", None)]
    fn android_thermal_status_maps_levels(
        #[case] dump: &str,
        #[case] want: Option<pipette_plan_types::AndroidThermalStatus>,
    ) {
        assert_eq!(parse_android_thermal_status(dump), want);
    }

    #[rstest]
    #[case("Headroom in 0 seconds: 0.53", Some(0.53))]
    #[case("Headroom in 5 seconds: 1.07\n", Some(1.07))]
    #[case("Headroom in 0 seconds: NaN", None)] // rate-limited
    #[case("unrelated", None)]
    fn android_headroom_parses_and_rejects_nan(#[case] out: &str, #[case] want: Option<f32>) {
        assert_eq!(parse_android_headroom(out), want);
    }

    /// Modeled on a Galaxy S25 Ultra's (SM-S938W, Android 16) output: a stale
    /// `Cached` block, the GPU at `-FLT_MAX` and a `BCL_V` sensor, all dropped.
    const S25_DUMPSYS: &str = "\
IsStatusOverride: false
Thermal Status: 1
Cached temperatures:
\tTemperature{mValue=99.0, mType=0, mName=AP, mStatus=3}
HAL Ready: true
Current temperatures from HAL:
\tTemperature{mValue=33.1, mType=0, mName=AP, mStatus=0}
\tTemperature{mValue=31.1, mType=2, mName=BAT, mStatus=1}
\tTemperature{mValue=34.3, mType=5, mName=PA, mStatus=0}
\tTemperature{mValue=32.7, mType=3, mName=SKIN, mStatus=0}
\tTemperature{mValue=-3.4028235E38, mType=1, mName=GPU, mStatus=0}
\tTemperature{mValue=0.0, mType=6, mName=BCL_V, mStatus=0}
\tTemperature{mValue=30.1, mType=4, mName=USB, mStatus=0}
Current cooling devices from HAL:
\tCoolingDevice{mValue=0, mType=1, mName=x}
";

    #[test]
    fn android_hal_sensors_parses_live_block_only() -> anyhow::Result<()> {
        let sensors = parse_android_hal_sensors(S25_DUMPSYS).context("live sensors present")?;
        assert_eq!(
            sensors,
            vec![
                pipette_plan_types::AndroidTemperatureSensor {
                    iteration: 0,
                    sensor_type: "cpu".into(),
                    name: "AP".into(),
                    celsius: 33,
                    throttling_status: pipette_plan_types::AndroidThrottlingSeverity::None,
                },
                pipette_plan_types::AndroidTemperatureSensor {
                    iteration: 0,
                    sensor_type: "battery".into(),
                    name: "BAT".into(),
                    celsius: 31,
                    throttling_status: pipette_plan_types::AndroidThrottlingSeverity::Light,
                },
                pipette_plan_types::AndroidTemperatureSensor {
                    iteration: 0,
                    sensor_type: "power_amplifier".into(),
                    name: "PA".into(),
                    celsius: 34,
                    throttling_status: pipette_plan_types::AndroidThrottlingSeverity::None,
                },
                pipette_plan_types::AndroidTemperatureSensor {
                    iteration: 0,
                    sensor_type: "skin".into(),
                    name: "SKIN".into(),
                    celsius: 33,
                    throttling_status: pipette_plan_types::AndroidThrottlingSeverity::None,
                },
                pipette_plan_types::AndroidTemperatureSensor {
                    iteration: 0,
                    sensor_type: "usb_port".into(),
                    name: "USB".into(),
                    celsius: 30,
                    throttling_status: pipette_plan_types::AndroidThrottlingSeverity::None,
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn android_hal_sensors_none_when_no_live_block() {
        assert_eq!(parse_android_hal_sensors("Thermal Status: 0\n"), None);
    }
}
