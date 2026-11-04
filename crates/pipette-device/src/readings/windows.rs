//! Windows: the counter sets below through PDH, in-process, and the GPU's
//! adapter and node data ([`gpu`]). None of it needs a driver.
//! `MSAcpi_ThermalZoneTemperature` is not read: it needs WMI, its temperatures
//! are the counter set's, and its trip points do not change.

use serde_json::Value;

use super::DeviceReadings;

/// Which field a counter's value lands in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Hardware,
    OsThrottling,
}

/// One counter asked of PDH. Its name keys its value in the record.
struct Counter {
    facility: &'static str,
    set: &'static str,
    /// `*` for every instance.
    instance: &'static str,
    name: &'static str,
    side: Side,
}

impl Counter {
    /// The English path PDH reads it at.
    fn path(&self) -> String {
        format!("\\{}({})\\{}", self.set, self.instance, self.name)
    }
}

const THERMAL_ZONES: &str = "windows_thermal_zone_information_counters";
const ESIF: &str = "windows_esif_device_information_counters";
const PROCESSOR: &str = "windows_processor_information_counters";

const fn zone(name: &'static str, side: Side) -> Counter {
    Counter {
        facility: THERMAL_ZONES,
        set: "Thermal Zone Information",
        instance: "*",
        name,
        side,
    }
}

const fn processor(name: &'static str, side: Side) -> Counter {
    Counter {
        facility: PROCESSOR,
        set: "Processor Information",
        instance: "_Total",
        name,
        side,
    }
}

/// The counters read, in record order.
///
/// `Throttle Reasons`, `% Passive Limit`, `% Performance Limit` and
/// `Performance Limit Flags` are the OS's verdict rather than a sensor, so they
/// go to `os_throttling`, under the facility and instance of the sensors beside
/// them. No power is recorded, so `\Energy Meter` is not read.
const COUNTERS: [Counter; 10] = [
    zone("Temperature", Side::Hardware),
    zone("High Precision Temperature", Side::Hardware),
    zone("Throttle Reasons", Side::OsThrottling),
    zone("% Passive Limit", Side::OsThrottling),
    Counter {
        facility: ESIF,
        set: "EsifDeviceInformation",
        instance: "*",
        name: "Temperature",
        side: Side::Hardware,
    },
    processor("Actual Frequency", Side::Hardware),
    processor("% Processor Performance", Side::Hardware),
    processor("% Processor Utility", Side::Hardware),
    processor("% Performance Limit", Side::OsThrottling),
    processor("Performance Limit Flags", Side::OsThrottling),
];

/// One value PDH formatted: which of [`COUNTERS`], for which instance.
struct CounterValue {
    counter: usize,
    instance: String,
    value: f64,
}

/// One record per facility and instance: `InstanceName` beside each counter's
/// value, under the counter's own name.
fn group(values: &[CounterValue]) -> DeviceReadings {
    values
        .iter()
        .filter_map(|v| Some((COUNTERS.get(v.counter)?, v, json_number(v.value)?)))
        .fold(
            DeviceReadings::default(),
            |mut readings, (counter, v, number)| {
                let side = match counter.side {
                    Side::Hardware => &mut readings.hardware,
                    Side::OsThrottling => &mut readings.os_throttling,
                };
                let Value::Array(records) = side
                    .entry(counter.facility)
                    .or_insert_with(|| Value::Array(Vec::new()))
                else {
                    return readings;
                };
                let index = match records.iter().position(|r| {
                    r.get("InstanceName").and_then(Value::as_str) == Some(&v.instance)
                }) {
                    Some(index) => index,
                    None => {
                        records.push(serde_json::json!({ "InstanceName": v.instance }));
                        records.len().saturating_sub(1)
                    }
                };
                if let Some(Value::Object(record)) = records.get_mut(index) {
                    record.insert(counter.name.to_string(), number);
                }
                readings
            },
        )
}

/// A deci-°C reading in whole degrees, rounded like every other sensor the
/// gate reads; `None` for the 0 a driver leaves when it reports nothing.
fn whole_celsius(deci: u32) -> Option<u32> {
    Some(deci.saturating_add(5) / 10).filter(|c| *c > 0)
}

/// A counter's value as JSON. An integral one is an integer, as its unit has it
/// (`Temperature` in whole kelvins, a flags bitmask); `None` for a NaN or an
/// infinity, which JSON cannot hold.
fn json_number(value: f64) -> Option<Value> {
    // 2^53: past it a double no longer holds every integer.
    const EXACT: f64 = 9_007_199_254_740_992.0;
    if value.fract() == 0.0 && value.abs() < EXACT {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "integral and inside the range a double holds exactly, checked above"
        )]
        let integral = value as i64;
        return Some(Value::from(integral));
    }
    serde_json::Number::from_f64(value).map(Value::Number)
}

#[cfg(target_os = "windows")]
pub(super) fn read() -> DeviceReadings {
    let mut readings = group(&pdh::read_counters());
    let adapters = gpu::read_adapters();
    let perf: Vec<Value> = adapters
        .iter()
        .filter_map(|a| a.perf.as_ref().map(|p| gpu::adapter_record(a.luid, p)))
        .collect();
    let nodes: Vec<Value> = adapters
        .iter()
        .flat_map(|a| a.nodes.iter().map(|n| gpu::node_record(a.luid, n)))
        .collect();
    if !perf.is_empty() {
        readings.hardware.insert(
            "windows_d3dkmt_adapter_perf_data".into(),
            Value::Array(perf),
        );
    }
    if !nodes.is_empty() {
        readings
            .hardware
            .insert("windows_d3dkmt_node_perf_data".into(), Value::Array(nodes));
    }
    readings
}

#[cfg(target_os = "windows")]
mod pdh {
    use std::ffi::c_void;
    use std::time::Duration;

    use windows_sys::Win32::Foundation::ERROR_SUCCESS;
    use windows_sys::Win32::System::Performance::{
        PDH_CSTATUS_NEW_DATA, PDH_CSTATUS_VALID_DATA, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_DOUBLE,
        PDH_MORE_DATA, PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData,
        PdhGetFormattedCounterArrayW, PdhOpenQueryW,
    };

    use super::{COUNTERS, CounterValue};

    /// Between PDH's two collections, which the rate counters
    /// (`% Processor Performance`, `% Processor Utility`) are formatted over.
    const COUNTER_WINDOW: Duration = Duration::from_secs(1);
    /// `pdh.h`; windows-sys does not carry it. Without it PDH caps a percentage
    /// at 100, and `% Processor Performance` passes 100 under boost.
    const PDH_FMT_NOCAP100: u32 = 0x0000_8000;

    /// Every counter of [`COUNTERS`] this box has, over [`COUNTER_WINDOW`].
    pub(super) fn read_counters() -> Vec<CounterValue> {
        let Some(query) = Query::open() else {
            return Vec::new();
        };
        let added: Vec<(usize, *mut c_void)> = COUNTERS
            .iter()
            .enumerate()
            .filter_map(|(index, c)| query.add(&c.path()).map(|handle| (index, handle)))
            .collect();
        if added.is_empty() || !query.collect() {
            return Vec::new();
        }
        std::thread::sleep(COUNTER_WINDOW);
        if !query.collect() {
            return Vec::new();
        }
        let mut buffer = Vec::new();
        added
            .into_iter()
            .flat_map(|(counter, handle)| {
                formatted(handle, &mut buffer)
                    .into_iter()
                    .map(move |(instance, value)| CounterValue {
                        counter,
                        instance,
                        value,
                    })
            })
            .collect()
    }

    struct Query(*mut c_void);

    impl Query {
        fn open() -> Option<Self> {
            let mut query = std::ptr::null_mut();
            // SAFETY: `query` is a live out-pointer; a null data source means
            // real-time data.
            let rc = unsafe { PdhOpenQueryW(std::ptr::null(), 0, &raw mut query) };
            (rc == ERROR_SUCCESS).then_some(Self(query))
        }

        /// The counter at `path`, or `None` where this box has no such counter.
        fn add(&self, path: &str) -> Option<*mut c_void> {
            let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
            let mut counter = std::ptr::null_mut();
            // SAFETY: `wide` is NUL-terminated and outlives the call, and
            // `counter` is a live out-pointer.
            let rc = unsafe { PdhAddEnglishCounterW(self.0, wide.as_ptr(), 0, &raw mut counter) };
            (rc == ERROR_SUCCESS).then_some(counter)
        }

        fn collect(&self) -> bool {
            // SAFETY: the query is open until drop.
            unsafe { PdhCollectQueryData(self.0) == ERROR_SUCCESS }
        }
    }

    impl Drop for Query {
        fn drop(&mut self) {
            // SAFETY: opened in `open` and closed once, here.
            unsafe { PdhCloseQuery(self.0) };
        }
    }

    /// Each instance's value of `counter`, skipping one PDH could not format.
    /// `buffer` is `u64` words so the items PDH writes into it are aligned.
    fn formatted(counter: *mut c_void, buffer: &mut Vec<u64>) -> Vec<(String, f64)> {
        let format = PDH_FMT_DOUBLE | PDH_FMT_NOCAP100;
        let mut size: u32 = 0;
        let mut count: u32 = 0;
        // SAFETY: a null buffer with `size == 0` asks only for the size needed.
        let rc = unsafe {
            PdhGetFormattedCounterArrayW(
                counter,
                format,
                &raw mut size,
                &raw mut count,
                std::ptr::null_mut(),
            )
        };
        let Ok(bytes) = usize::try_from(size) else {
            return Vec::new();
        };
        if rc != PDH_MORE_DATA || bytes == 0 {
            return Vec::new();
        }
        buffer.clear();
        buffer.resize(bytes.div_ceil(size_of::<u64>()), 0);
        // SAFETY: `buffer` holds at least `size` bytes, the size PDH asked for.
        let rc = unsafe {
            PdhGetFormattedCounterArrayW(
                counter,
                format,
                &raw mut size,
                &raw mut count,
                buffer.as_mut_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>(),
            )
        };
        let Ok(count) = usize::try_from(count) else {
            return Vec::new();
        };
        let capacity = buffer.len().saturating_mul(size_of::<u64>());
        if rc != ERROR_SUCCESS
            || count.saturating_mul(size_of::<PDH_FMT_COUNTERVALUE_ITEM_W>()) > capacity
        {
            return Vec::new();
        }
        // SAFETY: PDH wrote `count` items at the start of `buffer`, checked to
        // fit above, and `u64` alignment covers the item's (asserted below).
        let items = unsafe {
            std::slice::from_raw_parts(buffer.as_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>(), count)
        };
        // PDH writes the names into the same buffer, after the items, so each
        // is read as an offset into it rather than through its pointer.
        // SAFETY: `buffer` is `capacity` initialized bytes, as many `u16`s as
        // fit in it.
        let words = unsafe {
            std::slice::from_raw_parts(buffer.as_ptr().cast::<u16>(), capacity / size_of::<u16>())
        };
        let base = buffer.as_ptr().addr();
        items
            .iter()
            .filter(|item| {
                [PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA].contains(&item.FmtValue.CStatus)
            })
            .filter_map(|item| {
                let start = item.szName.addr().checked_sub(base)? / size_of::<u16>();
                let name = words.get(start..)?;
                let name =
                    String::from_utf16(name.get(..name.iter().position(|w| *w == 0)?)?).ok()?;
                // SAFETY: the `doubleValue` arm is the one PDH_FMT_DOUBLE fills.
                let value = unsafe { item.FmtValue.Anonymous.doubleValue };
                Some((name, value))
            })
            .collect()
    }

    const _: () =
        assert!(std::mem::align_of::<PDH_FMT_COUNTERVALUE_ITEM_W>() <= std::mem::align_of::<u64>());
}

#[cfg(target_os = "windows")]
pub(super) mod gpu;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn value(facility: &str, name: &str, instance: &str, value: f64) -> CounterValue {
        CounterValue {
            counter: COUNTERS
                .iter()
                .position(|c| c.facility == facility && c.name == name)
                .unwrap_or(usize::MAX),
            instance: instance.to_string(),
            value,
        }
    }

    /// Zone values land in one record per zone, split between the two fields
    /// by what each counter is.
    #[test]
    fn values_group_by_facility_and_instance() {
        let readings = group(&[
            value(THERMAL_ZONES, "Temperature", r"\_TZ.THZ0", 316.0),
            value(
                THERMAL_ZONES,
                "High Precision Temperature",
                r"\_TZ.THZ0",
                3162.0,
            ),
            value(THERMAL_ZONES, "Throttle Reasons", r"\_TZ.THZ0", 0.0),
            value(THERMAL_ZONES, "Temperature", r"\_TZ.TZ01", 293.0),
            value(ESIF, "Temperature", "IETM", 41.0),
            value(PROCESSOR, "% Processor Performance", "_Total", 99.6),
            value(PROCESSOR, "Performance Limit Flags", "_Total", 2.0),
        ]);
        assert_eq!(
            Value::Object(readings.hardware),
            json!({
                "windows_thermal_zone_information_counters": [
                    {
                        "InstanceName": r"\_TZ.THZ0",
                        "Temperature": 316,
                        "High Precision Temperature": 3162,
                    },
                    { "InstanceName": r"\_TZ.TZ01", "Temperature": 293 },
                ],
                "windows_esif_device_information_counters": [
                    { "InstanceName": "IETM", "Temperature": 41 },
                ],
                "windows_processor_information_counters": [
                    { "InstanceName": "_Total", "% Processor Performance": 99.6 },
                ],
            })
        );
        assert_eq!(
            Value::Object(readings.os_throttling),
            json!({
                "windows_thermal_zone_information_counters": [
                    { "InstanceName": r"\_TZ.THZ0", "Throttle Reasons": 0 },
                ],
                "windows_processor_information_counters": [
                    { "InstanceName": "_Total", "Performance Limit Flags": 2 },
                ],
            })
        );
    }

    /// Nothing read is nothing posted: no empty facility, no empty record.
    #[test]
    fn no_values_is_no_readings() {
        assert_eq!(group(&[]), DeviceReadings::default());
    }

    #[rstest]
    #[case::rounds_down(434, Some(43))]
    #[case::rounds_half_up(435, Some(44))]
    #[case::unset(0, None)]
    fn the_gpu_temperature_is_in_whole_degrees(#[case] deci: u32, #[case] expected: Option<u32>) {
        assert_eq!(whole_celsius(deci), expected);
    }

    #[test]
    fn integral_values_are_integers_and_the_rest_as_read() {
        assert_eq!(json_number(293.0), Some(json!(293)));
        assert_eq!(json_number(99.6), Some(json!(99.6)));
        assert_eq!(json_number(-1.0), Some(json!(-1)));
        assert_eq!(json_number(f64::NAN), None);
        assert_eq!(json_number(f64::INFINITY), None);
    }

    /// The paths are PDH's English ones: a typo reads as a counter the box
    /// does not have, silently.
    #[test]
    fn counters_are_read_at_their_english_paths() {
        let paths: Vec<String> = COUNTERS.iter().map(Counter::path).collect();
        assert!(paths.contains(&r"\Thermal Zone Information(*)\High Precision Temperature".into()));
        assert!(paths.contains(&r"\Processor Information(_Total)\Performance Limit Flags".into()));
        assert!(paths.contains(&r"\EsifDeviceInformation(*)\Temperature".into()));
    }

    /// No power is recorded: no counter reads one.
    #[test]
    fn no_counter_reads_power() {
        assert!(
            COUNTERS
                .iter()
                .all(|c| c.set != "Energy Meter" && c.name != "Power"),
            "a counter reads power"
        );
    }

    /// No power is recorded: the GPU adapter's record leaves out the struct's
    /// `Power`.
    #[cfg(target_os = "windows")]
    #[test]
    fn the_adapter_record_carries_no_power() {
        let record = gpu::adapter_record(
            windows_sys::Win32::Foundation::LUID::default(),
            &windows_sys::Wdk::Graphics::Direct3D::D3DKMT_ADAPTER_PERFDATA::default(),
        );
        assert!(record.get("Power").is_none(), "{record}");
        assert!(record.get("Temperature").is_some(), "{record}");
    }

    /// The whole reader on this host, run by the Windows CI leg: whatever the
    /// box has, each facility is a list of records, and every Windows host has
    /// `\Processor Information(_Total)`.
    #[cfg(target_os = "windows")]
    #[test]
    fn this_host_reads_records() {
        let readings = read();
        assert!(readings.hardware.contains_key(PROCESSOR), "{readings:?}");
        assert!(
            readings
                .hardware
                .values()
                .chain(readings.os_throttling.values())
                .all(|f| f.as_array().is_some_and(|r| r.iter().all(Value::is_object))),
            "{readings:?}"
        );
    }
}
