//! The client configuration file's `[readiness.<gate>]` table: measured facts
//! about the device type that tune its gate. Results carry only Android's
//! effective skin threshold, alongside the cell's other readiness flags.
//!
//! ```toml
//! [readiness.windows]
//! max_cpu_busy_pct = 15            # CPU busy share that still counts as idle
//! ceilings = { "acpi:THZ0" = 43 }  # hold each rep until the sensor reads <= °C
//! ```
//!
//! The other gates take the numbers they compare against. A `*_below_*` key
//! replaces the gate's own number in the same strict comparison.
//!
//! ```toml
//! [readiness.macos]
//! die_below_c = 60
//! cpu_busy_below_cores = 2
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer};

use crate::ReadinessPlatform;

/// The `[readiness]` table: the settings of one gate. No arm for
/// [`ReadinessPlatform::Other`], which checks nothing.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub enum ReadinessConfiguration {
    #[serde(rename = "android")]
    Android(AndroidReadinessConfiguration),
    #[serde(rename = "linux")]
    Linux(LinuxReadinessConfiguration),
    #[serde(rename = "pi5")]
    RaspberryPi5(RaspberryPi5ReadinessConfiguration),
    #[serde(rename = "orin")]
    JetsonOrin(JetsonOrinReadinessConfiguration),
    #[serde(rename = "macos")]
    MacOs(MacOsReadinessConfiguration),
    #[serde(rename = "windows")]
    Windows(WindowsReadinessConfiguration),
}

impl ReadinessConfiguration {
    pub fn platform(&self) -> ReadinessPlatform {
        match self {
            Self::Android(_) => ReadinessPlatform::Android,
            Self::Linux(_) => ReadinessPlatform::Linux,
            Self::RaspberryPi5(_) => ReadinessPlatform::RaspberryPi5,
            Self::JetsonOrin(_) => ReadinessPlatform::JetsonOrin,
            Self::MacOs(_) => ReadinessPlatform::MacOs,
            Self::Windows(_) => ReadinessPlatform::Windows,
        }
    }

    pub fn android(&self) -> Option<&AndroidReadinessConfiguration> {
        match self {
            Self::Android(android) => Some(android),
            Self::Linux(_)
            | Self::RaspberryPi5(_)
            | Self::JetsonOrin(_)
            | Self::MacOs(_)
            | Self::Windows(_) => None,
        }
    }

    pub fn linux(&self) -> Option<&LinuxReadinessConfiguration> {
        match self {
            Self::Linux(linux) => Some(linux),
            Self::Android(_)
            | Self::RaspberryPi5(_)
            | Self::JetsonOrin(_)
            | Self::MacOs(_)
            | Self::Windows(_) => None,
        }
    }

    pub fn pi5(&self) -> Option<&RaspberryPi5ReadinessConfiguration> {
        match self {
            Self::RaspberryPi5(pi5) => Some(pi5),
            Self::Android(_)
            | Self::Linux(_)
            | Self::JetsonOrin(_)
            | Self::MacOs(_)
            | Self::Windows(_) => None,
        }
    }

    pub fn orin(&self) -> Option<&JetsonOrinReadinessConfiguration> {
        match self {
            Self::JetsonOrin(orin) => Some(orin),
            Self::Android(_)
            | Self::Linux(_)
            | Self::RaspberryPi5(_)
            | Self::MacOs(_)
            | Self::Windows(_) => None,
        }
    }

    pub fn macos(&self) -> Option<&MacOsReadinessConfiguration> {
        match self {
            Self::MacOs(macos) => Some(macos),
            Self::Android(_)
            | Self::Linux(_)
            | Self::RaspberryPi5(_)
            | Self::JetsonOrin(_)
            | Self::Windows(_) => None,
        }
    }

    pub fn windows(&self) -> Option<&WindowsReadinessConfiguration> {
        match self {
            Self::Windows(windows) => Some(windows),
            Self::Android(_)
            | Self::Linux(_)
            | Self::RaspberryPi5(_)
            | Self::JetsonOrin(_)
            | Self::MacOs(_) => None,
        }
    }
}

impl fmt::Display for ReadinessConfiguration {
    /// For the `run` log line.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "readiness.{}", self.platform())?;
        let celsius = |c: Option<CeilingCelsius>| c.map(|c| c.to_string());
        let percent = |pct: Option<CpuPercent>| pct.map(|pct| pct.to_string());
        match self {
            Self::Android(android) => write_set_keys(
                f,
                [
                    ("hottest_below_c", celsius(android.hottest_below_c)),
                    ("skin_below_c", celsius(android.skin_below_c)),
                    ("cpu_busy_below_pct", percent(android.cpu_busy_below_pct)),
                    ("held_touch", android.held_touch.map(|h| h.to_string())),
                ],
            ),
            Self::Linux(linux) => write_set_keys(
                f,
                [
                    ("hottest_below_c", celsius(linux.hottest_below_c)),
                    ("cpu_load_below_pct", percent(linux.cpu_load_below_pct)),
                ],
            ),
            Self::RaspberryPi5(pi5) => write_set_keys(
                f,
                [
                    ("hottest_below_c", celsius(pi5.hottest_below_c)),
                    ("cpu_load_below_pct", percent(pi5.cpu_load_below_pct)),
                ],
            ),
            Self::JetsonOrin(orin) => write_set_keys(
                f,
                [
                    ("hottest_below_c", celsius(orin.hottest_below_c)),
                    ("cpu_load_below_pct", percent(orin.cpu_load_below_pct)),
                    ("gpu_load_below_pct", percent(orin.gpu_load_below_pct)),
                ],
            ),
            Self::MacOs(macos) => write_set_keys(
                f,
                [
                    ("die_below_c", celsius(macos.die_below_c)),
                    (
                        "cpu_busy_below_cores",
                        macos.cpu_busy_below_cores.map(|c| c.get().to_string()),
                    ),
                ],
            ),
            Self::Windows(windows) => write!(f, ": {windows}"),
        }
    }
}

fn write_set_keys<const N: usize>(
    f: &mut fmt::Formatter<'_>,
    keys: [(&str, Option<String>); N],
) -> fmt::Result {
    let set: Vec<String> = keys
        .into_iter()
        .filter_map(|(key, value)| value.map(|value| format!("{key}={value}")))
        .collect();
    if set.is_empty() {
        return Ok(());
    }
    write!(f, ": {}", set.join(" "))
}

/// `[readiness.android]`, the CLI's gate on an Android device. The app's own
/// gate does not read it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AndroidReadinessConfiguration {
    /// °C the hottest CPU-cluster zone must read below.
    #[serde(default)]
    pub hottest_below_c: Option<CeilingCelsius>,
    /// °C the hottest live HAL skin sensor must read below. Unset disables it.
    #[serde(default)]
    pub skin_below_c: Option<CeilingCelsius>,
    /// CPU busy share, over all cores, the device must be below.
    #[serde(default)]
    pub cpu_busy_below_pct: Option<CpuPercent>,
    /// What the gate does about a touch held on the touchscreen. Unset is
    /// [`HeldTouch::Wait`].
    #[serde(default)]
    pub held_touch: Option<HeldTouch>,
}

/// What the Android CLI gate does if a touch is held on the touchscreen.
///
/// A phantom touch (for example, from a screen cooler) stays until an
/// operator removes the cause, so a wait can only end at the deadline.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HeldTouch {
    /// Do not check touchscreen interrupts or pointers. Thermal and CPU load
    /// criteria still apply.
    Off,
    /// Hold the gate until the touch clears or the deadline passes, as for a
    /// busy CPU.
    #[default]
    Wait,
    /// Fail after a few consecutive polls detect the touch. Report the failure
    /// within seconds, before the deadline.
    Fail,
}

#[derive(Debug, thiserror::Error)]
#[error("`{0}` is not a held-touch mode; expected `off`, `wait`, or `fail`")]
pub struct HeldTouchError(String);

impl FromStr for HeldTouch {
    type Err = HeldTouchError;

    /// The grammar of the `--readiness-held-touch` flags and of
    /// `PIPETTE_READINESS_HELD_TOUCH`: any of the three words, in any case.
    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "wait" => Ok(Self::Wait),
            "fail" => Ok(Self::Fail),
            _ => Err(HeldTouchError(raw.to_string())),
        }
    }
}

impl fmt::Display for HeldTouch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Off => "off",
            Self::Wait => "wait",
            Self::Fail => "fail",
        })
    }
}

/// `[readiness.linux]`, a Linux host other than a Raspberry Pi 5 or a Jetson Orin.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinuxReadinessConfiguration {
    /// °C the hottest thermal zone must read below. A host with no thermal zone
    /// fails a wait that sets it.
    #[serde(default)]
    pub hottest_below_c: Option<CeilingCelsius>,
    /// CPU load, as a share of the cores the client may run on, the host must
    /// be below: the 1-minute load average, or the busy share of those cores
    /// for a client confined to some of the host's.
    #[serde(default)]
    pub cpu_load_below_pct: Option<CpuPercent>,
}

/// `[readiness.pi5]`. The firmware's throttle bits, where `vcgencmd` reads
/// them, hold the gate whatever is set here.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RaspberryPi5ReadinessConfiguration {
    /// °C the hottest thermal zone must read below. The gate's own is the
    /// firmware's soft limit.
    #[serde(default)]
    pub hottest_below_c: Option<CeilingCelsius>,
    /// CPU load, as a share of the cores the client may run on, the host must
    /// be below: the 1-minute load average, or the busy share of those cores
    /// for a client confined to some of the host's.
    #[serde(default)]
    pub cpu_load_below_pct: Option<CpuPercent>,
}

/// `[readiness.orin]`. A cooling device throttling a clock, or an
/// over-current event since the previous poll, holds the gate whatever is set
/// here.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JetsonOrinReadinessConfiguration {
    /// °C the hottest thermal zone must read below. The gate's own is the
    /// board's hot-surface trip.
    #[serde(default)]
    pub hottest_below_c: Option<CeilingCelsius>,
    /// CPU load, as a share of the cores the client may run on, the host must
    /// be below: the 1-minute load average, or the busy share of those cores
    /// for a client confined to some of the host's.
    #[serde(default)]
    pub cpu_load_below_pct: Option<CpuPercent>,
    /// GPU busy share the board must be below. Not gated where the kernel
    /// exposes no GPU load.
    // shortcut: reuses CpuPercent, so a bad value is refused as a "CPU
    // percentage"; add a GPU range type if that misleads an operator.
    #[serde(default)]
    pub gpu_load_below_pct: Option<CpuPercent>,
}

/// `[readiness.macos]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MacOsReadinessConfiguration {
    /// °C the die must read below. The gate's own was measured on one machine
    /// type. An unreadable die holds a wait that sets it.
    #[serde(default)]
    pub die_below_c: Option<CeilingCelsius>,
    /// Busy share × online CPUs, which the machine must be below.
    #[serde(default)]
    pub cpu_busy_below_cores: Option<BusyCores>,
}

/// `[readiness.windows]`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WindowsReadinessConfiguration {
    /// Highest CPU busy share, over all logical CPUs, that counts as idle.
    /// `None` keeps the gate's own threshold.
    #[serde(default)]
    pub max_cpu_busy_pct: Option<CpuPercent>,
    /// Sensors the gate holds each rep for until they read at or below their
    /// ceiling, up to `windows.rs`'s `CEILING_MAX_WAIT`.
    #[serde(default, deserialize_with = "unique_ceilings")]
    pub ceilings: BTreeMap<WindowsSensor, CeilingCelsius>,
}

impl fmt::Display for WindowsReadinessConfiguration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ceilings: Vec<String> = self
            .ceilings
            .iter()
            .map(|(sensor, c)| format!("{sensor}<={c}C"))
            .collect();
        write!(f, "ceilings=[{}]", ceilings.join(", "))?;
        match self.max_cpu_busy_pct {
            Some(pct) => write!(f, " max_cpu_busy_pct={pct}"),
            None => Ok(()),
        }
    }
}

/// A sensor a Windows ceiling names, spelled `<source>:<name>` or a bare
/// source.
///
/// - `acpi:<zone>`: an ACPI thermal zone, named by its instance's last segment
///   (`THZ0` for `\_TZ.THZ0`), matched in any case and stored upper-case.
/// - `gpu`: the GPU's own temperature.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WindowsSensor {
    AcpiZone(String),
    Gpu,
}

#[derive(Debug, thiserror::Error)]
#[error("`{raw}` is not a sensor; expected `acpi:<zone>` (letters, digits, `_`) or `gpu`")]
pub struct WindowsSensorError {
    raw: String,
}

#[derive(Debug, thiserror::Error)]
pub enum RangeError {
    #[error("temperature {0} is out of range; it is 1-{max} °C", max = CeilingCelsius::MAX)]
    Ceiling(u32),
    #[error("CPU percentage {0} is out of range; it is 1-100")]
    CpuPercent(u32),
    #[error("core count {0} is out of range; it is 1 or more")]
    BusyCores(u32),
}

impl FromStr for WindowsSensor {
    type Err = WindowsSensorError;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let refused = || WindowsSensorError {
            raw: raw.to_string(),
        };
        match raw.split_once(':') {
            None if raw == "gpu" => Ok(Self::Gpu),
            Some(("acpi", zone))
                if !zone.is_empty()
                    && zone.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') =>
            {
                Ok(Self::AcpiZone(zone.to_ascii_uppercase()))
            }
            _ => Err(refused()),
        }
    }
}

impl fmt::Display for WindowsSensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AcpiZone(zone) => write!(f, "acpi:{zone}"),
            Self::Gpu => write!(f, "gpu"),
        }
    }
}

impl<'de> Deserialize<'de> for WindowsSensor {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// Two spellings of one sensor (`acpi:thz0`, `acpi:THZ0`) would otherwise
/// collapse silently, the later one winning.
fn unique_ceilings<'de, D: Deserializer<'de>>(
    d: D,
) -> Result<BTreeMap<WindowsSensor, CeilingCelsius>, D::Error> {
    BTreeMap::<String, CeilingCelsius>::deserialize(d)?
        .into_iter()
        .try_fold(BTreeMap::new(), |mut ceilings, (name, ceiling)| {
            let sensor: WindowsSensor = name.parse().map_err(serde::de::Error::custom)?;
            match ceilings.insert(sensor, ceiling) {
                None => Ok(ceilings),
                Some(_) => Err(serde::de::Error::custom(format!(
                    "sensor `{name}` appears twice, in another case"
                ))),
            }
        })
}

/// A temperature bound in whole °C, 1 to [`CeilingCelsius::MAX`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(try_from = "u32")]
pub struct CeilingCelsius(u32);

impl CeilingCelsius {
    /// Above every critical trip measured (103-110 °C): a higher ceiling could
    /// never hold anything the platform would not already have shut down for.
    pub const MAX: u32 = 110;

    pub fn get(self) -> u32 {
        self.0
    }
}

impl TryFrom<u32> for CeilingCelsius {
    type Error = RangeError;

    fn try_from(c: u32) -> Result<Self, Self::Error> {
        (1..=Self::MAX)
            .contains(&c)
            .then_some(Self(c))
            .ok_or(RangeError::Ceiling(c))
    }
}

impl fmt::Display for CeilingCelsius {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A share of the CPUs in whole percent, 1 to 100.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(try_from = "u32")]
pub struct CpuPercent(u32);

impl CpuPercent {
    pub fn get(self) -> u32 {
        self.0
    }

    /// As the 0-1 ratio the Android and Linux gates compare.
    pub fn ratio(self) -> f64 {
        f64::from(self.0) / 100.0
    }
}

impl TryFrom<u32> for CpuPercent {
    type Error = RangeError;

    fn try_from(pct: u32) -> Result<Self, Self::Error> {
        (1..=100)
            .contains(&pct)
            .then_some(Self(pct))
            .ok_or(RangeError::CpuPercent(pct))
    }
}

impl fmt::Display for CpuPercent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A count of busy cores, 1 or more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(try_from = "u32")]
pub struct BusyCores(u32);

impl BusyCores {
    pub fn get(self) -> u32 {
        self.0
    }
}

impl TryFrom<u32> for BusyCores {
    type Error = RangeError;

    fn try_from(cores: u32) -> Result<Self, Self::Error> {
        (cores >= 1)
            .then_some(Self(cores))
            .ok_or(RangeError::BusyCores(cores))
    }
}

#[cfg(test)]
pub(crate) fn parse_readiness(text: &str) -> Result<ReadinessConfiguration, toml::de::Error> {
    #[derive(Deserialize)]
    struct File {
        readiness: ReadinessConfiguration,
    }
    toml::from_str::<File>(text).map(|file| file.readiness)
}

/// `[readiness.<gate>]` with `keys` under it, so a gate's tests use the keys as
/// an operator spells them.
#[cfg(test)]
pub(crate) fn table(
    gate: ReadinessPlatform,
    keys: &str,
) -> Result<ReadinessConfiguration, toml::de::Error> {
    parse_readiness(&format!("[readiness.{gate}]\n{keys}"))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::parse_readiness as parse;
    use super::*;

    #[test]
    fn the_windows_table_parses_normalizes_and_renders() -> anyhow::Result<()> {
        let r = parse(
            "[readiness.windows]\nmax_cpu_busy_pct = 15\nceilings = { \"acpi:thz0\" = 43, gpu = 70 }\n",
        )?;
        let windows = r
            .windows()
            .ok_or_else(|| anyhow::anyhow!("expected the windows arm, got {r:?}"))?;
        assert_eq!(windows.max_cpu_busy_pct.map(CpuPercent::get), Some(15));
        assert_eq!(
            windows
                .ceilings
                .iter()
                .map(|(s, c)| (s.to_string(), c.get()))
                .collect::<Vec<_>>(),
            vec![("acpi:THZ0".to_string(), 43), ("gpu".to_string(), 70)]
        );
        assert_eq!(
            r.to_string(),
            "readiness.windows: ceilings=[acpi:THZ0<=43C, gpu<=70C] max_cpu_busy_pct=15"
        );
        Ok(())
    }

    #[rstest]
    #[case(ReadinessPlatform::Android)]
    #[case(ReadinessPlatform::Linux)]
    #[case(ReadinessPlatform::RaspberryPi5)]
    #[case(ReadinessPlatform::JetsonOrin)]
    #[case(ReadinessPlatform::MacOs)]
    #[case(ReadinessPlatform::Windows)]
    fn each_table_is_named_as_its_gate(#[case] gate: ReadinessPlatform) -> anyhow::Result<()> {
        let r = parse(&format!("[readiness.{gate}]\n"))?;
        assert_eq!(r.platform(), gate);
        assert_eq!(
            r.to_string().split(':').next(),
            Some(format!("readiness.{gate}").as_str())
        );
        assert_eq!(
            [
                r.android().is_some(),
                r.linux().is_some(),
                r.pi5().is_some(),
                r.orin().is_some(),
                r.macos().is_some(),
                r.windows().is_some(),
            ],
            [
                ReadinessPlatform::Android,
                ReadinessPlatform::Linux,
                ReadinessPlatform::RaspberryPi5,
                ReadinessPlatform::JetsonOrin,
                ReadinessPlatform::MacOs,
                ReadinessPlatform::Windows,
            ]
            .map(|arm| arm == gate)
        );
        Ok(())
    }

    /// The `run` line names only the keys a table sets.
    #[rstest]
    #[case::android(
        "[readiness.android]\nhottest_below_c = 36\ncpu_busy_below_pct = 40\n",
        "readiness.android: hottest_below_c=36 cpu_busy_below_pct=40"
    )]
    #[case::android_skin(
        "[readiness.android]\nskin_below_c = 34\n",
        "readiness.android: skin_below_c=34"
    )]
    #[case::android_held_touch(
        "[readiness.android]\nhottest_below_c = 36\nheld_touch = \"fail\"\n",
        "readiness.android: hottest_below_c=36 held_touch=fail"
    )]
    #[case::android_held_touch_alone(
        "[readiness.android]\nheld_touch = \"wait\"\n",
        "readiness.android: held_touch=wait"
    )]
    #[case::android_held_touch_off(
        "[readiness.android]\nheld_touch = \"off\"\n",
        "readiness.android: held_touch=off"
    )]
    #[case::linux(
        "[readiness.linux]\nhottest_below_c = 80\ncpu_load_below_pct = 50\n",
        "readiness.linux: hottest_below_c=80 cpu_load_below_pct=50"
    )]
    #[case::pi5(
        "[readiness.pi5]\nhottest_below_c = 70\ncpu_load_below_pct = 20\n",
        "readiness.pi5: hottest_below_c=70 cpu_load_below_pct=20"
    )]
    #[case::orin(
        "[readiness.orin]\nhottest_below_c = 75\ncpu_load_below_pct = 20\ngpu_load_below_pct = 10\n",
        "readiness.orin: hottest_below_c=75 cpu_load_below_pct=20 gpu_load_below_pct=10"
    )]
    #[case::macos(
        "[readiness.macos]\ndie_below_c = 60\ncpu_busy_below_cores = 2\n",
        "readiness.macos: die_below_c=60 cpu_busy_below_cores=2"
    )]
    #[case::only_the_second_key(
        "[readiness.macos]\ncpu_busy_below_cores = 2\n",
        "readiness.macos: cpu_busy_below_cores=2"
    )]
    #[case::an_empty_table("[readiness.linux]\n", "readiness.linux")]
    fn each_gates_keys_parse_and_render(
        #[case] text: &str,
        #[case] rendered: &str,
    ) -> anyhow::Result<()> {
        assert_eq!(parse(text)?.to_string(), rendered);
        Ok(())
    }

    /// Bit for bit the `0.30` the Android and Linux gates default to, so
    /// restating the default changes nothing.
    #[test]
    fn thirty_percent_is_the_gates_own_ratio() -> anyhow::Result<()> {
        assert_eq!(
            CpuPercent::try_from(30)?.ratio().to_bits(),
            0.30_f64.to_bits()
        );
        Ok(())
    }

    #[rstest]
    #[case::no_source(
        "[readiness.windows]\nceilings = { THZ0 = 43 }",
        "`THZ0` is not a sensor"
    )]
    #[case::unknown_source(
        "[readiness.windows]\nceilings = { \"linux:x86_pkg_temp\" = 70 }",
        "not a sensor"
    )]
    #[case::bad_zone(
        "[readiness.windows]\nceilings = { \"acpi:THZ 0\" = 43 }",
        "not a sensor"
    )]
    #[case::empty_zone("[readiness.windows]\nceilings = { \"acpi:\" = 43 }", "not a sensor")]
    #[case::zero(
        "[readiness.windows]\nceilings = { \"acpi:THZ0\" = 0 }",
        "temperature 0 is out of range"
    )]
    #[case::past_every_trip(
        "[readiness.windows]\nceilings = { \"acpi:THZ0\" = 111 }",
        "temperature 111 is out of range"
    )]
    #[case::fractional("[readiness.windows]\nceilings = { \"acpi:THZ0\" = 43.5 }", "43.5")]
    #[case::same_zone_twice(
        "[readiness.windows]\nceilings = { \"acpi:THZ0\" = 43, \"acpi:thz0\" = 44 }",
        "appears twice"
    )]
    #[case::cpu_zero(
        "[readiness.windows]\nmax_cpu_busy_pct = 0",
        "CPU percentage 0 is out of range"
    )]
    #[case::unknown_key("[readiness.windows]\nceiling = { \"acpi:THZ0\" = 43 }", "ceiling")]
    // A key belongs to the gate it is named for.
    #[case::a_setting_another_gate_lacks(
        "[readiness.linux]\nmax_cpu_busy_pct = 15",
        "max_cpu_busy_pct"
    )]
    #[case::busy_share_on_linux(
        "[readiness.linux]\ncpu_busy_below_pct = 30",
        "unknown field `cpu_busy_below_pct`"
    )]
    #[case::percent_on_macos(
        "[readiness.macos]\ncpu_busy_below_pct = 30",
        "unknown field `cpu_busy_below_pct`"
    )]
    #[case::die_on_android("[readiness.android]\ndie_below_c = 50", "unknown field `die_below_c`")]
    #[case::bound_zero(
        "[readiness.linux]\nhottest_below_c = 0",
        "temperature 0 is out of range"
    )]
    #[case::bound_past_every_trip(
        "[readiness.macos]\ndie_below_c = 111",
        "temperature 111 is out of range"
    )]
    #[case::bound_fractional("[readiness.macos]\ndie_below_c = 50.5", "floating point `50.5`")]
    #[case::load_over_all_cores(
        "[readiness.pi5]\ncpu_load_below_pct = 101",
        "CPU percentage 101 is out of range"
    )]
    #[case::gpu_on_pi5(
        "[readiness.pi5]\ngpu_load_below_pct = 5",
        "unknown field `gpu_load_below_pct`"
    )]
    #[case::gpu_zero(
        "[readiness.orin]\ngpu_load_below_pct = 0",
        "percentage 0 is out of range"
    )]
    #[case::busy_zero(
        "[readiness.android]\ncpu_busy_below_pct = 0",
        "CPU percentage 0 is out of range"
    )]
    #[case::skin_zero(
        "[readiness.android]\nskin_below_c = 0",
        "temperature 0 is out of range"
    )]
    #[case::skin_high(
        "[readiness.android]\nskin_below_c = 111",
        "temperature 111 is out of range"
    )]
    #[case::held_touch_unknown(
        "[readiness.android]\nheld_touch = \"maybe\"",
        "unknown variant `maybe`"
    )]
    #[case::held_touch_on_linux(
        "[readiness.linux]\nheld_touch = \"fail\"",
        "unknown field `held_touch`"
    )]
    #[case::no_cores(
        "[readiness.macos]\ncpu_busy_below_cores = 0",
        "core count 0 is out of range"
    )]
    #[case::fractional_cores(
        "[readiness.macos]\ncpu_busy_below_cores = 1.5",
        "floating point `1.5`"
    )]
    #[case::negative("[readiness.android]\nhottest_below_c = -5", "integer `-5`")]
    #[case::a_gate_that_checks_nothing("[readiness.other]\n", "unknown variant `other`")]
    #[case::two_gates("[readiness.windows]\n[readiness.linux]\n", "wanted exactly 1 element")]
    #[case::settings_outside_a_gate(
        "[readiness]\nmax_cpu_busy_pct = 15",
        "unknown variant `max_cpu_busy_pct`"
    )]
    fn a_malformed_table_is_refused(
        #[case] text: &str,
        #[case] needle: &str,
    ) -> anyhow::Result<()> {
        let err = parse(text)
            .err()
            .ok_or_else(|| anyhow::anyhow!("should be refused: {text}"))?;
        assert!(err.to_string().contains(needle), "{text}: {err}");
        Ok(())
    }
}
