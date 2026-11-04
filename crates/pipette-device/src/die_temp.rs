//! macOS SoC die temperature, read from the private IOHID thermal sensors.
//!
//! macOS has no public die-temperature API. `ProcessInfo.thermalState` and the
//! thermal-pressure notification it wraps, measured on a fanless MacBook Neo
//! (A18 Pro), are a fixed ~318 s hold-off from when the CPU went quiet,
//! whatever the load length or peak temperature: they say work happened
//! recently, not that the SoC is hot.
//!
//! This takes the hottest `PMU tdie*` sensor among the AppleVendor temperature
//! services, as the iOS client does (`ios/Pipette/Pipette/Native/PipetteThermal.m`).
//! No root needed. The symbols are private, so they are `dlsym`'d: a macOS that
//! drops them yields `None` instead of failing to launch. Unlike iOS there is no
//! build flag: `PIPETTE_PRIVATE_THERMAL` exists for App Review.
//!
//! Cost: ~26 ms per read with 20 sensors (one ~1.3 ms IPC round trip each),
//! not the ~58 ms of a stateless read, because the client and services are
//! resolved once per thread. Fine for a 10 s poll or outside a timed window,
//! not for a hot loop. The sensor names are the same on every Apple Silicon Mac.

use std::cell::OnceCell;
use std::ffi::{CStr, c_void};

use core_foundation::array::CFArray;
use core_foundation::base::{CFRelease, CFTypeRef, TCFType};
use core_foundation::dictionary::CFDictionary;
use core_foundation::number::CFNumber;
use core_foundation::string::{CFString, CFStringRef};

/// `kIOHIDEventTypeTemperature`.
const EVENT_TYPE_TEMPERATURE: i64 = 15;
/// The temperature field selector: event type in the high 16 bits.
const TEMPERATURE_FIELD: i32 = 15 << 16;
/// `kHIDPage_AppleVendor`.
const APPLE_VENDOR_PAGE: i32 = 0xff00;
/// `kHIDUsage_AppleVendor_TemperatureSensor`.
const TEMPERATURE_SENSOR_USAGE: i32 = 0x0005;
/// Product-name prefix of the on-die sensors. The same services also expose
/// `PMU tdev*` (board/device probes, ~10 C cooler) and `gas gauge battery`;
/// only the die sensors track what throttles inference.
const DIE_SENSOR_PREFIX: &str = "PMU tdie";
/// Readings outside this range are treated as sensor noise rather than data.
const PLAUSIBLE_C: std::ops::Range<f64> = -50.0..150.0;

type ClientRef = *mut c_void;
type ServiceRef = *mut c_void;
type EventRef = *mut c_void;

// Private symbols have no header, so a wrong signature here is silent UB:
// these aliases are the only record of what was assumed.
type FnCreate = unsafe extern "C" fn(*const c_void) -> ClientRef;
type FnSetMatching = unsafe extern "C" fn(ClientRef, *const c_void);
type FnCopyServices = unsafe extern "C" fn(ClientRef) -> *const c_void;
type FnCopyProperty = unsafe extern "C" fn(ServiceRef, CFStringRef) -> CFTypeRef;
type FnCopyEvent = unsafe extern "C" fn(ServiceRef, i64, i32, i64) -> EventRef;
type FnEventFloat = unsafe extern "C" fn(EventRef, i32) -> f64;

/// Grouped so a partial resolution can't leave some usable and some null.
struct Symbols {
    create: FnCreate,
    set_matching: FnSetMatching,
    copy_services: FnCopyServices,
    copy_property: FnCopyProperty,
    copy_event: FnCopyEvent,
    event_float: FnEventFloat,
}

/// `RTLD_DEFAULT` searches only images already loaded. IOKit arrives today
/// only by way of CoreFoundation and the dyld shared cache, which nothing
/// promises, so load it explicitly: microseconds once resident, once per
/// thread. A failure needs no handling, the lookups then fail and log.
fn ensure_iokit_loaded() {
    // SAFETY: a literal, NUL-terminated framework path. The handle is
    // deliberately not retained: unloading IOKit is never wanted.
    unsafe {
        libc::dlopen(
            c"/System/Library/Frameworks/IOKit.framework/IOKit".as_ptr(),
            libc::RTLD_LAZY,
        );
    }
}

/// Warns on failure: a missing symbol means the API moved, and otherwise the
/// SoC-temperature column vanishes from every macOS result unexplained.
fn symbol(name: &CStr) -> Option<*mut c_void> {
    // SAFETY: `name` is a NUL-terminated C string, and `RTLD_DEFAULT` searches
    // the images already loaded; the result is only an address until the
    // caller transmutes it.
    let addr = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
    if addr.is_null() {
        tracing::warn!(
            symbol = %name.to_string_lossy(),
            "IOHID symbol did not resolve; SoC temperature will not be reported on this host"
        );
        return None;
    }
    Some(addr)
}

macro_rules! resolve {
    ($name:literal, $ty:ty) => {{
        let addr = symbol($name)?;
        // SAFETY: `$ty` is the documented signature of the IOHID entry point
        // `$name`, and `addr` was checked non-null by `symbol`.
        unsafe { std::mem::transmute::<*mut c_void, $ty>(addr) }
    }};
}

impl Symbols {
    fn load() -> Option<Self> {
        ensure_iokit_loaded();
        Some(Self {
            create: resolve!(c"IOHIDEventSystemClientCreate", FnCreate),
            set_matching: resolve!(c"IOHIDEventSystemClientSetMatching", FnSetMatching),
            copy_services: resolve!(c"IOHIDEventSystemClientCopyServices", FnCopyServices),
            copy_property: resolve!(c"IOHIDServiceClientCopyProperty", FnCopyProperty),
            copy_event: resolve!(c"IOHIDServiceClientCopyEvent", FnCopyEvent),
            event_float: resolve!(c"IOHIDEventGetFloatValue", FnEventFloat),
        })
    }
}

struct Sensors {
    syms: Symbols,
    /// Owns the service refs held in `dies`, so it must outlive them. The
    /// IOHID client backing it is deliberately leaked in [`Sensors::resolve`].
    _services: CFArray<*const c_void>,
    dies: Vec<ServiceRef>,
}

impl Sensors {
    fn resolve() -> Option<Self> {
        let syms = Symbols::load()?;
        // SAFETY: a null allocator means kCFAllocatorDefault.
        let client = unsafe { (syms.create)(std::ptr::null()) };
        if client.is_null() {
            tracing::debug!("no IOHID client: IOHIDEventSystemClientCreate returned null");
            return None;
        }
        let matching = CFDictionary::from_CFType_pairs(&[
            (
                CFString::new("PrimaryUsagePage").as_CFType(),
                CFNumber::from(APPLE_VENDOR_PAGE).as_CFType(),
            ),
            (
                CFString::new("PrimaryUsage").as_CFType(),
                CFNumber::from(TEMPERATURE_SENSOR_USAGE).as_CFType(),
            ),
        ]);
        // SAFETY: `client` is non-null and `matching` outlives the call, which
        // retains what it needs.
        unsafe { (syms.set_matching)(client, matching.as_concrete_TypeRef().cast()) };
        // SAFETY: `client` is non-null.
        let raw = unsafe { (syms.copy_services)(client) };
        if raw.is_null() {
            tracing::debug!("no sensor services: CopyServices returned null");
            return None;
        }
        // SAFETY: CopyServices follows the Create rule, so we own this array
        // and the wrapper takes over releasing it.
        let services: CFArray<*const c_void> =
            unsafe { CFArray::wrap_under_create_rule(raw.cast()) };

        let product_key = CFString::new("Product");
        let mut dies = Vec::new();
        for service in services.iter() {
            let service: ServiceRef = (*service).cast_mut();
            if service.is_null() {
                continue;
            }
            // SAFETY: `service` is a non-null element of the matched array.
            let name = unsafe { (syms.copy_property)(service, product_key.as_concrete_TypeRef()) };
            if name.is_null() {
                continue;
            }
            // CopyProperty follows the Create rule. Wrapping as a CFString is
            // only valid if it really is one; when it isn't, we release the raw
            // ref instead of mistyping it.
            // SAFETY: `name` is a non-null CF object.
            let is_string = unsafe { CFGetTypeID(name) } == CFString::type_id();
            let is_die = if is_string {
                // SAFETY: `name` is an owned CFString, checked just above.
                unsafe { CFString::wrap_under_create_rule(name.cast()) }
                    .to_string()
                    .starts_with(DIE_SENSOR_PREFIX)
            } else {
                // SAFETY: we own `name` and nothing else references it.
                unsafe { CFRelease(name) };
                false
            };
            if is_die {
                dies.push(service);
            }
        }
        if dies.is_empty() {
            tracing::debug!(
                services = services.len(),
                "no die sensors among the matched services"
            );
            return None;
        }
        tracing::debug!(sensors = dies.len(), "die sensors resolved");
        // `client` is intentionally not released: the service refs above
        // borrow from it and are cached for the life of the thread, so the
        // client has to outlive them. One leaked handle per thread.
        Some(Self {
            syms,
            _services: services,
            dies,
        })
    }

    /// Hottest current die reading, or `None` if every sensor failed to
    /// produce a plausible value.
    fn read_max(&self) -> Option<f64> {
        self.dies
            .iter()
            .filter_map(|&service| {
                // SAFETY: `service` came from the matched array, which is still
                // alive, as is the client it borrows from.
                let event =
                    unsafe { (self.syms.copy_event)(service, EVENT_TYPE_TEMPERATURE, 0, 0) };
                if event.is_null() {
                    return None;
                }
                // SAFETY: `event` is non-null.
                let celsius = unsafe { (self.syms.event_float)(event, TEMPERATURE_FIELD) };
                // SAFETY: `event` follows the Create rule, so we own it, and
                // nothing reads it after this.
                unsafe { CFRelease(event) };
                PLAUSIBLE_C.contains(&celsius).then_some(celsius)
            })
            // NaN never reaches here: `PLAUSIBLE_C.contains` rejects it.
            .max_by(f64::total_cmp)
    }
}

unsafe extern "C" {
    /// Not re-exported by the `core-foundation` wrapper, so declared here.
    fn CFGetTypeID(cf: CFTypeRef) -> usize;
}

thread_local! {
    /// Per-thread rather than behind an unsound `Send`/`Sync` claim on the
    /// CF/IOHID handles; resolution costs ~24 ms.
    static SENSORS: OnceCell<Option<Sensors>> = const { OnceCell::new() };
}

/// Hottest SoC die temperature in °C, or `None` when the private sensors are
/// unreadable (symbols gone, no matching services, every read failed).
/// `None` is a normal outcome: degrade to a coarser signal rather than fail.
pub fn die_temp_max_c() -> Option<f64> {
    SENSORS.with(|cell| {
        cell.get_or_init(Sensors::resolve)
            .as_ref()
            .and_then(Sensors::read_max)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole FFI path against the live kernel. Passes when the sensors are
    /// unavailable: CI may run where they aren't exposed.
    #[test]
    fn die_temp_reads_a_plausible_value() {
        match die_temp_max_c() {
            Some(celsius) => assert!(
                PLAUSIBLE_C.contains(&celsius),
                "implausible die temp {celsius}C",
            ),
            None => tracing::warn!("die temp unavailable on this host; nothing to assert"),
        }
    }

    /// A double-release or stale cached ref would surface here, not on first use.
    #[test]
    fn repeated_reads_stay_valid() {
        let readings: Vec<_> = (0..5).filter_map(|_| die_temp_max_c()).collect();
        if readings.is_empty() {
            return; // unavailable on this host
        }
        assert_eq!(readings.len(), 5, "cached sensors stopped reading mid-run");
        for celsius in readings {
            assert!(PLAUSIBLE_C.contains(&celsius), "implausible {celsius}C");
        }
    }
}
