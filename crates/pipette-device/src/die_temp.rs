//! macOS SoC die temperature, read from the private IOHID thermal sensors.
//!
//! macOS exposes no public die-temperature API, and the public signals are too
//! coarse to gate a benchmark on. `ProcessInfo.thermalState` and the
//! `com.apple.system.thermalpressurelevel` notification it wraps are, measured
//! on a fanless MacBook Neo (A18 Pro), a **fixed ~318 s hold-off anchored to
//! when the CPU went quiet** — invariant across a 30x range of load duration
//! and 10 C of peak die temperature, and still reading `moderate` at a
//! temperature it had held for five minutes. They report that work happened
//! recently, not that the SoC is hot.
//!
//! This reads the hardware instead: the `PMU tdie*` sensors, the same ones the
//! iOS client's `pipette_soc_temp` uses
//! (`ios/Pipette/Pipette/Native/PipetteThermal.m`) — matching AppleVendor
//! temperature services (`PrimaryUsagePage` 0xff00, `PrimaryUsage` 0x0005) and
//! taking the hottest reading. Sensor names are identical on Apple Silicon
//! Macs, and no root is required.
//!
//! Private API, hence `dlsym` rather than linking: a future macOS that drops
//! these symbols degrades to `None` — the column simply goes unreported —
//! instead of failing to launch. Unlike iOS there's no build flag
//! gating it — App Review is what forces `PIPETTE_PRIVATE_THERMAL` there, and
//! a locally built CLI has no such constraint.
//!
//! **Cost: ~26 ms per read** on a Mac with 20 `tdie` sensors, dominated by one
//! IPC round trip per sensor (~1.3 ms each); the client and service list are
//! resolved once per thread, which is why it isn't the ~58 ms a stateless read
//! costs. Cheap enough for a 10 s poll loop and for bracketing a timed rep
//! from outside the timed window, but not free — don't call it in a hot loop.

use std::cell::OnceCell;
use std::ffi::{c_void, CStr};

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

// The signatures the dlsym'd addresses are transmuted to. Spelled out as
// aliases rather than inline so each `transmute` names its target: for private
// symbols with no header to check against, a wrong signature is silent UB, and
// these are the only record of what was assumed.
type FnCreate = unsafe extern "C" fn(*const c_void) -> ClientRef;
type FnSetMatching = unsafe extern "C" fn(ClientRef, *const c_void);
type FnCopyServices = unsafe extern "C" fn(ClientRef) -> *const c_void;
type FnCopyProperty = unsafe extern "C" fn(ServiceRef, CFStringRef) -> CFTypeRef;
type FnCopyEvent = unsafe extern "C" fn(ServiceRef, i64, i32, i64) -> EventRef;
type FnEventFloat = unsafe extern "C" fn(EventRef, i32) -> f64;

/// The six private entry points, resolved once. Grouped in one struct so a
/// partial resolution can't leave some usable and some null.
struct Symbols {
    create: FnCreate,
    set_matching: FnSetMatching,
    copy_services: FnCopyServices,
    copy_property: FnCopyProperty,
    copy_event: FnCopyEvent,
    event_float: FnEventFloat,
}

/// Make sure IOKit is in the process image before [`symbol`] goes looking.
///
/// `RTLD_DEFAULT` searches only images *already loaded*, so every lookup below
/// depends on something having pulled IOKit in. Today something does — linking
/// `CoreFoundation` (which the `core-foundation` crate forces) is enough, and a
/// binary linking only libSystem resolves none of these symbols — but nothing in
/// this crate's build declares that dependency, and IOKit is not a direct
/// dependency of CoreFoundation either. It arrives by way of the dyld shared
/// cache, which is an implementation detail rather than a promise.
///
/// Asking for it explicitly costs a `dlopen` of an already-resident framework
/// (microseconds; the 13 ms cold-load case does not arise once CF has loaded
/// it), and this runs once per thread. A failure needs no handling: the symbol
/// lookups then fail and report themselves.
fn ensure_iokit_loaded() {
    // SAFETY: a literal, NUL-terminated framework path; the handle is
    // deliberately not retained — this only guarantees the image is resident,
    // and unloading IOKit is never wanted.
    unsafe {
        libc::dlopen(
            c"/System/Library/Frameworks/IOKit.framework/IOKit".as_ptr(),
            libc::RTLD_LAZY,
        );
    }
}

/// Look up one symbol in the already-loaded images.
///
/// Logs at `warn` when a lookup fails, because this is the failure mode that
/// would otherwise be invisible: the whole SoC-temperature column disappears
/// from every macOS result with nothing in the log to explain it. The other
/// failure points in this file are host conditions; this one means the API
/// moved, which is worth saying out loud once.
///
/// # Safety
/// The caller must transmute the result to a signature matching the real
/// symbol; a mismatch is undefined behavior.
unsafe fn symbol(name: &CStr) -> Option<*mut c_void> {
    let addr = libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr());
    if addr.is_null() {
        log::warn!(
            "die temp: {name:?} did not resolve; SoC temperature will not be \
             reported on this host"
        );
        return None;
    }
    Some(addr)
}

impl Symbols {
    fn load() -> Option<Self> {
        ensure_iokit_loaded();
        // SAFETY: each transmute target matches the documented signature of
        // the corresponding IOHID entry point, and every lookup is checked
        // non-null before use.
        unsafe {
            Some(Self {
                create: std::mem::transmute::<*mut c_void, FnCreate>(symbol(
                    c"IOHIDEventSystemClientCreate",
                )?),
                set_matching: std::mem::transmute::<*mut c_void, FnSetMatching>(symbol(
                    c"IOHIDEventSystemClientSetMatching",
                )?),
                copy_services: std::mem::transmute::<*mut c_void, FnCopyServices>(symbol(
                    c"IOHIDEventSystemClientCopyServices",
                )?),
                copy_property: std::mem::transmute::<*mut c_void, FnCopyProperty>(symbol(
                    c"IOHIDServiceClientCopyProperty",
                )?),
                copy_event: std::mem::transmute::<*mut c_void, FnCopyEvent>(symbol(
                    c"IOHIDServiceClientCopyEvent",
                )?),
                event_float: std::mem::transmute::<*mut c_void, FnEventFloat>(symbol(
                    c"IOHIDEventGetFloatValue",
                )?),
            })
        }
    }
}

/// A resolved set of die sensors, reusable for the life of the thread.
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
            log::debug!("die temp: IOHIDEventSystemClientCreate returned null");
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
            log::debug!("die temp: CopyServices returned null");
            return None;
        }
        // SAFETY: CopyServices follows the Create rule, so we own this array
        // and the wrapper takes over releasing it.
        let services: CFArray<*const c_void> =
            unsafe { CFArray::wrap_under_create_rule(raw.cast()) };

        let product_key = CFString::new("Product");
        let mut dies = Vec::new();
        for service in services.iter() {
            let service = (*service) as ServiceRef;
            if service.is_null() {
                continue;
            }
            // SAFETY: `service` is a non-null element of the matched array.
            let name = unsafe { (syms.copy_property)(service, product_key.as_concrete_TypeRef()) };
            if name.is_null() {
                continue;
            }
            // SAFETY: CopyProperty follows the Create rule. Wrapping as a
            // CFString is only valid if it really is one; when it isn't, we
            // release the raw ref instead of mistyping it.
            let is_die = unsafe {
                if CFGetTypeID(name) == CFString::type_id() {
                    CFString::wrap_under_create_rule(name.cast())
                        .to_string()
                        .starts_with(DIE_SENSOR_PREFIX)
                } else {
                    CFRelease(name);
                    false
                }
            };
            if is_die {
                dies.push(service);
            }
        }
        if dies.is_empty() {
            log::debug!(
                "die temp: no {DIE_SENSOR_PREFIX}* sensors among {} services",
                services.len()
            );
            return None;
        }
        log::debug!(
            "die temp: {} {DIE_SENSOR_PREFIX}* sensors resolved",
            dies.len()
        );
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
                // SAFETY: `event` is non-null and follows the Create rule, so we
                // read then release it.
                let celsius = unsafe {
                    let value = (self.syms.event_float)(event, TEMPERATURE_FIELD);
                    CFRelease(event);
                    value
                };
                PLAUSIBLE_C.contains(&celsius).then_some(celsius)
            })
            // `total_cmp` rather than `partial_cmp` + a fallback: it is a total
            // order, so there is no unreachable tie-break branch to justify.
            // NaN cannot reach it anyway — `PLAUSIBLE_C.contains` rejects it.
            .max_by(f64::total_cmp)
    }
}

extern "C" {
    /// Not re-exported by the `core-foundation` wrapper, so declared here.
    fn CFGetTypeID(cf: CFTypeRef) -> usize;
}

thread_local! {
    /// Resolved lazily and kept per-thread: the CF/IOHID handles are confined
    /// to their creating thread rather than shared behind an unsound
    /// `Send`/`Sync` claim, and resolution costs ~24 ms.
    static SENSORS: OnceCell<Option<Sensors>> = const { OnceCell::new() };
}

/// Hottest SoC die temperature in C, or `None` when the private sensors are
/// unreadable (symbols gone, no matching services, every read failed).
///
/// `None` is a normal outcome to design around, not an error: callers should
/// degrade to a coarser signal rather than fail. See the module docs for cost.
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

    /// Exercises the whole FFI path against the live kernel: dlsym, matching,
    /// service filtering, and an event read. Asserts a *plausible* value
    /// rather than a specific one, since it depends on how hot the host is.
    /// Skips rather than fails if the sensors are unavailable — the point is
    /// that the code degrades, and CI may run where they aren't exposed.
    #[test]
    fn die_temp_reads_a_plausible_value() {
        match die_temp_max_c() {
            Some(celsius) => assert!(
                PLAUSIBLE_C.contains(&celsius),
                "implausible die temp {celsius}C",
            ),
            None => log::warn!("die temp unavailable on this host; nothing to assert"),
        }
    }

    /// The resolved sensor set is cached, so repeated calls must keep working
    /// (a double-release or a stale ref would surface here, not on first use).
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
