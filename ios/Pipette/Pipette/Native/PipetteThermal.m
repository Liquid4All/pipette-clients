#import <Foundation/Foundation.h>

#include <stdint.h>
#include <dlfcn.h>

// Native (app-target) home for the SoC die-temperature probe that used to live
// in the Rust crate's `crates/pipette-ios/metal_counter.m`. The iOS benchmark
// path no longer links the Rust static lib, so this is compiled directly into
// the app target and exposed to Swift via Pipette-Bridging-Header.h.
// (Metal allocation is read directly in Swift via `MTLDevice.currentAllocatedSize`.)

// ---------------------------------------------------------------------------
// SoC die temperature (passive) for the benchmark thermal-cooldown gate.
//
// iOS exposes no public SoC temperature API, and `ProcessInfo.thermalState` is
// too coarse to gate benchmark readiness (it stays `.nominal` while the GPU
// down-clocks ~30%). We read the on-die temperature directly from the IOHID
// thermal sensors, resolved via dlsym so the private symbols can't break the
// link. Returns the max "PMU tdie*" reading in C, or -1 if unavailable (caller
// should then fall back to thermalState).
// ---------------------------------------------------------------------------

typedef struct __IOHIDEvent *PIPETTE_HIDEventRef;
typedef struct __IOHIDServiceClient *PIPETTE_HIDServiceClientRef;
typedef struct __IOHIDEventSystemClient *PIPETTE_HIDEventSystemClientRef;

typedef PIPETTE_HIDEventSystemClientRef (*PIPETTE_Create)(CFAllocatorRef);
typedef void (*PIPETTE_SetMatching)(PIPETTE_HIDEventSystemClientRef, CFDictionaryRef);
typedef CFArrayRef (*PIPETTE_CopyServices)(PIPETTE_HIDEventSystemClientRef);
typedef CFTypeRef (*PIPETTE_SvcCopyProp)(PIPETTE_HIDServiceClientRef, CFStringRef);
typedef PIPETTE_HIDEventRef (*PIPETTE_SvcCopyEvent)(PIPETTE_HIDServiceClientRef, int64_t, int32_t, int64_t);
typedef double (*PIPETTE_EventFloat)(PIPETTE_HIDEventRef, int32_t);

#define PIPETTE_kIOHIDEventTypeTemperature 15
#define PIPETTE_TempField (PIPETTE_kIOHIDEventTypeTemperature << 16)

double pipette_soc_temp(void) {
#ifdef PIPETTE_PRIVATE_THERMAL
    @autoreleasepool {
        PIPETTE_Create create        = (PIPETTE_Create)dlsym(RTLD_DEFAULT, "IOHIDEventSystemClientCreate");
        PIPETTE_SetMatching setMatch = (PIPETTE_SetMatching)dlsym(RTLD_DEFAULT, "IOHIDEventSystemClientSetMatching");
        PIPETTE_CopyServices copySvc = (PIPETTE_CopyServices)dlsym(RTLD_DEFAULT, "IOHIDEventSystemClientCopyServices");
        PIPETTE_SvcCopyProp copyProp = (PIPETTE_SvcCopyProp)dlsym(RTLD_DEFAULT, "IOHIDServiceClientCopyProperty");
        PIPETTE_SvcCopyEvent copyEvt = (PIPETTE_SvcCopyEvent)dlsym(RTLD_DEFAULT, "IOHIDServiceClientCopyEvent");
        PIPETTE_EventFloat evtFloat  = (PIPETTE_EventFloat)dlsym(RTLD_DEFAULT, "IOHIDEventGetFloatValue");
        if (!create || !setMatch || !copySvc || !copyEvt || !evtFloat || !copyProp) return -1.0;

        PIPETTE_HIDEventSystemClientRef client = create(kCFAllocatorDefault);
        if (!client) return -1.0;
        // AppleVendor temperature sensors: PrimaryUsagePage 0xff00, PrimaryUsage 0x0005.
        NSDictionary *m = @{ @"PrimaryUsagePage": @(0xff00), @"PrimaryUsage": @(0x0005) };
        setMatch(client, (__bridge CFDictionaryRef)m);
        CFArrayRef svcs = copySvc(client);
        long n = svcs ? CFArrayGetCount(svcs) : 0;
        double best = -1000.0;
        for (long i = 0; i < n; i++) {
            PIPETTE_HIDServiceClientRef s = (PIPETTE_HIDServiceClientRef)CFArrayGetValueAtIndex(svcs, i);
            CFTypeRef nm = copyProp(s, CFSTR("Product"));
            BOOL isDie = (nm && CFGetTypeID(nm) == CFStringGetTypeID() &&
                          [(__bridge NSString *)nm hasPrefix:@"PMU tdie"]);
            if (isDie) {
                PIPETTE_HIDEventRef ev = copyEvt(s, PIPETTE_kIOHIDEventTypeTemperature, 0, 0);
                double t = ev ? evtFloat(ev, PIPETTE_TempField) : -1000.0;
                if (ev) CFRelease(ev);
                if (t > -50.0 && t < 150.0 && t > best) best = t;
            }
            if (nm) CFRelease(nm);
        }
        if (svcs) CFRelease(svcs);
        CFRelease(client);
        return best > -1000.0 ? best : -1.0;
    }
#else
    // Private IOHID thermal read is compiled out by default — keeps the private
    // symbols/strings out of the binary. Enable with the PIPETTE_PRIVATE_THERMAL
    // build flag for benchmark builds; otherwise the host gates on thermalState.
    return -1.0;
#endif
}

// ---------------------------------------------------------------------------
// Whether this build carries the private read at all.
//
// Compiled from the same `#ifdef` as `pipette_soc_temp`, so it cannot disagree with it —
// a Swift-side condition could, because the Swift and ObjC preprocessors take separate
// flags and a build that sets only one produces an app whose gate is silently coarse.
// `Runtime.thisBuild` reports this, which is what lets a plan name the build it needs.
// ---------------------------------------------------------------------------
int pipette_private_thermal_build(void) {
#ifdef PIPETTE_PRIVATE_THERMAL
    return 1;
#else
    return 0;
#endif
}
