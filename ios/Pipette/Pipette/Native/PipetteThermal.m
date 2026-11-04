#import <Foundation/Foundation.h>

#include <stdint.h>
#include <dlfcn.h>

// SoC die temperature for the benchmark thermal-cooldown gate, exposed to Swift via
// Pipette-Bridging-Header.h.
//
// iOS has no public SoC temperature API, and `ProcessInfo.thermalState` is too coarse
// (it stays `.nominal` while the GPU down-clocks ~30%). The IOHID thermal sensors are
// private, resolved via dlsym so the symbols cannot break the link.

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

#ifdef PIPETTE_PRIVATE_THERMAL
// Every AppleVendor temperature service. `wants` filters which to read, as each read is
// one IPC round trip. Returns NO when the private symbols or the client are unavailable.
static BOOL pipette_each_temperature_service(BOOL (^wants)(NSString *product),
                                             void (^each)(NSString *product, NSNumber *level)) {
    PIPETTE_Create create        = (PIPETTE_Create)dlsym(RTLD_DEFAULT, "IOHIDEventSystemClientCreate");
    PIPETTE_SetMatching setMatch = (PIPETTE_SetMatching)dlsym(RTLD_DEFAULT, "IOHIDEventSystemClientSetMatching");
    PIPETTE_CopyServices copySvc = (PIPETTE_CopyServices)dlsym(RTLD_DEFAULT, "IOHIDEventSystemClientCopyServices");
    PIPETTE_SvcCopyProp copyProp = (PIPETTE_SvcCopyProp)dlsym(RTLD_DEFAULT, "IOHIDServiceClientCopyProperty");
    PIPETTE_SvcCopyEvent copyEvt = (PIPETTE_SvcCopyEvent)dlsym(RTLD_DEFAULT, "IOHIDServiceClientCopyEvent");
    PIPETTE_EventFloat evtFloat  = (PIPETTE_EventFloat)dlsym(RTLD_DEFAULT, "IOHIDEventGetFloatValue");
    if (!create || !setMatch || !copySvc || !copyEvt || !evtFloat || !copyProp) return NO;

    PIPETTE_HIDEventSystemClientRef client = create(kCFAllocatorDefault);
    if (!client) return NO;
    NSDictionary *m = @{ @"PrimaryUsagePage": @(0xff00), @"PrimaryUsage": @(0x0005) };
    setMatch(client, (__bridge CFDictionaryRef)m);
    CFArrayRef svcs = copySvc(client);
    long n = svcs ? CFArrayGetCount(svcs) : 0;
    for (long i = 0; i < n; i++) {
        PIPETTE_HIDServiceClientRef s = (PIPETTE_HIDServiceClientRef)CFArrayGetValueAtIndex(svcs, i);
        CFTypeRef nm = copyProp(s, CFSTR("Product"));
        NSString *product = (nm && CFGetTypeID(nm) == CFStringGetTypeID())
            ? [(__bridge NSString *)nm copy] : nil;
        if (nm) CFRelease(nm);
        NSNumber *level = nil;
        if (wants(product)) {
            PIPETTE_HIDEventRef ev = copyEvt(s, PIPETTE_kIOHIDEventTypeTemperature, 0, 0);
            if (ev) {
                level = @(evtFloat(ev, PIPETTE_TempField));
                CFRelease(ev);
            }
        }
        each(product, level);
    }
    if (svcs) CFRelease(svcs);
    CFRelease(client);
    return YES;
}
#endif

// The max "PMU tdie*" reading in degrees C, or -1 if unavailable (callers then fall back
// to thermalState).
double pipette_soc_temp(void) {
#ifdef PIPETTE_PRIVATE_THERMAL
    @autoreleasepool {
        __block double best = -1000.0;
        BOOL read = pipette_each_temperature_service(
            ^BOOL(NSString *product) { return [product hasPrefix:@"PMU tdie"]; },
            ^(__unused NSString *product, NSNumber *level) {
                double t = level ? level.doubleValue : -1000.0;
                if (t > -50.0 && t < 150.0 && t > best) best = t;
            });
        return read && best > -1000.0 ? best : -1.0;
    }
#else
    // Compiled out by default to keep the private symbols out of the binary;
    // PIPETTE_PRIVATE_THERMAL enables it for benchmark builds.
    return -1.0;
#endif
}

// Every temperature sensor, unfiltered and unrounded, for
// `apple_iohid_temperature_sensors`.
NSArray<NSDictionary<NSString *, id> *> *_Nullable pipette_temperature_sensors(void) {
#ifdef PIPETTE_PRIVATE_THERMAL
    NSMutableArray<NSDictionary<NSString *, id> *> *sensors = [NSMutableArray array];
    BOOL read = pipette_each_temperature_service(
        ^BOOL(__unused NSString *product) { return YES; },
        ^(NSString *product, NSNumber *level) {
            NSMutableDictionary<NSString *, id> *sensor = [NSMutableDictionary dictionary];
            if (product) sensor[@"Product"] = product;
            if (level) sensor[@"TemperatureLevel"] = level;
            [sensors addObject:sensor];
        });
    return read ? sensors : nil;
#else
    return nil;
#endif
}

// Whether this build carries the private read. Uses the same `#ifdef` as
// `pipette_soc_temp` because Swift and ObjC take separate preprocessor flags, so a Swift
// condition could disagree. `PrivateApis.thisBuild` reports it.
int pipette_private_thermal_build(void) {
#ifdef PIPETTE_PRIVATE_THERMAL
    return 1;
#else
    return 0;
#endif
}
