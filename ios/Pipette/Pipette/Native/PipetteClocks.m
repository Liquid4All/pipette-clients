#import <Foundation/Foundation.h>

#include <dlfcn.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// CPU-cluster and GPU clock residency over one measured rep, for the log.
//
// `ProcessInfo.thermalState` stays `.nominal` while the GPU down-clocks, so a rep that
// ran ~30% slow reads the same as a fast one in every field the result carries. IOReport's
// performance-state channels (what `powermetrics` reads on a Mac) report how long each CPU
// cluster and the GPU sat in each clock state between two samples. Resolved with
// dlsym like the IOHID read, and compiled out of builds without PIPETTE_PRIVATE_THERMAL.
// ---------------------------------------------------------------------------

#ifdef PIPETTE_PRIVATE_THERMAL
typedef struct IOReportSubscriptionRef *PIPETTE_IOReportSubscriptionRef;
typedef CFDictionaryRef PIPETTE_IOReportSampleRef;

typedef CFMutableDictionaryRef (*PIPETTE_CopyChannelsInGroup)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
typedef void (*PIPETTE_MergeChannels)(CFMutableDictionaryRef, CFMutableDictionaryRef, CFTypeRef);
typedef PIPETTE_IOReportSubscriptionRef (*PIPETTE_CreateSubscription)(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
typedef CFDictionaryRef (*PIPETTE_CreateSamples)(PIPETTE_IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
typedef CFDictionaryRef (*PIPETTE_CreateSamplesDelta)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef void (*PIPETTE_Iterate)(CFDictionaryRef, int (^)(PIPETTE_IOReportSampleRef));
typedef CFStringRef (*PIPETTE_ChannelGetString)(PIPETTE_IOReportSampleRef);
typedef int32_t (*PIPETTE_StateGetCount)(PIPETTE_IOReportSampleRef);
typedef CFStringRef (*PIPETTE_StateGetNameForIndex)(PIPETTE_IOReportSampleRef, int32_t);
typedef int64_t (*PIPETTE_StateGetResidency)(PIPETTE_IOReportSampleRef, int32_t);

static struct {
    PIPETTE_CreateSamples createSamples;
    PIPETTE_CreateSamplesDelta createDelta;
    PIPETTE_Iterate iterate;
    PIPETTE_ChannelGetString channelName;
    PIPETTE_StateGetCount stateCount;
    PIPETTE_StateGetNameForIndex stateName;
    PIPETTE_StateGetResidency stateResidency;
    PIPETTE_IOReportSubscriptionRef subscription;
    CFMutableDictionaryRef subscribed;
} pipette_clocks;

// One subscription for the process: subscribing walks every driver's report legend, while
// a sample is one read of the channels already chosen.
static BOOL pipette_clocks_ready(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        void *lib = dlopen("/usr/lib/libIOReport.dylib", RTLD_LAZY);
        if (!lib) return;
        PIPETTE_CopyChannelsInGroup copyGroup = (PIPETTE_CopyChannelsInGroup)dlsym(lib, "IOReportCopyChannelsInGroup");
        PIPETTE_MergeChannels merge = (PIPETTE_MergeChannels)dlsym(lib, "IOReportMergeChannels");
        PIPETTE_CreateSubscription subscribe = (PIPETTE_CreateSubscription)dlsym(lib, "IOReportCreateSubscription");
        pipette_clocks.createSamples = (PIPETTE_CreateSamples)dlsym(lib, "IOReportCreateSamples");
        pipette_clocks.createDelta = (PIPETTE_CreateSamplesDelta)dlsym(lib, "IOReportCreateSamplesDelta");
        pipette_clocks.iterate = (PIPETTE_Iterate)dlsym(lib, "IOReportIterate");
        pipette_clocks.channelName = (PIPETTE_ChannelGetString)dlsym(lib, "IOReportChannelGetChannelName");
        pipette_clocks.stateCount = (PIPETTE_StateGetCount)dlsym(lib, "IOReportStateGetCount");
        pipette_clocks.stateName = (PIPETTE_StateGetNameForIndex)dlsym(lib, "IOReportStateGetNameForIndex");
        pipette_clocks.stateResidency = (PIPETTE_StateGetResidency)dlsym(lib, "IOReportStateGetResidency");
        if (!copyGroup || !merge || !subscribe || !pipette_clocks.createSamples || !pipette_clocks.createDelta
            || !pipette_clocks.iterate || !pipette_clocks.channelName
            || !pipette_clocks.stateCount || !pipette_clocks.stateName || !pipette_clocks.stateResidency) return;

        CFMutableDictionaryRef channels = copyGroup(CFSTR("CPU Stats"), CFSTR("CPU Complex Performance States"), 0, 0, 0);
        CFMutableDictionaryRef gpu = copyGroup(CFSTR("GPU Stats"), CFSTR("GPU Performance States"), 0, 0, 0);
        if (!channels || !gpu) {
            if (channels) CFRelease(channels);
            if (gpu) CFRelease(gpu);
            return;
        }
        merge(channels, gpu, NULL);
        CFRelease(gpu);
        CFMutableDictionaryRef subscribed = NULL;
        PIPETTE_IOReportSubscriptionRef sub = subscribe(NULL, channels, &subscribed, 0, NULL);
        CFRelease(channels);
        if (!sub || !subscribed) return;
        pipette_clocks.subscription = sub;
        pipette_clocks.subscribed = subscribed;
    });
    return pipette_clocks.subscription != NULL;
}
#endif

// A retained sample of the clock channels, or NULL when this build or device can't read them.
const void *_Nullable pipette_clock_sample(void) {
#ifdef PIPETTE_PRIVATE_THERMAL
    if (!pipette_clocks_ready()) return NULL;
    return pipette_clocks.createSamples(pipette_clocks.subscription, pipette_clocks.subscribed, NULL);
#else
    return NULL;
#endif
}

void pipette_clock_release(const void *_Nullable sample) {
    if (sample) CFRelease(sample);
}

// Each channel's share of time per clock state between two samples, states under 1% left
// out, e.g. `ECPU[IDLE 58% V0P5 14%] PCPU[V19P0 91%] GPUPH[OFF 69% P1 16%]`. Nil when either
// sample is missing. The samples stay the caller's.
NSString *_Nullable pipette_clock_residency(const void *_Nullable start, const void *_Nullable end) {
#ifdef PIPETTE_PRIVATE_THERMAL
    if (!start || !end || !pipette_clocks_ready()) return nil;
    @autoreleasepool {
        CFDictionaryRef delta = pipette_clocks.createDelta((CFDictionaryRef)start, (CFDictionaryRef)end, NULL);
        if (!delta) return nil;
        NSMutableArray<NSString *> *parts = [NSMutableArray array];
        pipette_clocks.iterate(delta, ^int(PIPETTE_IOReportSampleRef s) {
            NSString *channel = (__bridge NSString *)pipette_clocks.channelName(s) ?: @"?";
            // The power manager's own votes (ECPM, PCPM_IDLE, ...) beside each cluster's clock.
            if ([channel containsString:@"CPM"]) return 0;
            int32_t count = pipette_clocks.stateCount(s);
            if (count <= 0) return 0;
            int64_t total = 0;
            for (int32_t i = 0; i < count; i++) total += pipette_clocks.stateResidency(s, i);
            if (total <= 0) return 0;
            NSMutableArray<NSString *> *states = [NSMutableArray array];
            for (int32_t i = 0; i < count; i++) {
                int64_t r = pipette_clocks.stateResidency(s, i);
                if (r * 100 < total) continue;
                NSString *name = (__bridge NSString *)pipette_clocks.stateName(s, i) ?: [NSString stringWithFormat:@"S%d", i];
                [states addObject:[NSString stringWithFormat:@"%@ %lld%%", name, (long long)(r * 100 / total)]];
            }
            [parts addObject:[NSString stringWithFormat:@"%@[%@]", channel, [states componentsJoinedByString:@" "]]];
            return 0;
        });
        CFRelease(delta);
        return parts.count ? [parts componentsJoinedByString:@" "] : nil;
    }
#else
    return nil;
#endif
}
