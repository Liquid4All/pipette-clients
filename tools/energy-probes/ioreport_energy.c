// Reads the IOReport "Energy Model" counters twice, N ms apart, and prints
// per-channel energy and average power. No root needed.
#include <CoreFoundation/CoreFoundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
typedef struct IOReportSubscriptionRef *IOReportSubscriptionRef;
typedef CFDictionaryRef IOReportSampleRef;
extern CFMutableDictionaryRef IOReportCopyChannelsInGroup(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
extern IOReportSubscriptionRef IOReportCreateSubscription(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef, CFMutableDictionaryRef, CFTypeRef);
extern CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
extern CFStringRef IOReportChannelGetChannelName(IOReportSampleRef);
extern CFStringRef IOReportChannelGetUnitLabel(IOReportSampleRef);
extern int64_t IOReportSimpleGetIntegerValue(IOReportSampleRef, int32_t *);
extern void IOReportIterate(CFDictionaryRef, int (^)(IOReportSampleRef));
int main(int argc, char **argv) {
  int ms = argc > 1 ? atoi(argv[1]) : 1000;
  CFMutableDictionaryRef ch = IOReportCopyChannelsInGroup(CFSTR("Energy Model"), NULL, 0, 0, 0);
  CFMutableDictionaryRef sub_ch = NULL;
  IOReportSubscriptionRef sub = IOReportCreateSubscription(NULL, ch, &sub_ch, 0, NULL);
  if (!sub) { fprintf(stderr, "subscription failed\n"); return 1; }
  CFDictionaryRef a = IOReportCreateSamples(sub, sub_ch, NULL);
  usleep(ms * 1000);
  CFDictionaryRef b = IOReportCreateSamples(sub, sub_ch, NULL);
  CFDictionaryRef d = IOReportCreateSamplesDelta(a, b, NULL);
  IOReportIterate(d, ^int(IOReportSampleRef s) {
    char name[128] = "", unit[16] = "";
    CFStringGetCString(IOReportChannelGetChannelName(s), name, sizeof name, kCFStringEncodingUTF8);
    CFStringRef u = IOReportChannelGetUnitLabel(s);
    if (u) CFStringGetCString(u, unit, sizeof unit, kCFStringEncodingUTF8);
    int64_t v = IOReportSimpleGetIntegerValue(s, NULL);
    double scale = !strcmp(unit, "nJ") ? 1e-9 : !strcmp(unit, "uJ") ? 1e-6 : !strcmp(unit, "mJ") ? 1e-3 : 0;
    if (v && scale) printf("%-24s %12lld %-3s %8.3f W\n", name, (long long)v, unit, v * scale / (ms / 1000.0));
    return 0;
  });
  return 0;
}
