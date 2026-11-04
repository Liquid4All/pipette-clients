// ewrap: run a command; report IOReport Energy Model deltas (SoC) bracketing it,
// plus SMC PSTR (system W) polled at 4 Hz during it. No root.
// usage: ewrap <cmd> [args...]   -> prints one "EWRAP elapsed_s=.. soc_cpu_j=.. soc_gpu_j=.. soc_all_j=.. pstr_mean_w=.. pstr_n=.. pstr_j=.. exit=.." line
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <pthread.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
extern char **environ;
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
typedef struct { char major, minor, build, reserved; unsigned short release; } Vers;
typedef struct { unsigned short version, length; unsigned int cpuPLimit, gpuPLimit, memPLimit; } PLimit;
typedef struct { unsigned int dataSize, dataType; char dataAttributes; } KeyInfo;
typedef struct { unsigned int key; Vers vers; PLimit pLimitData; KeyInfo keyInfo; char result, status, data8; unsigned int data32; unsigned char bytes[32]; } SMCParam;
static io_connect_t conn;
static int smccall(SMCParam *in, SMCParam *out) { size_t n = sizeof *out; return IOConnectCallStructMethod(conn, 2, in, sizeof *in, out, &n); }
static int readf(const char *key, float *v) {
  SMCParam in = {0}, out = {0};
  in.key = (key[0] << 24) | (key[1] << 16) | (key[2] << 8) | key[3]; in.data8 = 9;
  if (smccall(&in, &out) || out.result) return -1;
  in.keyInfo.dataSize = out.keyInfo.dataSize; in.data8 = 5; memset(&out, 0, sizeof out);
  if (smccall(&in, &out) || out.result) return -1;
  memcpy(v, out.bytes, 4); return 0;
}
static volatile int running = 1; static double psum = 0; static int pn = 0;
static void *poll(void *_) { while (running) { float v; if (!readf("PSTR", &v)) { psum += v; pn++; } usleep(250000); } return NULL; }
int main(int argc, char **argv) {
  if (argc < 2) return 2;
  io_service_t s = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSMC"));
  IOServiceOpen(s, mach_task_self(), 0, &conn);
  CFMutableDictionaryRef ch = IOReportCopyChannelsInGroup(CFSTR("Energy Model"), NULL, 0, 0, 0), sub_ch = NULL;
  IOReportSubscriptionRef sub = IOReportCreateSubscription(NULL, ch, &sub_ch, 0, NULL);
  pthread_t th; pthread_create(&th, NULL, poll, NULL);
  struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
  CFDictionaryRef a = IOReportCreateSamples(sub, sub_ch, NULL);
  pid_t pid; int st = 0;
  if (posix_spawnp(&pid, argv[1], NULL, NULL, argv + 1, environ)) { perror("spawn"); return 1; }
  waitpid(pid, &st, 0);
  CFDictionaryRef b = IOReportCreateSamples(sub, sub_ch, NULL);
  clock_gettime(CLOCK_MONOTONIC, &t1); running = 0; pthread_join(th, NULL);
  double el = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
  __block double cpu = 0, gpu = 0, all = 0;
  IOReportIterate(IOReportCreateSamplesDelta(a, b, NULL), ^int(IOReportSampleRef smp) {
    char name[128] = "", unit[16] = "";
    CFStringGetCString(IOReportChannelGetChannelName(smp), name, sizeof name, kCFStringEncodingUTF8);
    CFStringRef u = IOReportChannelGetUnitLabel(smp); if (u) CFStringGetCString(u, unit, sizeof unit, kCFStringEncodingUTF8);
    double sc = !strcmp(unit, "nJ") ? 1e-9 : !strcmp(unit, "uJ") ? 1e-6 : !strcmp(unit, "mJ") ? 1e-3 : !strcmp(unit, "J") ? 1 : 0;
    double j = IOReportSimpleGetIntegerValue(smp, NULL) * sc;
    if (strstr(name, "SRAM") || strstr(name, "DTL")) return 0;
    if (strstr(name, "CPU Energy")) { cpu += j; all += j; }
    else if (strstr(name, "GPU Energy")) { gpu += j; all += j; }
    else if (!strncmp(name, "DRAM", 4) || !strncmp(name, "AMCC", 4) || !strncmp(name, "DCS", 3) || !strncmp(name, "ANE", 3) || !strncmp(name, "FAB", 3)) all += j;
    return 0;
  });
  printf("EWRAP elapsed_s=%.3f soc_cpu_j=%.2f soc_gpu_j=%.2f soc_all_j=%.2f pstr_mean_w=%.2f pstr_n=%d pstr_j=%.1f exit=%d\n",
         el, cpu, gpu, all, pn ? psum / pn : 0, pn, pn ? psum / pn * el : 0, WEXITSTATUS(st));
  return 0;
}
