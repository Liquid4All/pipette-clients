// Reads SMC float keys (PSTR = total system power, W) N times at 1 s. No root.
#include <IOKit/IOKitLib.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
typedef struct { char major, minor, build, reserved; unsigned short release; } Vers;
typedef struct { unsigned short version, length; unsigned int cpuPLimit, gpuPLimit, memPLimit; } PLimit;
typedef struct { unsigned int dataSize, dataType; char dataAttributes; } KeyInfo;
typedef struct { unsigned int key; Vers vers; PLimit pLimitData; KeyInfo keyInfo; char result, status, data8; unsigned int data32; unsigned char bytes[32]; } SMCParam;
static io_connect_t conn;
static unsigned int k4(const char *s) { return (s[0] << 24) | (s[1] << 16) | (s[2] << 8) | s[3]; }
static int call(SMCParam *in, SMCParam *out) { size_t n = sizeof *out; return IOConnectCallStructMethod(conn, 2, in, sizeof *in, out, &n); }
static int readf(const char *key, float *v) {
  SMCParam in = {0}, out = {0};
  in.key = k4(key); in.data8 = 9; if (call(&in, &out) || out.result) return -1;
  in.keyInfo.dataSize = out.keyInfo.dataSize; in.data8 = 5; memset(&out, 0, sizeof out);
  if (call(&in, &out) || out.result) return -1;
  memcpy(v, out.bytes, 4); return 0;
}
int main(int argc, char **argv) {
  io_service_t s = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSMC"));
  if (IOServiceOpen(s, mach_task_self(), 0, &conn)) { fprintf(stderr, "open failed\n"); return 1; }
  int n = argc > 1 ? atoi(argv[1]) : 3;
  for (int i = 0; i < n; i++) {
    for (int k = 2; k < argc; k++) { float v; if (!readf(argv[k], &v)) printf("%s=%.2f ", argv[k], v); else printf("%s=NA ", argv[k]); }
    printf("\n"); fflush(stdout); if (i + 1 < n) sleep(1);
  }
}
