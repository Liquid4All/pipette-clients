// peakmtl — DYLD_INSERT_LIBRARIES shim for measuring peak Metal device
// memory from outside the inference process.
//
// Polls [MTLDevice currentAllocatedSize] across every device every 20 ms
// and tracks the running peak. Output is written to a tempfile whose
// path is supplied by the parent via the `PIPETTE_MEMPROBE_OUT`
// environment variable. The shim is dormant when that env is unset.
//
// File format (latest snapshot, truncate+write each time):
//
//   metal_peak_allocated_bytes=<u64>
//   metal_peak_recommended_max_bytes=<u64>
//   metal_unified=<0|1>
//   metal_devices=<u32>
//
// Why a tempfile and not stderr (as before):
//   - The runtime's own stderr (llama-bench progress, llama.cpp init
//     logs) is no longer interleaved with shim output. The parent can
//     inherit stderr to the operator console without filtering.
//   - The parent owns the file path lifecycle (`tempfile::TempDir`),
//     guaranteeing cleanup on success and on parent panic.
//   - Truncate-write-on-grow makes abnormal exits (`_exit()` / SIGKILL)
//     safe: the file always contains the highest peak observed so far.
//
// Build:
//   clang -O2 -dynamiclib -framework Metal -framework Foundation \
//         -o peakmtl.dylib peakmtl.m
//
// Use (from parent):
//   tmp=$(mktemp)
//   DYLD_INSERT_LIBRARIES=/path/to/peakmtl.dylib \
//   PIPETTE_MEMPROBE_OUT=$tmp \
//     <binary> <args>
//   cat $tmp
//
// Caveats: see ../../../pipette-mgmt/docs/methodology/peak-memory.md.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

// All cross-thread state goes through C11 atomics. `volatile` was UB
// per the C/Obj-C memory model — it inhibits compiler reordering but
// provides no synchronization. The atomic_* loads/stores below are the
// portable, defined alternative.
static _Atomic uint64_t g_peak_allocated = 0;
static _Atomic uint64_t g_peak_recommended_max = 0;
static _Atomic int g_unified = -1;
static _Atomic int g_n_devices = 0;
static _Atomic int g_done = 0;
static _Atomic int g_thread_started = 0;
static pthread_t g_thread;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

// Captured once at startup (see start_once). Empty string ⇒ shim
// dormant. We copy the env value rather than holding the pointer
// because the environment can be modified by the program at runtime.
static char g_outpath[4096];

static void write_snapshot(void) {
    if (g_outpath[0] == '\0') {
        return;
    }
    int fd = open(g_outpath, O_WRONLY | O_TRUNC | O_CREAT, 0600);
    if (fd < 0) {
        return;
    }
    char buf[256];
    int n = snprintf(buf, sizeof(buf),
        "metal_peak_allocated_bytes=%llu\n"
        "metal_peak_recommended_max_bytes=%llu\n"
        "metal_unified=%d\n"
        "metal_devices=%d\n",
        (unsigned long long)atomic_load_explicit(&g_peak_allocated, memory_order_relaxed),
        (unsigned long long)atomic_load_explicit(&g_peak_recommended_max, memory_order_relaxed),
        atomic_load_explicit(&g_unified, memory_order_relaxed),
        atomic_load_explicit(&g_n_devices, memory_order_relaxed));
    if (n > 0) {
        // Best-effort; partial writes here are bounded by `buf` size
        // (256 bytes) which is well within PIPE_BUF / sector-write
        // atomicity guarantees on macOS for regular files.
        ssize_t _ = write(fd, buf, (size_t)n);
        (void)_;
    }
    close(fd);
}

static void *poller(void *arg) {
    (void)arg;
    @autoreleasepool {
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        atomic_store_explicit(&g_n_devices, (int)devices.count, memory_order_relaxed);
        if (devices.count > 0) {
            id<MTLDevice> first = devices[0];
            atomic_store_explicit(&g_unified, [first hasUnifiedMemory] ? 1 : 0,
                                  memory_order_relaxed);
        }
        while (!atomic_load_explicit(&g_done, memory_order_relaxed)) {
            uint64_t total_alloc = 0;
            uint64_t total_rec = 0;
            for (id<MTLDevice> dev in devices) {
                total_alloc += (uint64_t)[dev currentAllocatedSize];
                total_rec   += (uint64_t)[dev recommendedMaxWorkingSetSize];
            }
            int grew = 0;
            uint64_t prev_alloc = atomic_load_explicit(&g_peak_allocated, memory_order_relaxed);
            if (total_alloc > prev_alloc) {
                atomic_store_explicit(&g_peak_allocated, total_alloc, memory_order_relaxed);
                grew = 1;
            }
            uint64_t prev_rec = atomic_load_explicit(&g_peak_recommended_max, memory_order_relaxed);
            if (total_rec > prev_rec) {
                atomic_store_explicit(&g_peak_recommended_max, total_rec, memory_order_relaxed);
            }
            // Snapshot on every new peak so abnormal exits (CPython
            // _exit, SIGKILL, abort) don't lose the high watermark —
            // the file always contains the latest max.
            if (grew) {
                write_snapshot();
            }
            usleep(20000);
        }
    }
    return NULL;
}

static void atexit_finalize(void) {
    atomic_store_explicit(&g_done, 1, memory_order_relaxed);
    if (atomic_load_explicit(&g_thread_started, memory_order_acquire)) {
        pthread_join(g_thread, NULL);
    }
    // Final write picks up the last-iteration peak the poller observed
    // before stopping. Idempotent with the on-grow writes.
    write_snapshot();
}

static void start_once(void) {
    const char *env = getenv("PIPETTE_MEMPROBE_OUT");
    if (env == NULL || env[0] == '\0') {
        // No output channel configured — shim is dormant. Don't spawn
        // the poller, don't register atexit. This makes the dylib safe
        // to leave injected via system-wide DYLD_INSERT_LIBRARIES if a
        // developer wants to (it just costs the constructor's branch).
        return;
    }
    strncpy(g_outpath, env, sizeof(g_outpath) - 1);
    g_outpath[sizeof(g_outpath) - 1] = '\0';

    atexit(atexit_finalize);
    int rc = pthread_create(&g_thread, NULL, poller, NULL);
    if (rc == 0) {
        atomic_store_explicit(&g_thread_started, 1, memory_order_release);
    } else {
        // We can't write a useful snapshot if the poller never ran;
        // emit a diagnostic to stderr and let the parent handle the
        // empty-file case.
        fprintf(stderr,
                "peakmtl: pthread_create failed (rc=%d); "
                "GPU peak measurement disabled for this run\n",
                rc);
    }
}

__attribute__((constructor))
static void peakmtl_init(void) {
    pthread_once(&g_once, start_once);
}
