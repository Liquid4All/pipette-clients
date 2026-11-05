// peakmtl: a DYLD_INSERT_LIBRARIES shim that polls every Metal device's
// currentAllocatedSize every 20 ms and writes the running peak to the file
// named by `PIPETTE_MEMPROBE_OUT` (dormant when unset). `src/metal.rs` parses
// its `key=value` lines.
//
// A file rather than stderr keeps the runtime's own stderr clean, and
// rewriting it on every new peak keeps the peak through `_exit()` or SIGKILL.
//
// Caveats: docs/methodology/peak-memory-macos.md.

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

// Cross-thread state is C11 atomics: `volatile` gives no synchronization.
static _Atomic uint64_t g_peak_allocated = 0;
static _Atomic uint64_t g_peak_recommended_max = 0;
static _Atomic int g_unified = -1;
static _Atomic int g_n_devices = 0;
static _Atomic int g_done = 0;
static _Atomic int g_thread_started = 0;
static pthread_t g_thread;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

// Copied, not pointed at: the program can change its environment later.
// Empty means dormant.
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
            // On every new peak, so `_exit`, SIGKILL or abort keep it.
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
    write_snapshot();
}

static void start_once(void) {
    const char *env = getenv("PIPETTE_MEMPROBE_OUT");
    if (env == NULL || env[0] == '\0') {
        // Dormant, so the dylib is harmless injected into any process.
        return;
    }
    strncpy(g_outpath, env, sizeof(g_outpath) - 1);
    g_outpath[sizeof(g_outpath) - 1] = '\0';

    atexit(atexit_finalize);
    int rc = pthread_create(&g_thread, NULL, poller, NULL);
    if (rc == 0) {
        atomic_store_explicit(&g_thread_started, 1, memory_order_release);
    } else {
        // The atexit snapshot still lands, with a zero peak.
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
