#include "crash_reporter.h"

// PIPETTE_CRASH_REPORTING_ENABLED is set by native/CMakeLists.txt when the root
// PIPETTE_ENABLE_CRASH_REPORTING option is ON. When it is off, sentry-native was
// never built, so <sentry.h> does not exist and there is nothing to link against.
//
// The BODY is what's conditional, not the file: `ee_crash_reporter_init` is still
// compiled and exported in both configurations. The Rust cdylib declares it in an
// `extern "C"` block (crates/pipette-android/src/lib.rs) and Kotlin declares the
// matching `external fun nativeInit` (BenchmarkCrashReporter.kt), so keeping the
// symbol present means neither of those has to know this option exists.

// Needed by both configurations (each logs its state at init), so it sits above the
// guard rather than being repeated in both branches.
#include <android/log.h>

#ifdef PIPETTE_CRASH_REPORTING_ENABLED

#include <sentry.h>

#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

#include <unistd.h>

namespace {

constexpr const char *kTag = "pipette-crash";

// Owns the envelope-output dir for the process lifetime — sentry keeps the
// transport `state` pointer, so this must outlive `sentry_init`. Static storage
// (not heap) so there's nothing to leak and the ownership is explicit; assigned
// under g_init_mutex before the transport's address is handed out. Lifetime is
// safe: the transport callback runs only at flush (next init) and crash time,
// never during static destruction at process exit (we never call sentry_close
// and the inproc backend doesn't flush at exit), so the handed-out address is
// never used after this global's destructor runs.
std::string g_envelope_dir;
std::mutex g_init_mutex;
// Set true only after a SUCCESSFUL sentry_init, under g_init_mutex. A failed
// init leaves it false so a later call can retry — a transient failure must not
// permanently disable capture, matching the empty-DSN/empty-dir guards below
// (a consumed std::call_once flag would defeat that).
bool g_initialized = false;
// Distinguishes envelopes written within one process.
std::atomic<uint64_t> g_envelope_seq{0};

// A per-incarnation token. pid+seq alone is NOT unique across process lifetimes
// (the OS reuses pids and seq resets to 0 each start), so a recycled pid could
// name-collide with an earlier still-undrained envelope and clobber it on
// rename. A MONOTONIC-nanos token, sampled once on first use (the first envelope
// write of this process) and cached for the process lifetime, makes the name
// unique per incarnation — two incarnations sample at different monotonic times.
// Monotonic (not CLOCK_REALTIME) so a wall-clock jump — NTP step or a manual time
// set — can't make it repeat. Falls back to 0 on the (unexpected) clock failure;
// pid+seq still separate envelopes within a process. Function-local static init
// is thread-safe.
uint64_t process_token() {
  static const uint64_t token = []() -> uint64_t {
    struct timespec ts {};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
      return 0;
    }
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(ts.tv_nsec);
  }();
  return token;
}

// Custom transport send hook. sentry hands us an envelope and transfers
// ownership (we must free it). We serialize it to a uniquely-named file under
// g_envelope_dir — no network — so `:benchmark` spins up no transport thread and
// touches no socket, which is what keeps it from perturbing the memory
// measurement. The main process drains this dir and uploads via its JVM SDK.
//
// Write to a `<final>.tmp` name then rename so the main-process drainer never
// observes a half-written envelope (and the tmp/final always share a base).
void disk_send_envelope(sentry_envelope_t *envelope, void *state) {
  auto *dir = static_cast<std::string *>(state);
  if (dir != nullptr && !dir->empty()) {
    const uint64_t seq = g_envelope_seq.fetch_add(1);
    char final_path[PATH_MAX];
    char tmp[PATH_MAX];
    const int n_final = std::snprintf(
        final_path, sizeof(final_path), "%s/%d-%llu-%llu.envelope",
        dir->c_str(), getpid(), static_cast<unsigned long long>(process_token()),
        static_cast<unsigned long long>(seq));
    const int n_tmp =
        std::snprintf(tmp, sizeof(tmp), "%s.tmp", final_path);
    if (n_final < 0 || n_final >= static_cast<int>(sizeof(final_path)) ||
        n_tmp < 0 || n_tmp >= static_cast<int>(sizeof(tmp))) {
      // Truncated path: bail rather than write to an unintended/partial name.
      __android_log_print(ANDROID_LOG_ERROR, kTag,
                          "crash envelope path too long; dropping");
    } else if (sentry_envelope_write_to_file(envelope, tmp) == 0 &&
               rename(tmp, final_path) == 0) {
      __android_log_print(ANDROID_LOG_INFO, kTag, "wrote crash envelope %s",
                          final_path);
    } else {
      __android_log_print(ANDROID_LOG_ERROR, kTag,
                          "failed to persist crash envelope to %s", final_path);
      remove(tmp);
    }
  }
  sentry_envelope_free(envelope);
}

}  // namespace

void ee_crash_reporter_init(const char *dsn, const char *environment,
                            const char *release, const char *db_path,
                            const char *envelope_dir) {
  // Guard the required inputs BEFORE the init path so a transiently-bad value
  // can't permanently disable capture: each guard returns without marking init
  // done, so a later call with good values still initializes.
  if (dsn == nullptr || dsn[0] == '\0') {
    __android_log_print(ANDROID_LOG_INFO, kTag,
                        "no DSN — native crash reporting disabled");
    return;
  }
  // Without an envelope dir the disk transport has nowhere to persist crashes:
  // sentry_init would still "succeed", but disk_send_envelope's empty-dir guard
  // would then free every crash envelope unwritten. Refuse to init (and say why)
  // rather than look healthy while silently dropping every crash.
  if (envelope_dir == nullptr || envelope_dir[0] == '\0') {
    __android_log_print(ANDROID_LOG_ERROR, kTag,
                        "no envelope dir — native crash reporting disabled");
    return;
  }

  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_initialized) {
    return;  // already initialized this process; sentry_init is not re-entrant
  }
  g_envelope_dir = envelope_dir;

  sentry_options_t *options = sentry_options_new();
  sentry_options_set_dsn(options, dsn);
  // Keep `:benchmark` free of gratuitous background work. sentry-native
  // defaults enable_logs / enable_metrics / auto_session_tracking to ON: the
  // first two spawn batcher threads that wake periodically, and session
  // tracking writes a session envelope (which our disk transport would then
  // emit and the main process would upload as if it were a crash). None are
  // used here, and any resident thread/allocation risks perturbing the memory
  // measurement this process exists to take — so disable all three. Combined
  // with SENTRY_TRANSPORT=none, the only sentry thread left is the inproc
  // backend's idle signal-handler thread.
  sentry_options_set_enable_logs(options, 0);
  sentry_options_set_enable_metrics(options, 0);
  sentry_options_set_auto_session_tracking(options, 0);
  if (environment != nullptr && environment[0] != '\0') {
    sentry_options_set_environment(options, environment);
  }
  if (release != nullptr && release[0] != '\0') {
    sentry_options_set_release(options, release);
  }
  if (db_path != nullptr && db_path[0] != '\0') {
    sentry_options_set_database_path(options, db_path);
  }
  // Backend (inproc) and transport (none) are fixed at build time via
  // SENTRY_BACKEND / SENTRY_TRANSPORT; install our disk-writing transport.
  sentry_transport_t *transport = sentry_transport_new(disk_send_envelope);
  sentry_transport_set_state(transport, &g_envelope_dir);
  sentry_options_set_transport(options, transport);

  const int rc = sentry_init(options);
  if (rc != 0) {
    // Leave g_initialized false so a later call can retry.
    __android_log_print(ANDROID_LOG_ERROR, kTag, "sentry_init failed (%d)", rc);
    return;
  }
  // Filter these events in the dashboard: they all come from `:benchmark`.
  sentry_set_tag("process", "benchmark");
  g_initialized = true;
  __android_log_print(
      ANDROID_LOG_INFO, kTag,
      "native crash reporter initialized (inproc backend, disk transport)");
}

#else  // !PIPETTE_CRASH_REPORTING_ENABLED

// No-op stub for builds configured with -DPIPETTE_ENABLE_CRASH_REPORTING=OFF.
//
// Signature-compatible with the real implementation so the Rust FFI declaration and
// the JNI `nativeInit` binding are unchanged; the parameters are deliberately
// unnamed because nothing here consumes them.
//
// It logs once per `:benchmark` process launch (init is called exactly once) so a
// build that silently captures nothing is still diagnosable from logcat. Without
// this line the only symptom would be crash envelopes that never appear, which
// looks identical to a reporting bug.
//
// JVM uncaught-exception capture is unaffected: that lives entirely in
// BenchmarkCrashReporter.installJvmCrashHandler and never enters native code.
void ee_crash_reporter_init(const char * /*dsn*/, const char * /*environment*/,
                            const char * /*release*/, const char * /*db_path*/,
                            const char * /*envelope_dir*/) {
  __android_log_print(ANDROID_LOG_INFO, "pipette-crash",
                      "built without native crash capture "
                      "(PIPETTE_ENABLE_CRASH_REPORTING=OFF)");
}

#endif  // PIPETTE_CRASH_REPORTING_ENABLED
