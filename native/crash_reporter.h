// Native crash capture for the isolated Android `:benchmark` process.
//
// The `:benchmark` process runs the llama.cpp model and exists to measure model
// memory cleanly, so it must carry NO JVM Sentry SDK (that skews the measurement
// — see PR #527 review). Instead we link sentry-native's inproc signal handler
// into this process's own `.so` and install a custom transport that writes each
// crash envelope to a file (no network, no transport threads in `:benchmark`).
// The main process (which has the JVM Sentry SDK) drains that dir and uploads.
//
// C-ABI so the Rust cdylib can call it via FFI (mirrors the `ee_*` shim funcs).

#ifndef PIPETTE_CRASH_REPORTER_H
#define PIPETTE_CRASH_REPORTER_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize native crash capture. Idempotent: only the first call per process
// takes effect. A null/empty `dsn` disables reporting (init becomes a no-op).
//   dsn          : Sentry DSN (read from the JVM manifest by the caller)
//   environment  : "debug" / "production" (may be null/empty)
//   release      : release identifier, e.g. "<pkg>@<versionName>" (may be null)
//   db_path      : sentry-native's private working/database dir (persists a
//                  crash across the process death until the next init flushes it)
//   envelope_dir : dir where crash envelopes are written for the main process
//                  to pick up and upload
void ee_crash_reporter_init(const char *dsn, const char *environment,
                            const char *release, const char *db_path,
                            const char *envelope_dir);

#ifdef __cplusplus
}
#endif

#endif  // PIPETTE_CRASH_REPORTER_H
