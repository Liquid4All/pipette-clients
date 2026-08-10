mod engine;
mod error;
mod llama;
mod loader;

// The benchmark measurement kernel lives in the repo-root `native/` directory
// (alongside the shared `llama_shim`) and is textually included here via
// `#[path]`. It refers to this crate's `error::PipetteError`, `llama`, and
// callback types through `crate::`-relative paths. (The iOS client shared this
// kernel until iOS moved to a native-Swift llama.cpp engine.)
#[path = "../../../native/benchmarks.rs"]
mod benchmarks;

pub use engine::LlamaEngine;
pub use error::PipetteError;

use std::ptr;
use std::sync::atomic::AtomicBool;
#[cfg(target_os = "android")]
use std::sync::atomic::Ordering;
use std::sync::Arc;

use jni::errors::ThrowRuntimeExAndDefault;
use jni::objects::{JClass, JObject, JString};
use jni::refs::Global;
use jni::strings::JNIString;
use jni::sys::{jboolean, jint, jlong, jstring};
use jni::{jni_sig, jni_str, Env, EnvUnowned, JValue, JavaVM};

use pipette_plan_types::thermal::{AndroidTemperatureSensor, AndroidThermalStatus};

/// Route the crate's `log::*` records to logcat under the `pipette-native`
/// tag. Idempotent (`init_once` guards against re-init), so it's cheap to call
/// at the top of every JNI entry point — whichever lands first wins and the
/// rest are no-ops. Without this the in-kernel instrumentation (e.g. the
/// per-rep `end_to_end_latency: measurement run i/5` lines, decode timings, and
/// readiness verdicts) is emitted into a process with no registered logger and
/// silently dropped, leaving on-device runs un-observable.
#[cfg(target_os = "android")]
fn init_native_logging() {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("pipette-native"),
    );
}

#[cfg(not(target_os = "android"))]
fn init_native_logging() {}

// Native crash capture (sentry-native inproc + disk transport), implemented in
// the C++ bridge (native/crash_reporter.cpp). Rust only forwards the JVM-side
// strings; all Sentry state lives in C so no JVM Sentry SDK is loaded here.
// Android-only — the bridge is only built there; host `cargo check`/tooling uses
// the no-op branch in nativeInit, matching the extern gating in llama.rs.
#[cfg(target_os = "android")]
extern "C" {
    fn ee_crash_reporter_init(
        dsn: *const std::ffi::c_char,
        environment: *const std::ffi::c_char,
        release: *const std::ffi::c_char,
        db_path: *const std::ffi::c_char,
        envelope_dir: *const std::ffi::c_char,
    );
}

#[derive(Debug, Clone)]
pub struct ModelHandle {
    pub ptr: u64,
}

pub trait ProgressCallback: Send + Sync {
    fn on_progress(&self, completed: u32, total: u32, message: String) -> bool;
}

/// Mirrors the iOS-shared kernel's `ReadinessOutcome` (the shared
/// `benchmarks.rs` is reused via `#[path]` and refers to this name), and the
/// Kotlin `ReadinessOutcome` the JNI boundary carries an encoding of.
///
/// All three variants are produced here. `TimedOut` was previously unreachable
/// on Android. The Kotlin gate returned `Unit` and its callers reported a bare
/// "not cancelled" boolean, which meant a device that never cooled reported
/// "proceed" and its throttled numbers were recorded as an ordinary result
/// (PIP-143).
pub enum ReadinessOutcome {
    Ready,
    Cancelled,
    TimedOut { observed: String },
}

pub trait ReadinessCallback: Send + Sync {
    fn wait_until_ready(&self) -> ReadinessOutcome;
}

/// Samples device thermal telemetry for per-rep benchmark cells. `headroom` and
/// `status` come from the app-SDK `PowerManager` (Kotlin-side, no permission);
/// `sensors` is read natively from `dumpsys` (privileged, see below). Unlike
/// [`ReadinessCallback`], a sampling failure is never fatal — each method yields
/// `None` for that rep, so telemetry can't abort a benchmark.
pub trait ThermalSampler: Send + Sync {
    /// Current thermal headroom (0.0 cool → 1.0 at the severe-throttle
    /// threshold, may exceed 1.0), or `None` when unavailable (unsupported API,
    /// not yet sampled, or a read failure).
    fn headroom(&self) -> Option<f32>;

    /// Current device-level thermal status
    /// (`PowerManager.getCurrentThermalStatus()`), or `None` when unavailable.
    /// Always returns a value on a supported device, so its all-or-nothing
    /// [`pipette_plan_types::thermal::ThermalTelemetry`] scalar series reliably populates.
    fn status(&self) -> Option<AndroidThermalStatus>;

    /// Per-sensor thermal-HAL temperatures for this snapshot, or `None` when the
    /// privileged read is unavailable (no `DUMP` grant, exec denied, or no
    /// sensor reported). Best-effort: the sensor series flattens whatever reps
    /// report, so `None` here simply omits the family.
    fn sensors(&self) -> Option<Vec<AndroidTemperatureSensor>>;
}

struct JavaProgressCallback {
    vm: JavaVM,
    callback: Global<JObject<'static>>,
}

impl ProgressCallback for JavaProgressCallback {
    fn on_progress(&self, completed: u32, total: u32, message: String) -> bool {
        self.vm
            .attach_current_thread(|env: &mut Env| -> jni::errors::Result<bool> {
                let message = env.new_string(message)?;
                let message = JObject::from(message);
                let value = env.call_method(
                    self.callback.as_obj(),
                    jni_str!("onProgress"),
                    jni_sig!("(IILjava/lang/String;)Z"),
                    &[
                        JValue::Int(completed as jint),
                        JValue::Int(total as jint),
                        JValue::Object(&message),
                    ],
                )?;
                value.z()
            })
            .unwrap_or(false)
    }
}

struct JavaReadinessCallback {
    vm: JavaVM,
    callback: Global<JObject<'static>>,
}

/// Tags marking an encoded non-ready [`ReadinessOutcome`] on the wire. Must stay
/// in step with `ReadinessOutcome.CANCELLED_PREFIX` / `TIMED_OUT_PREFIX` in
/// `Readiness.kt`, which documents the encoding.
const CANCELLED_PREFIX: &str = "cancelled:";
const TIMED_OUT_PREFIX: &str = "timedout:";

impl ReadinessCallback for JavaReadinessCallback {
    fn wait_until_ready(&self) -> ReadinessOutcome {
        // Kotlin `BenchmarkCooldownCallback.waitUntilReady(): String?` returns the
        // gate's outcome encoded as null (ready) or a tagged string (see
        // `ReadinessOutcome.encode` in `Readiness.kt`). It was a `Boolean` until
        // PIP-143, which could not express "gave up while the device was still
        // hot" and so admitted throttled reps as ordinary ones.
        let encoded =
            self.vm
                .attach_current_thread(|env: &mut Env| -> jni::errors::Result<Option<String>> {
                    let value = env.call_method(
                        self.callback.as_obj(),
                        jni_str!("waitUntilReady"),
                        jni_sig!("()Ljava/lang/String;"),
                        &[],
                    )?;
                    let object = value.l()?;
                    if object.is_null() {
                        return Ok(None);
                    }
                    let text = env.cast_local::<JString>(object)?;
                    // Bound to a local so the `MUTF8Chars` borrow of `text` ends
                    // before `text` itself goes out of scope at the block's end.
                    let decoded = text.mutf8_chars(env)?.to_string();
                    Ok(Some(decoded))
                });
        match encoded {
            Ok(None) => ReadinessOutcome::Ready,
            Ok(Some(reason)) if reason.starts_with(CANCELLED_PREFIX) => ReadinessOutcome::Cancelled,
            // An untagged or unknown-tagged string falls here too, deliberately:
            // if the two sides of this encoding ever drift, the safe reading is
            // the one that fails the cell rather than the one that admits a rep.
            Ok(Some(reason)) => ReadinessOutcome::TimedOut {
                observed: reason
                    .strip_prefix(TIMED_OUT_PREFIX)
                    .unwrap_or(&reason)
                    .to_string(),
            },
            // A failed JNI hop, like a cancel, records nothing. That is the safe
            // direction, and it must not be reported as a thermal verdict this
            // side never actually observed.
            Err(_) => ReadinessOutcome::Cancelled,
        }
    }
}

struct JavaThermalSampler {
    vm: JavaVM,
    callback: Global<JObject<'static>>,
    // Latches once a `sensors()` read comes back empty (no `DUMP` grant, exec
    // denied, or no HAL block) so the rest of the run stops forking `dumpsys` —
    // on the common ungranted device we pay at most one failed fork instead of
    // 2×N. Mirrors `Readiness.kt`'s `nativeUnavailable` short-circuit.
    // Only read on Android (the sensors path is `#[cfg(target_os = "android")]`);
    // on host builds the field is constructed but unused.
    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    sensors_unavailable: AtomicBool,
}

impl ThermalSampler for JavaThermalSampler {
    fn headroom(&self) -> Option<f32> {
        // Kotlin `BenchmarkThermalCallback.sampleHeadroom(): Float` returns the
        // current headroom or `Float.NaN` when unavailable. A JNI failure is
        // treated as a missing sample (`None`), never a cancellation — the
        // opposite of `JavaReadinessCallback`: thermal telemetry must never abort
        // a benchmark. Any non-finite reading (`NaN` or `±inf`) also maps to
        // `None` via `is_finite()`.
        let value = self
            .vm
            .attach_current_thread(|env: &mut Env| -> jni::errors::Result<f32> {
                let value = env.call_method(
                    self.callback.as_obj(),
                    jni_str!("sampleHeadroom"),
                    jni_sig!("()F"),
                    &[],
                )?;
                value.f()
            })
            .ok()?;
        value.is_finite().then_some(value)
    }

    fn status(&self) -> Option<AndroidThermalStatus> {
        // Kotlin `BenchmarkThermalCallback.sampleStatus(): Int` returns the
        // `PowerManager` `THERMAL_STATUS_*` ordinal (0–6). A JNI failure → `None`,
        // never a cancellation (like `headroom`); the ordinal→enum map drops any
        // out-of-range value to `None` as a defensive guard.
        let code = self
            .vm
            .attach_current_thread(|env: &mut Env| -> jni::errors::Result<i32> {
                let value = env.call_method(
                    self.callback.as_obj(),
                    jni_str!("sampleStatus"),
                    jni_sig!("()I"),
                    &[],
                )?;
                value.i()
            })
            .ok()?;
        pipette_device::android_thermal_status_from_code(code)
    }

    fn sensors(&self) -> Option<Vec<AndroidTemperatureSensor>> {
        // Read natively (no Java hop): the thermal-HAL parser lives in Rust and
        // the data source is `dumpsys thermalservice`, which needs
        // `android.permission.DUMP` (grantable on lab devices via
        // `adb shell pm grant`). Best-effort — `None` when the grant is absent or
        // the exec is denied. Once a read fails, latch off to avoid re-forking
        // `dumpsys` for every remaining rep on an ungranted device.
        #[cfg(target_os = "android")]
        {
            if self.sensors_unavailable.load(Ordering::Relaxed) {
                return None;
            }
            match pipette_device::android_hal_sensors() {
                Some(sensors) => Some(sensors),
                None => {
                    self.sensors_unavailable.store(true, Ordering::Relaxed);
                    None
                }
            }
        }
        #[cfg(not(target_os = "android"))]
        {
            None
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_ai_liquid_pipette_LlamaEngine_nativeLlamaCppCommit(
    mut env: EnvUnowned,
    _class: JClass,
) -> jstring {
    env.with_env(|env| -> jni::errors::Result<Option<jstring>> {
        Ok(Some(string_result(env, engine::llama_cpp_commit())))
    })
    .resolve::<ThrowRuntimeExAndDefault>()
    .unwrap_or(ptr::null_mut())
}

/// The active CPU-backend feature descriptor (e.g. `"dotprod,fp16_va,neon"`),
/// or `null` when no model has been loaded yet (the backend registers lazily)
/// or no CPU feature list is available. Surfaced into the submission payload as
/// `runtime_cpu_variant` — Android does runtime CPU-feature dispatch, so this
/// fingerprints the selected variant.
#[no_mangle]
pub extern "system" fn Java_ai_liquid_pipette_LlamaEngine_nativeCpuBackendDescriptor(
    mut env: EnvUnowned,
    _class: JClass,
) -> jstring {
    env.with_env(|env| -> jni::errors::Result<Option<jstring>> {
        Ok(Some(match llama::cpu_backend_descriptor() {
            Some(desc) => string_result(env, desc),
            None => ptr::null_mut(),
        }))
    })
    .resolve::<ThrowRuntimeExAndDefault>()
    .unwrap_or(ptr::null_mut())
}

/// Load a model and box the resulting [`LlamaEngine`]; the returned `jlong` is
/// a `*mut LlamaEngine` owned by the Kotlin side until `nativeDestroy`.
#[no_mangle]
pub extern "system" fn Java_ai_liquid_pipette_LlamaEngine_nativeCreate(
    mut env: EnvUnowned,
    _class: JClass,
    path: JString,
    n_gpu_layers: jint,
    context_size: jint,
    n_ubatch: jint,
) -> jlong {
    init_native_logging();
    env.with_env(|env| -> jni::errors::Result<jlong> {
        let result = (|| -> Result<jlong, PipetteError> {
            let path = java_string(env, path)?;
            let engine = LlamaEngine::create(&loader::LoadOptions {
                model_path: path,
                n_gpu_layers: n_gpu_layers as u32,
                context_size: context_size as u32,
                n_ubatch: n_ubatch as u32,
            })?;
            Ok(Box::into_raw(Box::new(engine)) as jlong)
        })();
        Ok(match result {
            Ok(ptr) => ptr,
            Err(error) => {
                throw_error(env, error);
                0
            }
        })
    })
    .resolve::<ThrowRuntimeExAndDefault>()
}

/// Free a boxed [`LlamaEngine`] (and the model it holds, via `Drop`).
#[no_mangle]
pub extern "system" fn Java_ai_liquid_pipette_LlamaEngine_nativeDestroy(
    mut env: EnvUnowned,
    _class: JClass,
    engine_ptr: jlong,
) {
    env.with_env(|_env| -> jni::errors::Result<()> {
        if engine_ptr != 0 {
            // SAFETY: `engine_ptr` came from `Box::into_raw` in `nativeCreate`
            // and is destroyed exactly once (the actor nulls its handle after).
            drop(unsafe { Box::from_raw(engine_ptr as *mut LlamaEngine) });
        }
        Ok(())
    })
    .resolve::<ThrowRuntimeExAndDefault>();
}

/// Initialize native crash capture for the `:benchmark` process. Forwards the
/// JVM-side config (DSN/environment/release from the manifest, plus the
/// SDK-private db dir and the envelope output dir under `filesDir`) to the C++
/// bridge, which owns all Sentry state. Idempotent on the native side.
#[no_mangle]
pub extern "system" fn Java_ai_liquid_pipette_BenchmarkCrashReporter_nativeInit(
    mut env: EnvUnowned,
    _class: JClass,
    dsn: JString,
    environment: JString,
    release: JString,
    db_path: JString,
    envelope_dir: JString,
) {
    init_native_logging();
    env.with_env(|env| -> jni::errors::Result<()> {
        // Each CString outlives the FFI call; the C side copies what it retains.
        let c = |env: &mut Env, s: JString| {
            std::ffi::CString::new(java_string(env, s).unwrap_or_default()).unwrap_or_default()
        };
        let (c_dsn, c_env, c_release, c_db, c_envelope) = (
            c(env, dsn),
            c(env, environment),
            c(env, release),
            c(env, db_path),
            c(env, envelope_dir),
        );
        // These are required; an empty value means the JNI string read failed (or misconfiguration), and the `unwrap_or_default()` above collapses that
        // to "". Surface it rather than let the native side fail quietly. Independent (not else-if) checks so EVERY empty field is reported — several
        // can be empty at once (e.g. a broken JNIEnv) and hiding all but the first would obscure the diagnosis. Worded to match the C++ guards: an
        // empty DSN or envelope_dir DISABLES native capture entirely (ee_crash_reporter_init bails), whereas an empty db_path only degrades
        // cross-process-death persistence/flush.
        if c_dsn.as_bytes().is_empty() {
            log::warn!("benchmark crash reporter: empty DSN at native init — native crash capture disabled");
        }
        if c_envelope.as_bytes().is_empty() {
            log::warn!(
                "benchmark crash reporter: empty envelope_dir at native init — native crash capture disabled (crashes cannot be persisted)"
            );
        }
        if c_db.as_bytes().is_empty() {
            log::warn!("benchmark crash reporter: empty db_path at native init — crashes may not persist/flush across process death");
        }
        // SAFETY: all five pointers reference live CString buffers for the
        // duration of the call; the bridge copies the strings it keeps.
        #[cfg(target_os = "android")]
        unsafe {
            ee_crash_reporter_init(
                c_dsn.as_ptr(),
                c_env.as_ptr(),
                c_release.as_ptr(),
                c_db.as_ptr(),
                c_envelope.as_ptr(),
            );
        }
        // Host stub: the C++ bridge is Android-only, so keep the CStrings "used" for a clean `cargo check` on non-Android targets.
        #[cfg(not(target_os = "android"))]
        let _ = (&c_dsn, &c_env, &c_release, &c_db, &c_envelope);
        Ok(())
    })
    .resolve::<ThrowRuntimeExAndDefault>();
}

/// Run a benchmark against the model held by the boxed engine `engine_ptr`.
#[no_mangle]
pub extern "system" fn Java_ai_liquid_pipette_LlamaEngine_nativeRunBenchmark(
    mut env: EnvUnowned,
    _class: JClass,
    engine_ptr: jlong,
    benchmark_json: JString,
    n_gpu_layers: jint,
    mmproj_path: JString,
    progress_callback: JObject,
    cooldown_callback: JObject,
    thermal_callback: JObject,
) -> jstring {
    env.with_env(|env| -> jni::errors::Result<Option<jstring>> {
        let result = (|| -> Result<String, PipetteError> {
            if engine_ptr == 0 {
                return Err(PipetteError::ModelLoad {
                    msg: "engine pointer is null".to_string(),
                });
            }
            // SAFETY: `engine_ptr` is a live `Box<LlamaEngine>` (see nativeCreate);
            // the actor serializes calls so there is no concurrent &mut.
            let engine = unsafe { &*(engine_ptr as *const LlamaEngine) };
            let benchmark_json = java_string(env, benchmark_json)?;
            let mmproj_path = nullable_java_string(env, mmproj_path)?;
            let progress = progress_callback_from_java(env, progress_callback)?;
            let readiness = readiness_callback_from_java(env, cooldown_callback)?;
            let thermal = thermal_sampler_from_java(env, thermal_callback)?;
            engine.run_benchmark(
                &benchmark_json,
                n_gpu_layers as u32,
                mmproj_path.as_deref(),
                progress,
                readiness,
                thermal,
            )
        })();
        Ok(Some(string_or_throw(env, result)))
    })
    .resolve::<ThrowRuntimeExAndDefault>()
    .unwrap_or(ptr::null_mut())
}

/// Run a benchmark that loads its own model and unloads when done — the
/// `max_memory_usage` case, where the load is part of the measurement.
#[no_mangle]
pub extern "system" fn Java_ai_liquid_pipette_LlamaEngine_nativeRunFresh(
    mut env: EnvUnowned,
    _class: JClass,
    benchmark_json: JString,
    model_path: JString,
    n_gpu_layers: jint,
    context_size: jint,
    n_ubatch: jint,
    mmproj_path: JString,
    progress_callback: JObject,
    cooldown_callback: JObject,
    thermal_callback: JObject,
) -> jstring {
    init_native_logging();
    env.with_env(|env| -> jni::errors::Result<Option<jstring>> {
        let result = (|| -> Result<String, PipetteError> {
            let benchmark_json = java_string(env, benchmark_json)?;
            let model_path = java_string(env, model_path)?;
            let mmproj_path = nullable_java_string(env, mmproj_path)?;
            let progress = progress_callback_from_java(env, progress_callback)?;
            let readiness = readiness_callback_from_java(env, cooldown_callback)?;
            let thermal = thermal_sampler_from_java(env, thermal_callback)?;
            engine::run_fresh(
                &benchmark_json,
                &loader::LoadOptions {
                    model_path,
                    n_gpu_layers: n_gpu_layers as u32,
                    context_size: context_size as u32,
                    n_ubatch: n_ubatch as u32,
                },
                mmproj_path.as_deref(),
                progress,
                readiness,
                thermal,
            )
        })();
        Ok(Some(string_or_throw(env, result)))
    })
    .resolve::<ThrowRuntimeExAndDefault>()
    .unwrap_or(ptr::null_mut())
}

/// Block until the device is thermally/charge ready to run a benchmark,
/// delegating to the shared `pipette_readiness` gate. Returns Java
/// `null` once ready, or the readiness error message as a string.
///
/// `skip_thermal` carries the app's waiver (PIP-434). It waives only the
/// thermal criterion; the load criterion still applies, so a skipped call is
/// still a real gate rather than a no-op. That is why the Kotlin caller keeps
/// taking this probe under the waiver instead of returning early.
#[no_mangle]
pub extern "system" fn Java_ai_liquid_pipette_Readiness_nativeWaitUntilReady(
    mut env: EnvUnowned,
    _class: JClass,
    max_wait_millis: jlong,
    skip_thermal: jboolean,
) -> jstring {
    env.with_env(|env| -> jni::errors::Result<Option<jstring>> {
        let max_wait =
            (max_wait_millis > 0).then(|| std::time::Duration::from_millis(max_wait_millis as u64));
        // `Unset` rather than `Enforce` when the app doesn't waive: an app that
        // left the toggle off has expressed no opinion, not an objection, so
        // `PIPETTE_READINESS_SKIP_THERMAL` must still decide. That env var is
        // how an `adb shell` invocation waives the criterion, and `Enforce`
        // would override it. `Skip` is an opinion and does override it, which
        // is the point of the toggle.
        let gate = if skip_thermal {
            pipette_readiness::ThermalGate::Skip
        } else {
            pipette_readiness::ThermalGate::Unset
        };
        let resolved = pipette_readiness::resolve_readiness(max_wait, gate);
        let result = pipette_readiness::wait_until_ready(&resolved);
        // Discriminate so the Kotlin driver can fall back to a coarser signal
        // (PowerManager thermal status) when the fine-grained probes aren't
        // readable in this process — the app sandbox denies dumpsys (DUMP
        // permission), /sys/class/thermal, and /proc/stat. `TimedOut` means the
        // probes WERE readable but the device stayed hot (keep waiting on the
        // real signal); any other error means the probes themselves failed
        // (fall back to the OS thermal status).
        Ok(match result {
            Ok(()) => None,
            Err(e @ pipette_readiness::ReadinessError::TimedOut { .. }) => {
                Some(string_result(env, format!("notready:{e}")))
            }
            Err(e) => Some(string_result(env, format!("unavailable:{e}"))),
        })
    })
    .resolve::<ThrowRuntimeExAndDefault>()
    .unwrap_or(std::ptr::null_mut())
}

fn java_string(env: &mut Env, value: JString) -> Result<String, PipetteError> {
    value
        .mutf8_chars(env)
        .map(|s| s.to_string())
        .map_err(|e| PipetteError::Inference { msg: e.to_string() })
}

fn nullable_java_string(env: &mut Env, value: JString) -> Result<Option<String>, PipetteError> {
    if value.is_null() {
        return Ok(None);
    }
    java_string(env, value).map(Some)
}

fn progress_callback_from_java(
    env: &mut Env,
    value: JObject,
) -> Result<Option<Arc<dyn ProgressCallback>>, PipetteError> {
    if value.is_null() {
        return Ok(None);
    }
    let vm = env
        .get_java_vm()
        .map_err(|e| PipetteError::Inference { msg: e.to_string() })?;
    let callback = env
        .new_global_ref(value)
        .map_err(|e| PipetteError::Inference { msg: e.to_string() })?;
    Ok(Some(Arc::new(JavaProgressCallback { vm, callback })))
}

fn readiness_callback_from_java(
    env: &mut Env,
    value: JObject,
) -> Result<Option<Arc<dyn ReadinessCallback>>, PipetteError> {
    if value.is_null() {
        return Ok(None);
    }
    let vm = env
        .get_java_vm()
        .map_err(|e| PipetteError::Inference { msg: e.to_string() })?;
    let callback = env
        .new_global_ref(value)
        .map_err(|e| PipetteError::Inference { msg: e.to_string() })?;
    Ok(Some(Arc::new(JavaReadinessCallback { vm, callback })))
}

fn thermal_sampler_from_java(
    env: &mut Env,
    value: JObject,
) -> Result<Option<Arc<dyn ThermalSampler>>, PipetteError> {
    if value.is_null() {
        return Ok(None);
    }
    let vm = env
        .get_java_vm()
        .map_err(|e| PipetteError::Inference { msg: e.to_string() })?;
    let callback = env
        .new_global_ref(value)
        .map_err(|e| PipetteError::Inference { msg: e.to_string() })?;
    Ok(Some(Arc::new(JavaThermalSampler {
        vm,
        callback,
        sensors_unavailable: AtomicBool::new(false),
    })))
}

fn string_or_throw(env: &mut Env, result: Result<String, PipetteError>) -> jstring {
    match result {
        Ok(value) => string_result(env, value),
        Err(error) => {
            throw_error(env, error);
            ptr::null_mut()
        }
    }
}

fn string_result(env: &mut Env, value: String) -> jstring {
    match env.new_string(value) {
        Ok(value) => value.into_raw(),
        Err(error) => {
            let _ = env.throw_new(
                jni_str!("java/lang/IllegalStateException"),
                JNIString::from(error.to_string()),
            );
            ptr::null_mut()
        }
    }
}

fn throw_error(env: &mut Env, error: PipetteError) {
    let _ = env.throw_new(
        jni_str!("java/lang/IllegalStateException"),
        JNIString::from(error.to_string()),
    );
}
