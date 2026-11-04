mod engine;
mod error;
mod llama;
mod loader;

// The measurement kernel lives beside the shim in `android/native/`; it reaches
// this crate's types through `crate::` paths.
#[path = "../../../android/native/benchmarks.rs"]
mod benchmarks;

// Owned by `build.rs`; included here only so its tests run.
#[cfg(test)]
#[path = "../tag.rs"]
mod tag;

pub use engine::LlamaEngine;
pub use error::PipetteError;

use std::ptr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
#[cfg(target_os = "android")]
use std::sync::atomic::Ordering;

use jni::errors::ThrowRuntimeExAndDefault;
use jni::objects::{JClass, JObject, JString};
use jni::refs::Global;
use jni::strings::JNIString;
use jni::sys::{jint, jlong, jstring};
use jni::{Env, EnvUnowned, JValue, JavaVM, jni_sig, jni_str};

use pipette_plan_types::{AndroidTemperatureSensor, AndroidThermalStatus};

/// Route `log` records (and pipette-ops/pipette-device `tracing` events) to
/// logcat. Idempotent, so each JNI entry point calls it; without a logger the
/// kernel's instrumentation is silently dropped.
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

// Native crash capture lives in android/native/crash_reporter.cpp, which owns all
// Sentry state; Rust only forwards the JVM-side strings.
#[cfg(target_os = "android")]
unsafe extern "C" {
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

fn saturating_jint(n: u32) -> jint {
    jint::try_from(n).unwrap_or(jint::MAX)
}

pub trait ProgressCallback: Send + Sync {
    fn on_progress(&self, completed: u32, total: u32, message: String) -> bool;
}

/// Mirrors the Kotlin `ReadinessOutcome`, which crosses JNI encoded (see
/// [`CANCELLED_PREFIX`]).
pub enum ReadinessOutcome {
    Ready,
    Cancelled,
    TimedOut { observed: String },
}

pub trait ReadinessCallback: Send + Sync {
    fn wait_until_ready(&self) -> ReadinessOutcome;
}

/// Per-rep thermal telemetry. Unlike [`ReadinessCallback`], a failure is never
/// fatal: the method yields `None` for that rep.
pub trait ThermalSampler: Send + Sync {
    /// Thermal headroom (0.0 cool, 1.0 at the severe-throttle threshold, may
    /// exceed 1.0).
    fn headroom(&self) -> Option<f32>;

    /// `PowerManager.getCurrentThermalStatus()`. Always present on a supported
    /// device, so its all-or-nothing
    /// [`pipette_plan_types::ThermalTelemetry`] series reliably populates.
    fn status(&self) -> Option<AndroidThermalStatus>;

    /// Thermal-HAL temperatures from the privileged `dumpsys` read; `None`
    /// omits the sensor family for this rep.
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
                        JValue::Int(saturating_jint(completed)),
                        JValue::Int(saturating_jint(total)),
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

/// Tags of an encoded non-ready [`ReadinessOutcome`]. Must match
/// `ReadinessOutcome.CANCELLED_PREFIX` / `TIMED_OUT_PREFIX` in `Readiness.kt`,
/// which documents the encoding.
const CANCELLED_PREFIX: &str = "cancelled:";
const TIMED_OUT_PREFIX: &str = "timedout:";

impl ReadinessCallback for JavaReadinessCallback {
    fn wait_until_ready(&self) -> ReadinessOutcome {
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
            // An untagged or unknown tag lands here on purpose: if the two sides
            // drift, failing the cell is safer than admitting a rep.
            Ok(Some(reason)) => ReadinessOutcome::TimedOut {
                observed: reason
                    .strip_prefix(TIMED_OUT_PREFIX)
                    .unwrap_or(&reason)
                    .to_string(),
            },
            // A failed JNI hop records nothing, and must not be reported as a
            // thermal verdict this side never observed.
            Err(_) => ReadinessOutcome::Cancelled,
        }
    }
}

struct JavaThermalSampler {
    vm: JavaVM,
    callback: Global<JObject<'static>>,
    // Latches after a failed `sensors()` read so an ungranted device forks
    // `dumpsys` once per run rather than before and after every rep.
    #[cfg_attr(
        not(target_os = "android"),
        allow(dead_code, reason = "read only by the android sensors path")
    )]
    sensors_unavailable: AtomicBool,
}

impl ThermalSampler for JavaThermalSampler {
    fn headroom(&self) -> Option<f32> {
        // Kotlin returns `Float.NaN` when unavailable. Unlike in
        // `JavaReadinessCallback`, a JNI failure is a missing sample, not a cancel.
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
        // A `PowerManager` `THERMAL_STATUS_*` ordinal.
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
        // `dumpsys thermalservice` needs `android.permission.DUMP`, granted on lab
        // devices via `adb shell pm grant`.
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

#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_liquid_pipette_LlamaEngine_nativeLlamaCppCommit(
    env: EnvUnowned,
    class: JClass,
) -> jstring {
    Java_ai_liquid_pipette_NativeLib_nativeLlamaCppCommit(env, class)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_liquid_pipette_NativeLib_nativeLlamaCppCommit(
    mut env: EnvUnowned,
    _class: JClass,
) -> jstring {
    env.with_env(|env| -> jni::errors::Result<Option<jstring>> {
        Ok(Some(string_result(env, engine::llama_cpp_commit())))
    })
    .resolve::<ThrowRuntimeExAndDefault>()
    .unwrap_or(ptr::null_mut())
}

/// The upstream `bNNNN` tag on the vendored llama.cpp pin, or `""` when it has
/// none. How it is advertised: `Capabilities.flags`.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_liquid_pipette_LlamaEngine_nativeLlamaCppTag(
    env: EnvUnowned,
    class: JClass,
) -> jstring {
    Java_ai_liquid_pipette_NativeLib_nativeLlamaCppTag(env, class)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_liquid_pipette_NativeLib_nativeLlamaCppTag(
    mut env: EnvUnowned,
    _class: JClass,
) -> jstring {
    env.with_env(|env| -> jni::errors::Result<Option<jstring>> {
        Ok(Some(string_result(env, engine::llama_cpp_tag())))
    })
    .resolve::<ThrowRuntimeExAndDefault>()
    .unwrap_or(ptr::null_mut())
}

/// The active CPU-backend feature descriptor (see
/// [`llama::cpu_backend_descriptor`]), or `null` before the first model load.
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_liquid_pipette_LlamaEngine_nativeCpuBackendDescriptor(
    env: EnvUnowned,
    class: JClass,
) -> jstring {
    Java_ai_liquid_pipette_NativeLib_nativeCpuBackendDescriptor(env, class)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_liquid_pipette_NativeLib_nativeCpuBackendDescriptor(
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
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_liquid_pipette_LlamaEngine_nativeCreate(
    mut env: EnvUnowned,
    _class: JClass,
    path: JString,
    n_gpu_layers: jint,
    context_size: jint,
    n_ubatch: jint,
    load_mode: jint,
) -> jlong {
    init_native_logging();
    env.with_env(|env| -> jni::errors::Result<jlong> {
        let result = (|| -> Result<jlong, PipetteError> {
            let path = java_string(env, path)?;
            let engine = LlamaEngine::create(&loader::LoadOptions {
                model_path: path,
                n_gpu_layers: load_count("n_gpu_layers", n_gpu_layers)?,
                context_size: load_count("context_size", context_size)?,
                n_ubatch: load_count("n_ubatch", n_ubatch)?,
                load_mode,
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
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_liquid_pipette_LlamaEngine_nativeDestroy(
    mut env: EnvUnowned,
    _class: JClass,
    engine_ptr: jlong,
) {
    env.with_env(|_env| -> jni::errors::Result<()> {
        if engine_ptr != 0 {
            // SAFETY: `engine_ptr` came from `Box::into_raw` in `nativeCreate`
            // and is destroyed exactly once (the actor swaps its state to empty
            // before destroying).
            drop(unsafe { Box::from_raw(engine_ptr as *mut LlamaEngine) });
        }
        Ok(())
    })
    .resolve::<ThrowRuntimeExAndDefault>();
}

/// Initialize native crash capture for the `:benchmark` process. Idempotent on
/// the native side.
#[unsafe(no_mangle)]
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
        // Empty means the JNI read failed or config is missing; the native side
        // would fail quietly. Separate checks so every empty field is reported.
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
        // The bridge is Android-only; keeps the host build free of unused warnings.
        #[cfg(not(target_os = "android"))]
        let _ = (&c_dsn, &c_env, &c_release, &c_db, &c_envelope);
        Ok(())
    })
    .resolve::<ThrowRuntimeExAndDefault>();
}

/// Run a benchmark against the model held by the boxed engine `engine_ptr`.
#[unsafe(no_mangle)]
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
                load_count("n_gpu_layers", n_gpu_layers)?,
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

/// See [`engine::run_fresh`].
#[unsafe(no_mangle)]
pub extern "system" fn Java_ai_liquid_pipette_LlamaEngine_nativeRunFresh(
    mut env: EnvUnowned,
    _class: JClass,
    benchmark_json: JString,
    model_path: JString,
    n_gpu_layers: jint,
    context_size: jint,
    n_ubatch: jint,
    load_mode: jint,
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
                    n_gpu_layers: load_count("n_gpu_layers", n_gpu_layers)?,
                    context_size: load_count("context_size", context_size)?,
                    n_ubatch: load_count("n_ubatch", n_ubatch)?,
                    load_mode,
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

/// A JNI `int` load option as `u32`; a negative one is refused rather than wrapped.
fn load_count(name: &str, value: jint) -> Result<u32, PipetteError> {
    u32::try_from(value).map_err(|_| PipetteError::ModelLoad {
        msg: format!("{name} must be non-negative, got {value}"),
    })
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
