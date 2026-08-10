//! In-process llama.cpp wrapper for Android.
//!
//! Unlike ee-cli which spawns `llama-bench`/`llama-server` as subprocesses,
//! Mobile clients require in-process execution. This module wraps the llama.cpp
//! C API that is compiled into the Android shared library via build.rs.

use std::{
    sync::mpsc::{self, RecvTimeoutError},
    thread::{self, JoinHandle},
    time::Duration,
};

#[cfg(target_os = "android")]
use pipette_doomloop::DoomloopPipeline;
use pipette_plan_types::result::BenchmarkEvalCompletionStopReason;

use crate::error::PipetteError;
use crate::ModelHandle;

// ---------------------------------------------------------------------------
// C API bindings (linked via build.rs on mobile targets)
// ---------------------------------------------------------------------------

// On non-mobile targets these symbols won't be available, so we gate the extern
// block behind cfg and provide stub implementations for cargo check on the host.

#[cfg(target_os = "android")]
extern "C" {
    fn ee_llama_model_load(
        path: *const std::ffi::c_char,
        n_gpu_layers: i32,
        n_ctx: i32,
        n_ubatch: i32,
    ) -> *mut std::ffi::c_void;
    fn ee_llama_last_error(buf: *mut std::ffi::c_char, buf_size: i32) -> i32;
    fn ee_cpu_backend_descriptor(buf: *mut std::ffi::c_char, buf_size: i32) -> i32;
    fn ee_default_threads() -> i32;
    fn ee_llama_n_threads(model: *mut std::ffi::c_void) -> i32;
    fn ee_llama_model_free(model: *mut std::ffi::c_void);
    fn ee_llama_tokenize_text(
        model: *mut std::ffi::c_void,
        text: *const std::ffi::c_char,
        out_tokens: *mut i32,
        max_tokens: i32,
        add_special: bool,
    ) -> i32;
    fn ee_llama_decode_batch(
        model: *mut std::ffi::c_void,
        tokens: *const i32,
        n_tokens: i32,
    ) -> i32;
    fn ee_llama_sample_greedy(model: *mut std::ffi::c_void) -> i32;
    fn ee_llama_token_to_str(
        model: *mut std::ffi::c_void,
        token: i32,
        buf: *mut u8,
        buf_size: i32,
    ) -> i32;
    fn ee_llama_apply_chat_template(
        model: *mut std::ffi::c_void,
        messages_json: *const std::ffi::c_char,
        buf: *mut u8,
        buf_size: i32,
    ) -> i32;
    fn ee_llama_kv_cache_clear_ctx(model: *mut std::ffi::c_void);
    fn ee_llama_token_eos_id(model: *mut std::ffi::c_void) -> i32;
    fn ee_llama_sampler_reset(model: *mut std::ffi::c_void);
    fn ee_mtmd_init(
        model: *mut std::ffi::c_void,
        mmproj_path: *const std::ffi::c_char,
        use_gpu: bool,
    ) -> *mut std::ffi::c_void;
    fn ee_mtmd_free(mtmd_ctx: *mut std::ffi::c_void);
    fn ee_mtmd_alloc_gray_chunks(
        mtmd_ctx: *mut std::ffi::c_void,
        text_with_marker: *const std::ffi::c_char,
        nx: u32,
        ny: u32,
        out_n_tokens: *mut usize,
    ) -> *mut std::ffi::c_void;
    fn ee_mtmd_free_chunks(chunks: *mut std::ffi::c_void);
    fn ee_mtmd_eval_chunks(
        mtmd_ctx: *mut std::ffi::c_void,
        model: *mut std::ffi::c_void,
        chunks: *mut std::ffi::c_void,
    ) -> i32;
}

/// Opaque handle to an mtmd (multimodal) context, scoped to a specific
/// llama model + mmproj pairing. Must be freed via `mtmd_free` before the
/// underlying model is unloaded.
///
/// `ptr` is only read by mobile-target code; `allow(dead_code)` silences the
/// host-target lint.
#[derive(Debug)]
#[allow(dead_code)]
pub struct MtmdHandle {
    pub ptr: u64,
}

// ---------------------------------------------------------------------------
// Error classification
// ---------------------------------------------------------------------------

/// Translate an `llama_decode` return code into a Rust-side error message
/// that names the failure mode instead of just echoing the int. Code 1 in
/// particular ("could not find a KV slot for the batch") used to surface to
/// the iOS UI as a bare "decode failed with code 1" which told nobody
/// anything; now it reads like a sentence. New codes can be added as we
/// encounter them in the wild.
#[cfg(target_os = "android")]
fn decode_error_message(rc: i32) -> String {
    match rc {
        1 => "decode failed: KV cache is full. The benchmark's prompt+decode \
              exceeds the allocated context size. (llama_decode rc=1)"
            .to_string(),
        -1 => "decode failed: invalid arguments to llama_decode (rc=-1)".to_string(),
        // GPU/Metal compute failure — leaves the context in a poisoned state,
        // every subsequent decode on the same handle returns -3.
        -3 => "decode failed: compute graph error, likely a Metal/GPU \
               failure (rc=-3). The model handle is unusable; the runner \
               will reload before the next cell."
            .to_string(),
        _ => format!("decode failed with code {rc}"),
    }
}

/// Returns true if `msg` looks like an out-of-memory error from the shim's
/// `g_last_load_error` capture. The fragment list mirrors what
/// `llama_shim.cpp::edge_log_callback` decides to record — see UNSUPPORTED_ARCHS
/// and the strstr() filter there. Pattern matching on text is fragile in
/// principle but stable in practice: these strings come from llama.cpp's own
/// log statements and Apple's Metal compiler, both of which change rarely.
#[cfg(target_os = "android")]
fn is_out_of_memory_error(msg: &str) -> bool {
    const OOM_FRAGMENTS: &[&str] = &[
        "posix_memalign failed",
        "failed to allocate buffer",
        "out of memory",
        "MTLCompiler", // Metal compiler errors during load are
        // typically allocation-driven on iOS
        "Compiler failed to build request",
    ];
    OOM_FRAGMENTS.iter().any(|frag| msg.contains(frag))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn load_model(
    path: &str,
    n_gpu_layers: u32,
    context_size: u32,
    // Prefill micro-batch (llama.cpp `n_ubatch`); 0 lets the shim apply
    // llama.cpp's default (512). The shared `ee_llama_model_load` shim accepts
    // it, so it's threaded straight through.
    n_ubatch: u32,
) -> Result<ModelHandle, PipetteError> {
    #[cfg(target_os = "android")]
    {
        use std::ffi::CString;
        let c_path =
            CString::new(path).map_err(|e| PipetteError::ModelLoad { msg: e.to_string() })?;
        let ptr = unsafe {
            ee_llama_model_load(
                c_path.as_ptr(),
                n_gpu_layers as i32,
                context_size as i32,
                n_ubatch as i32,
            )
        };
        if ptr.is_null() {
            let detail = unsafe {
                let mut buf = [0u8; 2048];
                let n = ee_llama_last_error(
                    buf.as_mut_ptr() as *mut std::ffi::c_char,
                    buf.len() as i32,
                );
                if n > 0 {
                    String::from_utf8_lossy(&buf[..n as usize]).into_owned()
                } else {
                    String::new()
                }
            };
            let msg = if !detail.is_empty() {
                detail
            } else {
                format!("failed to load model from {path}")
            };
            // The shim's log-capture (`g_last_load_error` in llama_shim.cpp)
            // includes a known set of allocator/Metal error fragments — pattern-
            // match against them to surface OOM as a distinct error variant so
            // the Swift runner can show a useful message and skip remaining
            // cells in this model group instead of just propagating the raw text.
            return Err(if is_out_of_memory_error(&msg) {
                PipetteError::OutOfMemory { msg }
            } else {
                PipetteError::ModelLoad { msg }
            });
        }
        Ok(ModelHandle { ptr: ptr as u64 })
    }

    #[cfg(not(target_os = "android"))]
    {
        let _ = (path, n_gpu_layers, context_size, n_ubatch);
        Err(PipetteError::ModelLoad {
            msg: "llama.cpp only available on iOS/Android targets".to_string(),
        })
    }
}

/// The active CPU-backend feature descriptor — a stable, sorted, lowercased
/// list of the enabled CPU features of the registered ggml CPU backend (e.g.
/// `"dotprod,fp16_va,matmul_int8,neon"`). It reflects the runtime-selected DL
/// variant on this device and so implies the effective CPU compile flags; the
/// management server stores it as `runtime_cpu_variant` so result analysis can
/// tell when the selected variant changed.
///
/// Returns `None` until a model has been loaded (the backend registers lazily)
/// or when the shim reports no CPU feature list.
pub fn cpu_backend_descriptor() -> Option<String> {
    #[cfg(target_os = "android")]
    {
        let mut buf = [0u8; 512];
        let n = unsafe {
            ee_cpu_backend_descriptor(buf.as_mut_ptr() as *mut std::ffi::c_char, buf.len() as i32)
        };
        if n > 0 {
            Some(String::from_utf8_lossy(&buf[..n as usize]).into_owned())
        } else {
            None
        }
    }

    #[cfg(not(target_os = "android"))]
    {
        None
    }
}

/// The compute-thread count the runtime would pick for a model loaded *now* under
/// the current cpuset — usable performant cores, or all permitted cores when no
/// performant core is permitted by the cpuset (fallback). A fresh recompute;
/// prefer [`n_threads`] for a loaded model. `None` on non-Android hosts.
pub fn default_thread_count() -> Option<i32> {
    #[cfg(target_os = "android")]
    {
        let n = unsafe { ee_default_threads() };
        if n > 0 {
            Some(n)
        } else {
            None
        }
    }

    #[cfg(not(target_os = "android"))]
    {
        None
    }
}

/// The compute-thread count a loaded model was configured with (fixed at load).
/// Authoritative for reproducibility — the cpuset can change between load and a
/// later run, so this stays matched to the pinned pool the run used, unlike
/// [`default_thread_count`]. `None` on non-Android hosts / null count.
pub fn n_threads(model: &ModelHandle) -> Option<i32> {
    #[cfg(target_os = "android")]
    {
        let n = unsafe { ee_llama_n_threads(model.ptr as *mut std::ffi::c_void) };
        if n > 0 {
            Some(n)
        } else {
            None
        }
    }

    #[cfg(not(target_os = "android"))]
    {
        let _ = model;
        None
    }
}

/// Tokenizes `text` against the model's vocabulary.
///
/// `add_special` controls whether the tokenizer prepends special tokens
/// (BOS / chat-format markers) before the text. Pass `true` when the text is a
/// full prompt that the model will consume as-is (e.g. `chat_completion` after
/// applying a chat template, or a dummy prefill prompt). Pass `false` when
/// counting text-content tokens for benchmarks, to match the CLI's HTTP
/// `/tokenize` default and keep token-count parameters comparable across
/// platforms.
pub fn tokenize(
    model: &ModelHandle,
    text: &str,
    add_special: bool,
) -> Result<Vec<i32>, PipetteError> {
    #[cfg(target_os = "android")]
    {
        use std::ffi::CString;
        let c_text =
            CString::new(text).map_err(|e| PipetteError::Tokenize { msg: e.to_string() })?;
        let ptr = model.ptr as *mut std::ffi::c_void;
        let mut tokens = vec![0i32; text.len() + 32];
        let n = unsafe {
            ee_llama_tokenize_text(
                ptr,
                c_text.as_ptr(),
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                add_special,
            )
        };
        if n < 0 {
            return Err(PipetteError::Tokenize {
                msg: "tokenization failed".to_string(),
            });
        }
        tokens.truncate(n as usize);
        Ok(tokens)
    }

    #[cfg(not(target_os = "android"))]
    {
        let _ = (model, text, add_special);
        Err(PipetteError::Tokenize {
            msg: "llama.cpp only available on iOS/Android targets".to_string(),
        })
    }
}

pub fn prefill(model: &ModelHandle, tokens: &[i32]) -> Result<(), PipetteError> {
    #[cfg(target_os = "android")]
    {
        let ptr = model.ptr as *mut std::ffi::c_void;
        let rc = unsafe { ee_llama_decode_batch(ptr, tokens.as_ptr(), tokens.len() as i32) };
        if rc != 0 {
            return Err(PipetteError::Inference {
                msg: format!("prefill failed with code {rc}"),
            });
        }
        Ok(())
    }

    #[cfg(not(target_os = "android"))]
    {
        let _ = (model, tokens);
        Err(PipetteError::Inference {
            msg: "llama.cpp only available on iOS/Android targets".to_string(),
        })
    }
}

/// A finished generation plus the stop metadata captured while producing it,
/// in the shape the eval submission records
/// ([`pipette_plan_types::result::BenchmarkEvalCompletion`]).
///
/// The generating loop is the only place that can tell an EOS from a token cap
/// from a doom-loop abort, so it carries the answer out rather than returning
/// bare text a caller would have to guess about.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    pub text: String,
    pub stop_reason: BenchmarkEvalCompletionStopReason,
    /// The raw why behind a non-clean stop. `None` for a clean `eos`/`truncated`.
    pub stop_detail: Option<String>,
    /// Tokens generated, counted by the decode loop itself. Excludes the EOS
    /// token, which is a stop signal rather than output.
    pub completion_tokens: u64,
}

/// Which branch ended a decode loop, before it is mapped onto the wire enum.
// Constructed only by the android decode loop; the host build sees the tests as
// its only user, same as `decode_greedy` below.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RawStop {
    /// A stop token was sampled.
    StopToken,
    /// The doom-loop detector fired, carrying the detector's name.
    DoomLoop(String),
    /// The loop ran to `max_tokens` without stopping on its own.
    CapReached,
}

/// Map a decode loop's exit onto the wire enum.
///
/// A doom-loop abort is the only source of `doom_loop`, and a stop token is the
/// only source of `eos`, so neither is ever inferred from the token count. The
/// count travels alongside rather than deciding anything: it is what lets a
/// reader separate an `eos` under the cap from one that landed exactly on it.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
pub(crate) fn classify_stop(
    stop: RawStop,
    completion_tokens: u64,
) -> (BenchmarkEvalCompletionStopReason, Option<String>) {
    match stop {
        RawStop::StopToken => (BenchmarkEvalCompletionStopReason::Eos, None),
        RawStop::DoomLoop(detector) => (
            BenchmarkEvalCompletionStopReason::DoomLoop,
            Some(format!(
                "doom-loop detector {detector} fired after {completion_tokens} tokens"
            )),
        ),
        RawStop::CapReached => (BenchmarkEvalCompletionStopReason::Truncated, None),
    }
}

// Used only on the android target (via `chat_completion`); on the host build
// the call sites are behind `#[cfg(target_os = "android")]`, so it reads as dead.
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
pub fn decode_greedy(
    model: &ModelHandle,
    max_tokens: u32,
    stop_tokens: &[i32],
) -> Result<Generation, PipetteError> {
    #[cfg(target_os = "android")]
    {
        let ptr = model.ptr as *mut std::ffi::c_void;
        let pipeline = DoomloopPipeline::default();
        let mut output = String::new();
        let mut buf = [0u8; 256];
        let mut completion_tokens: u64 = 0;
        // Only reached unchanged when the loop runs its full budget; every
        // early exit below sets it to the branch that fired.
        let mut stop = RawStop::CapReached;

        for _ in 0..max_tokens {
            let token = unsafe { ee_llama_sample_greedy(ptr) };
            if stop_tokens.contains(&token) {
                stop = RawStop::StopToken;
                break;
            }
            // Counted before the text is appended: a token the detokenizer
            // can't render is still a token the model produced, and the count
            // is what separates a capped generation from an early stop.
            completion_tokens += 1;
            let n =
                unsafe { ee_llama_token_to_str(ptr, token, buf.as_mut_ptr(), buf.len() as i32) };
            if n > 0 && (n as usize) <= buf.len() {
                if let Ok(s) = std::str::from_utf8(&buf[..n as usize]) {
                    output.push_str(s);
                }
            }
            // Feed token back for autoregressive generation
            let rc = unsafe { ee_llama_decode_batch(ptr, &token as *const i32, 1) };
            if rc != 0 {
                return Err(PipetteError::Inference {
                    msg: decode_error_message(rc),
                });
            }
            // Check for doom loop (repetitive generation)
            if output.len() > 256 {
                if let Some(name) = pipeline.check(&output) {
                    log::warn!(
                        "{}",
                        pipette_doomloop::format_trigger_log(name, output.len())
                    );
                    stop = RawStop::DoomLoop(name.to_string());
                    break;
                }
            }
        }
        let (stop_reason, stop_detail) = classify_stop(stop, completion_tokens);
        Ok(Generation {
            text: output,
            stop_reason,
            stop_detail,
            completion_tokens,
        })
    }

    #[cfg(not(target_os = "android"))]
    {
        let _ = (model, max_tokens, stop_tokens);
        Err(PipetteError::Inference {
            msg: "llama.cpp only available on iOS/Android targets".to_string(),
        })
    }
}

pub fn chat_completion(
    model: &ModelHandle,
    messages_json: &str,
    max_tokens: u32,
    mcq_choices_json: Option<&str>,
) -> Result<Generation, PipetteError> {
    #[cfg(target_os = "android")]
    {
        use std::ffi::CString;
        let ptr = model.ptr as *mut std::ffi::c_void;

        // Apply chat template to get the prompt string
        let c_messages = CString::new(messages_json)
            .map_err(|e| PipetteError::Inference { msg: e.to_string() })?;
        let mut prompt_buf = vec![0u8; 32768];
        let prompt_len = unsafe {
            ee_llama_apply_chat_template(
                ptr,
                c_messages.as_ptr(),
                prompt_buf.as_mut_ptr(),
                prompt_buf.len() as i32,
            )
        };
        if prompt_len < 0 {
            return Err(PipetteError::Inference {
                msg: "failed to apply chat template".to_string(),
            });
        }
        if prompt_len as usize >= prompt_buf.len() {
            return Err(PipetteError::Inference {
                msg: format!(
                    "chat template output ({prompt_len} bytes) exceeds buffer size ({})",
                    prompt_buf.len()
                ),
            });
        }
        let prompt = std::str::from_utf8(&prompt_buf[..prompt_len as usize])
            .map_err(|e| PipetteError::Inference { msg: e.to_string() })?;

        // Tokenize the prompt
        let tokens = tokenize(model, prompt, true)?;

        // Prefill
        prefill(model, &tokens)?;

        // Decode
        if let Some(choices_json) = mcq_choices_json {
            // MCQ constrained decoding: only allow choice tokens, max_tokens=1
            let choices: Vec<String> = serde_json::from_str(choices_json)
                .map_err(|e| PipetteError::Json { msg: e.to_string() })?;
            // For MCQ we sample a single token and match to closest choice
            let token = unsafe { ee_llama_sample_greedy(ptr) };
            let mut buf = [0u8; 256];
            let n =
                unsafe { ee_llama_token_to_str(ptr, token, buf.as_mut_ptr(), buf.len() as i32) };
            let generated = if n > 0 && (n as usize) <= buf.len() {
                std::str::from_utf8(&buf[..n as usize])
                    .unwrap_or("")
                    .trim()
                    .to_string()
            } else {
                String::new()
            };
            // Return the generated token if it matches a choice, otherwise
            // return the first choice (constrained decoding fallback)
            let matched = choices.iter().any(|c| c == &generated);
            let text = if matched {
                generated
            } else {
                choices.into_iter().next().unwrap_or_default()
            };
            // `unknown`, which the wire enum names the MCQ arm as a case of.
            // There is no stop signal to classify: exactly one token is sampled
            // and the loop that would observe an EOS or a cap never runs. The
            // detail records which of the two MCQ paths produced the text,
            // because Android's arm is unconstrained sampling with a fallback
            // rather than the CLI's grammar, so a fallback answer is the model
            // failing to name a choice rather than choosing the first one.
            let stop_detail = if matched {
                "mcq arm (single greedy token, matched a choice)"
            } else {
                "mcq arm (single greedy token, no choice matched; fell back to the first)"
            };
            Ok(Generation {
                text,
                stop_reason: BenchmarkEvalCompletionStopReason::Unknown,
                stop_detail: Some(stop_detail.to_string()),
                completion_tokens: 1,
            })
        } else {
            // Free-form greedy decoding — stop on EOS token
            let eos = unsafe { ee_llama_token_eos_id(ptr) };
            decode_greedy(model, max_tokens, &[eos])
        }
    }

    #[cfg(not(target_os = "android"))]
    {
        let _ = (model, messages_json, max_tokens, mcq_choices_json);
        Err(PipetteError::Inference {
            msg: "llama.cpp only available on iOS/Android targets".to_string(),
        })
    }
}

/// Clear the KV cache so the next inference starts with a fresh context.
/// This must be called between independent eval samples to avoid context
/// pollution from previous prompts/completions.
pub fn reset_context(model: &ModelHandle) -> Result<(), PipetteError> {
    #[cfg(target_os = "android")]
    {
        let ptr = model.ptr as *mut std::ffi::c_void;
        unsafe { ee_llama_kv_cache_clear_ctx(ptr) };
        Ok(())
    }

    #[cfg(not(target_os = "android"))]
    {
        let _ = model;
        Ok(())
    }
}

pub fn unload_model(model: &ModelHandle) -> Result<(), PipetteError> {
    #[cfg(target_os = "android")]
    {
        let ptr = model.ptr as *mut std::ffi::c_void;
        if !ptr.is_null() {
            unsafe { ee_llama_model_free(ptr) };
        }
        Ok(())
    }

    #[cfg(not(target_os = "android"))]
    {
        let _ = model;
        Ok(())
    }
}

/// Reset the sampler state. Called between measurement runs so accumulated
/// accept history does not leak across runs.
pub fn sampler_reset(model: &ModelHandle) {
    #[cfg(target_os = "android")]
    {
        let ptr = model.ptr as *mut std::ffi::c_void;
        unsafe { ee_llama_sampler_reset(ptr) };
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = model;
    }
}

/// Decode exactly `n` tokens greedily starting from the current context
/// position, ignoring EoG tokens. Used by the VL throughput benchmark to
/// measure a fixed-count decode phase — unlike `decode_greedy`, this does
/// NOT stop early on stop tokens or doom-loop detection.
///
/// Returns Ok(()) on success, Err on decode failure mid-loop.
#[allow(unused_variables)]
pub fn decode_n_greedy_ignore_eog(model: &ModelHandle, n: u32) -> Result<(), PipetteError> {
    #[cfg(target_os = "android")]
    {
        let ptr = model.ptr as *mut std::ffi::c_void;
        for _ in 0..n {
            let token = unsafe { ee_llama_sample_greedy(ptr) };
            if token < 0 {
                return Err(PipetteError::Inference {
                    msg: "sampler returned invalid token".to_string(),
                });
            }
            let rc = unsafe { ee_llama_decode_batch(ptr, &token as *const i32, 1) };
            if rc != 0 {
                return Err(PipetteError::Inference {
                    msg: decode_error_message(rc),
                });
            }
        }
        Ok(())
    }
    #[cfg(not(target_os = "android"))]
    {
        Err(PipetteError::Inference {
            msg: "llama.cpp only available on iOS/Android targets".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// mtmd (multimodal) wrappers
// ---------------------------------------------------------------------------

#[allow(unused_variables)]
pub fn mtmd_init(
    model: &ModelHandle,
    mmproj_path: &str,
    use_gpu: bool,
) -> Result<MtmdHandle, PipetteError> {
    #[cfg(target_os = "android")]
    {
        use std::ffi::CString;
        let c_path = CString::new(mmproj_path)
            .map_err(|e| PipetteError::ModelLoad { msg: e.to_string() })?;
        let model_ptr = model.ptr as *mut std::ffi::c_void;
        let ptr = unsafe { ee_mtmd_init(model_ptr, c_path.as_ptr(), use_gpu) };
        if ptr.is_null() {
            return Err(PipetteError::ModelLoad {
                msg: format!("failed to load mmproj from {mmproj_path}"),
            });
        }
        Ok(MtmdHandle { ptr: ptr as u64 })
    }
    #[cfg(not(target_os = "android"))]
    {
        Err(PipetteError::ModelLoad {
            msg: "mtmd only available on iOS/Android targets".to_string(),
        })
    }
}

#[allow(unused_variables)]
pub fn mtmd_free(mtmd: MtmdHandle) {
    #[cfg(target_os = "android")]
    {
        let ptr = mtmd.ptr as *mut std::ffi::c_void;
        if !ptr.is_null() {
            unsafe { ee_mtmd_free(ptr) };
        }
    }
}

/// Allocate and tokenize a synthetic gray RGB image + the given text prompt.
/// The text MUST contain the mtmd media marker `<__media__>` exactly once.
///
/// Returns `(chunks_ptr, n_tokens)` where `n_tokens` is the total token count
/// (text + image) reported by `mtmd_helper_get_n_tokens`. The chunks pointer
/// must be freed via `mtmd_free_chunks`.
#[allow(unused_variables)]
pub fn mtmd_alloc_gray_chunks(
    mtmd: &MtmdHandle,
    text_with_marker: &str,
    width: u32,
    height: u32,
) -> Result<(u64, usize), PipetteError> {
    #[cfg(target_os = "android")]
    {
        use std::ffi::CString;
        let c_text = CString::new(text_with_marker)
            .map_err(|e| PipetteError::Inference { msg: e.to_string() })?;
        let mut n_tokens: usize = 0;
        let ptr = unsafe {
            ee_mtmd_alloc_gray_chunks(
                mtmd.ptr as *mut std::ffi::c_void,
                c_text.as_ptr(),
                width,
                height,
                &mut n_tokens as *mut usize,
            )
        };
        if ptr.is_null() {
            return Err(PipetteError::Inference {
                msg: "mtmd_tokenize failed (check that prompt contains <__media__> marker and fits in context)".to_string(),
            });
        }
        Ok((ptr as u64, n_tokens))
    }
    #[cfg(not(target_os = "android"))]
    {
        Err(PipetteError::Inference {
            msg: "mtmd only available on iOS/Android targets".to_string(),
        })
    }
}

#[allow(unused_variables)]
pub fn mtmd_free_chunks(chunks_ptr: u64) {
    #[cfg(target_os = "android")]
    {
        let ptr = chunks_ptr as *mut std::ffi::c_void;
        if !ptr.is_null() {
            unsafe { ee_mtmd_free_chunks(ptr) };
        }
    }
}

/// Evaluate the given chunks (text + image), starting from KV position 0.
/// The caller MUST clear the KV cache beforehand via `reset_context`.
///
/// On return, the context is positioned at the end of the prompt and the
/// last-token logits are available for greedy sampling.
#[allow(unused_variables)]
pub fn mtmd_eval_chunks(
    mtmd: &MtmdHandle,
    model: &ModelHandle,
    chunks_ptr: u64,
) -> Result<(), PipetteError> {
    #[cfg(target_os = "android")]
    {
        let rc = unsafe {
            ee_mtmd_eval_chunks(
                mtmd.ptr as *mut std::ffi::c_void,
                model.ptr as *mut std::ffi::c_void,
                chunks_ptr as *mut std::ffi::c_void,
            )
        };
        if rc != 0 {
            return Err(PipetteError::Inference {
                msg: format!("mtmd_helper_eval_chunks failed with code {rc}"),
            });
        }
        Ok(())
    }
    #[cfg(not(target_os = "android"))]
    {
        Err(PipetteError::Inference {
            msg: "mtmd only available on iOS/Android targets".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Memory measurement
// ---------------------------------------------------------------------------

/// Android currently builds the CPU backend, so use process RSS as the closest
/// equivalent peak-memory sample for benchmark result metadata.
#[cfg(target_os = "android")]
pub fn metal_allocated_size_bytes() -> u64 {
    current_rss_bytes().unwrap_or(0)
}

#[cfg(not(target_os = "android"))]
pub fn metal_allocated_size_bytes() -> u64 {
    0
}

#[cfg(target_os = "android")]
fn current_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmRSS:")?;
        let kb = value.split_whitespace().next()?.parse::<u64>().ok()?;
        Some(kb * 1024)
    })
}

#[must_use = "MetalAllocationPoller::stop_and_join_with_sample must be called to retrieve the peak"]
pub struct MetalAllocationPoller {
    baseline: u64,
    stop: mpsc::Sender<()>,
    handle: JoinHandle<u64>,
}

impl MetalAllocationPoller {
    pub(crate) fn stop_and_join_with_sample(self, extra_sample: u64) -> Result<u64, PipetteError> {
        let final_sample = metal_allocated_size_bytes().max(extra_sample);
        let _ = self.stop.send(());
        let peak = self.handle.join().map_err(|_| PipetteError::Benchmark {
            msg: "Metal allocation poller thread panicked".to_string(),
        })?;
        Ok(memory_peak_delta(self.baseline, peak, final_sample))
    }
}

pub fn spawn_metal_allocation_poller() -> MetalAllocationPoller {
    let baseline = metal_allocated_size_bytes();
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let handle = thread::spawn(move || -> u64 {
        let mut peak = baseline;
        loop {
            peak = peak.max(metal_allocated_size_bytes());
            match stop_rx.recv_timeout(Duration::from_millis(20)) {
                Ok(_) | Err(RecvTimeoutError::Disconnected) => break peak,
                Err(RecvTimeoutError::Timeout) => continue,
            }
        }
    });
    MetalAllocationPoller {
        baseline,
        stop: stop_tx,
        handle,
    }
}

fn memory_peak_delta(baseline: u64, peak: u64, final_sample: u64) -> u64 {
    peak.max(final_sample).saturating_sub(baseline)
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::result::BenchmarkEvalCompletionStopReason;

    use super::{classify_stop, memory_peak_delta, RawStop};

    #[test]
    fn memory_peak_delta_subtracts_baseline() {
        assert_eq!(memory_peak_delta(100, 450, 300), 350);
    }

    #[test]
    fn memory_peak_delta_uses_final_sample() {
        assert_eq!(memory_peak_delta(100, 300, 450), 350);
    }

    #[test]
    fn memory_peak_delta_saturates_when_counter_drops_below_baseline() {
        assert_eq!(memory_peak_delta(450, 300, 200), 0);
    }

    /// A stop token is the only source of `eos`, and a clean stop carries no
    /// detail: there is nothing to explain when the model ended its own turn.
    #[test]
    fn a_stop_token_classifies_as_eos() {
        assert_eq!(
            classify_stop(RawStop::StopToken, 12),
            (BenchmarkEvalCompletionStopReason::Eos, None)
        );
    }

    /// Running the budget out is the cap, which is the case the whole field
    /// exists to separate from `eos`: same text, different meaning.
    #[test]
    fn exhausting_the_budget_classifies_as_truncated() {
        assert_eq!(
            classify_stop(RawStop::CapReached, 256),
            (BenchmarkEvalCompletionStopReason::Truncated, None)
        );
    }

    /// `doom_loop` is client-only and unrecoverable from the stored text, so
    /// the detail has to name the detector that fired.
    #[test]
    fn a_doom_loop_abort_names_the_detector_that_fired() {
        let (reason, detail) = classify_stop(RawStop::DoomLoop("ExactRepeat".to_string()), 4096);
        assert_eq!(reason, BenchmarkEvalCompletionStopReason::DoomLoop);
        // Empty rather than unwrapped: an absent detail then fails the two
        // assertions below, which is the thing actually under test.
        let detail = detail.unwrap_or_default();
        assert!(detail.contains("ExactRepeat"), "detail was {detail}");
        assert!(detail.contains("4096"), "detail was {detail}");
    }

    /// The count travels alongside the classification and never decides it. A
    /// generation that emitted EOS on the very last allowed token is still an
    /// `eos`, not a `truncated`, even though the counts are identical.
    #[test]
    fn the_token_count_does_not_override_the_stop_signal() {
        assert_eq!(
            classify_stop(RawStop::StopToken, 256).0,
            BenchmarkEvalCompletionStopReason::Eos
        );
        assert_eq!(
            classify_stop(RawStop::CapReached, 256).0,
            BenchmarkEvalCompletionStopReason::Truncated
        );
    }
}
