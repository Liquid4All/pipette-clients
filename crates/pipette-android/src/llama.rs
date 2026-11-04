//! In-process llama.cpp for Android, over the C API of the shim in
//! `android/native/llama_shim.cpp`. Desktop clients spawn llama.cpp
//! subprocesses instead.

use std::{
    sync::mpsc::{self, RecvTimeoutError},
    thread::{self, JoinHandle},
    time::Duration,
};

use pipette_plan_types::BenchmarkEvalCompletionStopReason;

use crate::ModelHandle;
use crate::error::PipetteError;

// The shim exists only in Android builds; the host gets stubs so it can check.

#[cfg(target_os = "android")]
unsafe extern "C" {
    fn ee_llama_model_load(
        path: *const std::ffi::c_char,
        n_gpu_layers: i32,
        n_ctx: i32,
        n_ubatch: i32,
        load_mode: i32,
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
        out_image_tokens: *mut usize,
    ) -> *mut std::ffi::c_void;
    fn ee_mtmd_free_chunks(chunks: *mut std::ffi::c_void);
    fn ee_mtmd_eval_chunks(
        mtmd_ctx: *mut std::ffi::c_void,
        model: *mut std::ffi::c_void,
        chunks: *mut std::ffi::c_void,
    ) -> i32;
}

/// An mtmd (multimodal) context for one model + mmproj pairing. Must be freed
/// via `mtmd_free` before the model is unloaded.
#[derive(Debug)]
#[cfg_attr(
    not(target_os = "android"),
    allow(dead_code, reason = "`ptr` is read only by mobile-target code")
)]
pub struct MtmdHandle {
    pub ptr: u64,
}

/// Name the failure behind a `llama_decode` return code instead of echoing it.
#[cfg(target_os = "android")]
fn decode_error_message(rc: i32) -> String {
    match rc {
        1 => "decode failed: KV cache is full. The benchmark's prompt+decode \
              exceeds the allocated context size. (llama_decode rc=1)"
            .to_string(),
        -1 => "decode failed: invalid arguments to llama_decode (rc=-1)".to_string(),
        // Poisons the context: every later decode on the handle returns -3.
        -3 => "decode failed: compute graph error, likely a Metal/GPU \
               failure (rc=-3). The model handle is unusable; the runner \
               will reload before the next cell."
            .to_string(),
        _ => format!("decode failed with code {rc}"),
    }
}

/// Whether `msg` from the shim's `g_last_load_error` capture looks like OOM.
/// The fragments target lines that `edge_log_callback` in `llama_shim.cpp`
/// records; they come from llama.cpp and Metal log text, which rarely changes.
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

pub fn load_model(
    path: &str,
    n_gpu_layers: u32,
    context_size: u32,
    // 0 lets the shim apply its default (512).
    n_ubatch: u32,
    // llama's `llama_load_mode` discriminant, kept raw because it crosses two FFI
    // boundaries unchanged (JNI from Kotlin, then C into the shim).
    load_mode: i32,
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
                load_mode,
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
        let _ = (path, n_gpu_layers, context_size, n_ubatch, load_mode);
        Err(PipetteError::ModelLoad {
            msg: "llama.cpp only available on iOS/Android targets".to_string(),
        })
    }
}

/// The sorted, lowercased CPU features of the runtime-selected ggml CPU variant
/// (e.g. `"dotprod,fp16_va,matmul_int8,neon"`), submitted as
/// `runtime_cpu_variant`. `None` until a model has loaded (the backend
/// registers lazily) or when the shim reports no feature list.
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

/// The thread count a model loaded *now* would get under the current cpuset.
/// Prefer [`n_threads`] for a loaded model.
pub fn default_thread_count() -> Option<i32> {
    #[cfg(target_os = "android")]
    {
        let n = unsafe { ee_default_threads() };
        if n > 0 { Some(n) } else { None }
    }

    #[cfg(not(target_os = "android"))]
    {
        None
    }
}

/// The thread count fixed when `model` was loaded. Authoritative: the cpuset can
/// change between load and run, which [`default_thread_count`] would miss.
pub fn n_threads(model: &ModelHandle) -> Option<i32> {
    #[cfg(target_os = "android")]
    {
        let n = unsafe { ee_llama_n_threads(model.ptr as *mut std::ffi::c_void) };
        if n > 0 { Some(n) } else { None }
    }

    #[cfg(not(target_os = "android"))]
    {
        let _ = model;
        None
    }
}

/// `add_special` prepends special tokens (BOS, chat markers): `true` for a full
/// prompt the model consumes as-is, `false` when counting content tokens, to
/// match the CLI's `/tokenize` default so token counts compare across platforms.
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

/// A finished generation plus its stop metadata, as
/// [`pipette_plan_types::BenchmarkEvalCompletion`] records it. Only the decode
/// loop can tell an EOS from a token cap, so it reports which.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    pub text: String,
    pub stop_reason: BenchmarkEvalCompletionStopReason,
    /// The raw why behind a non-clean stop. `None` for a clean `eos`/`truncated`.
    pub stop_detail: Option<String>,
    /// Excludes the EOS token, which is a stop signal rather than output.
    pub completion_tokens: u64,
}

/// Which branch ended a decode loop, before it is mapped onto the wire enum.
#[cfg_attr(
    not(target_os = "android"),
    allow(dead_code, reason = "constructed only by the android decode loop")
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RawStop {
    StopToken,
    CapReached,
}

/// `eos` comes only from a stop token, never from the token count.
impl From<RawStop> for BenchmarkEvalCompletionStopReason {
    fn from(stop: RawStop) -> Self {
        match stop {
            RawStop::StopToken => Self::Eos,
            RawStop::CapReached => Self::Truncated,
        }
    }
}

#[cfg_attr(
    not(target_os = "android"),
    allow(dead_code, reason = "called only on android, via `chat_completion`")
)]
pub fn decode_greedy(
    model: &ModelHandle,
    max_tokens: u32,
    stop_tokens: &[i32],
) -> Result<Generation, PipetteError> {
    #[cfg(target_os = "android")]
    {
        let ptr = model.ptr as *mut std::ffi::c_void;
        let mut output = String::new();
        let mut buf = [0u8; 256];
        let mut completion_tokens: u64 = 0;
        let mut stop = RawStop::CapReached;

        for _ in 0..max_tokens {
            let token = unsafe { ee_llama_sample_greedy(ptr) };
            if stop_tokens.contains(&token) {
                stop = RawStop::StopToken;
                break;
            }
            // Counted even if the detokenizer can't render it: the model still
            // produced it.
            completion_tokens += 1;
            let n =
                unsafe { ee_llama_token_to_str(ptr, token, buf.as_mut_ptr(), buf.len() as i32) };
            if n > 0 && (n as usize) <= buf.len() {
                if let Ok(s) = std::str::from_utf8(&buf[..n as usize]) {
                    output.push_str(s);
                }
            }
            let rc = unsafe { ee_llama_decode_batch(ptr, &token as *const i32, 1) };
            if rc != 0 {
                return Err(PipetteError::Inference {
                    msg: decode_error_message(rc),
                });
            }
        }
        Ok(Generation {
            text: output,
            stop_reason: stop.into(),
            stop_detail: None,
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

        let tokens = tokenize(model, prompt, true)?;

        prefill(model, &tokens)?;

        if let Some(choices_json) = mcq_choices_json {
            // MCQ: one unconstrained greedy token, which must equal a choice
            // exactly or the first choice is returned.
            let choices: Vec<String> = serde_json::from_str(choices_json)
                .map_err(|e| PipetteError::Json { msg: e.to_string() })?;
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
            let matched = choices.iter().any(|c| c == &generated);
            let text = if matched {
                generated
            } else {
                choices.into_iter().next().unwrap_or_default()
            };
            // `unknown`: one token has no stop signal to classify. The detail
            // says which path produced the text: with no grammar (unlike the
            // CLI), a fallback means the model named no choice, not that it
            // chose the first.
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

/// Clear the KV cache. Required between independent samples, or earlier
/// prompts leak into the next.
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

/// Reset the sampler so accept history does not leak across runs.
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

/// Decode exactly `n` greedy tokens, ignoring EoG, so benchmarks time a
/// fixed-count decode (unlike `decode_greedy`).
#[cfg_attr(
    not(target_os = "android"),
    expect(unused_variables, reason = "the body is android-only")
)]
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

#[cfg_attr(
    not(target_os = "android"),
    expect(unused_variables, reason = "the body is android-only")
)]
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

#[cfg_attr(
    not(target_os = "android"),
    expect(unused_variables, reason = "the body is android-only")
)]
pub fn mtmd_free(mtmd: MtmdHandle) {
    #[cfg(target_os = "android")]
    {
        let ptr = mtmd.ptr as *mut std::ffi::c_void;
        if !ptr.is_null() {
            unsafe { ee_mtmd_free(ptr) };
        }
    }
}

/// Tokenize a synthetic gray image plus `text_with_marker`, which must contain
/// `<__media__>` exactly once.
///
/// Returns `(chunks_ptr, total_tokens, image_tokens)`, both counts straight from
/// mtmd. Free the chunks via `mtmd_free_chunks`.
pub fn mtmd_alloc_gray_chunks(
    mtmd: &MtmdHandle,
    text_with_marker: &str,
    width: u32,
    height: u32,
) -> Result<(u64, usize, usize), PipetteError> {
    #[cfg(target_os = "android")]
    {
        use std::ffi::CString;
        let c_text = CString::new(text_with_marker)
            .map_err(|e| PipetteError::Inference { msg: e.to_string() })?;
        let mut n_tokens: usize = 0;
        let mut image_tokens: usize = 0;
        let ptr = unsafe {
            ee_mtmd_alloc_gray_chunks(
                mtmd.ptr as *mut std::ffi::c_void,
                c_text.as_ptr(),
                width,
                height,
                &mut n_tokens as *mut usize,
                &mut image_tokens as *mut usize,
            )
        };
        if ptr.is_null() {
            return Err(PipetteError::Inference {
                msg: "mtmd_tokenize failed (check that prompt contains <__media__> marker and fits in context)".to_string(),
            });
        }
        Ok((ptr as u64, n_tokens, image_tokens))
    }
    #[cfg(not(target_os = "android"))]
    {
        let _ = (mtmd, text_with_marker, width, height);
        Err(PipetteError::Inference {
            msg: "mtmd only available on iOS/Android targets".to_string(),
        })
    }
}

#[cfg_attr(
    not(target_os = "android"),
    expect(unused_variables, reason = "the body is android-only")
)]
pub fn mtmd_free_chunks(chunks_ptr: u64) {
    #[cfg(target_os = "android")]
    {
        let ptr = chunks_ptr as *mut std::ffi::c_void;
        if !ptr.is_null() {
            unsafe { ee_mtmd_free_chunks(ptr) };
        }
    }
}

/// Evaluate the chunks from KV position 0, so the caller must `reset_context`
/// first. Leaves the last-token logits ready for greedy sampling.
#[cfg_attr(
    not(target_os = "android"),
    expect(unused_variables, reason = "the body is android-only")
)]
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

/// Process RSS: Android builds the CPU backend, so there is no Metal allocation
/// to sample and RSS is the closest equivalent.
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
    use pipette_plan_types::BenchmarkEvalCompletionStopReason;

    use super::{RawStop, memory_peak_delta};

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

    #[test]
    fn a_stop_token_classifies_as_eos() {
        assert_eq!(
            BenchmarkEvalCompletionStopReason::from(RawStop::StopToken),
            BenchmarkEvalCompletionStopReason::Eos
        );
    }

    #[test]
    fn exhausting_the_budget_classifies_as_truncated() {
        assert_eq!(
            BenchmarkEvalCompletionStopReason::from(RawStop::CapReached),
            BenchmarkEvalCompletionStopReason::Truncated
        );
    }
}
