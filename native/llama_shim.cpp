#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__ANDROID__)
#include <fstream>
#include <sched.h>

// Implemented in crates/pipette-android/native_loader.cpp (Android-only).
// The CPU backend ships as separate, runtime-loaded `libggml-cpu-*.so` variants
// (GGML_CPU_ALL_VARIANTS + GGML_BACKEND_DL), so it is registered at startup by
// the loader, and the threadpool API — which lives in the variant `.so` — is
// reached through these wrappers rather than direct (unresolvable at link time)
// calls. See native_loader.cpp for the rationale.
extern "C" void pipette_native_load_backends(void);
extern "C" ggml_threadpool_t pipette_ggml_threadpool_new(struct ggml_threadpool_params * params);
extern "C" void pipette_ggml_threadpool_free(ggml_threadpool_t tp);
#endif

#include <nlohmann/json.hpp>

namespace {

struct EdgeLlamaHandle {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    llama_sampler * sampler = nullptr;
    uint32_t n_ctx_captured = 0;
    // The compute-thread count chosen at load under the then-current cpuset.
    // Captured so the benchmark result records the count the run actually used,
    // not a value recomputed later (the cpuset can change between load and run).
    int n_threads = 0;
    // Performance-core-pinned compute pool (Android only; nullptr elsewhere and
    // when topology detection fails — then llama.cpp uses its auto threadpool).
    ggml_threadpool_t threadpool = nullptr;
};

// Shared across threads: ggml/llama.cpp log callbacks can fire from Metal
// compiler or other worker threads during model load, so a thread_local here
// would discard the most informative GPU error messages.
static std::mutex g_last_load_error_mutex;
static std::string g_last_load_error;
static std::atomic<bool> g_capturing_load_errors{false};

static void edge_log_callback(enum ggml_log_level /*level*/, const char * text, void * /*user_data*/) {
    if (text == nullptr || !g_capturing_load_errors.load(std::memory_order_acquire)) return;
    if (strstr(text, "fused Gated Delta Net not supported") ||
        strstr(text, "posix_memalign failed") ||
        strstr(text, "failed to allocate buffer") ||
        strstr(text, "Compiler failed to build request") ||
        strstr(text, "ggml_metal") ||
        strstr(text, "MTLCompiler") ||
        strstr(text, "unsupported op") ||
        strstr(text, "not supported") ||
        strstr(text, "error:")) {
        std::lock_guard<std::mutex> lock(g_last_load_error_mutex);
        g_last_load_error += text;
    }
}

void ensure_backend_init() {
    static std::once_flag once;
    std::call_once(once, []() {
#if defined(__ANDROID__)
        // Score + register the best-supported CPU backend variant before
        // llama touches the backend registry.
        pipette_native_load_backends();
#endif
        llama_backend_init();
        llama_log_set(edge_log_callback, nullptr);
    });
}

#if defined(__ANDROID__)
// Performance CPU indices = every core with `cpu_capacity` >= half the max
// (i.e. exclude only the genuine little tier — see the threshold rationale in
// the body).
//
// `/sys/devices/system/cpu/cpuN/cpu_capacity` reports each core's normalized
// DMIPS capacity (the prime core is 1024; little cores are far lower — e.g.
// 280/855/1024 on a 3-tier SoC). Inference decode synchronizes all worker
// threads at every token, so a single little core in the pool stalls the whole
// step — pinning to the non-little cores is what keeps decode off the slow
// path. (Mirrors inference_engine's `cpu_capacity`-based performant-core
// selection.) Returns empty only if `/sys` is unreadable, in which case callers
// fall back to the all-core default. A flat topology now returns all cores (all
// pass the threshold), so those devices get an all-core strict pin rather than
// the auto pool — harmless, since there is no little tier to avoid.
std::vector<int> performant_cpu_indices() {
    std::vector<uint32_t> caps;
    for (int i = 0;; ++i) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/cpu_capacity");
        if (!f.is_open()) break;
        uint32_t c = 0;
        f >> c;
        if (f.fail()) { caps.clear(); break; }
        caps.push_back(c);
    }
    std::vector<int> perf;
    if (caps.empty()) return perf;
    const uint32_t hi = *std::max_element(caps.begin(), caps.end());
    // Keep every core within the top capacity band; exclude only genuine little
    // cores. A "> lowest tier" test was too aggressive on 2+N SoCs with no little
    // cores — e.g. Snapdragon 8 Elite (6 perf @ cap 765 + 2 prime @ 1024) — where
    // it kept only the 2 primes. Decode is memory-bandwidth-bound and gains from
    // the perf cores too, so 6 perf cores can beat 2 primes. Threshold at half the
    // max: little cores (~20-40% of max, e.g. Pixel's 207/1024) are excluded while
    // perf/prime (>= ~75%) are kept.
    const uint32_t threshold = hi / 2;
    for (int i = 0; i < static_cast<int>(caps.size()); ++i) {
        if (caps[i] >= threshold) perf.push_back(i);
    }
    return perf;
}
#endif

#if defined(__ANDROID__)
// The CPUs the compute pool should actually use: the performant tier
// (`performant_cpu_indices`) intersected with the cpuset the process is *allowed*
// on (`sched_getaffinity`). This matters because the OEM cpuset can forbid the
// very cores the capacity heuristic prefers — on Samsung a non-`top-app`
// `:benchmark` process lands in `/foreground = [0-5]`, which excludes the prime
// cores the heuristic would pick. Pinning to a forbidden core fails EINVAL and
// silently drops the strict threadpool, so we filter to permitted cores.
//
// Fallbacks: if no performant core is permitted (e.g. all primes forbidden), use
// every permitted core rather than none; if affinity can't be read, fall back to
// the raw performant tier (prior behavior).
std::vector<int> usable_compute_cpus() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return performant_cpu_indices();
    }
    std::vector<int> usable;
    for (int cpu : performant_cpu_indices()) {
        if (cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed)) usable.push_back(cpu);
    }
    if (!usable.empty()) return usable;
    // No performant core is permitted — use whatever the cgroup does allow.
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) usable.push_back(cpu);
    }
    return usable;
}
#endif

int default_threads() {
#if defined(__ANDROID__)
    const auto usable = usable_compute_cpus();
    if (!usable.empty()) {
        return static_cast<int>(usable.size());
    }
#endif
    const auto hc = std::thread::hardware_concurrency();
    return static_cast<int>(hc == 0 ? 1u : hc);
}

} // namespace

#if !defined(GGML_USE_METAL)
extern "C" uint64_t ee_metal_current_allocated_size(void) {
    return 0;
}
#endif

extern "C" int ee_llama_last_error(char * buf, int buf_size) {
    // Copies the last load-error message into the caller-provided buffer while
    // holding the mutex, so a concurrent ee_llama_model_load cannot invalidate
    // the source string mid-read. Returns the number of bytes written
    // (excluding the trailing null terminator), which may be less than the
    // message's true length if `buf_size` is too small — the copy is
    // truncated and null-terminated in that case.
    if (buf == nullptr || buf_size <= 0) return 0;
    std::lock_guard<std::mutex> lock(g_last_load_error_mutex);
    if (g_last_load_error.empty()) {
        buf[0] = '\0';
        return 0;
    }
    const size_t n = std::min<size_t>(g_last_load_error.size(), static_cast<size_t>(buf_size) - 1);
    std::memcpy(buf, g_last_load_error.data(), n);
    buf[n] = '\0';
    return static_cast<int>(n);
}

// Active CPU-backend feature descriptor, e.g. "dotprod,fp16_va,matmul_int8,neon".
//
// Built from the registered "CPU" backend's `ggml_backend_get_features()` — the
// same source `llama_print_system_info()` reads — filtered to enabled (value
// "1") features, lowercased and sorted so the string is stable and directly
// comparable across runs. It reflects the *active* backend and thus implies the
// effective CPU compile flags: on Android that's the runtime-selected DL
// `libggml-cpu-*` variant; on iOS / desktop the single static build. The
// backend-registry API used here is part of ggml-base (linked on every
// platform), unlike the `ggml_cpu_has_*` helpers which live in the
// runtime-loaded variant `.so` on Android and would not resolve at link time.
//
// Empty when no CPU backend is registered yet — call after a model load — or
// when the backend exposes no feature list. Writes into the caller buffer like
// `ee_llama_last_error`; returns bytes written (excluding the null terminator).
extern "C" int ee_cpu_backend_descriptor(char * buf, int buf_size) {
    if (buf == nullptr || buf_size <= 0) return 0;
    std::vector<std::string> feats;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (reg == nullptr || ggml_backend_reg_name(reg) == nullptr ||
            std::strcmp(ggml_backend_reg_name(reg), "CPU") != 0) {
            continue;
        }
        auto get_features = reinterpret_cast<ggml_backend_get_features_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features"));
        if (get_features == nullptr) continue;
        for (ggml_backend_feature * f = get_features(reg); f != nullptr && f->name != nullptr; f++) {
            if (f->value != nullptr && std::strcmp(f->value, "1") == 0) {
                std::string name(f->name);
                std::transform(name.begin(), name.end(), name.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                feats.push_back(std::move(name));
            }
        }
    }
    std::sort(feats.begin(), feats.end());
    feats.erase(std::unique(feats.begin(), feats.end()), feats.end());

    std::string out;
    for (size_t i = 0; i < feats.size(); i++) {
        if (i != 0) out += ',';
        out += feats[i];
    }
    const size_t n = std::min<size_t>(out.size(), static_cast<size_t>(buf_size) - 1);
    std::memcpy(buf, out.data(), n);
    buf[n] = '\0';
    return static_cast<int>(n);
}

// The compute-thread count the runtime would pick for a model loaded *now* under
// the current cpuset — usable performant cores, or (when none are permitted) all
// permitted cores, or (when affinity/topology is unreadable) hardware_concurrency.
// This is a fresh recompute; prefer `ee_llama_n_threads` for a loaded model, whose
// count was fixed at load. Used only for the fresh-load benchmark path where load
// immediately precedes the run. Called in the
// :benchmark process, so it reflects that process's cpuset.
extern "C" int ee_default_threads(void) {
    return default_threads();
}

// The compute-thread count a specific loaded model was configured with (fixed at
// load; see EdgeLlamaHandle::n_threads). Authoritative for reproducibility: the
// cpuset can change between load and a later run, so recording this rather than a
// recompute keeps the value matched to the pinned pool the run used. 0 on a null
// handle.
extern "C" int ee_llama_n_threads(void * raw_handle) {
    if (raw_handle == nullptr) return 0;
    return static_cast<EdgeLlamaHandle *>(raw_handle)->n_threads;
}

extern "C" void * ee_llama_model_load(const char * path, int n_gpu_layers, int n_ctx, int n_ubatch) {
    ensure_backend_init();
    {
        std::lock_guard<std::mutex> lock(g_last_load_error_mutex);
        g_last_load_error.clear();
    }
    g_capturing_load_errors.store(true, std::memory_order_release);
    struct CaptureGuard {
        ~CaptureGuard() { g_capturing_load_errors.store(false, std::memory_order_release); }
    } capture_guard;

    auto handle = std::make_unique<EdgeLlamaHandle>();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    // Keep iOS aligned with desktop llama-bench, where our CLI injects
    // `--mmap 0` for throughput and max-memory benchmarks. mmap=true can let
    // prefill/decode succeed by keeping GGUF weights file-backed / partially
    // resident; under the no-mmap benchmark contract, models that require that
    // path should fail instead of producing rows.
    mparams.load_mode = LLAMA_LOAD_MODE_NONE;

    handle->model = llama_model_load_from_file(path, mparams);
    if (handle->model == nullptr) {
        return nullptr;
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx > 0 ? static_cast<uint32_t>(n_ctx) : cparams.n_ctx;
    // Match upstream llama-bench defaults so iOS numbers are directly
    // comparable to the macOS CLI runner, which shells out to llama-bench.
    // Prefill micro-batch is configurable via `n_ubatch` (<=0 → default 512, the
    // upstream llama-bench default). The logical batch (n_batch) must be >= the
    // micro-batch, so grow it to fit a larger ubatch.
    uint32_t ub = n_ubatch > 0 ? static_cast<uint32_t>(n_ubatch) : 512u;
    cparams.n_ubatch = std::min<uint32_t>(ub, cparams.n_ctx);
    cparams.n_batch  = std::min<uint32_t>(std::max<uint32_t>(2048u, ub), cparams.n_ctx);
    cparams.n_seq_max = 1;
    cparams.no_perf = true;
#if defined(__ANDROID__)
    // Compute the usable compute-core set ONCE (reads /sys cpu_capacity +
    // sched_getaffinity) and derive n_threads from it, so the count here and the
    // strict pin below share an identical set: no redundant I/O, and no TOCTOU
    // where a changed affinity would leave n_threads != popcount(cpumask) and
    // make ggml's strict round-robin oversubscribe a core.
    const std::vector<int> usable = usable_compute_cpus();
    const unsigned hc = std::thread::hardware_concurrency();
    const int nthreads = usable.empty() ? static_cast<int>(hc == 0 ? 1u : hc) : static_cast<int>(usable.size());
#else
    const unsigned hc = std::thread::hardware_concurrency();
    const int nthreads = static_cast<int>(hc == 0 ? 1u : hc);
#endif
    cparams.n_threads = nthreads;
    cparams.n_threads_batch = nthreads;
    handle->n_threads = nthreads;

    handle->ctx = llama_init_from_model(handle->model, cparams);
    if (handle->ctx == nullptr) {
        llama_model_free(handle->model);
        return nullptr;
    }
    handle->n_ctx_captured = cparams.n_ctx;

    handle->sampler = llama_sampler_init_greedy();
    if (handle->sampler == nullptr) {
        llama_free(handle->ctx);
        llama_model_free(handle->model);
        return nullptr;
    }

#if defined(__ANDROID__)
    // Pin the compute pool to the same `usable` core set `n_threads` was derived
    // from (single pool for generation and batch/prefill), so we never target a
    // core the cpuset forbids (which fails EINVAL and silently drops the strict
    // pool) and n_threads always equals the pinned-core count. Best-effort: if
    // creation fails, llama.cpp falls back to its auto threadpool.
    {
        if (!usable.empty() && usable.size() <= static_cast<size_t>(GGML_MAX_N_THREADS)) {
            ggml_threadpool_params tpp = ggml_threadpool_params_default(static_cast<int>(usable.size()));
            for (int cpu : usable) {
                if (cpu >= 0 && cpu < GGML_MAX_N_THREADS) {
                    tpp.cpumask[cpu] = true;
                }
            }
            tpp.strict_cpu = true;
            ggml_threadpool_t tp = pipette_ggml_threadpool_new(&tpp);
            if (tp != nullptr) {
                llama_attach_threadpool(handle->ctx, tp, tp);
                handle->threadpool = tp;
            }
        }
    }
#endif

    return handle.release();
}

extern "C" void ee_llama_model_free(void * raw_handle) {
    auto * handle = static_cast<EdgeLlamaHandle *>(raw_handle);
    if (handle == nullptr) {
        return;
    }

    if (handle->sampler != nullptr) {
        llama_sampler_free(handle->sampler);
    }
    if (handle->ctx != nullptr) {
#if defined(__ANDROID__)
        // Detach the pinned pool before freeing the context that uses it.
        if (handle->threadpool != nullptr) {
            llama_detach_threadpool(handle->ctx);
        }
#endif
        llama_free(handle->ctx);
    }
#if defined(__ANDROID__)
    if (handle->threadpool != nullptr) {
        pipette_ggml_threadpool_free(handle->threadpool);
    }
#endif
    if (handle->model != nullptr) {
        llama_model_free(handle->model);
    }

    delete handle;
}

extern "C" int ee_llama_tokenize_text(
    void * raw_handle,
    const char * text,
    int32_t * out_tokens,
    int max_tokens,
    bool add_special
) {
    auto * handle = static_cast<EdgeLlamaHandle *>(raw_handle);
    if (handle == nullptr || handle->model == nullptr || text == nullptr || out_tokens == nullptr) {
        return -1;
    }

    const auto * vocab = llama_model_get_vocab(handle->model);
    return llama_tokenize(
        vocab,
        text,
        static_cast<int32_t>(std::strlen(text)),
        out_tokens,
        max_tokens,
        add_special,
        true
    );
}

extern "C" int ee_llama_decode_batch(void * raw_handle, const int32_t * tokens, int n_tokens) {
    auto * handle = static_cast<EdgeLlamaHandle *>(raw_handle);
    if (handle == nullptr || handle->ctx == nullptr || tokens == nullptr || n_tokens <= 0) {
        return -1;
    }

    // llama_decode asserts that the submitted batch is no larger than
    // cparams.n_batch. Split the input into n_batch-sized chunks so the
    // caller can hand us any prefill length without hitting that assert.
    // llama_batch_get_one continues from the current KV cache position,
    // so consecutive calls append naturally.
    const int n_batch = static_cast<int>(llama_n_batch(handle->ctx));
    int offset = 0;
    while (offset < n_tokens) {
        const int chunk = std::min(n_tokens - offset, n_batch);
        auto batch = llama_batch_get_one(
            const_cast<llama_token *>(tokens + offset), chunk);
        const int rc = llama_decode(handle->ctx, batch);
        if (rc != 0) {
            return rc;
        }
        offset += chunk;
    }
    llama_synchronize(handle->ctx);
    return 0;
}

extern "C" int ee_llama_sample_greedy(void * raw_handle) {
    auto * handle = static_cast<EdgeLlamaHandle *>(raw_handle);
    if (handle == nullptr || handle->ctx == nullptr || handle->sampler == nullptr) {
        return -1;
    }

    const auto token = llama_sampler_sample(handle->sampler, handle->ctx, -1);
    llama_sampler_accept(handle->sampler, token);
    return token;
}

extern "C" int ee_llama_token_to_str(void * raw_handle, int token, uint8_t * buf, int buf_size) {
    auto * handle = static_cast<EdgeLlamaHandle *>(raw_handle);
    if (handle == nullptr || handle->model == nullptr || buf == nullptr || buf_size <= 0) {
        return -1;
    }

    const auto * vocab = llama_model_get_vocab(handle->model);
    return llama_token_to_piece(vocab, token, reinterpret_cast<char *>(buf), buf_size, 0, true);
}

extern "C" int ee_llama_apply_chat_template(
    void * raw_handle,
    const char * messages_json,
    uint8_t * buf,
    int buf_size
) {
    auto * handle = static_cast<EdgeLlamaHandle *>(raw_handle);
    if (handle == nullptr || handle->model == nullptr || messages_json == nullptr || buf == nullptr) {
        return -1;
    }

    const char * tmpl = llama_model_chat_template(handle->model, nullptr);
    if (tmpl == nullptr) {
        return -1;
    }

    std::vector<std::string> roles;
    std::vector<std::string> contents;
    std::vector<llama_chat_message> chat;

    try {
        nlohmann::json parsed = nlohmann::json::parse(messages_json);
        if (!parsed.is_array()) {
            return -1;
        }

        roles.reserve(parsed.size());
        contents.reserve(parsed.size());
        chat.reserve(parsed.size());

        for (const auto & item : parsed) {
            if (!item.is_object()) {
                return -1;
            }
            const auto role_it    = item.find("role");
            const auto content_it = item.find("content");
            if (role_it == item.end() || !role_it->is_string() ||
                content_it == item.end() || !content_it->is_string()) {
                return -1;
            }
            roles.push_back(role_it->get<std::string>());
            contents.push_back(content_it->get<std::string>());
        }
    } catch (...) {
        return -1;
    }

    for (size_t i = 0; i < roles.size(); ++i) {
        llama_chat_message msg {
            roles[i].c_str(),
            contents[i].c_str(),
        };
        chat.push_back(msg);
    }

    return llama_chat_apply_template(
        tmpl,
        chat.data(),
        chat.size(),
        true,
        reinterpret_cast<char *>(buf),
        buf_size
    );
}

extern "C" void ee_llama_kv_cache_clear_ctx(void * raw_handle) {
    auto * handle = static_cast<EdgeLlamaHandle *>(raw_handle);
    if (handle == nullptr || handle->ctx == nullptr) {
        return;
    }

    llama_memory_clear(llama_get_memory(handle->ctx), false);
}

extern "C" int ee_llama_token_eos_id(void * raw_handle) {
    auto * handle = static_cast<EdgeLlamaHandle *>(raw_handle);
    if (handle == nullptr || handle->model == nullptr) {
        return -1;
    }

    return llama_vocab_eos(llama_model_get_vocab(handle->model));
}

extern "C" void ee_llama_sampler_reset(void * raw_handle) {
    auto * handle = static_cast<EdgeLlamaHandle *>(raw_handle);
    if (handle == nullptr || handle->sampler == nullptr) {
        return;
    }
    llama_sampler_reset(handle->sampler);
}

// ---------------------------------------------------------------------------
// mtmd (multimodal) primitives — used by the VL throughput benchmark.
// Each function is a thin wrapper around the mtmd C API. Orchestration
// (measurement loop, timing, stats) stays in Rust for consistency with the
// existing ee_llama_* primitives.
// ---------------------------------------------------------------------------

extern "C" void * ee_mtmd_init(
    void * raw_handle,
    const char * mmproj_path,
    bool use_gpu
) {
    auto * handle = static_cast<EdgeLlamaHandle *>(raw_handle);
    if (handle == nullptr || handle->model == nullptr || mmproj_path == nullptr) {
        return nullptr;
    }

    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu       = use_gpu;
    params.print_timings = false;
    params.n_threads     = default_threads();
    params.warmup        = false; // we run our own warmup in the benchmark loop

    return mtmd_init_from_file(mmproj_path, handle->model, params);
}

extern "C" void ee_mtmd_free(void * mtmd_ctx) {
    if (mtmd_ctx == nullptr) {
        return;
    }
    mtmd_free(static_cast<mtmd_context *>(mtmd_ctx));
}

// Allocate a synthetic solid-gray RGB bitmap of the given dimensions, tokenize
// the given text prompt (which must contain the mtmd media marker) together
// with the bitmap, and return an opaque chunks pointer. Writes the total
// token count (text + image) to *out_n_tokens. Returns nullptr on failure.
extern "C" void * ee_mtmd_alloc_gray_chunks(
    void * mtmd_ctx_raw,
    const char * text_with_marker,
    uint32_t nx,
    uint32_t ny,
    size_t * out_n_tokens
) {
    auto * mctx = static_cast<mtmd_context *>(mtmd_ctx_raw);
    if (mctx == nullptr || text_with_marker == nullptr || nx == 0 || ny == 0) {
        return nullptr;
    }

    const size_t n_bytes = static_cast<size_t>(nx) * static_cast<size_t>(ny) * 3;
    std::vector<unsigned char> pixels(n_bytes, 128);

    mtmd_bitmap * bitmap = mtmd_bitmap_init(nx, ny, pixels.data());
    if (bitmap == nullptr) {
        return nullptr;
    }

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (chunks == nullptr) {
        mtmd_bitmap_free(bitmap);
        return nullptr;
    }

    mtmd_input_text text;
    text.text          = text_with_marker;
    text.add_special   = true;
    text.parse_special = true;

    const mtmd_bitmap * bitmaps_arr[] = { bitmap };
    int32_t rc = mtmd_tokenize(mctx, chunks, &text, bitmaps_arr, 1);

    // The bitmap's pixel data has been copied into the chunk's preprocessed
    // tensors by this point, so we can free it now regardless of rc.
    mtmd_bitmap_free(bitmap);

    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        return nullptr;
    }

    if (out_n_tokens != nullptr) {
        *out_n_tokens = mtmd_helper_get_n_tokens(chunks);
    }
    return chunks;
}

extern "C" void ee_mtmd_free_chunks(void * chunks_raw) {
    if (chunks_raw == nullptr) {
        return;
    }
    mtmd_input_chunks_free(static_cast<mtmd_input_chunks *>(chunks_raw));
}

// Run llama_decode over text chunks and mtmd_encode + llama_decode over image
// chunks, starting from KV position 0 (the caller must have cleared the KV
// cache beforehand). Leaves the context positioned at new_n_past so the
// subsequent greedy decode loop can sample off the last-token logits.
// Returns 0 on success, non-zero on failure.
extern "C" int ee_mtmd_eval_chunks(
    void * mtmd_ctx_raw,
    void * raw_handle,
    void * chunks_raw
) {
    auto * mctx   = static_cast<mtmd_context *>(mtmd_ctx_raw);
    auto * handle = static_cast<EdgeLlamaHandle *>(raw_handle);
    auto * chunks = static_cast<mtmd_input_chunks *>(chunks_raw);
    if (mctx == nullptr || handle == nullptr || handle->ctx == nullptr || chunks == nullptr) {
        return -1;
    }

    llama_pos new_n_past = 0;
    // Must match the context's configured n_batch — cparams.n_batch is
    // clamped to std::min(2048, n_ctx) at init, so passing n_ctx here would
    // exceed the limit and mtmd_helper_eval_chunks's sub-batched llama_decode
    // calls would fail for contexts > 2048.
    const int32_t n_batch = static_cast<int32_t>(llama_n_batch(handle->ctx));
    const int32_t rc = mtmd_helper_eval_chunks(
        mctx,
        handle->ctx,
        chunks,
        /*n_past=*/0,
        /*seq_id=*/0,
        /*n_batch=*/n_batch,
        /*logits_last=*/true,
        &new_n_past
    );
    if (rc != 0) {
        return rc;
    }
    // Block until the GPU work has completed so wall-clock timing captured
    // around this call on the Rust side reflects actual compute time.
    llama_synchronize(handle->ctx);
    return 0;
}
