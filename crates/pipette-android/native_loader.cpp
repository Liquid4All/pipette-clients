// Android CPU-backend variant loader.
//
// The Android build compiles ggml with `GGML_CPU_ALL_VARIANTS=ON` +
// `GGML_BACKEND_DL=ON`, so the CPU backend is NOT statically linked into
// `libpipette_android.so`. Instead each ARM feature level is built as a
// separate `libggml-cpu-android_armv8.x_*.so` shipped alongside the engine in
// the APK's `lib/<abi>/` dir. This loader runs once before the first model
// load: it finds those sibling `.so`s, asks each for its `ggml_backend_score()`
// (which returns 0 when the running CPU lacks the variant's required features),
// and registers the highest-scoring supported variant via `ggml_backend_init()`
// + `ggml_backend_register()`. That is exactly how a single APK runs optimally
// across the Android fleet without SIGILL'ing on devices below the build's
// feature floor.
//
// That automatic choice can be overridden for measurement (see
// `forced_variant_tag()` below), because the score is a feature bitmask rather
// than a throughput ranking, so the highest-scoring variant is not necessarily
// the fastest one.
//
// Ported from inference_engine's `native_loader.cpp` (the on-disk-scan path
// only — Pipette sets `useLegacyPackaging = true`, so the variants are
// extracted to a real directory and `dladdr` on this TU yields the lib dir to
// scan).
//
// The armv9 variants ARE packaged: on an armv9 device this loader scores and
// selects one (highest feature level wins) even though its LFM2A audio encoder
// produces incorrect output. That is an accepted tradeoff — Pipette is a
// benchmark harness, so we measure every backend the device exposes rather than
// patch the inference engine for the harness (see cmake/stage_android_jnilibs.cmake).
//
// This file is Android-only and is compiled exclusively into
// `libpipette_android.so` by `crates/pipette-android/build.rs`. It also exposes
// `pipette_ggml_threadpool_{new,free}` so the (shared, iOS+Android) llama_shim
// can build its performance-core-pinned threadpool: `ggml_threadpool_new` /
// `ggml_threadpool_free` live in the CPU variant `.so` (runtime-loaded), so they
// are not link-time symbols of the engine lib — they are resolved once (below)
// from the variant we load.

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <android/log.h>
#include <atomic>
#include <cstdlib>
#include <dirent.h>
#include <dlfcn.h>
#include <sys/system_properties.h>

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

#define PIPETTE_LOG_TAG "pipette-cpudispatch"
#define PIPETTE_LOGI(...) __android_log_print(ANDROID_LOG_INFO, PIPETTE_LOG_TAG, __VA_ARGS__)
#define PIPETTE_LOGW(...) __android_log_print(ANDROID_LOG_WARN, PIPETTE_LOG_TAG, __VA_ARGS__)
#define PIPETTE_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, PIPETTE_LOG_TAG, __VA_ARGS__)

namespace {

typedef int (*ggml_backend_score_fn)(void);
typedef ggml_backend_reg_t (*ggml_backend_init_fn)(void);
typedef ggml_threadpool_t (*ggml_threadpool_new_fn)(struct ggml_threadpool_params *);
typedef void (*ggml_threadpool_free_fn)(ggml_threadpool_t);

// Resolved once from the winning variant (inside the call_once below) and read
// by the threadpool wrappers. Atomic so a future caller reaching the wrappers
// from a thread that did not drive the loader's call_once still observes a
// consistent value (nullptr or the real pointer), never a torn one. The owning
// dlopen handle is intentionally never closed, so these pointers stay valid for
// the process lifetime.
std::atomic<ggml_threadpool_new_fn> g_tp_new{nullptr};
std::atomic<ggml_threadpool_free_fn> g_tp_free{nullptr};

// Directory containing this shared library, via dladdr on a function in this
// TU. With useLegacyPackaging=true the engine lib is extracted to disk in the
// app's nativeLibraryDir, so its siblings (the variant .so's) are in the same
// dir. If the lib is instead mmap'd inside the APK (useLegacyPackaging=false),
// dladdr returns an `<apk>!/lib/...` path that is not a real directory; detect
// that and fail with an actionable message rather than a confusing opendir error.
std::string self_dir() {
    Dl_info info;
    if (dladdr(reinterpret_cast<void *>(&self_dir), &info) == 0 || info.dli_fname == nullptr) {
        PIPETTE_LOGE("dladdr could not resolve this library's path; CPU backend not loaded");
        return "";
    }
    std::string path(info.dli_fname);
    if (path.find('!') != std::string::npos) {
        PIPETTE_LOGE("native libs are mmap'd inside the APK (%s); the CPU-variant loader "
                     "requires them extracted to disk. Set android.packaging.jniLibs."
                     "useLegacyPackaging = true.",
                     path.c_str());
        return "";
    }
    size_t slash = path.find_last_of('/');
    return slash != std::string::npos ? path.substr(0, slash) : "";
}

bool is_cpu_variant(const std::string &name) {
    static const std::string prefix = "libggml-cpu-";
    static const std::string suffix = ".so";
    return name.size() > prefix.size() + suffix.size() &&
           name.compare(0, prefix.size(), prefix) == 0 &&
           name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The feature tag of a variant path, i.e. the `<tag>` in
// `libggml-cpu-<tag>.so` ("android_armv8.6_1"). Empty if the name does not match.
std::string variant_tag(const std::string &path) {
    static const std::string prefix = "libggml-cpu-";
    static const std::string suffix = ".so";
    size_t slash = path.find_last_of('/');
    std::string name = slash != std::string::npos ? path.substr(slash + 1) : path;
    if (!is_cpu_variant(name)) {
        return "";
    }
    return name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
}

// Diagnostic override: pin the CPU backend to one variant instead of taking the
// highest `ggml_backend_score()`. Reads `PIPETTE_FORCE_CPU_VARIANT` from the
// environment first (host/CLI contexts), then the Android system property
// `debug.pipette.cpu_variant`, which is settable on retail `user` builds with
// `adb shell setprop debug.pipette.cpu_variant armv8.6_1` and readable here
// without any app-side plumbing (this loader runs from a dlopen'd native lib
// before any Kotlin could hand it a path).
//
// Why this exists: `ggml_backend_score()` is a feature bitmask
// (`ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp`), so it is monotone in feature
// *count*, not throughput. A device therefore always runs its highest feature
// level and can never be measured against a lower one, yet SVE is not additive
// in ggml (it displaces the NEON/i8mm paths in `simd-mappings.h` and the K-quant
// dots), so "more features" is not automatically "faster". This makes that
// comparison a single `setprop` instead of a rebuild per variant.
//
// Matching is on the tag, with or without the `android_` prefix, so both
// `armv8.6_1` and `android_armv8.6_1` work.
std::string forced_variant_tag() {
    if (const char *env = getenv("PIPETTE_FORCE_CPU_VARIANT"); env != nullptr && env[0] != '\0') {
        return std::string(env);
    }
    char prop[PROP_VALUE_MAX] = {};
    if (__system_property_get("debug.pipette.cpu_variant", prop) > 0 && prop[0] != '\0') {
        return std::string(prop);
    }
    return "";
}

// True when `tag` names the variant at `path`, accepting the tag with or without
// the platform prefix ("armv8.6_1" matches "android_armv8.6_1").
bool variant_matches(const std::string &path, const std::string &tag) {
    const std::string actual = variant_tag(path);
    if (actual.empty()) {
        return false;
    }
    if (actual == tag) {
        return true;
    }
    size_t underscore = actual.find('_');
    return underscore != std::string::npos && actual.substr(underscore + 1) == tag;
}

std::vector<std::string> discover_variants(const std::string &dir) {
    std::vector<std::string> out;
    DIR *d = opendir(dir.c_str());
    if (d == nullptr) {
        PIPETTE_LOGE("cannot open lib dir %s", dir.c_str());
        return out;
    }
    while (struct dirent *e = readdir(d)) {
        std::string name(e->d_name);
        if (is_cpu_variant(name)) {
            out.push_back(dir + "/" + name);
        }
    }
    closedir(d);
    return out;
}

// dlopen the variant locally, read its score, close it again (scoring must not
// leave a half-registered backend behind, and RTLD_LOCAL keeps a non-winning
// variant's symbols out of the global namespace). Returns the score, or -1 if it
// could not be loaded / has no score function.
int score_variant(const std::string &path) {
    void *h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr) {
        PIPETTE_LOGW("dlopen failed for %s: %s", path.c_str(), dlerror());
        return -1;
    }
    auto score_fn = reinterpret_cast<ggml_backend_score_fn>(dlsym(h, "ggml_backend_score"));
    int score = score_fn != nullptr ? score_fn() : -1;
    dlclose(h);
    return score;
}

} // namespace

extern "C" {

// Resolve + register the best-scoring CPU backend variant. Idempotent.
void pipette_native_load_backends(void) {
    static std::once_flag once;
    std::call_once(once, []() {
        const std::string dir = self_dir();
        if (dir.empty()) {
            return; // self_dir already logged the actionable reason.
        }
        PIPETTE_LOGI("scanning %s for CPU backend variants", dir.c_str());

        std::vector<std::string> variants = discover_variants(dir);
        const std::string forced = forced_variant_tag();
        std::string best_path;
        int best_score = 0;
        std::string forced_path;
        int forced_score = -1;
        std::string available_tags;
        for (const auto &path : variants) {
            int score = score_variant(path);
            PIPETTE_LOGI("variant %s score=%d", path.c_str(), score);
            if (score > best_score) {
                best_score = score;
                best_path = path;
            }
            if (!forced.empty()) {
                if (!available_tags.empty()) {
                    available_tags += ", ";
                }
                available_tags += variant_tag(path);
                if (variant_matches(path, forced)) {
                    forced_path = path;
                    forced_score = score;
                }
            }
        }

        // Apply the override, but never at the cost of a SIGILL: score 0 means the
        // running CPU lacks a feature this variant was compiled for. Forcing
        // *downward* (the case the A/B needs) always scores > 0 and is honoured.
        if (!forced.empty()) {
            if (forced_path.empty()) {
                PIPETTE_LOGE("cpu-variant override '%s' matches no packaged variant; "
                             "ignoring it and selecting by score. Available: %s",
                             forced.c_str(), available_tags.c_str());
            } else if (forced_score <= 0) {
                PIPETTE_LOGE("cpu-variant override '%s' scored %d, so this CPU lacks a feature "
                             "it was built for and running it would SIGILL. Ignoring the "
                             "override and selecting by score.",
                             forced.c_str(), forced_score);
            } else {
                PIPETTE_LOGW("cpu-variant override ACTIVE: forcing %s (score=%d) instead of "
                             "the highest-scoring %s (score=%d). Unset with "
                             "`setprop debug.pipette.cpu_variant \"\"`.",
                             forced_path.c_str(), forced_score,
                             best_path.empty() ? "(none)" : best_path.c_str(), best_score);
                best_path = forced_path;
                best_score = forced_score;
            }
        }

        if (best_path.empty()) {
            PIPETTE_LOGE("no supported CPU backend variant found among %zu candidate(s) in %s; "
                         "this device is below the build's feature floor — model loads will fail.",
                         variants.size(), dir.c_str());
            return;
        }

        // Load the winner with RTLD_LOCAL (its symbols reach us via the registry
        // function table + the explicit dlsym below, so they need not be global).
        // The handle is intentionally never closed: its code backs the registered
        // backend and the threadpool entry points for the process lifetime.
        //
        // The registered reg's api_version is NOT re-checked here (unlike ggml's
        // own generic loader): every variant is built from the same vendored
        // llama.cpp tree as ggml-base in one CMake invocation, so a version skew
        // between them is impossible by construction.
        void *handle = dlopen(best_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            PIPETTE_LOGE("failed to load winning variant %s: %s", best_path.c_str(), dlerror());
            return;
        }
        auto init_fn = reinterpret_cast<ggml_backend_init_fn>(dlsym(handle, "ggml_backend_init"));
        if (init_fn == nullptr) {
            PIPETTE_LOGE("variant %s has no ggml_backend_init", best_path.c_str());
            return;
        }
        ggml_backend_reg_t reg = init_fn();
        if (reg == nullptr) {
            PIPETTE_LOGE("ggml_backend_init returned null for %s", best_path.c_str());
            return;
        }
        ggml_backend_register(reg);

        // Resolve the threadpool entry points once from this handle (they live in
        // the variant .so, not in any lib the engine links at build time). If
        // they are ever missing, the wrappers below fall back to llama.cpp's auto
        // threadpool — log it once so a lost perf-core pinning isn't silent.
        g_tp_new.store(reinterpret_cast<ggml_threadpool_new_fn>(dlsym(handle, "ggml_threadpool_new")),
                       std::memory_order_release);
        g_tp_free.store(reinterpret_cast<ggml_threadpool_free_fn>(dlsym(handle, "ggml_threadpool_free")),
                        std::memory_order_release);
        if (g_tp_new.load(std::memory_order_relaxed) == nullptr ||
            g_tp_free.load(std::memory_order_relaxed) == nullptr) {
            PIPETTE_LOGW("ggml_threadpool_{new,free} not found in %s; "
                         "performance-core pinning disabled (using auto threadpool)",
                         best_path.c_str());
        }

        PIPETTE_LOGI("registered CPU backend %s (score=%d)", best_path.c_str(), best_score);
    });
}

// Performance-core threadpool helpers. Resolved once by the loader above (the
// symbols live in the runtime-loaded variant `.so`, so the shim cannot call them
// directly). If unavailable, return null / no-op so the shim falls back to
// llama.cpp's auto threadpool.
ggml_threadpool_t pipette_ggml_threadpool_new(struct ggml_threadpool_params *params) {
    auto fn = g_tp_new.load(std::memory_order_acquire);
    return fn != nullptr ? fn(params) : nullptr;
}

void pipette_ggml_threadpool_free(ggml_threadpool_t tp) {
    auto fn = g_tp_free.load(std::memory_order_acquire);
    if (fn != nullptr && tp != nullptr) {
        fn(tp);
    }
}

} // extern "C"
