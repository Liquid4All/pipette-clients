#include <jni.h>
#include <android/log.h>
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define PIPETTE_LOG_TAG "pipette-jni"
#define PIPETTE_LOGW(...) __android_log_print(ANDROID_LOG_WARN, PIPETTE_LOG_TAG, __VA_ARGS__)
#define PIPETTE_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, PIPETTE_LOG_TAG, __VA_ARGS__)

#ifndef LLAMA_CPP_COMMIT
#define LLAMA_CPP_COMMIT "unknown"
#endif

#ifndef LLAMA_CPP_TAG
#define LLAMA_CPP_TAG ""
#endif

extern "C" {
int ee_llama_last_error(char * buf, int buf_size);
int ee_cpu_backend_descriptor(char * buf, int buf_size);
int ee_default_threads(void);
int ee_llama_n_threads(void * raw_handle);
void * ee_llama_model_load(const char * path, int n_gpu_layers, int n_ctx, int n_ubatch, int load_mode);
void ee_llama_model_free(void * raw_handle);
int ee_llama_tokenize_text(void * raw_handle, const char * text, int32_t * out_tokens, int max_tokens, bool add_special);
int ee_llama_decode_batch(void * raw_handle, const int32_t * tokens, int n_tokens);
int ee_llama_sample_greedy(void * raw_handle);
int ee_llama_token_to_str(void * raw_handle, int token, uint8_t * buf, int buf_size);
int ee_llama_apply_chat_template(void * raw_handle, const char * messages_json, uint8_t * buf, int buf_size);
void ee_llama_kv_cache_clear_ctx(void * raw_handle);
int ee_llama_token_eos_id(void * raw_handle);
void ee_llama_sampler_reset(void * raw_handle);

void * ee_mtmd_init(void * raw_handle, const char * mmproj_path, bool use_gpu);
void ee_mtmd_free(void * mtmd_ctx);
void * ee_mtmd_alloc_gray_chunks(void * mtmd_ctx_raw, const char * text_with_marker, uint32_t nx, uint32_t ny, size_t * out_n_tokens, size_t * out_image_tokens);
void * ee_mtmd_alloc_chunks_from_image(void * mtmd_ctx_raw, const char * text_with_marker, const unsigned char * image_bytes, size_t image_len, size_t * out_n_tokens, size_t * out_image_tokens);
void ee_mtmd_free_chunks(void * chunks_raw);
int ee_mtmd_eval_chunks(void * mtmd_ctx_raw, void * raw_handle, void * chunks_raw);
}

namespace {

class JniStringUtfChars {
public:
    JniStringUtfChars(JNIEnv *env, jstring jstr) : env_(env), jstr_(jstr), chars_(nullptr) {
        if (env_ != nullptr && jstr_ != nullptr) {
            chars_ = env_->GetStringUTFChars(jstr_, nullptr);
        }
    }

    ~JniStringUtfChars() {
        if (env_ != nullptr && jstr_ != nullptr && chars_ != nullptr) {
            env_->ReleaseStringUTFChars(jstr_, chars_);
        }
    }

    const char *get() const {
        return chars_ != nullptr ? chars_ : "";
    }

    bool is_null() const {
        return chars_ == nullptr;
    }

    size_t length() const {
        return chars_ != nullptr ? std::strlen(chars_) : 0;
    }

    JniStringUtfChars(const JniStringUtfChars &) = delete;
    JniStringUtfChars &operator=(const JniStringUtfChars &) = delete;

private:
    JNIEnv *env_;
    jstring jstr_;
    const char *chars_;
};

void throw_exception(JNIEnv *env, const char *exception_class, const char *message) {
    if (env == nullptr || env->ExceptionCheck()) return;
    jclass cls = env->FindClass(exception_class);
    if (cls != nullptr) {
        env->ThrowNew(cls, message != nullptr ? message : "");
        env->DeleteLocalRef(cls);
    }
}

void throw_illegal_state(JNIEnv *env, const char *message) {
    throw_exception(env, "java/lang/IllegalStateException", message);
}

void throw_illegal_argument(JNIEnv *env, const char *message) {
    throw_exception(env, "java/lang/IllegalArgumentException", message);
}

void throw_out_of_memory(JNIEnv *env, const char *message) {
    throw_exception(env, "java/lang/OutOfMemoryError", message);
}

bool is_out_of_memory_error(const char *msg) {
    if (msg == nullptr) return false;
    return std::strstr(msg, "posix_memalign failed") != nullptr ||
           std::strstr(msg, "failed to allocate buffer") != nullptr ||
           std::strstr(msg, "out of memory") != nullptr ||
           std::strstr(msg, "MTLCompiler") != nullptr ||
           std::strstr(msg, "Compiler failed to build request") != nullptr;
}

std::string decode_error_message(int rc) {
    switch (rc) {
        case 1:
            return "decode failed: KV cache is full. The benchmark prompt+decode exceeds the allocated context size. (llama_decode rc=1)";
        case -1:
            return "decode failed: invalid arguments to llama_decode (rc=-1)";
        case -3:
            return "decode failed: compute graph error, likely a Metal/GPU failure (rc=-3). The model handle is unusable; the runner will reload before the next cell.";
        default:
            return "decode failed with code " + std::to_string(rc);
    }
}

} // namespace

// =============================================================================
// NativeLib Metadata Entry Points
// =============================================================================

extern "C" JNIEXPORT jstring JNICALL
Java_ai_liquid_pipette_NativeLib_nativeLlamaCppCommit(
    JNIEnv *env,
    jclass /*clazz*/) {
    if (env == nullptr) {
        return nullptr;
    }
    return env->NewStringUTF(LLAMA_CPP_COMMIT);
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_liquid_pipette_NativeLib_nativeLlamaCppTag(
    JNIEnv *env,
    jclass /*clazz*/) {
    if (env == nullptr) {
        return nullptr;
    }
    return env->NewStringUTF(LLAMA_CPP_TAG);
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_liquid_pipette_NativeLib_nativeCpuBackendDescriptor(
    JNIEnv *env,
    jclass /*clazz*/) {
    if (env == nullptr) {
        return nullptr;
    }
    char buf[512];
    int len = ee_cpu_backend_descriptor(buf, sizeof(buf));
    if (len <= 0) {
        return nullptr;
    }
    return env->NewStringUTF(buf);
}

// =============================================================================
// LlamaCpp Engine Primitives
// =============================================================================

extern "C" JNIEXPORT jlong JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeModelLoad(
    JNIEnv *env,
    jclass /*clazz*/,
    jstring path,
    jint n_gpu_layers,
    jint context_size,
    jint n_ubatch,
    jint load_mode) {
    if (env == nullptr) {
        return 0;
    }
    if (path == nullptr) {
        throw_illegal_argument(env, "model path cannot be null");
        return 0;
    }

    JniStringUtfChars c_path(env, path);
    if (c_path.is_null()) {
        throw_illegal_argument(env, "failed to read model path");
        return 0;
    }

    void *handle = ee_llama_model_load(
        c_path.get(),
        n_gpu_layers,
        context_size,
        n_ubatch,
        load_mode
    );

    if (handle == nullptr) {
        char err_buf[2048];
        int err_len = ee_llama_last_error(err_buf, sizeof(err_buf));
        std::string err_msg;
        if (err_len > 0) {
            err_msg.assign(err_buf, err_len);
        } else {
            err_msg = "failed to load model from " + std::string(c_path.get());
        }

        if (is_out_of_memory_error(err_msg.c_str())) {
            throw_out_of_memory(env, err_msg.c_str());
        } else {
            throw_illegal_state(env, err_msg.c_str());
        }
        return 0;
    }

    return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeModelFree(
    JNIEnv * /*env*/,
    jclass /*clazz*/,
    jlong handle) {
    if (handle != 0) {
        ee_llama_model_free(reinterpret_cast<void *>(handle));
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeNThreads(
    JNIEnv * /*env*/,
    jclass /*clazz*/,
    jlong handle) {
    if (handle == 0) {
        return 0;
    }
    return ee_llama_n_threads(reinterpret_cast<void *>(handle));
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeDefaultThreads(
    JNIEnv * /*env*/,
    jclass /*clazz*/) {
    return ee_default_threads();
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeCurrentRssBytes(
    JNIEnv * /*env*/,
    jclass /*clazz*/) {
    int fd = open("/proc/self/statm", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return 0;
    }
    char buf[128];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (n <= 0) {
        return 0;
    }
    buf[n] = '\0';
    const char *p = buf;
    while (*p && *p != ' ') ++p;
    while (*p && *p == ' ') ++p;
    char *end = nullptr;
    long resident_pages = strtol(p, &end, 10);
    if (resident_pages <= 0) {
        return 0;
    }
    static const long page_size = []() {
        long sz = sysconf(_SC_PAGESIZE);
        return (sz > 0) ? sz : 4096;
    }();
    return static_cast<jlong>(resident_pages) * static_cast<jlong>(page_size);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeTokenize(
    JNIEnv *env,
    jclass /*clazz*/,
    jlong handle,
    jbyteArray text_utf8,
    jboolean add_special) {
    if (env == nullptr || handle == 0 || text_utf8 == nullptr) {
        if (env != nullptr) {
            throw_illegal_argument(env, "invalid arguments for tokenize");
        }
        return nullptr;
    }

    jsize text_len = env->GetArrayLength(text_utf8);
    if (text_len == 0 && !add_special) {
        return env->NewIntArray(0);
    }
    std::vector<char> c_text;
    try {
        c_text.resize(static_cast<size_t>(text_len) + 1);
    } catch (const std::bad_alloc &) {
        throw_out_of_memory(env, "native allocation failed in tokenize text buffer");
        return nullptr;
    }
    env->GetByteArrayRegion(text_utf8, 0, text_len, reinterpret_cast<jbyte *>(c_text.data()));
    if (env->ExceptionCheck()) {
        return nullptr;
    }
    c_text[text_len] = '\0';

    void *ptr = reinterpret_cast<void *>(handle);
    int capacity = static_cast<int>(text_len) + 32;
    std::vector<int32_t> tokens;
    try {
        tokens.resize(capacity);
    } catch (const std::bad_alloc &) {
        throw_out_of_memory(env, "native allocation failed in tokenize");
        return nullptr;
    }

    int n = ee_llama_tokenize_text(ptr, c_text.data(), tokens.data(), capacity, add_special);
    if (n < 0 && n != INT32_MIN) {
        int needed = -n;
        if (needed > capacity) {
            capacity = needed;
            try {
                tokens.resize(capacity);
            } catch (const std::bad_alloc &) {
                throw_out_of_memory(env, "native allocation failed in tokenize growth");
                return nullptr;
            }
            n = ee_llama_tokenize_text(ptr, c_text.data(), tokens.data(), capacity, add_special);
        }
    }

    if (n < 0) {
        throw_illegal_state(env, "tokenization failed");
        return nullptr;
    }

    jintArray result = env->NewIntArray(n);
    if (result == nullptr) {
        return nullptr;
    }
    env->SetIntArrayRegion(result, 0, n, tokens.data());
    return result;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeDecodeBatch(
    JNIEnv *env,
    jclass /*clazz*/,
    jlong handle,
    jintArray tokens,
    jint offset,
    jint length) {
    if (env == nullptr) {
        return -1;
    }
    if (handle == 0) {
        throw_illegal_argument(env, "model handle is null");
        return -1;
    }
    if (tokens == nullptr) {
        throw_illegal_argument(env, "tokens array cannot be null");
        return -1;
    }

    jsize array_len = env->GetArrayLength(tokens);
    if (offset < 0 || length < 0 || length > array_len - offset) {
        throw_illegal_argument(env, "offset and length exceed tokens array bounds");
        return -1;
    }
    if (length == 0) {
        return 0;
    }

    int32_t stack_tokens[32];
    int32_t *tokens_ptr = stack_tokens;
    std::vector<int32_t> dyn_tokens;
    if (length > 32) {
        try {
            dyn_tokens.resize(length);
            tokens_ptr = dyn_tokens.data();
        } catch (const std::bad_alloc &) {
            throw_out_of_memory(env, "native allocation failed in decodeBatch");
            return -1;
        }
    }
    env->GetIntArrayRegion(tokens, offset, length, tokens_ptr);
    if (env->ExceptionCheck()) {
        return -1;
    }

    int rc = ee_llama_decode_batch(reinterpret_cast<void *>(handle), tokens_ptr, length);
    if (rc != 0) {
        throw_illegal_state(env, decode_error_message(rc).c_str());
        return rc;
    }
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeSampleGreedy(
    JNIEnv *env,
    jclass /*clazz*/,
    jlong handle) {
    if (env == nullptr || handle == 0) {
        if (env != nullptr) {
            throw_illegal_argument(env, "model handle is null");
        }
        return -1;
    }
    return ee_llama_sample_greedy(reinterpret_cast<void *>(handle));
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeTokenToStr(
    JNIEnv *env,
    jclass /*clazz*/,
    jlong handle,
    jint token) {
    if (env == nullptr || handle == 0) {
        return nullptr;
    }

    void *ptr = reinterpret_cast<void *>(handle);
    uint8_t stack_buf[256];
    int n = ee_llama_token_to_str(ptr, token, stack_buf, sizeof(stack_buf));

    if (n < 0 && n != INT32_MIN) {
        // Two-pass buffer growth: -n indicates required size
        int needed = -n;
        std::vector<uint8_t> dyn_buf;
        try {
            dyn_buf.resize(needed);
        } catch (const std::bad_alloc &) {
            throw_out_of_memory(env, "native allocation failed in tokenToStr");
            return nullptr;
        }
        n = ee_llama_token_to_str(ptr, token, dyn_buf.data(), needed);
        if (n <= 0) {
            return env->NewByteArray(0);
        }
        int copy_len = std::min(n, needed);
        jbyteArray res = env->NewByteArray(copy_len);
        if (res == nullptr) return nullptr;
        env->SetByteArrayRegion(res, 0, copy_len, reinterpret_cast<const jbyte *>(dyn_buf.data()));
        return res;
    } else if (n > 0) {
        jbyteArray res = env->NewByteArray(n);
        if (res == nullptr) return nullptr;
        env->SetByteArrayRegion(res, 0, n, reinterpret_cast<const jbyte *>(stack_buf));
        return res;
    } else {
        return env->NewByteArray(0);
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeTokenToBuffer(
    JNIEnv *env,
    jclass /*clazz*/,
    jlong handle,
    jint token,
    jobject direct_buffer,
    jint offset,
    jint capacity) {
    if (env == nullptr) return -1;
    if (handle == 0) {
        throw_illegal_argument(env, "model handle is null");
        return -1;
    }
    if (direct_buffer == nullptr) {
        throw_illegal_argument(env, "buffer cannot be null");
        return -1;
    }

    void *addr = env->GetDirectBufferAddress(direct_buffer);
    if (addr == nullptr) {
        throw_illegal_argument(env, "buffer must be a direct ByteBuffer");
        return -1;
    }

    jlong buf_cap = env->GetDirectBufferCapacity(direct_buffer);
    if (offset < 0 || capacity <= 0 || static_cast<jlong>(capacity) > buf_cap - static_cast<jlong>(offset)) {
        throw_illegal_argument(env, "offset and capacity exceed direct buffer bounds");
        return -1;
    }

    uint8_t *dest = static_cast<uint8_t *>(addr) + offset;
    return ee_llama_token_to_str(reinterpret_cast<void *>(handle), token, dest, capacity);
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeApplyChatTemplate(
    JNIEnv *env,
    jclass /*clazz*/,
    jlong handle,
    jbyteArray messages_json_utf8) {
    if (env == nullptr || handle == 0 || messages_json_utf8 == nullptr) {
        if (env != nullptr) {
            throw_illegal_argument(env, "invalid arguments for applyChatTemplate");
        }
        return nullptr;
    }

    jsize json_len = env->GetArrayLength(messages_json_utf8);
    std::vector<char> c_json;
    try {
        c_json.resize(static_cast<size_t>(json_len) + 1);
    } catch (const std::bad_alloc &) {
        throw_out_of_memory(env, "native allocation failed in applyChatTemplate buffer");
        return nullptr;
    }
    env->GetByteArrayRegion(messages_json_utf8, 0, json_len, reinterpret_cast<jbyte *>(c_json.data()));
    if (env->ExceptionCheck()) {
        return nullptr;
    }
    c_json[json_len] = '\0';

    void *ptr = reinterpret_cast<void *>(handle);
    int capacity = 32768;
    std::vector<uint8_t> buf;
    try {
        buf.resize(capacity);
    } catch (const std::bad_alloc &) {
        throw_out_of_memory(env, "native allocation failed in applyChatTemplate");
        return nullptr;
    }
    int n = ee_llama_apply_chat_template(ptr, c_json.data(), buf.data(), capacity);
    if (n > capacity) {
        capacity = n + 1;
        try {
            buf.resize(capacity);
        } catch (const std::bad_alloc &) {
            throw_out_of_memory(env, "native allocation failed in applyChatTemplate growth");
            return nullptr;
        }
        n = ee_llama_apply_chat_template(ptr, c_json.data(), buf.data(), capacity);
    }
    if (n < 0) {
        throw_illegal_state(env, "failed to apply chat template");
        return nullptr;
    }

    int copy_len = std::min(n, capacity);
    jbyteArray res = env->NewByteArray(copy_len);
    if (res == nullptr) return nullptr;
    env->SetByteArrayRegion(res, 0, copy_len, reinterpret_cast<const jbyte *>(buf.data()));
    return res;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeTokenEosId(
    JNIEnv *env,
    jclass /*clazz*/,
    jlong handle) {
    if (env == nullptr || handle == 0) {
        if (env != nullptr) {
            throw_illegal_argument(env, "model handle is null");
        }
        return -1;
    }
    return ee_llama_token_eos_id(reinterpret_cast<void *>(handle));
}

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeKvCacheClear(
    JNIEnv * /*env*/,
    jclass /*clazz*/,
    jlong handle) {
    if (handle != 0) {
        ee_llama_kv_cache_clear_ctx(reinterpret_cast<void *>(handle));
    }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeSamplerReset(
    JNIEnv * /*env*/,
    jclass /*clazz*/,
    jlong handle) {
    if (handle != 0) {
        ee_llama_sampler_reset(reinterpret_cast<void *>(handle));
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeDecodeNGreedyIgnoreEog(
    JNIEnv *env,
    jclass /*clazz*/,
    jlong handle,
    jint n) {
    if (env == nullptr) {
        return -1;
    }
    if (handle == 0 || n <= 0) {
        return 0;
    }
    void *ptr = reinterpret_cast<void *>(handle);
    for (jint i = 0; i < n; ++i) {
        int token = ee_llama_sample_greedy(ptr);
        if (token < 0) {
            throw_illegal_state(env, "sampler returned invalid token");
            return -1;
        }
        int rc = ee_llama_decode_batch(ptr, &token, 1);
        if (rc != 0) {
            throw_illegal_state(env, decode_error_message(rc).c_str());
            return rc;
        }
    }
    return 0;
}

// =============================================================================
// mtmd (Multimodal) Primitives
// =============================================================================

extern "C" JNIEXPORT jlong JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeMtmdInit(
    JNIEnv *env,
    jclass /*clazz*/,
    jlong model_handle,
    jstring mmproj_path,
    jboolean use_gpu) {
    if (env == nullptr || model_handle == 0 || mmproj_path == nullptr) {
        if (env != nullptr) {
            throw_illegal_argument(env, "invalid arguments for mtmdInit");
        }
        return 0;
    }

    JniStringUtfChars c_path(env, mmproj_path);
    if (c_path.is_null()) {
        throw_illegal_argument(env, "failed to read mmproj path");
        return 0;
    }

    void *mtmd = ee_mtmd_init(reinterpret_cast<void *>(model_handle), c_path.get(), use_gpu);
    if (mtmd == nullptr) {
        std::string msg = "failed to load mmproj from " + std::string(c_path.get());
        throw_illegal_state(env, msg.c_str());
        return 0;
    }

    return reinterpret_cast<jlong>(mtmd);
}

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeMtmdFree(
    JNIEnv * /*env*/,
    jclass /*clazz*/,
    jlong mtmd_handle) {
    if (mtmd_handle != 0) {
        ee_mtmd_free(reinterpret_cast<void *>(mtmd_handle));
    }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeMtmdAllocGrayChunks(
    JNIEnv *env,
    jclass /*clazz*/,
    jlong mtmd_handle,
    jbyteArray text_with_marker_utf8,
    jint width,
    jint height) {
    if (env == nullptr || mtmd_handle == 0 || text_with_marker_utf8 == nullptr || width <= 0 || height <= 0) {
        if (env != nullptr) {
            throw_illegal_argument(env, "invalid arguments for mtmdAllocGrayChunks");
        }
        return nullptr;
    }

    jsize text_len = env->GetArrayLength(text_with_marker_utf8);
    std::vector<char> c_text;
    try {
        c_text.resize(static_cast<size_t>(text_len) + 1);
    } catch (const std::bad_alloc &) {
        throw_out_of_memory(env, "native allocation failed in mtmd text buffer");
        return nullptr;
    }
    env->GetByteArrayRegion(text_with_marker_utf8, 0, text_len, reinterpret_cast<jbyte *>(c_text.data()));
    if (env->ExceptionCheck()) {
        return nullptr;
    }
    c_text[text_len] = '\0';

    size_t n_tokens = 0;
    size_t image_tokens = 0;
    void *chunks = ee_mtmd_alloc_gray_chunks(
        reinterpret_cast<void *>(mtmd_handle),
        c_text.data(),
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        &n_tokens,
        &image_tokens
    );

    if (chunks == nullptr) {
        throw_illegal_state(env, "mtmd_tokenize failed (check that prompt contains <__media__> marker and fits in context)");
        return nullptr;
    }

    jlong res_data[3] = {
        reinterpret_cast<jlong>(chunks),
        static_cast<jlong>(n_tokens),
        static_cast<jlong>(image_tokens)
    };

    jlongArray result = env->NewLongArray(3);
    if (result == nullptr) {
        ee_mtmd_free_chunks(chunks);
        return nullptr;
    }
    env->SetLongArrayRegion(result, 0, 3, res_data);
    return result;
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeMtmdAllocChunksFromImage(
    JNIEnv *env,
    jclass /*clazz*/,
    jlong mtmd_handle,
    jbyteArray text_with_marker_utf8,
    jbyteArray image_bytes) {
    if (env == nullptr || mtmd_handle == 0 || text_with_marker_utf8 == nullptr || image_bytes == nullptr) {
        if (env != nullptr) {
            throw_illegal_argument(env, "invalid arguments for mtmdAllocChunksFromImage");
        }
        return nullptr;
    }

    jsize text_len = env->GetArrayLength(text_with_marker_utf8);
    std::vector<char> c_text;
    try {
        c_text.resize(static_cast<size_t>(text_len) + 1);
    } catch (const std::bad_alloc &) {
        throw_out_of_memory(env, "native allocation failed in mtmd text buffer");
        return nullptr;
    }
    env->GetByteArrayRegion(text_with_marker_utf8, 0, text_len, reinterpret_cast<jbyte *>(c_text.data()));
    if (env->ExceptionCheck()) {
        return nullptr;
    }
    c_text[text_len] = '\0';

    jsize image_len = env->GetArrayLength(image_bytes);
    if (image_len <= 0) {
        throw_illegal_argument(env, "mtmdAllocChunksFromImage: image_bytes is empty");
        return nullptr;
    }
    std::vector<unsigned char> c_image;
    try {
        c_image.resize(static_cast<size_t>(image_len));
    } catch (const std::bad_alloc &) {
        throw_out_of_memory(env, "native allocation failed in mtmd image buffer");
        return nullptr;
    }
    env->GetByteArrayRegion(image_bytes, 0, image_len, reinterpret_cast<jbyte *>(c_image.data()));
    if (env->ExceptionCheck()) {
        return nullptr;
    }

    size_t n_tokens = 0;
    size_t image_tokens = 0;
    void *chunks = ee_mtmd_alloc_chunks_from_image(
        reinterpret_cast<void *>(mtmd_handle),
        c_text.data(),
        c_image.data(),
        static_cast<size_t>(image_len),
        &n_tokens,
        &image_tokens
    );

    if (chunks == nullptr) {
        throw_illegal_state(env, "mtmd_tokenize failed (check that the image decodes and the prompt contains <__media__>)");
        return nullptr;
    }

    jlong res_data[3] = {
        reinterpret_cast<jlong>(chunks),
        static_cast<jlong>(n_tokens),
        static_cast<jlong>(image_tokens)
    };

    jlongArray result = env->NewLongArray(3);
    if (result == nullptr) {
        ee_mtmd_free_chunks(chunks);
        return nullptr;
    }
    env->SetLongArrayRegion(result, 0, 3, res_data);
    return result;
}

extern "C" JNIEXPORT jint JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeMtmdEvalChunks(
    JNIEnv *env,
    jclass /*clazz*/,
    jlong mtmd_handle,
    jlong model_handle,
    jlong chunks_ptr) {
    if (env == nullptr || mtmd_handle == 0 || model_handle == 0 || chunks_ptr == 0) {
        if (env != nullptr) {
            throw_illegal_argument(env, "invalid arguments for mtmdEvalChunks");
        }
        return -1;
    }

    int rc = ee_mtmd_eval_chunks(
        reinterpret_cast<void *>(mtmd_handle),
        reinterpret_cast<void *>(model_handle),
        reinterpret_cast<void *>(chunks_ptr)
    );

    if (rc != 0) {
        std::string msg = "mtmd_helper_eval_chunks failed with code " + std::to_string(rc);
        throw_illegal_state(env, msg.c_str());
        return rc;
    }
    return 0;
}

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_pipette_llama_LlamaCpp_nativeMtmdFreeChunks(
    JNIEnv * /*env*/,
    jclass /*clazz*/,
    jlong chunks_ptr) {
    if (chunks_ptr != 0) {
        ee_mtmd_free_chunks(reinterpret_cast<void *>(chunks_ptr));
    }
}
