// JNI primitives of the in-process ONNX Runtime GenAI engine (kotlinEngine), over genai's C
// API: every call returns an OgaResult, so no C++ exception reaches a JNI frame. The Kotlin
// side is ai.liquid.pipette.onnx.OnnxGenai, and the benchmarks are OnnxGenaiBenchmark.
//
// Built with PIPETTE_ONNX_GENAI_ENABLED when android/Pipette/build-onnx-genai-android.sh
// has produced the two libraries. Without it the same symbols exist and refuse, so the
// Kotlin declarations link either way.

#include <jni.h>
#include <cstdint>
#include <ctime>
#include <string>
#include <vector>

#ifndef ONNXRUNTIME_VERSION
#define ONNXRUNTIME_VERSION "unknown"
#endif

#ifndef ONNXRUNTIME_GENAI_COMMIT
#define ONNXRUNTIME_GENAI_COMMIT "unknown"
#endif

#ifdef PIPETTE_ONNX_GENAI_ENABLED
#include "ort_genai_c.h"
#endif

namespace {

void throw_illegal_state(JNIEnv *env, const std::string &message) {
    if (env == nullptr || env->ExceptionCheck()) return;
    jclass cls = env->FindClass("java/lang/IllegalStateException");
    if (cls != nullptr) {
        env->ThrowNew(cls, message.c_str());
        env->DeleteLocalRef(cls);
    }
}

#ifdef PIPETTE_ONNX_GENAI_ENABLED

class JniStringUtfChars {
public:
    JniStringUtfChars(JNIEnv *env, jstring jstr) : env_(env), jstr_(jstr), chars_(nullptr) {
        if (env_ != nullptr && jstr_ != nullptr) {
            chars_ = env_->GetStringUTFChars(jstr_, nullptr);
        }
    }

    ~JniStringUtfChars() {
        if (chars_ != nullptr) {
            env_->ReleaseStringUTFChars(jstr_, chars_);
        }
    }

    const char *get() const { return chars_; }

    JniStringUtfChars(const JniStringUtfChars &) = delete;
    JniStringUtfChars &operator=(const JniStringUtfChars &) = delete;

private:
    JNIEnv *env_;
    jstring jstr_;
    const char *chars_;
};

// Frees a genai object on every scope exit.
template <typename T, void (*Destroy)(T *)>
class Owned {
public:
    Owned() : ptr_(nullptr) {}
    ~Owned() { reset(); }

    T **out() { return &ptr_; }
    T *get() const { return ptr_; }

    T *release() {
        T *ptr = ptr_;
        ptr_ = nullptr;
        return ptr;
    }

    void reset() {
        if (ptr_ != nullptr) {
            Destroy(ptr_);
            ptr_ = nullptr;
        }
    }

    Owned(const Owned &) = delete;
    Owned &operator=(const Owned &) = delete;

private:
    T *ptr_;
};

// Together because the tokenizer is built from the model.
struct Loaded {
    OgaModel *model;
    OgaTokenizer *tokenizer;
};

// True when `result` is an error, which it throws as an IllegalStateException.
bool failed(JNIEnv *env, OgaResult *result, const char *what) {
    if (result == nullptr) return false;
    const char *error = OgaResultGetError(result);
    throw_illegal_state(env, std::string(what) + ": " + (error != nullptr ? error : "unknown error"));
    OgaDestroyResult(result);
    return true;
}

int64_t monotonic_nanos() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

#endif // PIPETTE_ONNX_GENAI_ENABLED

const char *const kNotBuilt = "this build has no ONNX Runtime GenAI engine";

} // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_liquid_pipette_onnx_OnnxGenai_nativeIsBuilt(JNIEnv *, jclass) {
#ifdef PIPETTE_ONNX_GENAI_ENABLED
    return JNI_TRUE;
#else
    return JNI_FALSE;
#endif
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_liquid_pipette_onnx_OnnxGenai_nativeOnnxRuntimeVersion(JNIEnv *env, jclass) {
    return env->NewStringUTF(ONNXRUNTIME_VERSION);
}

extern "C" JNIEXPORT jstring JNICALL
Java_ai_liquid_pipette_onnx_OnnxGenai_nativeGenaiCommit(JNIEnv *env, jclass) {
    return env->NewStringUTF(ONNXRUNTIME_GENAI_COMMIT);
}

// Through OgaConfig rather than OgaCreateModel(path): the config is where
// execution-provider options would be set.
extern "C" JNIEXPORT jlong JNICALL
Java_ai_liquid_pipette_onnx_OnnxGenai_nativeModelLoad(JNIEnv *env, jclass, jstring path) {
#ifdef PIPETTE_ONNX_GENAI_ENABLED
    JniStringUtfChars dir(env, path);
    if (dir.get() == nullptr) {
        throw_illegal_state(env, "the model path is null");
        return 0;
    }
    Owned<OgaConfig, OgaDestroyConfig> config;
    Owned<OgaModel, OgaDestroyModel> model;
    Owned<OgaTokenizer, OgaDestroyTokenizer> tokenizer;
    if (failed(env, OgaCreateConfig(dir.get(), config.out()), "ONNX Runtime GenAI config")) return 0;
    if (failed(env, OgaCreateModelFromConfig(config.get(), model.out()), "ONNX Runtime GenAI model load")) return 0;
    if (failed(env, OgaCreateTokenizer(model.get(), tokenizer.out()), "ONNX Runtime GenAI tokenizer")) return 0;
    return reinterpret_cast<jlong>(new Loaded{model.release(), tokenizer.release()});
#else
    (void)path;
    throw_illegal_state(env, kNotBuilt);
    return 0;
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_pipette_onnx_OnnxGenai_nativeModelFree(JNIEnv *, jclass, jlong handle) {
#ifdef PIPETTE_ONNX_GENAI_ENABLED
    auto *loaded = reinterpret_cast<Loaded *>(handle);
    if (loaded == nullptr) return;
    OgaDestroyTokenizer(loaded->tokenizer);
    OgaDestroyModel(loaded->model);
    delete loaded;
#else
    (void)handle;
#endif
}

// The text's token ids under the model's own tokenizer, with no chat template.
extern "C" JNIEXPORT jintArray JNICALL
Java_ai_liquid_pipette_onnx_OnnxGenai_nativeEncode(JNIEnv *env, jclass, jlong handle, jbyteArray text_utf8) {
#ifdef PIPETTE_ONNX_GENAI_ENABLED
    auto *loaded = reinterpret_cast<Loaded *>(handle);
    if (loaded == nullptr || text_utf8 == nullptr) {
        throw_illegal_state(env, "encode needs a loaded model and a text");
        return nullptr;
    }
    const jsize length = env->GetArrayLength(text_utf8);
    std::string text(static_cast<size_t>(length), '\0');
    env->GetByteArrayRegion(text_utf8, 0, length, reinterpret_cast<jbyte *>(text.data()));

    Owned<OgaSequences, OgaDestroySequences> sequences;
    if (failed(env, OgaCreateSequences(sequences.out()), "ONNX Runtime GenAI sequences")) return nullptr;
    if (failed(env, OgaTokenizerEncode(loaded->tokenizer, text.c_str(), sequences.get()), "ONNX Runtime GenAI encode")) {
        return nullptr;
    }
    const size_t count = OgaSequencesGetSequenceCount(sequences.get(), 0);
    const int32_t *ids = OgaSequencesGetSequenceData(sequences.get(), 0);
    jintArray out = env->NewIntArray(static_cast<jsize>(count));
    if (out == nullptr) return nullptr;
    if (count > 0) env->SetIntArrayRegion(out, 0, static_cast<jsize>(count), ids);
    return out;
#else
    (void)handle;
    (void)text_utf8;
    throw_illegal_state(env, kNotBuilt);
    return nullptr;
#endif
}

// One generation of exactly `decode_tokens` tokens over `prompt_ids`, timed here so the
// timed loop makes no JNI call per token. Returns three longs:
//   [0] nanoseconds over AppendTokens and the first GenerateNextToken together: the engine
//       may run the prefill in either, so neither alone is the time to the first token
//   [1] nanoseconds over the remaining decode_tokens - 1 tokens
//   [2] tokens generated
// min_length is this engine's ignore-EOS; with max_length it pins the generation's length.
extern "C" JNIEXPORT jlongArray JNICALL
Java_ai_liquid_pipette_onnx_OnnxGenai_nativeGenerate(JNIEnv *env, jclass, jlong handle, jintArray prompt_ids, jint decode_tokens) {
#ifdef PIPETTE_ONNX_GENAI_ENABLED
    auto *loaded = reinterpret_cast<Loaded *>(handle);
    if (loaded == nullptr || prompt_ids == nullptr) {
        throw_illegal_state(env, "generate needs a loaded model and a prompt");
        return nullptr;
    }
    const jsize prompt_length = env->GetArrayLength(prompt_ids);
    if (prompt_length <= 0 || decode_tokens <= 0) {
        throw_illegal_state(env, "generate needs a prompt and at least one token to generate");
        return nullptr;
    }
    std::vector<int32_t> ids(static_cast<size_t>(prompt_length));
    env->GetIntArrayRegion(prompt_ids, 0, prompt_length, ids.data());

    Owned<OgaGeneratorParams, OgaDestroyGeneratorParams> params;
    Owned<OgaGenerator, OgaDestroyGenerator> generator;
    const double total = static_cast<double>(prompt_length) + static_cast<double>(decode_tokens);
    if (failed(env, OgaCreateGeneratorParams(loaded->model, params.out()), "ONNX Runtime GenAI params")) return nullptr;
    if (failed(env, OgaGeneratorParamsSetSearchNumber(params.get(), "min_length", total), "ONNX Runtime GenAI min_length")) {
        return nullptr;
    }
    if (failed(env, OgaGeneratorParamsSetSearchNumber(params.get(), "max_length", total), "ONNX Runtime GenAI max_length")) {
        return nullptr;
    }
    if (failed(env, OgaCreateGenerator(loaded->model, params.get(), generator.out()), "ONNX Runtime GenAI generator")) {
        return nullptr;
    }

    const int64_t started = monotonic_nanos();
    if (failed(env, OgaGenerator_AppendTokens(generator.get(), ids.data(), ids.size()), "ONNX Runtime GenAI prefill")) {
        return nullptr;
    }
    if (failed(env, OgaGenerator_GenerateNextToken(generator.get()), "ONNX Runtime GenAI first token")) return nullptr;
    const int64_t first_token = monotonic_nanos();
    // A fixed count, not IsDone, which would make a silent early stop look complete.
    for (jint i = 1; i < decode_tokens; i++) {
        if (failed(env, OgaGenerator_GenerateNextToken(generator.get()), "ONNX Runtime GenAI decode")) return nullptr;
    }
    const int64_t finished = monotonic_nanos();

    const size_t sequence_length = OgaGenerator_GetSequenceCount(generator.get(), 0);
    const jlong generated = static_cast<jlong>(sequence_length) - prompt_length;
    const jlong values[3] = {first_token - started, finished - first_token, generated};
    jlongArray out = env->NewLongArray(3);
    if (out == nullptr) return nullptr;
    env->SetLongArrayRegion(out, 0, 3, values);
    return out;
#else
    (void)handle;
    (void)prompt_ids;
    (void)decode_tokens;
    throw_illegal_state(env, kNotBuilt);
    return nullptr;
#endif
}
