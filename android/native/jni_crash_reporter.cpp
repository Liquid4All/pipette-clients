#include <jni.h>
#include <android/log.h>
#include "crash_reporter.h"

#define PIPETTE_LOG_TAG "pipette-crash"
#define PIPETTE_LOGW(...) __android_log_print(ANDROID_LOG_WARN, PIPETTE_LOG_TAG, __VA_ARGS__)

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

    bool empty() const {
        return chars_ == nullptr || chars_[0] == '\0';
    }

    JniStringUtfChars(const JniStringUtfChars &) = delete;
    JniStringUtfChars &operator=(const JniStringUtfChars &) = delete;

private:
    JNIEnv *env_;
    jstring jstr_;
    const char *chars_;
};

} // namespace

extern "C" JNIEXPORT void JNICALL
Java_ai_liquid_pipette_BenchmarkCrashReporter_nativeInit(
    JNIEnv *env,
    jclass /*clazz*/,
    jstring dsn,
    jstring environment,
    jstring release,
    jstring db_path,
    jstring envelope_dir) {
    if (env == nullptr) {
        return;
    }

    JniStringUtfChars c_dsn(env, dsn);
    JniStringUtfChars c_env(env, environment);
    JniStringUtfChars c_release(env, release);
    JniStringUtfChars c_db_path(env, db_path);
    JniStringUtfChars c_envelope_dir(env, envelope_dir);

    if (c_dsn.empty()) {
        PIPETTE_LOGW("benchmark crash reporter: empty DSN at native init: native crash capture disabled");
    }
    if (c_envelope_dir.empty()) {
        PIPETTE_LOGW("benchmark crash reporter: empty envelope_dir at native init: native crash capture disabled (crashes cannot be persisted)");
    }
    if (c_db_path.empty()) {
        PIPETTE_LOGW("benchmark crash reporter: empty db_path at native init: crashes may not persist/flush across process death");
    }

    ee_crash_reporter_init(
        c_dsn.get(),
        c_env.get(),
        c_release.get(),
        c_db_path.get(),
        c_envelope_dir.get()
    );
}
