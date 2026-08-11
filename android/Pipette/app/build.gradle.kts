import com.ncorti.ktfmt.gradle.tasks.KtfmtCheckTask
import com.ncorti.ktfmt.gradle.tasks.KtfmtFormatTask
import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    // Compose Compiler plugin — required to enable Compose (buildFeatures.compose).
    // Pinned to AGP 9.2.1's built-in Kotlin version (2.2.10) so the compiler plugin
    // matches the Kotlin compiler doing the compilation. Compose is used ONLY to host
    // Clerk's prebuilt AuthView; the rest of the app stays XML/Material.
    id("org.jetbrains.kotlin.plugin.compose") version "2.2.10"
    id("org.jetbrains.kotlin.plugin.serialization") version "2.2.10"
    // ktfmt — Kotlin formatter in Google style. `ktfmtCheck` gates CI;
    // `ktfmtFormat` rewrites sources in place. See the ktfmt {} block at the bottom.
    alias(libs.plugins.ktfmt)
    // detekt — Kotlin static analysis (the linter half of the gate). `:app:detekt`
    // gates CI; config in config/detekt/detekt.yml. See the detekt {} block below.
    alias(libs.plugins.detekt)
    // Sentry crash/perf monitoring. `apply false` puts the plugin on the buildscript
    // classpath without applying it, so application can be conditional on whether a
    // Sentry DSN was supplied (see `crashReportingEnabled` and the apply(plugin = ...)
    // below). A plugins {} alias cannot be made conditional, which is why it is applied
    // imperatively. A build with no DSN, such as any fork, never applies it.
    //
    // Only the PLUGIN is optional. The sentry-android SDK stays an unconditional
    // dependency below, because Kotlin sources call Sentry.isEnabled() and
    // Sentry.feedback().capture(); dropping it would break compilation, not merely
    // disable a feature. It ships inert instead: with no DSN the SDK never initializes,
    // Sentry.isEnabled() stays false, and every UI surface gated on it stays hidden.
    alias(libs.plugins.sentry) apply false
}

// Build config resolved from `local.properties` (local dev) or environment
// variables (CI — set as secrets exposed to the Gradle step), kept OUT of VCS.
val localProperties = Properties().apply {
    val file = rootProject.file("local.properties")
    if (file.exists()) file.inputStream().use { load(it) }
}
fun localOrEnv(propertyName: String, envName: String): String? =
    (localProperties.getProperty(propertyName) ?: providers.environmentVariable(envName).orNull)?.trim()

// Clerk publishable keys. No key is baked into the repo: sign-in is enabled by the *presence* of a
// key and disabled by its absence, in every variant. A checkout with no key configured builds and
// runs with the auth gate open, which is what lets someone outside Liquid clone this and get a
// working app without a Clerk instance of their own.
//
// The key is public by design (the same value ships in an APK anyone can unzip), so this is not
// about secrecy. It is about a fork not silently authenticating against Liquid's instance, and
// about "no key" meaning "no auth" rather than a broken app.
//
// Point a build at an instance with either:
//   local.properties:  clerk.publishableKey.release / clerk.publishableKey.debug
//   env (CI):          CLERK_PUBLISHABLE_KEY / CLERK_PUBLISHABLE_KEY_DEBUG
val clerkPublishableKeyRelease = localOrEnv("clerk.publishableKey.release", "CLERK_PUBLISHABLE_KEY").orEmpty()
// Debug falls back to the release key so a single `clerk.publishableKey.release` opts both variants
// into the same instance, which is the common case when testing sign-in locally.
val clerkPublishableKeyDebug = localOrEnv("clerk.publishableKey.debug", "CLERK_PUBLISHABLE_KEY_DEBUG").orEmpty().ifBlank { clerkPublishableKeyRelease }

// Clerk SDK verbose logging, OFF by default and available to debug builds only. Turning it on
// makes the SDK log every Frontend-API request and response BODY (it installs an OkHttp
// HttpLoggingInterceptor at Level.BODY), which is genuinely useful for auth error codes but
// writes credentials to logcat in the clear: the email-code OTP, the session JWT, and the
// password submitted by the gate's password step. So it's an explicit opt-in a developer makes
// while debugging, not something every debug build does.
//   local.properties:  clerk.debugLogging=true
//   env (CI):          CLERK_DEBUG_LOGGING=true
val clerkDebugLogging = localOrEnv("clerk.debugLogging", "CLERK_DEBUG_LOGGING").toBoolean()

// Mirror of ClerkConfiguration.isCompleteKey, which decides at RUNTIME whether this build has auth.
// Reported and enforced from here in the same terms, so the build log and the runtime cannot
// disagree: a key that fails this is treated as no key at all on the device, however plausible it
// looks in the log. See the kdoc on isCompleteKey for why both placeholder spellings count.
fun clerkKeyIsUsable(key: String): Boolean = key.isNotBlank() && !key.contains("$(") && !key.contains("\${")

// Say which state this build is in, the same way the Sentry line below does. Without a baked-in
// default, "does this build have sign-in?" is answered by configuration rather than by the source,
// so it has to be visible in the build log: it is the only place a CI run reveals whether the
// CLERK_PUBLISHABLE_KEY secret actually arrived. A silent build is how a release ships with the
// gate open and nobody notices.
logger.lifecycle(
    when {
        clerkKeyIsUsable(clerkPublishableKeyRelease) && clerkKeyIsUsable(clerkPublishableKeyDebug) ->
            "Clerk: sign-in ENABLED for both variants (key configured)"
        clerkKeyIsUsable(clerkPublishableKeyDebug) -> "Clerk: sign-in enabled for debug only (no usable release key)"
        clerkKeyIsUsable(clerkPublishableKeyRelease) -> "Clerk: sign-in enabled for release only (no usable debug key)"
        // Set to something, but to something the runtime will reject. Never intended; verifyReleaseClerkKey fails the release for it.
        clerkPublishableKeyRelease.isNotBlank() || clerkPublishableKeyDebug.isNotBlank() ->
            "Clerk: key configured but BROKEN (unexpanded placeholder), so sign-in would be DISABLED at runtime"
        else -> "Clerk: sign-in DISABLED, auth gate open (no key supplied; set clerk.publishableKey.release or CLERK_PUBLISHABLE_KEY to enable)"
    },
)

// Sentry DSN. NO baked-in default: a build gets crash reporting only if it supplies a DSN.
//   local.properties:  sentry.dsn
//   env (CI):          SENTRY_DSN
//
// This is what makes the project fork-friendly. A fork has no SENTRY_DSN, so it compiles no
// sentry-native, applies no Sentry Gradle plugin, ships an empty io.sentry.dsn, and hides
// every Sentry-backed UI surface, without editing a line or knowing this option exists.
// Nothing points at Liquid's Sentry project by accident.
//
// The DSN is public by design (it ships in every distributed binary and can only ingest
// events), so supplying it via local.properties or a CI env var is about not aiming other
// people's builds at our project, not about secrecy.
val sentryDsn = localOrEnv("sentry.dsn", "SENTRY_DSN").orEmpty()

// Crash reporting is DERIVED from whether a DSN was supplied, then optionally forced off.
// Governs four things together, so a build can't end up half-configured: the Sentry Gradle
// plugin (mapping + source-context upload), the sentry-native library compiled into the
// :benchmark engine (forwarded to CMake through build-rust-android.sh), the native-symbol
// upload task, and the io.sentry.dsn manifest value the SDK and BenchmarkCrashReporter read.
//
// Deriving rather than defaulting ON is what satisfies "included only when a key was
// provided". The explicit switch remains, but only to force reporting OFF while a DSN is
// configured (e.g. measuring without the inproc signal handler resident):
//   local.properties:  sentry.crashReporting=false
//   env (CI):          PIPETTE_ENABLE_CRASH_REPORTING=0
// Setting it true does NOT conjure a DSN; with none supplied there is nothing to report to.
val crashReportingRaw = localOrEnv("sentry.crashReporting", "PIPETTE_ENABLE_CRASH_REPORTING")
// Three states, not two. Unset DERIVES from the DSN, which is the normal path. An explicit
// value overrides in either direction:
//   0/off/false : off even with a DSN (e.g. measuring without the inproc handler resident)
//   1/on/true   : wire Sentry in even without a DSN. Nothing can report (the manifest DSN is
//                 still empty), but the dependency graph is the one we distribute. That is
//                 exactly what ci/android-attribution.py needs, so the committed
//                 ThirdPartyLicenses.json describes our shipping build and stays stable
//                 whether or not the machine running the check has a DSN configured.
val crashReportingForced: Boolean? =
    when (crashReportingRaw?.lowercase()) {
        null, "" -> null
        "1", "on", "true" -> true
        "0", "off", "false" -> false
        else ->
            throw org.gradle.api.GradleException(
                "sentry.crashReporting / PIPETTE_ENABLE_CRASH_REPORTING must be 1/0 (on/off, true/false); got '$crashReportingRaw'",
            )
    }
val crashReportingEnabled = crashReportingForced ?: sentryDsn.isNotBlank()

// State the resolved mode once at configuration time. Deriving from a value's presence means
// the difference between "reporting" and "silently not reporting" is invisible in the source,
// so a CI log needs to say which one it built. This line is the only thing that does.
logger.lifecycle(
    when {
        crashReportingEnabled && sentryDsn.isNotBlank() -> "Sentry: ENABLED (DSN supplied)"
        // Forced on with no DSN: linked and packaged, but inert. The attribution check does this.
        crashReportingEnabled -> "Sentry: linked but INERT (forced on, no DSN, nothing will report)"
        crashReportingForced == false -> "Sentry: disabled (explicitly turned off)"
        else -> "Sentry: disabled (no DSN supplied; set sentry.dsn or SENTRY_DSN to enable)"
    },
)

// PostHog product analytics. The project API key (`phc_…`) is PUBLIC and write-only: it can
// only ingest events, never read them. So, exactly like the Clerk publishable key above and
// the Sentry DSN in AndroidManifest.xml, it is baked in as a default rather than treated as a
// secret. Without a baked-in default a distributed APK would ship analytics-disabled.
// Override to point a build at a different PostHog project:
//   local.properties:  posthog.apiKey / posthog.host
//   env (CI):          POSTHOG_API_KEY / POSTHOG_HOST
// Liquid AI PostHog project 456142, US Cloud.
val posthogApiKeyDefault = "phc_tGKa98yNH3WzFs7C2SKKisXYoKU2rLU8VkB3aYMRnyao"
val posthogApiKey = localOrEnv("posthog.apiKey", "POSTHOG_API_KEY").orEmpty().ifBlank { posthogApiKeyDefault }
// Ingest host, NOT the dashboard host (us.posthog.com): events POST to the `i` subdomain.
val posthogHost = localOrEnv("posthog.host", "POSTHOG_HOST").orEmpty().ifBlank { "https://us.i.posthog.com" }

// CI-injected build identity for monotonic versioning (e.g. Firebase App
// Distribution / Play Store, which require an ever-increasing versionCode).
//   env (CI):  PIPETTE_VERSION_NAME — base app version. Defaults to "1.0";
//              overridden with the tag's version when building from a release tag.
//   env (CI):  PIPETTE_VERSION_CODE. Monotonic integer: a fixed base plus the
//              release workflow's run number. Play's ceiling is 2_100_000_000
//              (below int32's 2_147_483_647), and its floor is the highest code
//              it has ever accepted. The base absorbs a run-number reset, which
//              GitHub triggers by renaming the workflow file; see that
//              workflow's "Resolve build identity" step for the bump rule.
//   env (CI):  PIPETTE_BUILD_TAG. Short commit tag (e.g. "g7d2a7f6") appended
//              to the base version as "1.0+g7d2a7f6", so the user-visible
//              version names the commit it was cut from.
//   env (CI):  PIPETTE_BUILD_VERSION. The version this build is PUBLISHED as —
//              ci/version.sh's output, which is also the GitHub release's tag
//              and name (e.g. "2026.08.1-3-ga1b2c3d4ab"). Surfaces as
//              BuildConfig.BUILD_VERSION and is submitted verbatim as
//              `client_version`, so a warehouse row maps to a downloadable
//              release by equality rather than by anyone's parsing rule.
// Uninjected/local builds fall back to versionCode 1 / versionName "1.0" /
// BUILD_VERSION "dev".
val baseVersionName = localOrEnv("pipette.versionName", "PIPETTE_VERSION_NAME") ?: "1.0"
val ciVersionCode = localOrEnv("pipette.versionCode", "PIPETTE_VERSION_CODE")?.toIntOrNull()
val ciBuildTag = localOrEnv("pipette.buildTag", "PIPETTE_BUILD_TAG")?.takeIf { it.isNotBlank() }

// Kept off versionName deliberately. versionName is user-visible and, on the
// Play path, constrained by what the store will accept; `client_version` wants
// the release string untouched. Two fields, so neither has to compromise.
// "dev" matches what the Rust client's build.rs defaults to, so a local build of
// either reads the same way in the warehouse.
val buildVersion = localOrEnv("pipette.buildVersion", "PIPETTE_BUILD_VERSION")?.takeIf { it.isNotBlank() } ?: "dev"

// Real release keystore, if configured (local.properties / env); otherwise null and
// the release build falls back to the shared debug keystore (see buildTypes.release).
//   local.properties:  release.storeFile / release.storePassword
//                      release.keyAlias / release.keyPassword
//   env (CI):          RELEASE_STORE_FILE / RELEASE_STORE_PASSWORD
//                      RELEASE_KEY_ALIAS / RELEASE_KEY_PASSWORD
// Resolve ALL release-signing values; only treat the keystore as configured when
// every one is present (and the file exists). A partial config would otherwise
// create a broken signingConfig with empty passwords; instead we leave it null and
// the release build falls back to the debug keystore (see buildTypes.release).
val releaseStorePassword = localOrEnv("release.storePassword", "RELEASE_STORE_PASSWORD")
val releaseKeyAlias = localOrEnv("release.keyAlias", "RELEASE_KEY_ALIAS")
val releaseKeyPassword = localOrEnv("release.keyPassword", "RELEASE_KEY_PASSWORD")
val releaseStoreFile = localOrEnv("release.storeFile", "RELEASE_STORE_FILE")
    ?.let { rootProject.file(it) }
    ?.takeIf {
        it.exists() &&
            !releaseStorePassword.isNullOrBlank() &&
            !releaseKeyAlias.isNullOrBlank() &&
            !releaseKeyPassword.isNullOrBlank()
    }

val buildRustAndroidArm64 by tasks.registering(Exec::class) {
    group = "build"
    description = "Build and copy the Rust llama.cpp bridge into jniLibs for arm64-v8a."
    commandLine(rootProject.file("build-rust-android.sh").absolutePath)

    // Declare inputs/outputs so Gradle can mark the task UP-TO-DATE and skip the
    // whole CMake + cargo + stage/strip pipeline on no-op rebuilds (e.g. a
    // Kotlin-only edit). Covers the native sources the script consumes; a
    // vendored-submodule bump or a cross-crate change still needs --rerun-tasks
    // (or a clean), which is rare.
    val repoRoot = rootProject.file("../..")
    inputs.file(rootProject.file("build-rust-android.sh"))
    inputs.dir(File(repoRoot, "crates/pipette-android"))
    inputs.dir(File(repoRoot, "native"))
    // CMake is the build's source of truth now — track the root project, presets,
    // and the shared cmake/ modules so an edit to any of them re-runs the build.
    inputs.file(File(repoRoot, "CMakeLists.txt"))
    inputs.file(File(repoRoot, "CMakePresets.json"))
    inputs.dir(File(repoRoot, "cmake"))
    inputs.dir(File(repoRoot, "vendor/llama.cpp/ggml"))
    inputs.dir(File(repoRoot, "vendor/llama.cpp/src"))
    inputs.dir(File(repoRoot, "vendor/llama.cpp/include"))
    inputs.dir(File(repoRoot, "vendor/llama.cpp/tools/mtmd"))
    // pipette_bridge also includes llama.cpp's bundled deps (nlohmann/json).
    inputs.dir(File(repoRoot, "vendor/llama.cpp/vendor"))
    // sentry-native is linked into the engine, so its source is a build input too — otherwise a submodule bump could serve a stale UP-TO-DATE build.
    // Only when crash reporting is on: with it off the submodule may legitimately not be checked out at all, and declaring a missing dir as an input
    // would make configuration depend on a path that does not exist.
    if (crashReportingEnabled) {
        inputs.dir(File(repoRoot, "vendor/sentry-native/src"))
        inputs.dir(File(repoRoot, "vendor/sentry-native/include"))
        inputs.file(File(repoRoot, "vendor/sentry-native/CMakeLists.txt"))
    }
    inputs.property("kleidiai", providers.environmentVariable("PIPETTE_ENABLE_KLEIDIAI").orElse(""))
    // Flipping crash reporting changes what the engine links, so it must invalidate the cached task the same way the KleidiAI knob does. Without this
    // an OFF-then-ON rebuild would be served UP-TO-DATE and silently ship the previous configuration's engine.
    inputs.property("crashReporting", crashReportingEnabled)
    // The script re-parses this for CMake; pass the normalized boolean rather than the raw property so Gradle and CMake cannot disagree about how a
    // value like "off" or "TRUE" was interpreted.
    environment("PIPETTE_ENABLE_CRASH_REPORTING", if (crashReportingEnabled) "1" else "0")
    outputs.dir(layout.buildDirectory.dir("generated/rustJniLibs/arm64-v8a"))
    // The staging step also emits UNSTRIPPED native symbols here (see cmake/stage_android_jnilibs.cmake). Declare it as an output so it's tracked by
    // up-to-date checks and restored from the build cache — otherwise a warm-cache CI run skips this task and uploadBenchmarkNativeSymbols finds no
    // symbols to upload.
    outputs.dir(layout.buildDirectory.dir("generated/rustNativeSymbols/arm64-v8a"))
}

android {
    namespace = "ai.liquid.pipette"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "ai.liquid.pipette"
        minSdk = 31
        targetSdk = 36
        // Monotonic when CI injects PIPETTE_VERSION_CODE (base + run number);
        // otherwise 1 for local/uninjected builds. versionName is deliberately
        // NOT derived from it: a run-number reset would make it ambiguous as
        // well as colliding the code. Local builds keep the bare base version.
        versionCode = ciVersionCode ?: 1
        versionName = ciBuildTag?.let { "$baseVersionName+$it" } ?: baseVersionName

        // The published release this APK is, submitted verbatim as
        // `client_version` (see LocalStorage.kt). The release build type takes
        // it as-is; debug appends "-debug" below, mirroring versionNameSuffix,
        // so a developer's submissions stay distinguishable from a release's.
        buildConfigField("String", "BUILD_VERSION", "\"$buildVersion\"")

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk {
            abiFilters += "arm64-v8a"
        }

        // Release/default Clerk key (production "Liquid AI" instance). Resolved from
        // local.properties / env above; debug overrides it below. Read via
        // BuildConfig.CLERK_PUBLISHABLE_KEY.
        buildConfigField("String", "CLERK_PUBLISHABLE_KEY", "\"$clerkPublishableKeyRelease\"")

        // Hard-false for the default (so RELEASE can never log request bodies, whatever the
        // property says); the debug build type below substitutes the opt-in value.
        buildConfigField("boolean", "CLERK_DEBUG_LOGGING", "false")

        // PostHog analytics. Both variants report to the same project; events carry an
        // `app_environment` property (debug/production, derived from BuildConfig.DEBUG in
        // Analytics.kt) so dev traffic can be filtered out, the PostHog analogue of the
        // per-build-type `io.sentry.environment` manifest value.
        buildConfigField("String", "POSTHOG_API_KEY", "\"$posthogApiKey\"")
        buildConfigField("String", "POSTHOG_HOST", "\"$posthogHost\"")

        // Sentry DSN for the io.sentry.dsn manifest meta-data, which both the JVM SDK's
        // auto-init and BenchmarkCrashReporter (for the `:benchmark` native reporter) read.
        // Set here rather than per-build-type because both variants report to the same
        // project; they are separated by io.sentry.environment instead (see buildTypes).
        // Empty when sentry.crashReporting=false, which no-ops both.
        manifestPlaceholders["sentryDsn"] = sentryDsn
    }

    signingConfigs {
        // Shared debug keystore checked into the repo (android/Pipette/debug.keystore)
        // so every developer's debug build is signed with the SAME key. Without it,
        // each machine falls back to its own ~/.android/debug.keystore, so a debug
        // APK built by one developer can't update one already installed from another
        // (signature mismatch forces an uninstall/reinstall). Standard Android debug
        // credentials — this key is intentionally public and NOT for release builds.
        getByName("debug") {
            storeFile = rootProject.file("debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
        // Real release keystore, only when FULLY configured (releaseStoreFile is
        // non-null only when the file + all three credentials are present — see
        // above). When absent, the release build falls back to the debug keystore
        // below so the production-Clerk variant is still installable for testing.
        if (releaseStoreFile != null) {
            create("release") {
                storeFile = releaseStoreFile
                storePassword = releaseStorePassword
                keyAlias = releaseKeyAlias
                keyPassword = releaseKeyPassword
            }
        }
    }

    buildTypes {
        debug {
            // Distinct applicationId so the debug (dev-Clerk) build installs ALONGSIDE
            // the release (prod-Clerk) build instead of replacing it — lets you compare
            // dev vs production login side by side. `android:process=":benchmark"` is
            // relative, so it becomes ai.liquid.pipette.debug:benchmark; the main-process
            // guard compares getProcessName() to the runtime packageName, so it still holds.
            applicationIdSuffix = ".debug"
            versionNameSuffix = "-debug"
            // The `client_version` counterpart of versionNameSuffix above: that
            // suffix no longer reaches the wire, since client_version is
            // BUILD_VERSION rather than VERSION_NAME. Without this a debug
            // build of a release commit would submit as that release.
            buildConfigField("String", "BUILD_VERSION", "\"$buildVersion-debug\"")
            signingConfig = signingConfigs.getByName("debug")
            // Sentry environment (read by io.sentry.environment in the manifest): keep
            // dev/test crashes + feedback out of the production environment. Mirrors the
            // iOS SentryConfiguration #if DEBUG branch.
            manifestPlaceholders["sentryEnvironment"] = "debug"
            // Debug builds use the development Clerk instance (set
            // clerk.publishableKey.debug in local.properties, or the
            // CLERK_PUBLISHABLE_KEY_DEBUG env var in CI) so test emails (`+clerk_test`)
            // verify with the fixed code 424242 and we never touch production while
            // developing. Falls back to the release key when unset.
            buildConfigField("String", "CLERK_PUBLISHABLE_KEY", "\"$clerkPublishableKeyDebug\"")
            // Verbose Clerk logging, opt-in per developer (clerk.debugLogging). See the
            // credential-logging note where clerkDebugLogging is resolved.
            buildConfigField("boolean", "CLERK_DEBUG_LOGGING", "$clerkDebugLogging")
        }
        release {
            // Use the real release keystore when configured; otherwise fall back to
            // the shared debug keystore so the release/production-Clerk variant is
            // still installable for dev-vs-prod login testing (no release keystore yet).
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
            // Sentry environment (read by io.sentry.environment in the manifest).
            manifestPlaceholders["sentryEnvironment"] = "production"
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlin {
        jvmToolchain(17)
    }
    buildFeatures {
        viewBinding = true
        aidl = true
        // Compose is enabled ONLY to host Clerk's prebuilt AuthView in a ComposeView
        // (the auth gate). The rest of the app stays XML/Material views.
        compose = true
        buildConfig = true
    }
    packaging {
        jniLibs {
            // Extract native libs to disk on install (extractNativeLibs=true).
            // The CPU-backend variant loader (native_loader.cpp) discovers the
            // sibling libggml-cpu-android_*.so via dladdr + a directory scan,
            // which needs the libs on the filesystem rather than mmap'd inside
            // the APK.
            useLegacyPackaging = true
        }
    }
    sourceSets {
        getByName("main") {
            jniLibs.directories.add(
                layout.buildDirectory.dir("generated/rustJniLibs").get().asFile.absolutePath,
            )
        }
    }
    testOptions {
        unitTests {
            // Robolectric reads merged manifest/resources for the sandboxed
            // application under test (LocalStoragePayloadTest).
            isIncludeAndroidResources = true
        }
    }
}

tasks.configureEach {
    if (name.matches(Regex("merge[A-Z].*JniLibFolders"))) {
        dependsOn(buildRustAndroidArm64)
    }
}

dependencies {
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.core.splashscreen)
    implementation(libs.androidx.datastore.preferences)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)
    implementation(libs.androidx.work.runtime.ktx)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.material)
    // Clerk auth gate (F3): SDK core only. The custom email-code UI lives in the
    // Compose app (AuthGateScreen), so the prebuilt clerk-android-ui AuthView is no
    // longer needed.
    implementation(libs.clerk.android.api)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.material3)
    // Compose Preview tooling (versions from the BOM): -preview for @Preview annotations,
    // -tooling for the Android Studio preview renderer (debug-only, kept out of release).
    implementation(libs.androidx.compose.ui.tooling.preview)
    debugImplementation(libs.androidx.compose.ui.tooling)
    // Sentry SDK. The Gradle plugin would auto-install this, but we declare it
    // explicitly so Sentry.feedback().capture() (the in-app feedback flow) resolves at
    // compile time. Pinned to the plugin's bundled SdkVersion so autoInstall skips it.
    implementation(libs.sentry.android)
    // The integrations the plugin's autoInstall would add implicitly, declared explicitly and
    // gated on the same condition as everything else Sentry: a build with no DSN ships none of
    // them. Explicit rather than autoInstalled so the shipped set is visible in the build file
    // and pinned to sentryAndroid, instead of varying with plugin application.
    if (crashReportingEnabled) {
        implementation(libs.sentry.android.fragment)
        implementation(libs.sentry.android.navigation)
        implementation(libs.sentry.android.sqlite)
        implementation(libs.sentry.compose.android)
        implementation(libs.sentry.kotlin.extensions)
        implementation(libs.sentry.okhttp)
    }
    implementation(libs.androidx.compose.material.icons.core)
    // activity-compose: setContent for the Compose launcher (ComposeMainActivity).
    implementation(libs.androidx.activity.compose)
    // Per-screen ViewModels surfaced to composables + collectAsStateWithLifecycle.
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)

    // Navigation 3 — Compose-first backstack navigation for the tabbed screens.
    implementation(libs.androidx.navigation3.runtime)
    implementation(libs.androidx.navigation3.ui)
    implementation(libs.kotlinx.serialization.core)
    // Ed25519 for the device signing identity (Secrets.kt); see the version catalog.
    implementation(libs.tink.android)
    // PostHog product analytics: manual events only, initialized in the main process
    // only (see Analytics.kt / PipetteApp).
    implementation(libs.posthog.android)
    testImplementation(libs.json.java)
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    // Robolectric: exercise Context-dependent code (LocalStorage.writePayload,
    // DeviceInfo) on the JVM without an emulator — the real production method
    // runs against a sandboxed app filesystem and shadowed system services.
    testImplementation(libs.robolectric)
    testImplementation(libs.androidx.test.core)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
}

// Format all Kotlin sources (main + test + androidTest) in Google style. Run
// `./gradlew :app:ktfmtFormat` to fix locally; CI runs `:app:ktfmtCheck` to gate.
// maxWidth is bumped from ktfmt's default 100 to 150 so the line-length budget
// matches detekt's MaxLineLength (config/detekt/detekt.yml) — the two tools agree
// on when a line is "too long" instead of ktfmt wrapping at 100 while detekt
// allows 150. googleStyle() only sets indents/trailing-commas, so setting maxWidth
// after it composes cleanly.
ktfmt {
    googleStyle()
    maxWidth.set(150)
}

// ktfmt-gradle (0.25.0) discovers Android source sets only inside its
// `plugins.withId("kotlin-android")` hook. This module uses AGP 9's built-in
// Kotlin support instead (no org.jetbrains.kotlin.android applied — see the
// compose plugin note above), so that hook never fires, the plugin finds zero
// source sets, and its generated :app:ktfmtCheck / :app:ktfmtFormat tasks become
// silent no-ops that pass on any input. (Verified upstream 0.26.0 doesn't fix
// this — it still gates Android discovery behind kotlin-android and only adds
// .gradle.kts *script* checking; same gap as ktlint-gradle #1008.) So register
// the source tasks explicitly over app/src. The plugin's
// `tasks.withType(KtfmtBaseTask).configureEach {}` still applies googleStyle() and
// the ktfmt classpath to these, so they behave exactly like the auto-generated
// per-source-set tasks would. Wired into the conventional ktfmtCheck/ktfmtFormat
// aggregators so `:app:ktfmtCheck` (CI) and `:app:ktfmtFormat` (local) keep working.
val ktfmtCheckAppSources by
    tasks.registering(KtfmtCheckTask::class) {
        source = fileTree("src")
        include("**/*.kt")
    }
val ktfmtFormatAppSources by
    tasks.registering(KtfmtFormatTask::class) {
        source = fileTree("src")
        include("**/*.kt")
    }
tasks.named("ktfmtCheck") { dependsOn(ktfmtCheckAppSources) }
tasks.named("ktfmtFormat") { dependsOn(ktfmtFormatAppSources) }

// Clerk key sanity check for release artifacts. Absence is a valid configuration (no key means the
// auth gate is open, see the key resolution at the top of this file), so this no longer requires a
// key. What it still catches is a key that was *meant* to arrive and got mangled: a truncated
// secret, a quoted-empty string, a placeholder that never expanded. Those would otherwise reach
// Clerk.initialize as garbage and fail at runtime on a user's device instead of here.
//
// It also announces the auth-disabled case, because a release built with no key is right for a fork
// and almost certainly wrong for us. That distinction can't be made from inside Gradle, so this
// warns and CI asserts (see the "Verify the Clerk key" step in android-release-distribution.yml,
// which runs only in the canonical repo).
//
// AGP 9 only generates unit tests for the debug variant, so this can't be asserted from
// testDebugUnitTest and is wired onto the release artifact tasks instead.
val verifyReleaseClerkKey by
    tasks.registering {
        // Capture the resolved key into a local String (not the script-level val) so the doLast action
        // stays configuration-cache compatible — capturing the outer val would drag the script in.
        val releaseKey = clerkPublishableKeyRelease
        // Resolved here, at configuration time, for the same reason: calling clerkKeyIsUsable from
        // inside doLast would capture the build script itself.
        val releaseKeyUsable = clerkKeyIsUsable(releaseKey)
        val warn = logger
        doLast {
            if (releaseKey.isBlank()) {
                warn.warn(
                    "Clerk: no publishable key configured, so this release ships with sign-in DISABLED and the auth gate open. Set " +
                        "clerk.publishableKey.release in local.properties or CLERK_PUBLISHABLE_KEY in the environment to enable it."
                )
            } else {
                // Prefix AND the runtime's own rule. Checking only the prefix would pass a value like
                // `pk_$(CLERK_PUBLISHABLE_KEY)`, which the runtime predicate rejects, so the release would build clean
                // and then ship with sign-in silently disabled: the exact outcome this task exists to prevent.
                require(releaseKey.startsWith("pk_") && releaseKeyUsable) {
                    "Release CLERK_PUBLISHABLE_KEY is set to '$releaseKey', which is not a usable Clerk publishable key: it must start with 'pk_' " +
                        "and must not still contain an unexpanded '\$(...)' or '\${...}' placeholder. Leave it unset to build with sign-in " +
                        "disabled; a malformed key is always a mistake."
                }
            }
        }
    }
// The two tasks that produce a shippable release artifact: `assembleRelease` (APK, for Firebase App Distribution) and `bundleRelease` (AAB, for Play).
// Declared once so every release-time hook below wires to the same set; adding a third release task must not silently wire only half of them.
val releaseArtifactTasks = tasks.matching { it.name == "assembleRelease" || it.name == "bundleRelease" }

releaseArtifactTasks.configureEach { dependsOn(verifyReleaseClerkKey) }

// Deriving crash reporting from the DSN's presence introduces one regression risk: an official
// release built without SENTRY_DSN wired ships with no crash capture, and nothing fails. The
// `Sentry: disabled` line in the log is easy to miss in a few thousand lines of CI output.
//
// So fail the release outright in that case, but ONLY for an official build, identified by
// CI having injected PIPETTE_VERSION_CODE. That variable is set by the release workflow and by
// nothing else, so a fork (or a local `assembleRelease`) is never blocked by a secret it has
// no way to hold. No new knob: it reuses a signal the build already depends on.
val verifyReleaseSentryDsn by
    tasks.registering {
        // Capture into locals so the doLast action stays configuration-cache compatible rather
        // than dragging the build script in (same reason as verifyReleaseClerkKey above).
        val isOfficialBuild = ciVersionCode != null
        val haveDsn = sentryDsn.isNotBlank()
        val forcedOff = crashReportingForced == false
        doLast {
            require(!isOfficialBuild || haveDsn || forcedOff) {
                "This is an official release build (PIPETTE_VERSION_CODE is set) but no Sentry DSN was configured, so it would " +
                    "ship with crash reporting silently disabled. Set the SENTRY_DSN secret on the release workflow, or pass " +
                    "PIPETTE_ENABLE_CRASH_REPORTING=0 to state that a telemetry-free release is intended."
            }
        }
    }

releaseArtifactTasks.configureEach { dependsOn(verifyReleaseSentryDsn) }

// detekt — static analysis / linter. ktfmt owns formatting; detekt's `formatting`
// ruleset (a ktlint wrapper) is deliberately left off the classpath (no
// `detektPlugins(...)` dependency), so the two never disagree on style. Config is
// the curated rule set shared with leap-android-sdk (config/detekt/detekt.yml),
// whose MaxLineLength matches ktfmt's maxWidth (150). The detekt Gradle plugin
// wires its per-variant tasks (detektMain/detektDebug/...) through the Kotlin
// Android plugin, which AGP 9's built-in Kotlin never applies — so, like ktfmt
// above, point the top-level `detekt` task at app/src explicitly so `:app:detekt`
// actually analyzes our sources (main + test + androidTest) instead of nothing.
detekt {
    config.setFrom(rootProject.file("config/detekt/detekt.yml"))
    source.setFrom(files("src"))
    // Grandfather the findings that already exist in the codebase so the gate
    // turns red only on NEWLY introduced issues (leap-android-sdk does the same
    // per module). Regenerate with `./gradlew :app:detektBaseline` after a
    // deliberate cleanup; never to silence a fresh finding in review.
    baseline = file("detekt-baseline.xml")
}

// Sentry auth token for build-time uploads (ProGuard mappings + source context).
// Resolved like the Clerk keys above — local.properties: sentry.authToken, or
// SENTRY_AUTH_TOKEN in CI. Kept OUT of VCS (sentry.properties is gitignored too).
val sentryAuthToken = localOrEnv("sentry.authToken", "SENTRY_AUTH_TOKEN")

// Apply the Sentry Gradle plugin only when crash reporting is on. Declared `apply false`
// in the plugins {} block above, so it is on the classpath but inert until applied here.
//
// The typed `sentry { }` accessor is generated only for plugins applied via `plugins {}`,
// so with an imperative apply the extension has to be configured by type instead. Nothing
// else about the configuration changes.
if (crashReportingEnabled) {
    apply(plugin = "io.sentry.android.gradle")

    extensions.configure<io.sentry.android.gradle.extensions.SentryPluginExtension>("sentry") {
        org.set("liquid-ai")
        // MUST be the project that owns the DSN in AndroidManifest.xml (io.sentry.dsn) —
        // mappings/source-context upload here, while crashes ingest via the DSN, so a slug
        // that resolves to a different project would leave crashes unsymbolicated.
        projectName.set("pipette-client-android")
        // Only set when present — `set(null)` would override the plugin's own token
        // resolution, and a tokenless local build must still configure cleanly.
        sentryAuthToken?.let { authToken.set(it) }

        // Uploads source code so Sentry shows it inline in stack traces. The upload tasks
        // require an auth token, so only enable when one is configured — otherwise a local
        // build with no token would fail. (Disable entirely if you don't want to expose sources.)
        includeSourceContext.set(!sentryAuthToken.isNullOrBlank())

        // Performance tracing is off (see the manifest sample-rate 0). Also disable the
        // plugin's auto-instrumentation so no span-wrapping bytecode is injected around
        // DB/file/OkHttp IO — this app benchmarks on-device LLMs and that instrumentation
        // would add overhead to the very code paths being measured.
        tracingInstrumentation {
            enabled.set(false)
        }
    }
}

// Upload the UNSTRIPPED native debug symbols so `:benchmark` native-crash backtraces symbolicate in Sentry.
//
// The APK ships STRIPPED libs (small); the staging step (cmake/stage_android_jnilibs.cmake) keeps unstripped copies with the same GNU build-id under
// rustNativeSymbols/. Sentry matches them to the crashing image by that build-id. The Sentry Gradle plugin's own `uploadNativeSymbols` only understands
// AGP `externalNativeBuild` outputs, which this CMake-on-top build doesn't use — so we upload via sentry-cli directly.
//
// Best-effort (mirrors the iOS dSYM upload phase): a missing sentry-cli, an empty symbol dir, or an upload error WARNS but never fails the build.
// The auth token is read by sentry-cli from the ambient environment (CI sets SENTRY_AUTH_TOKEN on the release step) — it is NEVER passed through
// Gradle, so it can't be serialized into the on-disk configuration cache. Runs after `assembleRelease` and `bundleRelease`, only when the token is
// present in the env; can be run explicitly with `SENTRY_AUTH_TOKEN` exported (locally, also honors `~/.sentryclirc`).
val nativeSymbolsDir = layout.buildDirectory.dir("generated/rustNativeSymbols/arm64-v8a")
val uploadBenchmarkNativeSymbols =
    tasks.register<Exec>("uploadBenchmarkNativeSymbols") {
        group = "sentry"
        description = "Upload :benchmark native debug symbols to Sentry (best-effort; needs SENTRY_AUTH_TOKEN in the env + sentry-cli)."
        dependsOn(buildRustAndroidArm64)
        inputs.dir(nativeSymbolsDir)
        // Keep this a Provider<Boolean> (configuration-cache-serializable, unlike a build-script reference) and resolve it INSIDE onlyIf, so the
        // token-present decision is made at execution time from the current env — never frozen into a reused configuration-cache entry from an earlier
        // (e.g. no-token) run. require NON-BLANK, since GitHub substitutes an empty string for an unset `${{ secrets.SENTRY_AUTH_TOKEN }}`.
        val hasToken = providers.environmentVariable("SENTRY_AUTH_TOKEN").map { it.isNotBlank() }.orElse(false)
        // Also gated on crash reporting: with it off there are no unstripped sentry-native symbols to upload (the lib was never built), so running
        // sentry-cli could only warn about an empty dir. Captured into a local so the lambda stays configuration-cache friendly.
        val reportingOn = crashReportingEnabled
        onlyIf { reportingOn && hasToken.get() }
        val cli = localOrEnv("sentry.cli.path", "SENTRY_CLI") ?: "sentry-cli"
        val symbolsPath = nativeSymbolsDir.get().asFile.absolutePath
        // A shell wrapper degrades every failure (missing CLI, empty dir, upload error) to a warning + exit 0, so symbol upload can never fail the
        // release. It also asserts the dir has files, so a staging regression that empties it is visible in the log rather than a silent no-op. $1/$2
        // are shell positionals (the cli path and symbols dir) — passed as args so no secret and no Kotlin interpolation enters the script.
        commandLine(
            "sh",
            "-c",
            """
            if ! command -v "$1" >/dev/null 2>&1; then echo "warning: sentry-cli ($1) not on PATH — skipping native symbol upload"; exit 0; fi
            if [ -z "$(ls -A "$2" 2>/dev/null)" ]; then echo "warning: no native symbols in $2 — skipping upload (staging may have produced none)"; exit 0; fi
            "$1" debug-files upload --org liquid-ai --project pipette-client-android "$2" || echo "warning: native symbol upload failed (non-fatal)"
            """
                .trimIndent(),
            "upload-native-symbols", // $0
            cli, // $1
            symbolsPath, // $2
        )
    }

// Auto-upload on release assembly OR bundling (best-effort; no-ops without SENTRY_AUTH_TOKEN in the env, so local release builds are unaffected).
// Covering `bundleRelease` matters because a Play-only CI run builds that task alone: without it, a bundle shipped to Play would carry no uploaded
// native debug symbols and `:benchmark` native crashes would not symbolicate.
releaseArtifactTasks.configureEach { finalizedBy(uploadBenchmarkNativeSymbols) }
