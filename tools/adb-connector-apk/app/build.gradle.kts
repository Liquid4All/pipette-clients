import com.ncorti.ktfmt.gradle.tasks.KtfmtCheckTask
import com.ncorti.ktfmt.gradle.tasks.KtfmtFormatTask

plugins {
    id("com.android.application")
    // Same formatter and version as android/Pipette, so one style covers both Android builds.
    id("com.ncorti.ktfmt.gradle") version "0.25.0"
}

// Injected by CI (the code is the build's UTC hour as yyMMddHH, so it rises with time). A local
// build reports "dev", which tells you a phone runs someone's working tree, not a release.
val injectedVersionName = System.getenv("PIPETTE_ADB_CONNECTOR_VERSION_NAME")?.takeIf { it.isNotBlank() }
val injectedVersionCode = System.getenv("PIPETTE_ADB_CONNECTOR_VERSION_CODE")?.toIntOrNull()

// Reused rather than copied so the repo has exactly one debug key. Asserted below: a silent fall
// back to ~/.android/debug.keystore would produce an APK that cannot update the previous one.
val sharedDebugKeystore = rootProject.file("../../android/Pipette/debug.keystore")

android {
    namespace = "ai.liquid.pipette.adbconnector"
    compileSdk { version = release(36) }

    defaultConfig {
        applicationId = "ai.liquid.pipette.adbconnector"
        // adb_wifi_enabled (wireless debugging) arrived in Android 11; below that there is
        // nothing for this app to turn on.
        minSdk = 30
        targetSdk = 36
        versionCode = injectedVersionCode ?: 1
        versionName = injectedVersionName ?: "dev"
    }

    signingConfigs {
        // Pinned so every build signs with the same key. AGP's per-machine key on ephemeral CI
        // runners would make each release refuse to update the last
        // (`INSTALL_FAILED_UPDATE_INCOMPATIBLE`), and the forced uninstall drops the
        // WRITE_SECURE_SETTINGS grant and returns the package to FLAG_STOPPED, disarming the boot
        // receiver. Public Android debug credentials: this is a lab tool, never a distributed release.
        getByName("debug") {
            storeFile = sharedDebugKeystore
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
    }

    buildTypes {
        getByName("debug") { signingConfig = signingConfigs.getByName("debug") }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

// Outside android {} so it runs even for tasks that never reach the signing step.
require(sharedDebugKeystore.isFile) {
    "missing shared debug keystore at ${sharedDebugKeystore.path}. It pins the signing key so " +
        "builds can update each other in place; see the signingConfigs comment."
}

dependencies {
    // The only dependency, for waiting between two settings writes without blocking a thread.
    // Version matches android/Pipette's catalog. No AndroidX: framework widgets cover the UI.
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
}

ktfmt {
    googleStyle()
    maxWidth.set(150)
}

// ktfmt-gradle finds Android source sets only in its `plugins.withId("kotlin-android")` hook,
// which AGP 9's built-in Kotlin never fires, so ktfmtCheck would pass on any input. Same
// workaround as android/Pipette: register the source tasks explicitly.
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
