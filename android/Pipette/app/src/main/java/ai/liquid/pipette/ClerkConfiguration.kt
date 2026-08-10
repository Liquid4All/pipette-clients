package ai.liquid.pipette

/**
 * Clerk publishable-key configuration, mirroring the iOS client's `ClerkConfiguration`. The key is public by design (the same value ships inside an
 * APK anyone can unzip) and is injected as a `BuildConfig` field from `app/build.gradle.kts`.
 *
 * [isComplete] decides whether this build has auth at all. No key configured is a supported configuration, not a broken one: the SDK is never
 * initialized, `AppContainer.clerkAuth` stays null, and the gate opens. That is what lets a checkout with no Clerk instance build and run.
 */
object ClerkConfiguration {
  val publishableKey: String = BuildConfig.CLERK_PUBLISHABLE_KEY

  val isComplete: Boolean
    get() = isCompleteKey(publishableKey)
}

/**
 * Pure predicate behind [ClerkConfiguration.isComplete] (no Android / `BuildConfig` deps, so it's unit-testable directly): a key is usable only when
 * it's non-blank and not still an unsubstituted placeholder.
 *
 * The placeholder case is the one worth keeping separate from blank. `$(CLERK_PUBLISHABLE_KEY)` means a substitution was set up and did not run,
 * which is a broken build rather than an unconfigured one; treating it as a usable key would hand that literal to `Clerk.initialize`.
 *
 * Both placeholder spellings are rejected because the key reaches this build from two directions: `$(...)` is Xcode's xcconfig syntax, which the iOS
 * client's `Info.plist` uses and this predicate mirrors, and `${...}` is what an unexpanded Gradle or shell reference looks like. Either one arriving
 * intact means substitution failed somewhere upstream.
 *
 * `verifyReleaseClerkKey` in `app/build.gradle.kts` and the "Verify the Clerk key" step in `android-release-distribution.yml` apply the same rule, so
 * a key this predicate would reject fails the build instead of quietly disabling sign-in. The rule is stated three times in three languages on
 * purpose: Gradle and a workflow's shell cannot call into this. Change one, change all three.
 */
internal fun isCompleteKey(key: String): Boolean = key.isNotBlank() && !key.contains("$(") && !key.contains("\${")
