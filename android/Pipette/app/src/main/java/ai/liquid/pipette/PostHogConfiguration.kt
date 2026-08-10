package ai.liquid.pipette

/**
 * PostHog project configuration, mirroring the iOS client's `PostHogConfiguration` (and this app's [ClerkConfiguration]). The project API key is
 * public by design: a `phc_…` key is write-only, able to ingest events but never to read them, so it is baked into the build as a default rather than
 * treated as a secret, exactly like the Clerk publishable key and the Sentry DSN. Both are injected as `BuildConfig` fields from
 * `app/build.gradle.kts`, overridable via `local.properties` / env.
 *
 * [isComplete] guards initialization: a blank or unsubstituted key means the build wasn't configured with a real PostHog project, so analytics stays
 * off entirely rather than initializing the SDK with garbage and pushing events at an ingest endpoint that will reject them.
 */
object PostHogConfiguration {
  val apiKey: String = BuildConfig.POSTHOG_API_KEY

  val host: String = BuildConfig.POSTHOG_HOST

  val isComplete: Boolean
    get() = isCompletePostHogConfig(apiKey, host)
}

/**
 * Pure predicate behind [PostHogConfiguration.isComplete] (no Android / `BuildConfig` deps, so it's unit-testable directly).
 *
 * A config is usable only when both values are non-blank, neither is still an unsubstituted `$(...)` placeholder, and the key carries PostHog's
 * `phc_` project-key prefix. The prefix check is what catches the genuinely dangerous mix-up: a **personal API key** (`phx_…`) is a read-write
 * credential that must never ship in a client, so refusing to initialize with one turns a leaked-secret incident into a silently disabled feature.
 */
internal fun isCompletePostHogConfig(apiKey: String, host: String): Boolean =
  apiKey.isNotBlank() &&
    !apiKey.contains("$(") &&
    apiKey.startsWith("phc_") &&
    host.isNotBlank() &&
    !host.contains("$(") &&
    host.startsWith("https://")
