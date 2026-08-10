package ai.liquid.pipette

import java.net.URI

/** Which collector a registration is aimed at (iOS `CollectorEndpoint` / `CollectorEndpointOption` parity). */
object CollectorEndpoint {
  private const val HTTPS = "https"

  /**
   * The canonical form of a hand-typed collector URL, or null when it isn't a usable collector.
   *
   * A bare host gains an `https://` prefix; another scheme (including plain `http`) is rejected rather than silently upgraded, because a downgraded
   * collector would ship benchmark payloads in the clear. Credentials, query, and fragment mean nothing to the management API, and accepting them
   * would make two URLs for one collector — so a mistyped endpoint fails the form here rather than the first network call.
   */
  fun normalizedCustomUrl(value: String): String? {
    val trimmed = value.trim()
    val candidate = if (trimmed.contains("://")) trimmed else "$HTTPS://$trimmed"
    // A URL we can't parse is a URL we won't register against.
    val uri = runCatching { URI(candidate) }.getOrNull() ?: return null
    val host = uri.host?.lowercase().orEmpty()
    val usable = host.isNotEmpty() && uri.scheme?.lowercase() == HTTPS && uri.userInfo == null && uri.query == null && uri.fragment == null
    val port = if (uri.port == -1) "" else ":${uri.port}"
    return if (usable) "$HTTPS://$host$port${uri.path.orEmpty().trimEnd('/')}" else null
  }
}

/** The collector choice offered on the setup screen: the Liquid AI production collector, or one the user types in. */
enum class CollectorEndpointOption(val title: String) {
  PRODUCTION("Liquid AI"),
  CUSTOM("Custom");

  /** The server URL to register against, or null when this option has no valid URL yet (an empty or malformed custom entry). */
  fun serverUrl(customUrl: String): String? =
    when (this) {
      PRODUCTION -> AppSettingsStore.DEFAULT_SERVER_URL
      CUSTOM -> CollectorEndpoint.normalizedCustomUrl(customUrl)
    }
}
