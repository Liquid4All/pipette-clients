package ai.liquid.pipette

import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

private const val HF_HOST = "huggingface.co"
private const val USER_AGENT = "pipette-android"
private const val TIMEOUT_MS = 30_000
private const val MAX_REDIRECTS = 5
private const val HTTP_TEMP_REDIRECT = 307
private const val HTTP_PERM_REDIRECT = 308
private val redirectCodes =
  setOf(
    HttpURLConnection.HTTP_MOVED_PERM,
    HttpURLConnection.HTTP_MOVED_TEMP,
    HttpURLConnection.HTTP_SEE_OTHER,
    HTTP_TEMP_REDIRECT,
    HTTP_PERM_REDIRECT,
  )

internal fun modelDownloadSize(url: String, hfToken: String?): Long? {
  val connection = openModelConnection(url, hfToken, method = "HEAD")
  return try {
    if (connection.responseCode == HttpURLConnection.HTTP_OK) connection.contentLengthLong.takeIf { it > 0 } else null
  } finally {
    connection.disconnect()
  }
}

/** Follow redirects without forwarding the Hugging Face token to the CDN. */
internal fun openModelConnection(url: String, hfToken: String?, existing: Long = 0L, method: String = "GET"): HttpURLConnection {
  var target = URL(url)
  repeat(MAX_REDIRECTS + 1) {
    val connection =
      (target.openConnection() as HttpURLConnection).apply {
        requestMethod = method
        connectTimeout = TIMEOUT_MS
        readTimeout = TIMEOUT_MS
        instanceFollowRedirects = false
        setRequestProperty("User-Agent", USER_AGENT)
        if (target.host == HF_HOST || target.host.endsWith(".$HF_HOST")) {
          hfToken?.let { setRequestProperty("Authorization", "Bearer $it") }
        }
        if (existing > 0) setRequestProperty("Range", "bytes=$existing-")
      }
    val status = runCatching { connection.responseCode }.onFailure { connection.disconnect() }.getOrThrow()
    if (status !in redirectCodes) return connection
    val location = connection.getHeaderField("Location")
    connection.disconnect()
    if (location.isNullOrBlank()) throw IOException("Download redirect has no destination")
    target = URL(target, location)
  }
  throw IOException("Too many model download redirects")
}
