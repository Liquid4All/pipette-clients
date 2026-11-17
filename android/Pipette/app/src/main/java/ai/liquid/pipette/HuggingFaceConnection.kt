package ai.liquid.pipette

import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * One GET to HuggingFace or to wherever it redirects, for both of the app's downloaders: [DownloadWorker], which fetches a `.gguf` file, and
 * [HuggingFaceHttp], which fetches a package's files.
 *
 * Redirects are followed by hand so the HuggingFace bearer token is never replayed to a redirect target that is not `huggingface.co`: a file stored
 * in LFS redirects to a CDN. That rule is why this is one function and not two copies of it.
 */
object HuggingFaceConnection {
  // Exact host or a real subdomain — so the HF token isn't sent to look-alikes like "evil-huggingface.co".
  private const val HF_HOST = "huggingface.co"
  private const val USER_AGENT = "pipette-android"
  private const val TIMEOUT_MS = 30_000
  private const val HTTP_TEMP_REDIRECT = 307
  private const val HTTP_PERM_REDIRECT = 308
  private const val MAX_REDIRECTS = 5
  private val REDIRECT_CODES =
    setOf(
      HttpURLConnection.HTTP_MOVED_PERM,
      HttpURLConnection.HTTP_MOVED_TEMP,
      HttpURLConnection.HTTP_SEE_OTHER,
      HTTP_TEMP_REDIRECT,
      HTTP_PERM_REDIRECT,
    )

  fun isHuggingFaceHost(host: String?): Boolean = host == HF_HOST || (host != null && host.endsWith(".$HF_HOST"))

  /**
   * Open [urlString], following redirects, and return the connection that answered with something other than one. [offset] asks for the bytes from
   * there on, to continue a partial file. [token] is read for each hop it may be sent to, and sent to no other.
   */
  fun open(urlString: String, offset: Long, token: () -> String?): HttpURLConnection {
    var target = URL(urlString)
    var redirects = 0
    while (true) {
      val connection = connect(target, offset, token)
      if (connection.responseCode !in REDIRECT_CODES) return connection
      val location = connection.getHeaderField("Location")
      connection.disconnect()
      if (location.isNullOrBlank() || redirects++ >= MAX_REDIRECTS) throw IOException("Too many redirects for $urlString")
      target = URL(target, location) // resolve relative redirects against the current URL
    }
  }

  private fun connect(url: URL, offset: Long, token: () -> String?): HttpURLConnection =
    (url.openConnection() as HttpURLConnection).apply {
      requestMethod = "GET"
      connectTimeout = TIMEOUT_MS
      readTimeout = TIMEOUT_MS
      instanceFollowRedirects = false
      setRequestProperty("User-Agent", USER_AGENT)
      if (isHuggingFaceHost(url.host)) token()?.let { setRequestProperty("Authorization", "Bearer $it") }
      if (offset > 0) setRequestProperty("Range", "bytes=$offset-")
      connect()
    }
}
