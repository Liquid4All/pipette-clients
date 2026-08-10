package ai.liquid.pipette

import java.io.EOFException
import java.io.IOException
import java.net.ConnectException
import java.net.ProtocolException
import java.net.SocketException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import javax.net.ssl.SSLHandshakeException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins which download failures keep a `.part` alive for a retry and which drop it.
 *
 * This is the split that decides whether one flaky moment on a hotel network costs the user a multi-gigabyte re-download, so the classification is
 * worth asserting rather than inferring from the exception hierarchy. The set mirrors the iOS client's `downloadURLErrorInfo`, which is the same
 * decision expressed over `URLError.Code`.
 */
class DownloadRetryTest {
  /** Every transient case: the connection never opened, timed out, or died mid-body. Retrying these is exactly what fixes them. */
  @Test
  fun connectivityFailuresAreRecoverable() {
    assertTrue(isRecoverableNetworkError(SocketTimeoutException("read timed out")))
    assertTrue(isRecoverableNetworkError(UnknownHostException("huggingface.co")))
    assertTrue(isRecoverableNetworkError(ConnectException("Connection refused")))
    assertTrue(isRecoverableNetworkError(SocketException("Connection reset")))
  }

  /**
   * A truncated response is the case that matters most in practice: the transfer ran for a while, then the connection dropped.
   *
   * All three shapes have to be covered. `HttpURLConnection` on Android is OkHttp-backed, so a connection dropped mid-body normally surfaces as
   * `ProtocolException("unexpected end of stream")` or `EOFException`, **neither** of which is a [SocketException]; only a clean early `read() == -1`
   * reaches [IncompleteDownloadException]. Classifying just the last one would send the most common real-world drop to a terminal failure.
   */
  @Test
  fun truncatedResponseIsRecoverable() {
    assertTrue(isRecoverableNetworkError(IncompleteDownloadException(bytesRead = 1_000, total = 4_000)))
    assertTrue(isRecoverableNetworkError(ProtocolException("unexpected end of stream")))
    assertTrue(isRecoverableNetworkError(EOFException()))
    // A plain IOException is NOT assumed transient: the worker raises those for HTTP statuses and 416s too, and retrying those forever is worse than
    // failing once.
    assertFalse(isRecoverableNetworkError(IOException("HTTP 404 Not Found")))
  }

  /** The streaming stack wraps, so an interrupted transfer can arrive as a generic IOException carrying the real cause. */
  @Test
  fun wrappedConnectivityFailuresAreUnwrapped() {
    assertTrue(isRecoverableNetworkError(IOException("write failed", SocketException("Connection reset"))))
    assertFalse(isRecoverableNetworkError(IOException("HTTP 403", IllegalStateException("nope"))))
    assertFalse(isRecoverableNetworkError(null))
  }

  /** Nothing a retry can fix: the server answered, and its answer was no. Matches iOS, where these arrive as a non-`URLError` and fail terminally. */
  @Test
  fun serverAndProtocolFailuresAreTerminal() {
    assertFalse(isRecoverableNetworkError(IOException("HTTP 403 Forbidden")))
    assertFalse(isRecoverableNetworkError(IOException("HTTP 416 (range not satisfiable)")))
    assertFalse(isRecoverableNetworkError(IOException("Too many redirects for https://example.com/m.gguf")))
    assertFalse(isRecoverableNetworkError(SSLHandshakeException("cert path failed")))
    assertFalse(isRecoverableNetworkError(IllegalStateException("boom")))
  }

  /** The message still has to read as a sentence for the notification and the failed-row text. */
  @Test
  fun incompleteDownloadMessageNamesBothCounts() {
    assertEquals("Incomplete download: 1000 of 4000 bytes", IncompleteDownloadException(bytesRead = 1_000, total = 4_000).message)
  }

  /**
   * The waiting state must stay distinct from paused and failed. [DownloadRegistry.putIfActive] refuses to write over a PAUSED or FAILED row, so if
   * this state collided with either, a retrying worker could never publish progress again and the row would sit frozen while bytes moved underneath.
   */
  @Test
  fun waitingForNetworkIsItsOwnState() {
    assertEquals("waiting_network", DownloadWorker.STATE_WAITING_NETWORK)
    assertFalse(DownloadWorker.STATE_WAITING_NETWORK == DownloadWorker.STATE_PAUSED)
    assertFalse(DownloadWorker.STATE_WAITING_NETWORK == DownloadWorker.STATE_FAILED)

    val waiting = ActiveDownload("k", "m.gguf", "repo", 10, 100, "Waiting for network…", DownloadWorker.STATE_WAITING_NETWORK)
    assertTrue(waiting.isWaitingForNetwork)
    assertFalse(waiting.isFailed)
    assertFalse(waiting.isPaused)
    // Pausable (that is how the user stops the retry loop) but not "resumable", since it is already resuming itself.
    assertTrue(waiting.canPause)
    assertFalse(waiting.canResume)
  }

  /**
   * Both Models screens render the badge from this one property. They used to disagree: the Compose tab mapped the state, the legacy Views tab
   * printed it raw. That went unnoticed only because every state until now was a single lowercase word.
   */
  @Test
  fun everyStateHasAReadableLabel() {
    fun labelFor(state: String) = ActiveDownload("k", "m.gguf", null, 0, -1, "", state).displayLabel

    assertEquals("Queued", labelFor(DownloadWorker.STATE_QUEUED))
    assertEquals("Downloading", labelFor(DownloadWorker.STATE_RUNNING))
    assertEquals("Paused", labelFor(DownloadWorker.STATE_PAUSED))
    assertEquals("Failed", labelFor(DownloadWorker.STATE_FAILED))
    // The one that would otherwise leak an identifier into the UI as "Waiting_network".
    assertEquals("Waiting for network", labelFor(DownloadWorker.STATE_WAITING_NETWORK))
  }

  /**
   * Retries are bounded. WorkManager never gives up on its own, so without a cap a permanently unreachable host (a typo'd repo, a decommissioned CDN)
   * throws `UnknownHostException` on every attempt and leaves the row at "Waiting for network…" forever, with no failure text and (by design) no
   * Resume affordance. The cap is what still turns that into a readable failure.
   */
  @Test
  fun retriesAreBounded() {
    assertTrue(DownloadWorker.MAX_NETWORK_ATTEMPTS in 2..10)
  }
}
