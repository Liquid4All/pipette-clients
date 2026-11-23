package ai.liquid.pipette.compose

import ai.liquid.pipette.AppContainer
import ai.liquid.pipette.HttpResponseBody
import ai.liquid.pipette.ManagementClientException
import ai.liquid.pipette.PipetteApp
import ai.liquid.pipette.R
import ai.liquid.pipette.compose.theme.PipetteTheme
import android.graphics.ImageDecoder
import android.os.SystemClock
import android.util.Base64
import android.util.Base64DataException
import android.util.Base64InputStream
import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onVisibilityChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.io.IOException
import java.net.HttpURLConnection
import java.net.MalformedURLException
import java.net.URL
import java.nio.ByteBuffer
import java.util.concurrent.ConcurrentHashMap
import javax.net.ssl.SSLHandshakeException
import javax.net.ssl.SSLPeerUnverifiedException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.async

/**
 * Brand logo for a model, mirroring iOS `ModelBrand.detect` + `BrandLogoView`. Logos are the project's own assets, ported from the iOS asset catalog
 * into res/drawable. Brands without a bundled asset (Meta/Llama, DeepSeek, Microsoft) fall back to a neutral initial badge. A [logoDigest] from the
 * catalog replaces the bundled logo after it loads. The view requests the logo when it appears on screen.
 */
@Composable
fun BrandLogo(modelName: String, modifier: Modifier = Modifier, size: Dp = 24.dp, logoDigest: String? = null) {
  var isOnScreen by remember { mutableStateOf(false) }
  val container = (LocalContext.current.applicationContext as? PipetteApp)?.containerOrNull
  val remote by
    key(logoDigest) {
      produceState(logoDigest?.let(RemoteLogos::loaded), isOnScreen, container) {
        value =
          logoDigest?.let { digest ->
            RemoteLogos.loaded(digest) ?: if (isOnScreen && container != null) RemoteLogos.load(digest, container) else null
          }
      }
    }
  val sized = modifier.size(size).onVisibilityChanged { isOnScreen = it }
  val res = brandDrawable(modelName)
  val loaded = remote
  if (loaded != null) {
    androidx.compose.foundation.Image(bitmap = loaded, contentDescription = null, contentScale = ContentScale.Fit, modifier = sized)
  } else if (res != null) {
    androidx.compose.foundation.Image(painter = painterResource(res), contentDescription = null, contentScale = ContentScale.Fit, modifier = sized)
  } else {
    Box(modifier = sized.clip(RoundedCornerShape(percent = 50)).background(PipetteTheme.colors.gray5), contentAlignment = Alignment.Center) {
      Text(
        modelName.trim().take(1).uppercase().ifBlank { "?" },
        style = TextStyle(fontSize = (size.value * 0.5).sp, fontWeight = FontWeight.SemiBold),
        color = PipetteTheme.colors.gray,
      )
    }
  }
}

/** Cache logos and permanent failures for the app run. Retry transient failures on a later request. */
private object RemoteLogos {
  private const val TAG = "RemoteLogos"
  private const val TIMEOUT_MS = 10_000
  private const val RETRY_COOLDOWN_MS = 30_000L
  private const val HTTP_TOO_MANY_REQUESTS = 429
  private const val LAST_HTTP_SERVER_ERROR = 599
  private const val MAX_IMAGE_BYTES = 2 * 1024 * 1024
  private const val MAX_IMAGE_DIMENSION = 512
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
  private val loads = ConcurrentHashMap<String, Deferred<FetchResult>>()

  private sealed interface FetchResult {
    data class Image(val bitmap: ImageBitmap) : FetchResult

    data class RetryAfter(val deadlineMs: Long) : FetchResult

    data object Unavailable : FetchResult
  }

  suspend fun load(digest: String, container: AppContainer): ImageBitmap? {
    val load =
      loads.compute(digest) { key, existing ->
        val retry = existing?.resultIfCompleted() as? FetchResult.RetryAfter
        if (existing == null || (retry != null && SystemClock.elapsedRealtime() >= retry.deadlineMs)) {
          scope.async { fetch(key, container) }
        } else {
          existing
        }
      }
    return (load?.await() as? FetchResult.Image)?.bitmap
  }

  /** Return the cached logo for [digest] so a row can show it immediately when the row reappears. */
  fun loaded(digest: String): ImageBitmap? = (loads[digest]?.resultIfCompleted() as? FetchResult.Image)?.bitmap

  @OptIn(ExperimentalCoroutinesApi::class)
  private fun Deferred<FetchResult>.resultIfCompleted(): FetchResult? = if (isCompleted && !isCancelled) getCompleted() else null

  private fun retryAfterCooldown() = FetchResult.RetryAfter(SystemClock.elapsedRealtime() + RETRY_COOLDOWN_MS)

  private fun isRetryableStatus(status: Int): Boolean =
    status == HttpURLConnection.HTTP_CLIENT_TIMEOUT ||
      status == HTTP_TOO_MANY_REQUESTS ||
      status in HttpURLConnection.HTTP_INTERNAL_ERROR..LAST_HTTP_SERVER_ERROR

  private fun fetch(digest: String, container: AppContainer): FetchResult =
    runCatching {
        val uri = container.familyLogoUri(digest)
        if (uri == null) {
          Log.w(TAG, "family logo unavailable for unregistered device: digest=$digest")
          FetchResult.Unavailable
        } else {
          fetchImage(uri, digest)
        }
      }
      .getOrElse { error ->
        Log.w(TAG, "family logo fetch failed: digest=$digest", error)
        when (error) {
          is ManagementClientException -> if (isRetryableStatus(error.statusCode)) retryAfterCooldown() else FetchResult.Unavailable
          is Base64DataException,
          is MalformedURLException,
          is ImageDecoder.DecodeException,
          is SSLHandshakeException,
          is SSLPeerUnverifiedException -> FetchResult.Unavailable
          is IOException -> retryAfterCooldown()
          is Exception -> FetchResult.Unavailable
          else -> throw error
        }
      }

  private fun fetchImage(uri: String, digest: String): FetchResult =
    when {
      uri.startsWith("data:") -> {
        val payload = uri.substringAfter(";base64,", missingDelimiterValue = "")
        val bytes = Base64InputStream(payload.byteInputStream(), Base64.DEFAULT).use { HttpResponseBody.read(it, MAX_IMAGE_BYTES) }
        decodeImage(bytes)
      }
      uri.startsWith("https://") -> download(uri, digest)
      else -> {
        Log.w(TAG, "family logo URI unsupported: digest=$digest")
        FetchResult.Unavailable
      }
    }

  private fun decodeImage(bytes: ByteArray): FetchResult.Image {
    val source = ImageDecoder.createSource(ByteBuffer.wrap(bytes))
    val bitmap =
      ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        val longestSide = maxOf(info.size.width, info.size.height)
        if (longestSide > MAX_IMAGE_DIMENSION) {
          val scale = MAX_IMAGE_DIMENSION.toDouble() / longestSide
          decoder.setTargetSize(maxOf(1, (info.size.width * scale).toInt()), maxOf(1, (info.size.height * scale).toInt()))
        }
      }
    return FetchResult.Image(bitmap.asImageBitmap())
  }

  private fun download(url: String, digest: String): FetchResult {
    val connection = URL(url).openConnection() as HttpURLConnection
    connection.connectTimeout = TIMEOUT_MS
    connection.readTimeout = TIMEOUT_MS
    return try {
      val status = connection.responseCode
      if (status == HttpURLConnection.HTTP_OK) {
        decodeImage(connection.inputStream.use { HttpResponseBody.read(it, MAX_IMAGE_BYTES) })
      } else {
        Log.w(TAG, "family logo download rejected: digest=$digest status=$status")
        if (isRetryableStatus(status)) retryAfterCooldown() else FetchResult.Unavailable
      }
    } finally {
      connection.disconnect()
    }
  }
}

/** Maps a model name to a bundled brand drawable, or null when none exists (caller shows a fallback). Detection order matches iOS ModelBrand. */
private fun brandDrawable(name: String): Int? {
  val n = name.lowercase()
  return when {
    n.contains("lfm") || n.contains("liquid") -> R.drawable.brand_liquid
    n.contains("gemma") -> R.drawable.brand_google
    n.contains("granite") || n.contains("ibm") -> R.drawable.brand_ibm
    n.contains("qwen") -> R.drawable.brand_qwen
    n.contains("mistral") || n.contains("ministral") || n.contains("mixtral") -> R.drawable.brand_mistral
    else -> null // meta/llama, deepseek, microsoft, unknown → fallback badge
  }
}

@androidx.compose.ui.tooling.preview.Preview(name = "Brand logo", showBackground = true, backgroundColor = 0xFF000000)
@Composable
private fun BrandLogoPreview() {
  PipetteTheme(darkTheme = true) { BrandLogo("Qwen 3.5 2B", size = 40.dp) }
}
