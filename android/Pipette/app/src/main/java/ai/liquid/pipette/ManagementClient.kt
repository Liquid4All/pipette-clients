package ai.liquid.pipette

import android.net.Uri
import java.net.HttpURLConnection
import java.net.URI
import java.net.URISyntaxException
import java.net.URL
import java.security.SecureRandom
import java.time.Instant
import java.time.format.DateTimeFormatter
import org.json.JSONArray
import org.json.JSONObject

/**
 * Thrown for a non-2xx management-server response; carries the HTTP status so callers can map specific codes (e.g. a 401/403 pre-auth-key verdict on
 * registration) to friendly messages. Mirrors iOS `ManagementClientError.httpStatus`.
 */
class ManagementClientException(val statusCode: Int, val responseBody: String) :
  IllegalStateException(
    if (responseBody.isBlank()) "Management server returned HTTP $statusCode" else "Management server returned HTTP $statusCode: $responseBody"
  )

class ManagementClient(private val secrets: Secrets) {
  data class RegistrationResult(val clientId: String, val status: String)

  /**
   * The client record returned by `PATCH`/`GET /clients/me`, narrowed to the two fields this app acts on. [reindexPending] is `true` when the profile
   * change voided the client's queue standing; see [ProfileRefreshService] for why this client does not wait on it.
   */
  data class ClientProfile(val status: String, val reindexPending: Boolean)

  /**
   * Outcome of a conditional `GET` (`If-None-Match`). [body] is the raw response body on `200`, or `null` on `304` (the caller keeps its cached
   * copy); [etag] is the server's content hash to echo back as `If-None-Match` next time. Mirrors iOS `ManagementClient.ConditionalGet`.
   */
  data class ConditionalGet(val body: String?, val etag: String?)

  /**
   * Fetch the benchmark catalog (definitions only — no eval samples), ETag-conditional. The `GET /benchmarks` endpoints are public — no client id or
   * signature — so this needs no registration.
   */
  fun fetchBenchmarks(serverUrl: String, ifNoneMatch: String?): ConditionalGet = conditionalGet(serverUrl, "/benchmarks", ifNoneMatch)

  /**
   * Fetch a single benchmark by id, ETag-conditional. Unlike the list, the per-id response includes the eval `samples`, so every benchmark must be
   * fetched individually to be fully hydrated. Public — no auth.
   */
  fun fetchBenchmark(serverUrl: String, benchmarkId: String, ifNoneMatch: String?): ConditionalGet =
    // Percent-encode the id as a single path segment — ids come from server JSON, so an id with reserved characters must not reshape the URL path.
    conditionalGet(serverUrl, "/benchmarks/${Uri.encode(benchmarkId)}", ifNoneMatch)

  fun register(
    serverUrl: String,
    organization: String,
    contactEmail: String,
    clientDetails: String,
    publicKeyHex: String,
    preauthKey: String? = null,
  ): RegistrationResult {
    val body =
      JSONObject()
        .put("public_key", publicKeyHex)
        .put("organization", organization)
        .put("contact_email", contactEmail)
        .put("client_details", clientDetails)
    // Optional pre-auth key (`preauth_{key_id}.{secret}`): a valid one makes the server auto-approve
    // this registration instead of leaving it pending manual approval. Omitted when blank.
    if (!preauthKey.isNullOrBlank()) body.put("preauth_key", preauthKey)
    val response = request(serverUrl, "/clients/register", "POST", null, body.toString())
    val json = JSONObject(response)
    return RegistrationResult(clientId = json.getString("client_id"), status = json.getString("status"))
  }

  /**
   * `PATCH /clients/me` — refresh the device profile and capability set the planner matches jobs against. Authenticated like any other client call.
   *
   * [profile] is the request body; see [ProfileRefreshService] for which fields the server's schema accepts. `PATCH` is safe over [HttpURLConnection]
   * on Android (its implementation is OkHttp-backed, and OkHttp's method list includes `PATCH`) even though the desktop JVM's own implementation
   * rejects it.
   */
  fun updateMe(serverUrl: String, clientId: String, profile: JSONObject): ClientProfile {
    val response = request(serverUrl, "/clients/me", "PATCH", clientId, profile.toString())
    val json = JSONObject(response)
    return ClientProfile(
      status = json.getString("status"),
      // Defaulted rather than required so an older server that predates the field still parses, matching the CLI's `#[serde(default)]` and iOS's
      // `decodeIfPresent ?? false`.
      reindexPending = json.optBoolean("reindex_pending", false),
    )
  }

  fun submitResult(serverUrl: String, clientId: String, payloadJson: String): String =
    request(serverUrl, "/benchmarks", "POST", clientId, payloadJson)

  fun submitResultBatch(serverUrl: String, clientId: String, payloads: JSONArray): String {
    val body = JSONObject().put("submissions", payloads)
    return request(serverUrl, "/benchmarks/batch", "POST", clientId, body.toString())
  }

  private fun request(serverUrl: String, path: String, method: String, clientId: String?, body: String?): String {
    val url = URL(endpoint(serverUrl, path))
    val connection = url.openConnection() as HttpURLConnection
    connection.requestMethod = method
    // The `v1` signature covers the request target, so a followed redirect would present a signature over the pre-redirect target and earn a 401 —
    // and OkHttp forwards custom headers, `X-Signature` included, to the redirect host. Surface the 3xx instead.
    connection.instanceFollowRedirects = false
    connection.connectTimeout = 15_000
    connection.readTimeout = 120_000
    connection.setRequestProperty("Accept", "application/json")
    connection.setRequestProperty("User-Agent", "pipette-android")
    if (clientId != null) {
      val timestamp = DateTimeFormatter.ISO_INSTANT.format(Instant.now())
      val nonce = generateNonce()
      connection.setRequestProperty("X-Client-Id", clientId)
      connection.setRequestProperty("X-Timestamp", timestamp)
      connection.setRequestProperty("X-Nonce", nonce)
      connection.setRequestProperty("X-Signature", secrets.sign(signedPayload(method, url.requestTarget(), timestamp, clientId, nonce)))
    }
    if (body != null) {
      connection.doOutput = true
      connection.setRequestProperty("Content-Type", "application/json")
      connection.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }
    }

    val status = connection.responseCode
    val stream = if (status in 200..299) connection.inputStream else connection.errorStream
    val text = stream?.bufferedReader()?.use { it.readText() } ?: ""
    if (status !in 200..299) {
      throw ManagementClientException(status, text)
    }
    return text
  }

  /**
   * Unauthenticated `GET` that honors `If-None-Match`: sends the last-seen [ifNoneMatch] etag, and a `304` returns a [ConditionalGet] with a null
   * body so the caller keeps its cached copy. Used for the public `/benchmarks` endpoints. Kept separate from [request] because that helper treats a
   * `304` as a non-success status and throws.
   */
  private fun conditionalGet(serverUrl: String, path: String, ifNoneMatch: String?): ConditionalGet {
    val connection = URL(endpoint(serverUrl, path)).openConnection() as HttpURLConnection
    connection.requestMethod = "GET"
    connection.connectTimeout = 15_000
    connection.readTimeout = 120_000
    connection.setRequestProperty("Accept", "application/json")
    connection.setRequestProperty("User-Agent", "pipette-android")
    if (ifNoneMatch != null) connection.setRequestProperty("If-None-Match", ifNoneMatch)

    val status = connection.responseCode
    val etag = connection.getHeaderField("ETag")
    if (status == HttpURLConnection.HTTP_NOT_MODIFIED) return ConditionalGet(body = null, etag = etag)

    val stream = if (status in 200..299) connection.inputStream else connection.errorStream
    val text = stream?.bufferedReader()?.use { it.readText() } ?: ""
    if (status !in 200..299) {
      throw IllegalStateException(if (text.isBlank()) "Management server returned HTTP $status" else "Management server returned HTTP $status: $text")
    }
    return ConditionalGet(body = text, etag = etag)
  }
}

/**
 * The `v1` signed payload: six newline-separated fields — scheme tag, method, request target, timestamp, client id, nonce (mgmt `authentication.md`
 * §2.1). Binding the method and target scopes a signature to that method and target; the nonce makes it single-use, so a captured signature cannot be
 * replayed inside the freshness window. The request body is still not covered.
 */
internal fun signedPayload(method: String, pathAndQuery: String, timestamp: String, clientId: String, nonce: String): String =
  "v1\n$method\n$pathAndQuery\n$timestamp\n$clientId\n$nonce"

/** Shared because [SecureRandom] is thread-safe and reseeding per request buys nothing. */
private val nonceRandom = SecureRandom()

/** 128 bits: enough that a collision across the fleet stays negligible without the client tracking what it has already sent. */
private const val NONCE_BYTES = 16

/**
 * A fresh per-request nonce: 16 CSPRNG bytes, lowercase hex.
 *
 * Hex rather than an arbitrary byte string on purpose. The nonce is a field in a newline-delimited payload, so a value carrying a newline could forge
 * a field boundary and make two different requests hash to one payload; hex cannot. It also satisfies the server's non-empty and valid-UTF-8 rules by
 * construction. 128 bits makes a collision across the fleet negligible, so the server's replay cache can reject a repeat without the client
 * coordinating.
 */
internal fun generateNonce(): String {
  val bytes = ByteArray(NONCE_BYTES)
  nonceRandom.nextBytes(bytes)
  return bytes.joinToString("") { "%02x".format(it) }
}

/**
 * Assemble an absolute endpoint URL, rejecting any server URL that would not reach the wire verbatim.
 *
 * [HttpURLConnection] is OkHttp-backed and canonicalizes the target it sends — percent-encoding characters that are not legal in a URL and resolving
 * `.`/`..` segments — while [requestTarget] reads the path back off the parsed URL as written. A server URL needing either rewrite would sign one
 * target and transmit another, 401ing every authenticated request, so it fails here instead. Three complementary guards cover what OkHttp rewrites:
 * printable-ASCII catches spaces, control characters, and non-ASCII ([URI] permits non-ASCII by documented deviation from RFC 2396, so it cannot);
 * [URI] catches the remaining illegal punctuation (`<`, `>`, `{`, `}`, `|`, `\`, `^`, `"`); and the dot-segment check reads the *decoded* path,
 * because OkHttp resolves `%2e%2e` as `..` too.
 */
internal fun endpoint(serverUrl: String, path: String): String {
  val trimmed = serverUrl.trim().trimEnd('/')
  require(trimmed.startsWith("http://") || trimmed.startsWith("https://")) { "Invalid management server URL: $serverUrl" }
  val endpoint = trimmed + path
  // `'!'..'~'` is printable ASCII minus the space — the only characters that survive to the wire unescaped.
  require(endpoint.all { it in '!'..'~' }) { "Invalid management server URL: $serverUrl" }
  val parsed =
    try {
      URI(endpoint)
    } catch (error: URISyntaxException) {
      throw IllegalArgumentException("Invalid management server URL: $serverUrl", error)
    }
  require((parsed.path ?: "").split('/').none { it == "." || it == ".." }) { "Invalid management server URL: $serverUrl" }
  return endpoint
}

/**
 * The request target the server signs over: this URL's path plus its query. Taken off the parsed URL rather than the endpoint path, so it carries any
 * path prefix the configured server URL contributes — the server verifies against the target it received.
 */
internal fun URL.requestTarget(): String = if (query == null) path else "$path?$query"

class RegistrationService(
  private val storage: LocalStorage,
  private val secrets: Secrets,
  private val client: ManagementClient,
  private val analytics: Analytics = NoOpAnalytics,
) {
  fun register(
    serverUrl: String,
    organization: String,
    contactEmail: String,
    preauthKey: String? = null,
    clerkUserId: String? = null,
    clerkSessionId: String? = null,
    clerkPrimaryEmail: String? = null,
  ): ManagementClient.RegistrationResult {
    val publicKeyHex = secrets.generatePendingSigningKeyPair()
    val result =
      try {
        client.register(
          serverUrl = serverUrl,
          organization = organization,
          contactEmail = contactEmail,
          clientDetails = DeviceInfo.modelName(),
          publicKeyHex = publicKeyHex,
          preauthKey = preauthKey,
        )
      } catch (error: Throwable) {
        secrets.deletePendingPrivateKey()
        // Coarse error kind only: a registration failure message can embed the server URL
        // and the contact email, neither of which belongs in analytics.
        analytics.capture(AnalyticsEvents.DEVICE_REGISTRATION_FAILED, mapOf(AnalyticsEvents.ERROR_KIND to AnalyticsEvents.errorKind(error)))
        throw error
      }

    if (!secrets.promotePendingPrivateKey()) {
      secrets.deletePendingPrivateKey()
      // Instrumented like the network failure above. The server accepted this device, but without a
      // usable key it can never sign a request, so leaving it uncaptured would drop the device from
      // the funnel entirely (no `device_registered`, no failure), reading as a user who wandered off
      // rather than a Keystore fault. Same reason iOS captures on its keychain-save failure.
      val error = IllegalStateException("Failed to save private key")
      analytics.capture(AnalyticsEvents.DEVICE_REGISTRATION_FAILED, mapOf(AnalyticsEvents.ERROR_KIND to AnalyticsEvents.errorKind(error)))
      throw error
    }

    // Link the Clerk identity locally at registration time (Clerk is never sent
    // to the mgmt server — registration is Ed25519-signed, identical to iOS).
    val base =
      RegistrationData(
        clientId = result.clientId,
        status = result.status,
        serverUrl = serverUrl,
        organization = organization,
        contactEmail = contactEmail,
        registeredAt = DateFormats.isoNow(),
      )
    val data =
      if (clerkUserId != null) {
        base.withClerkLink(clerkUserId, clerkSessionId, clerkPrimaryEmail)
      } else {
        base
      }
    storage.saveRegistration(data)
    // From here on every event attributes to the server-assigned device id. Deliberately no
    // email / organization / Clerk identity: the management server already holds those, and
    // PostHog has no need for them.
    analytics.identify(result.clientId)
    analytics.capture(AnalyticsEvents.DEVICE_REGISTERED, mapOf(AnalyticsEvents.STATUS to result.status))
    return result
  }
}
