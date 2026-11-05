package ai.liquid.pipette

import com.google.crypto.tink.subtle.Ed25519Sign
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The bundle `auth-export` prints and `auth-import` reads.
 *
 * Worth its own suite for the reason iOS's `HeadlessAuthExportTests` is: this pair is the only path off a device for a signing key the Keystore
 * exports no other way, so a bundle that round-trips wrong costs the device its `client_id` and orphans every result it ever submitted. Nothing else
 * reports that, because the token an operator copied looks exactly as valid either way.
 *
 * A plain JVM test — no Robolectric. [IdentityBundle] touches no Android service: `org.json` is a test dependency and Tink's `Ed25519Sign` is pure
 * Java, so the whole codec is reachable here, which is what makes the refusal table cheap enough to be exhaustive.
 */
class IdentityBundleTest {

  /** A real seed, so the derivation in `publicKeyHexOf` is exercised rather than stubbed. Deterministic, since nothing here needs a fresh value. */
  private fun privateKeyHex(fill: Int = 3): PrivateKeyHex = PrivateKeyHex(ByteArray(32) { (it + fill).toByte() }.toHex())

  private fun registration() =
    RegistrationData(
      clientId = "ev1_a3f8c2d1",
      status = "approved",
      serverUrl = "https://pipette.liquid.ai",
      organization = "Liquid AI",
      contactEmail = "ops@liquid.ai",
      registeredAt = "2026-08-12T00:00:00Z",
      clerkUserId = "user_2abc",
      clerkSessionId = "sess_9xyz",
      clerkPrimaryEmail = "ops@liquid.ai",
      clerkLinkedAt = "2026-08-12T00:05:00Z",
    )

  private fun bundle(privateKeyHex: PrivateKeyHex = privateKeyHex(), userToken: String? = "hf_user") =
    IdentityBundle(privateKeyHex = privateKeyHex, registration = registration(), userToken = userToken)

  private fun decoded(token: String): IdentityBundle {
    val result = IdentityBundle.validated(token)
    assertTrue("expected a bundle, got $result", result is IdentityBundle.Result.Ok)
    return (result as IdentityBundle.Result.Ok).bundle
  }

  private fun refusal(token: String): IdentityBundle.Failure {
    val result = IdentityBundle.validated(token)
    assertTrue("expected a refusal, got $result", result is IdentityBundle.Result.Error)
    return (result as IdentityBundle.Result.Error).failure
  }

  /** The record a valid bundle encodes to, as JSON, so a test can patch one field and leave the rest faithful. */
  private fun recordOf(source: IdentityBundle): JSONObject =
    JSONObject(String(java.util.Base64.getUrlDecoder().decode(source.encode()), Charsets.UTF_8))

  private fun tokenOf(record: JSONObject): String = IdentityBundle.base64url(record.toString())

  @Test
  fun aBundleRoundTripsThroughItsToken() {
    val source = bundle()

    val restored = decoded(source.encode())

    assertEquals(source, restored)
    assertEquals("the clerk link rides along, so a restored device keeps its sign-in", registration(), restored.registration)
    assertEquals("hf_user", restored.userToken)
    assertEquals(1, restored.tokenCount)
  }

  /** One word, no shell metacharacters: base64url without padding, which is what lets the token be passed as an `am start` extra. */
  @Test
  fun aTokenIsShellSafe() {
    val token = bundle().encode()

    assertTrue(token.isNotEmpty())
    assertTrue(token, token.all { it.isLetterOrDigit() || it == '-' || it == '_' })
  }

  @Test
  fun theBundleDerivesThePublicKeyFromThePrivateOne() {
    val seed = ByteArray(32) { (it + 3).toByte() }

    assertEquals(Ed25519Sign.KeyPair.newKeyPairFromSeed(seed).publicKey.toHex(), IdentityBundle.publicKeyHexOf(bundle().privateKeyHex))
  }

  /**
   * The bundle carries a signing key, so its own `toString` must not print one.
   *
   * A `data class` generates a `toString` that a failing assertion prints and a `Log.d` would emit, which is why the key is a [PrivateKeyHex] here
   * rather than a `String`. Pinned because the protection is invisible: nothing about a data class suggests one of its fields is redacted.
   */
  @Test
  fun theBundleDoesNotPrintItsKey() {
    val rendered = bundle().toString()

    assertFalse(rendered, rendered.contains(privateKeyHex().value))
    assertTrue(rendered, rendered.contains("redacted"))
  }

  /**
   * The bundle's registration is `snake_case`, not the camelCase spelling [RegistrationData.toJson] writes to disk.
   *
   * The two encodings are separate on purpose. iOS reads the camelCase keys only through a `LegacyCodingKeys` path its own source documents as
   * read-only, so a bundle in the on-disk spelling would decode there by its compatibility route — and an iOS build that ever drops that route would
   * stop reading Android's bundles with nothing having changed on this side.
   */
  @Test
  fun theBundleWritesTheWireSpellingRatherThanTheOnDiskOne() {
    val record = recordOf(bundle()).getJSONObject("registration")

    assertEquals("ev1_a3f8c2d1", record.getString("client_id"))
    assertEquals("https://pipette.liquid.ai", record.getString("server_url"))
    assertEquals("ops@liquid.ai", record.getString("contact_email"))
    assertEquals("2026-08-12T00:00:00Z", record.getString("registered_at"))
    assertEquals("user_2abc", record.getString("clerk_user_id"))
    assertFalse("the on-disk spelling must not appear", record.has("clientId"))
    assertFalse(record.has("serverUrl"))
  }

  /** The three fields this client does not hold are emitted as the empty values iOS decodes, so one parser reads either platform's bundle. */
  @Test
  fun theBundleEmitsTheFieldsThisClientDoesNotHold() {
    val record = recordOf(bundle())

    assertEquals(IdentityBundle.CURRENT_VERSION, record.getInt("version"))
    assertEquals(0, record.getJSONObject("device_labels").length())
    assertTrue("settings is an explicit null, which iOS reads as absent", record.isNull("settings"))
    assertEquals(0, record.getJSONObject("model_tokens").length())
  }

  /**
   * An iOS-produced bundle whose device never recorded the four optional registration fields still imports.
   *
   * [RegistrationData.fromJson] requires all six through `getString`, so a decoder written against it would refuse this bundle and strand the
   * identity it was carrying. iOS defaults the same four to the empty string for the same reason: they are convenience copies of state the server
   * owns, and a missing one costs a Settings row rather than the ability to sign.
   */
  @Test
  fun aBundleMissingTheOptionalRegistrationFieldsImportsWithThemBlank() {
    val record = recordOf(bundle())
    val registration = record.getJSONObject("registration")
    listOf("status", "organization", "contact_email", "registered_at").forEach(registration::remove)

    val restored = decoded(tokenOf(record))

    assertEquals("ev1_a3f8c2d1", restored.registration.clientId)
    assertEquals("https://pipette.liquid.ai", restored.registration.serverUrl)
    assertEquals("", restored.registration.status)
    assertEquals("", restored.registration.organization)
    assertEquals("", restored.registration.contactEmail)
    assertEquals("", restored.registration.registeredAt)
  }

  /** A bundle from a device holding no token restores none, rather than an empty entry the import would count as a failed restore. */
  @Test
  fun aBundleWithNoTokenCarriesNone() {
    val restored = decoded(bundle(userToken = null).encode())

    assertNull(restored.userToken)
    assertEquals(0, restored.tokenCount)
  }

  /**
   * Per-model tokens only iOS can hold are counted and dropped.
   *
   * Counted rather than ignored because a silently discarded credential means a gated download 401s on the restored device with nothing in the log to
   * explain it. Dropped rather than stored because this client has one token slot, and picking one of several would restore a credential for a model
   * nobody named.
   */
  @Test
  fun perModelTokensAreCountedAndDropped() {
    val record = recordOf(bundle())
    record.put("model_tokens", JSONObject().put("mlx://repo=org/a", "token_a").put("mlx://repo=org/b", "token_b"))

    val restored = decoded(tokenOf(record))

    assertEquals(2, restored.modelTokenCount)
    assertEquals("only what this client can restore is counted as a token", 1, restored.tokenCount)
    assertEquals("re-encoding writes the empty object this client holds", 0, recordOf(restored).getJSONObject("model_tokens").length())
  }

  /**
   * The bundles `auth-import` must refuse, and what it must say about each.
   *
   * Rows over a faithfully encoded record rather than one hand-written malformed token per case: the interesting difference is one field, and writing
   * eight tokens by hand is how one of them stops resembling a real bundle.
   */
  @Test
  fun malformedBundlesAreRefusedWithTheReasonTheyFailedFor() {
    val cases =
      listOf(
        Triple("a later format", { r: JSONObject -> r.put("version", 2) }, "bundle version 2 is not readable by this build"),
        Triple("version 0", { r: JSONObject -> r.put("version", 0) }, "bundle version 0 is not readable by this build"),
        Triple(
          "no version at all",
          { r: JSONObject ->
            r.remove("version")
            r
          },
          "it carries no `version`",
        ),
        Triple(
          "a key that is not hex",
          { r: JSONObject -> r.put("private_key_hex", "zz".repeat(32)) },
          "the bundle's signing key is unusable: not hex",
        ),
        Triple(
          "hex of the wrong length",
          { r: JSONObject -> r.put("private_key_hex", "ab".repeat(16)) },
          "the bundle's signing key is unusable: 16 bytes is not an Ed25519 signing key",
        ),
        // `toInt(16)` reads a sign, so these are the right length and would decode to 32 bytes unless the codec checks for hex digits first.
        Triple("signed byte pairs", { r: JSONObject -> r.put("private_key_hex", "+a".repeat(32)) }, "the bundle's signing key is unusable: not hex"),
        Triple(
          "negative byte pairs",
          { r: JSONObject -> r.put("private_key_hex", "-0".repeat(32)) },
          "the bundle's signing key is unusable: not hex",
        ),
        Triple(
          "no key at all",
          { r: JSONObject ->
            r.remove("private_key_hex")
            r
          },
          "it carries no `private_key_hex`",
        ),
        Triple(
          "a registration naming no client",
          { r: JSONObject -> r.put("registration", JSONObject().put("status", "approved")) },
          "its `registration` has no `client_id` and `server_url`",
        ),
      )
    for ((description, patch, expected) in cases) {
      val token = tokenOf(patch(recordOf(bundle())))

      val reason = refusal(token).reason

      assertTrue("$description: expected '$expected', got '$reason'", reason.contains(expected))
    }
  }

  /** A version this build cannot read is reported *as a version*, which is why the probe reads it before the rest of the record. */
  @Test
  fun aLaterFormatReportsItsVersionRatherThanItsMissingFields() {
    val record = recordOf(bundle()).put("version", 9)
    // Every other field gone, so a decoder that read the record first would have plenty else to complain about.
    listOf("private_key_hex", "registration", "device_labels", "settings", "user_token", "model_tokens").forEach(record::remove)

    val failure = refusal(tokenOf(record))

    assertTrue("expected an unsupported version, got $failure", failure is IdentityBundle.Failure.UnsupportedVersion)
    assertEquals(9, (failure as IdentityBundle.Failure.UnsupportedVersion).found)
  }

  @Test
  fun aTokenThatIsNotBase64UrlIsRefusedAsSuch() {
    val failure = refusal("not a bundle!")

    assertEquals(IdentityBundle.Failure.NotBase64Url, failure)
    assertTrue(failure.reason, failure.reason.contains("not base64url"))
  }

  /** The standard alphabet is refused rather than half-decoded, because `+` and `/` are not characters this encoder can emit. */
  @Test
  fun aStandardBase64TokenIsRefusedAsNotBase64Url() {
    val standard = java.util.Base64.getEncoder().encodeToString(ByteArray(64) { (it * 5).toByte() })

    assertTrue(standard, standard.contains('+') || standard.contains('/') || standard.contains('='))
    assertEquals(IdentityBundle.Failure.NotBase64Url, refusal(standard))
  }

  @Test
  fun jsonThatIsNotABundleIsRefusedAsSuch() {
    val failure = refusal(IdentityBundle.base64url("""{"hello":"world"}"""))

    assertTrue("expected a not-a-bundle refusal, got $failure", failure is IdentityBundle.Failure.NotABundle)
  }

  @Test
  fun base64UrlThatIsNotJsonIsRefusedAsNotABundle() {
    val failure = refusal(IdentityBundle.base64url("this decodes but is not a record"))

    assertTrue("expected a not-a-bundle refusal, got $failure", failure is IdentityBundle.Failure.NotABundle)
  }
}
