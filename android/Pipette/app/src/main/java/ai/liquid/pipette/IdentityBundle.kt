package ai.liquid.pipette

import com.google.crypto.tink.subtle.Ed25519Sign
import java.util.Base64
import org.json.JSONObject

/**
 * Everything needed to restore this device's identity somewhere else: the signing key, the registration record, and the Hugging Face token.
 *
 * The signing key is why this type exists. It is sealed under an Android Keystore key scoped to this app's UID, and `pipette_secrets` plus
 * `registration.json` are excluded from both backup channels (`res/xml/data_extraction_rules.xml`), so an uninstall ends the identity. A device that
 * replaces this one comes back without that identity, and so without its `client_id`: `LocalStorage.isRegistered` needs the registration and
 * `Secrets.sign` needs the key, so a device holding one half can never submit. Every result already submitted under the old key then loses its
 * attribution. The CLI needs no counterpart, since its key is a `0600` file and a backup there is a file copy.
 *
 * ## Wire-compatible with iOS version 1
 *
 * The token is JSON, then base64url without padding, matching `ios/Pipette/Pipette/Identity/IdentityBundle.swift`. A bundle written here imports
 * there and the reverse, which is what makes a replacement device a free choice of platform.
 *
 * Android holds a strict subset of what iOS holds: no `DeviceLabels`, no `ClientSettings`, and one `hf_token` slot rather than a token per model. It
 * therefore writes `device_labels: {}`, `settings: null` and `model_tokens: {}`, and accepts and discards all three on import. Both iOS
 * `DeviceLabels` fields are optional, so `{}` decodes there. What a discarded per-model token costs is on [modelTokenCount].
 *
 * ## The registration inside is spelled `snake_case`
 *
 * Deliberately **not** [RegistrationData.toJson], which emits the camelCase on-disk spelling. iOS reads that spelling only through a
 * `LegacyCodingKeys` path its own source documents as read-only, so writing it here would produce a bundle iOS decodes by its compatibility route.
 * The two encodings are separate on purpose, and [registrationJson] is this one.
 *
 * Decoding is as lenient as iOS's (`IdentityRegistration.swift:85-109`): only `client_id` and `server_url` are required and the other four
 * registration fields default to the empty string. [RegistrationData.fromJson] requires all six, so a strict decoder here would reject an
 * iOS-produced bundle whose device never recorded, say, an organization.
 */
data class IdentityBundle(
  /** The signing key, typed so no generated `toString` can print it. See [PrivateKeyHex]. */
  val privateKeyHex: PrivateKeyHex,
  val registration: RegistrationData,
  /** The operator's Hugging Face token. Null when the device holds none. */
  val userToken: String?,
  /**
   * How many per-model tokens the bundle carried that this client cannot hold.
   *
   * Always zero in a bundle this client wrote. An iOS bundle can carry a plan-supplied token per model, and Android has one token slot, so those are
   * decoded, counted, and dropped. Counted rather than ignored because a silently discarded credential means a gated download 401s on the restored
   * device with nothing in the log to explain it; `auth-import` reports the count when it is not zero.
   */
  val modelTokenCount: Int = 0,
) {

  /** How many tokens this client can restore from the bundle, which is the count `auth-export` and `auth-import` print. */
  val tokenCount: Int
    get() = if (userToken == null) 0 else 1

  /** The bundle as one token: its JSON, base64url without padding. */
  fun encode(): String =
    base64url(
      JSONObject()
        .put(KEY_VERSION, CURRENT_VERSION)
        .put(KEY_PRIVATE_KEY_HEX, privateKeyHex.value)
        .put(KEY_REGISTRATION, registrationJson(registration))
        // The three fields this client does not hold, emitted as the empty values iOS decodes rather than omitted, so one parser reads both
        // platforms' bundles.
        .put(KEY_DEVICE_LABELS, JSONObject())
        .put(KEY_SETTINGS, JSONObject.NULL)
        .putOptString(KEY_USER_TOKEN, userToken)
        .put(KEY_MODEL_TOKENS, JSONObject())
        .toString()
    )

  /** Why a bundle cannot be restored. One place for the refusal text, since `auth-import` prints each case straight to an operator's log. */
  sealed class Failure(val reason: String) {
    data object NotBase64Url : Failure("the bundle is not base64url: pass the whole `bundle=` value from `auth-export`")

    class NotABundle(detail: String) : Failure("the bundle decoded but is not an identity record: $detail")

    class UnsupportedVersion(val found: Int) : Failure("bundle version $found is not readable by this build, which reads version $CURRENT_VERSION")

    class InvalidPrivateKey(detail: String) : Failure("the bundle's signing key is unusable: $detail")
  }

  sealed class Result {
    data class Ok(val bundle: IdentityBundle) : Result()

    data class Error(val failure: Failure) : Result()
  }

  companion object {
    /** The format this build writes, and the only one it reads. [decode] refuses anything else rather than guess at a record a later build wrote. */
    const val CURRENT_VERSION = 1

    // The bundle's own wire keys. Named so the encoder and the decoder cannot disagree about a spelling, which is the one way a token stops being
    // readable by the client that wrote it.
    private const val KEY_VERSION = "version"
    private const val KEY_PRIVATE_KEY_HEX = "private_key_hex"
    private const val KEY_REGISTRATION = "registration"
    private const val KEY_DEVICE_LABELS = "device_labels"
    private const val KEY_SETTINGS = "settings"
    private const val KEY_USER_TOKEN = "user_token"
    private const val KEY_MODEL_TOKENS = "model_tokens"

    /**
     * The public key a private-key hex yields, or null when the hex is not an Ed25519 seed.
     *
     * On the companion rather than only on an instance because `auth-whoami` needs the derivation without a bundle: it reports a public key and
     * prints no token. One derivation therefore serves both verbs, so neither can report a key the other would refuse.
     */
    fun publicKeyHexOf(privateKeyHex: PrivateKeyHex): String? = keyPair(privateKeyHex)?.publicKey?.toHex()

    /** The keypair a hex names, or null when it is not a usable seed. [keyFailure] says which way it failed. */
    private fun keyPair(privateKeyHex: PrivateKeyHex): Ed25519Sign.KeyPair? =
      privateKeyHex.value
        .hexToBytesOrNull()
        ?.takeIf { it.size == Secrets.ED25519_PRIVATE_KEY_BYTES }
        ?.let { seed -> runCatching { Ed25519Sign.KeyPair.newKeyPairFromSeed(seed) }.getOrNull() }

    /**
     * Why a key cannot sign, or null when it can.
     *
     * Three checks rather than one, because each accepts a value the next rejects: [hexToBytesOrNull] refuses a string that is not hex, the length
     * refuses hex of the wrong size, and the derivation refuses a seed Tink will not take. iOS makes the same split for the same reason
     * (`IdentityBundle.signingKey`), and its first two messages are reused verbatim.
     */
    private fun keyFailure(privateKeyHex: PrivateKeyHex): Failure? {
      val seed = privateKeyHex.value.hexToBytesOrNull()
      return when {
        seed == null -> Failure.InvalidPrivateKey("not hex")
        seed.size != Secrets.ED25519_PRIVATE_KEY_BYTES -> Failure.InvalidPrivateKey("${seed.size} bytes is not an Ed25519 signing key")
        keyPair(privateKeyHex) == null -> Failure.InvalidPrivateKey("the seed does not produce an Ed25519 keypair")
        else -> null
      }
    }

    /**
     * Read a token back, refusing anything this build cannot restore faithfully. [validated] is the entry point, and says why the stages run in this
     * order.
     */
    @Suppress("ReturnCount") // one per stage the token can fail at, in the order a reader has to trust them; a single exit would need a sentinel
    private fun decode(token: String): Result {
      val json = base64urlDecoded(token) ?: return Result.Error(Failure.NotBase64Url)
      val record = runCatching { JSONObject(json) }.getOrNull() ?: return Result.Error(Failure.NotABundle("it is not a JSON object"))
      val version = record.optNullableInt(KEY_VERSION) ?: return Result.Error(Failure.NotABundle("it carries no `$KEY_VERSION`"))
      if (version != CURRENT_VERSION) return Result.Error(Failure.UnsupportedVersion(version))
      val privateKeyHex =
        record.optNullableString(KEY_PRIVATE_KEY_HEX) ?: return Result.Error(Failure.NotABundle("it carries no `$KEY_PRIVATE_KEY_HEX`"))
      val registration =
        record.optJSONObject(KEY_REGISTRATION)?.let(::registrationFrom)
          ?: return Result.Error(Failure.NotABundle("its `$KEY_REGISTRATION` has no `client_id` and `server_url`"))
      return Result.Ok(
        IdentityBundle(
          privateKeyHex = PrivateKeyHex(privateKeyHex),
          registration = registration,
          userToken = record.optNullableString(KEY_USER_TOKEN),
          modelTokenCount = record.optJSONObject(KEY_MODEL_TOKENS)?.length() ?: 0,
        )
      )
    }

    /**
     * A decoded bundle whose key can sign — everything `auth-import` checks before it writes anything.
     *
     * The only way in. [decode] is private because a bundle whose key was never checked is not a thing this class hands out: `auth-import` promotes
     * the hex it carries, so a caller that skipped the check would store a key `readPrivateKey` reads back as absent.
     *
     * One function rather than a sequence each caller repeats, so the verb and its tests cannot disagree about what makes a bundle restorable. The
     * key is checked and the derived public key discarded: what reaches storage is the hex the bundle carries, and this is the proof that the hex is
     * a key.
     *
     * The **version is read before the rest of the record**, so a bundle from a later format reports its version rather than the fields this build
     * failed to find in it. iOS reads it through a `VersionProbe` for the same reason.
     */
    fun validated(token: String): Result =
      when (val decoded = decode(token)) {
        is Result.Error -> decoded
        is Result.Ok -> keyFailure(decoded.bundle.privateKeyHex)?.let { Result.Error(it) } ?: decoded
      }

    /**
     * JSON to one token: base64url without padding. The value then holds no `+`, `/` or `=`, so it survives a shell, a URL and a copy-paste as a
     * single word.
     *
     * Internal rather than private because the tests build tokens too. A second copy of this transform could drift from the encoder with nothing
     * reporting it.
     */
    internal fun base64url(json: String): String = Base64.getUrlEncoder().withoutPadding().encodeToString(json.toByteArray(Charsets.UTF_8))

    /**
     * base64url back to JSON text, or null for anything that is not base64url at all.
     *
     * `getUrlDecoder` accepts the unpadded form the encoder writes and refuses `+` and `/`, so a standard-alphabet token is reported as not base64url
     * rather than half-decoded. Bytes that are not UTF-8 survive this as replacement characters and are refused by [JSONObject] a line later, which
     * is the same answer by a different route.
     */
    private fun base64urlDecoded(token: String): String? =
      runCatching { String(Base64.getUrlDecoder().decode(token.trim()), Charsets.UTF_8) }.getOrNull()

    /**
     * The bundle's `registration` object, in the `snake_case` spelling iOS's `CodingKeys` declare.
     *
     * The clerk fields are emitted as explicit nulls when unset, which iOS's `decodeIfPresent` reads as absent. They ride along because they are
     * local-only state a restored device would otherwise lose: the Clerk link gates the app's own sign-in and is never sent to the management server.
     */
    private fun registrationJson(registration: RegistrationData): JSONObject =
      JSONObject()
        .put("client_id", registration.clientId)
        .put("status", registration.status)
        .put("server_url", registration.serverUrl)
        .put("organization", registration.organization)
        .put("contact_email", registration.contactEmail)
        .put("registered_at", registration.registeredAt)
        .putOptString("clerk_user_id", registration.clerkUserId)
        .putOptString("clerk_session_id", registration.clerkSessionId)
        .putOptString("clerk_primary_email", registration.clerkPrimaryEmail)
        .putOptString("clerk_linked_at", registration.clerkLinkedAt)

    /**
     * [RegistrationData] from a bundle's registration object, or null when it names no client to restore.
     *
     * Only the upload identity is required, which is the crate's own record (`pipette-cli/src/identity/registration.rs`) and iOS's rule. The other
     * four are convenience copies of state the server owns — they fill Settings rows and prefill re-registration — so a bundle missing one restores a
     * device with an empty row rather than being refused outright, which would strand the identity it was carrying.
     */
    private fun registrationFrom(json: JSONObject): RegistrationData? {
      val clientId = json.optNullableString("client_id")
      val serverUrl = json.optNullableString("server_url")
      if (clientId == null || serverUrl == null) return null
      return RegistrationData(
        clientId = clientId,
        status = json.optNullableString("status").orEmpty(),
        serverUrl = serverUrl,
        organization = json.optNullableString("organization").orEmpty(),
        contactEmail = json.optNullableString("contact_email").orEmpty(),
        registeredAt = json.optNullableString("registered_at").orEmpty(),
        clerkUserId = json.optNullableString("clerk_user_id"),
        clerkSessionId = json.optNullableString("clerk_session_id"),
        clerkPrimaryEmail = json.optNullableString("clerk_primary_email"),
        clerkLinkedAt = json.optNullableString("clerk_linked_at"),
      )
    }
  }
}
