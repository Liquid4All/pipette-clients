package ai.liquid.pipette

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import androidx.annotation.CheckResult
import com.google.crypto.tink.Aead
import com.google.crypto.tink.integration.android.AndroidKeystore
import com.google.crypto.tink.subtle.Ed25519Sign
import java.util.Base64

/**
 * The device signing identity: an Ed25519 keypair whose public key is registered with the management server and whose private key produces the
 * `X-Signature` on every authenticated request.
 *
 * **Why Tink and not the JCA.** No Android below API 37 can produce an *exportable* Ed25519 key through the JCA, so the previous
 * `KeyPairGenerator.getInstance("Ed25519")` failed on every device below that — and since key generation is the first step of
 * [RegistrationService.register], registration was impossible there. `getInstance` returns the first provider in preference order that offers the
 * algorithm, which makes the outcome depend on the API level:
 * - **API 31–34:** no provider has it. Conscrypt (`AndroidOpenSSL`) gained `EdDSA` and the `Ed25519` alias only in `android17-release`, and AOSP's
 *   repackaged BouncyCastle ships no EdEC provider, so the lookup throws `NoSuchAlgorithmException: Ed25519 KeyPairGenerator not available`.
 *   Reproduced on an API 33 emulator.
 * - **API 35–36:** `AndroidKeyStore` registers `KeyPairGenerator.ED25519` from `android15-release`, and JCA lookup is case-insensitive, so with
 *   Conscrypt still lacking it the call resolves *there*. No better: that SPI needs `initialize(KeyGenParameterSpec)` and throws
 *   `IllegalStateException("Not initialized")` without one, and Keystore keys are non-exportable, so the `private.encoded` the old code persisted
 *   would never have been readable. Reproduced on an API 36 emulator.
 * - **API 37+:** Conscrypt has it and precedes `AndroidKeyStore` in provider order, so it wins and yields an ordinary exportable software key.
 *   Confirmed on a device: the provider resolves to `AndroidOpenSSL`. This is why only API 37+ installs can hold a legacy blob for
 *   [migrateLegacyPkcs8Key] to convert.
 *
 * Tink carries its own implementation, so none of this depends on the platform's provider set any more.
 *
 * **Key format.** [Ed25519Sign] works in raw 32-byte keys, which is what the management server's wire format already wants — the public key goes out
 * as 64 hex characters and the signature as 128. That also aligns Android with the other two clients: iOS stores
 * `Curve25519.Signing.PrivateKey.rawRepresentation` as hex, and the Rust CLI stores the hex seed. Keys written by the previous JCA implementation are
 * base64 PKCS#8 blobs, converted in place the first time the signing path needs them — see [migrateLegacyPkcs8Key].
 *
 * **At rest.** That hex is what the code passes around; it is not what lands on disk. Every slot is encrypted under an AES-256-GCM key held in the
 * Android Keystore, which the app can use but never read, and stored base64 under `<slot>_enc`. Before that, the key and the Hugging Face token sat
 * in plaintext in `pipette_secrets.xml`, readable by anyone with filesystem access to the data dir. There are therefore two in-place upgrades, not
 * one: PKCS#8 to raw hex, and plaintext to ciphertext. Neither ever deletes its source until the replacement is stored, so a Keystore that is
 * momentarily unreachable can never cost the device its identity.
 *
 * @param aead injectable so tests can supply a fake; Robolectric has no working Keystore. Null means the Keystore was unreachable. Encrypted slots
 *   then read as absent, and nothing new is ever written unencrypted. A slot still holding pre-encryption plaintext stays readable, deliberately: a
 *   not-yet-migrated install keeps working through a Keystore outage instead of losing its identity to one.
 */
class Secrets(context: Context, private val aead: Aead? = defaultAead()) {
  private val prefs: SharedPreferences = context.getSharedPreferences("pipette_secrets", Context.MODE_PRIVATE)

  /**
   * Create a keypair and stage the private key, returning the public key hex to register. Staged rather than stored outright so an interrupted
   * registration can't leave behind a promoted key the server never saw; [promotePendingPrivateKey] commits it once the server accepts.
   *
   * @throws IllegalStateException if the key cannot be stored, so the caller never registers a public key no device can sign for.
   */
  fun generatePendingSigningKeyPair(): String {
    val pair = Ed25519Sign.KeyPair.newKeyPair()
    // Fail before the caller registers this public key with the server: a key we cannot store is an identity
    // the server would record and no device could ever use.
    synchronized(KEY_LOCK) {
      check(writeSecret(KEY_PENDING_PRIVATE_HEX, pair.privateKey.toHex())) { "Cannot store a signing key: the Android Keystore is unavailable" }
    }
    return pair.publicKey.toHex()
  }

  fun promotePendingPrivateKey(): Boolean =
    synchronized(KEY_LOCK) {
      // Seal before touching storage. Clearing the pending and legacy slots first and failing to write the
      // promoted key would destroy every copy of the identity at once.
      val promoted = readSecret(KEY_PENDING_PRIVATE_HEX, migrate = false)?.let { seal(KEY_PRIVATE_HEX, it) }
      if (promoted == null) {
        false
      } else {
        // Clear both legacy slots as well: the new key supersedes them, nothing reads
        // them again, and leaving them would strand plaintext key material forever.
        prefs
          .edit()
          .putString(encSlot(KEY_PRIVATE_HEX), promoted)
          .remove(KEY_PRIVATE_HEX)
          .removeSecret(KEY_PENDING_PRIVATE_HEX)
          .remove(LEGACY_KEY_PENDING_PRIVATE_PKCS8)
          .remove(LEGACY_KEY_PRIVATE_PKCS8)
          .apply()
        true
      }
    }

  fun deletePendingPrivateKey() {
    synchronized(KEY_LOCK) { prefs.edit().removeSecret(KEY_PENDING_PRIVATE_HEX).remove(LEGACY_KEY_PENDING_PRIVATE_PKCS8).apply() }
  }

  /**
   * Whether a usable promoted signing key exists — backs the "Present"/"Missing" line in the settings debug panel. Deliberately a pure read: it
   * reports a legacy blob, or a value still in pre-encryption plaintext, as present without converting either, so rendering a diagnostic can't mutate
   * key material. Both migrations happen on the signing path, where the key is actually needed.
   */
  fun hasPrivateKey(): Boolean = synchronized(KEY_LOCK) { readPrivateKey(migrate = false) != null }

  fun deletePrivateKey() {
    synchronized(KEY_LOCK) { prefs.edit().removeSecret(KEY_PRIVATE_HEX).remove(LEGACY_KEY_PRIVATE_PKCS8).apply() }
  }

  /** Signs [payload] — the `v1` signed payload built by [signedPayload] — returning the hex `X-Signature` value. */
  fun sign(payload: String): String {
    val privateKey = synchronized(KEY_LOCK) { readPrivateKey(migrate = true) } ?: error("No private signing key")
    return Ed25519Sign(privateKey).sign(payload.toByteArray(Charsets.UTF_8)).toHex()
  }

  /**
   * The raw 32-byte private key, or null when there is no usable one — either because the device was never registered, or because what is stored
   * won't parse.
   *
   * The *presence* of a current-format value in either form (encrypted, or not-yet-migrated plaintext), rather than its readability, decides which
   * slot wins. A corrupt current key therefore reads as "no key" rather than falling back, because the slot being set at all means this install is on
   * the new format, so a lingering legacy blob could only be a superseded identity. [migrate] applies only to the branch where no current key is set
   * at all: true converts the legacy blob in place, false decodes it and leaves storage untouched. It also rides down into the current slot, where it
   * governs the separate plaintext-to-ciphertext upgrade.
   */
  private fun readPrivateKey(migrate: Boolean): ByteArray? =
    when {
      hasSecret(KEY_PRIVATE_HEX) -> currentPrivateKey(migrate)
      migrate -> migrateLegacyPkcs8Key()
      else -> legacyPrivateKey()
    }

  /** The current-format key, or null if absent or unreadable. [migrate] false keeps this a pure read, upgrading no pre-encryption plaintext. */
  private fun currentPrivateKey(migrate: Boolean): ByteArray? =
    readSecret(KEY_PRIVATE_HEX, migrate)?.hexToBytesOrNull()?.takeIf { it.size == ED25519_PRIVATE_KEY_BYTES }

  /** Decodes the legacy blob without touching storage, so callers can test for it without committing to the migration. */
  private fun legacyPrivateKey(): ByteArray? =
    prefs.getString(LEGACY_KEY_PRIVATE_PKCS8, null)?.let { blob ->
      runCatching { Base64.getDecoder().decode(blob) }.getOrNull()?.let(::seedFromPkcs8)
    }

  /**
   * One-shot upgrade of a key written by the previous JCA implementation, recovering the *same* key rather than minting a new one — so the registered
   * `client_id` and the server's copy of the public key both stay valid.
   *
   * A blob that won't parse is left in place, because deleting it is irreversible and buys nothing. Registration state lives in `LocalStorage`, not
   * here, so dropping the key doesn't reset anything: the device still reports itself registered and every signed request still fails (visibly —
   * `ResultSubmissionService` records the failure per cell — but with no way back). Keeping the bytes means a later fix to [seedFromPkcs8] can still
   * recover the identity, where deleting them ends it permanently.
   */
  private fun migrateLegacyPkcs8Key(): ByteArray? {
    val seed = legacyPrivateKey() ?: return null
    // Drop the blob only once the converted key is safely encrypted. Removing it alongside a write that
    // silently did nothing would destroy a recoverable identity, which this method exists to preserve.
    if (writeSecret(KEY_PRIVATE_HEX, seed.toHex())) prefs.edit().remove(LEGACY_KEY_PRIVATE_PKCS8).apply()
    return seed
  }

  /**
   * The 32-byte seed inside an RFC 8410 Ed25519 PKCS#8 encoding, or null if [pkcs8] isn't one.
   *
   * Reads the seed from a *fixed offset*, not the tail: a v2 `OneAsymmetricKey` appends the public key, so its trailing 32 bytes are the public key,
   * and slicing from the end would install a well-formed but wrong identity that signs happily and fails every server-side check. This is a prefix
   * match, not a DER parser: it pins the seed's position by matching the id-Ed25519 AlgorithmIdentifier and the OCTET STRING headers that precede it,
   * and assumes the short-form lengths Conscrypt emits. It does not validate the outer length, the version, or anything after the seed.
   */
  private fun seedFromPkcs8(pkcs8: ByteArray): ByteArray? {
    val wellFormed =
      pkcs8.size >= SEED_OFFSET + ED25519_PRIVATE_KEY_BYTES &&
        pkcs8[0] == DER_SEQUENCE_TAG &&
        ED25519_PKCS8_PREFIX.indices.all { pkcs8[PKCS8_PREFIX_OFFSET + it] == ED25519_PKCS8_PREFIX[it] }
    return if (wellFormed) pkcs8.copyOfRange(SEED_OFFSET, SEED_OFFSET + ED25519_PRIVATE_KEY_BYTES) else null
  }

  /**
   * Stores the Hugging Face token, returning whether it landed. False means the Keystore was unreachable and the token was NOT saved; any previously
   * stored token is left intact. Callers that report success to the user should check it, or a download will later fail on a token the user believes
   * they saved.
   *
   * [CheckResult] states that contract in a form tools can act on, because a doc comment did not: both settings screens used to report "HF token
   * updated" unconditionally, so a Keystore outage discarded the token silently. Treat it as documentation and an IDE inspection, NOT as a gate.
   * Verified on this project that `:app:lintDebug` reports nothing for a deliberately ignored result, so the call sites below are what actually
   * prevents the bug, and a new caller that drops the result will not be caught by CI.
   */
  @CheckResult fun saveHfToken(token: String): Boolean = synchronized(KEY_LOCK) { writeSecret(KEY_HF_TOKEN, token.trim()) }

  fun loadHfToken(): String? = synchronized(KEY_LOCK) { readSecret(KEY_HF_TOKEN)?.takeIf { it.isNotBlank() } }

  fun deleteHfToken() {
    synchronized(KEY_LOCK) { prefs.edit().removeSecret(KEY_HF_TOKEN).apply() }
  }

  /**
   * Reads a slot, opportunistically upgrading a value written before encryption landed. Returns null when neither form is present, and when the slot
   * holds ciphertext that cannot be opened, whether because the Keystore is unreachable or because the ciphertext won't authenticate.
   *
   * An unreachable Keystore is therefore not on its own enough to make this return null: a slot still holding pre-encryption plaintext is returned
   * regardless. That is the deliberate concession described in the class KDoc, which keeps a not-yet-migrated install working through a Keystore
   * outage rather than losing its identity to one. The upgrade attempt on that path simply no-ops and leaves the plaintext in place.
   *
   * [migrate] false makes this a pure read, which is what lets [hasPrivateKey] stay a diagnostic that cannot mutate key material. Even with [migrate]
   * true the upgrade is non-destructive: see [upgradePlaintext].
   */
  private fun readSecret(slot: String, migrate: Boolean = true): String? {
    val ciphertext = prefs.getString(encSlot(slot), null)
    return if (ciphertext != null) {
      unseal(slot, ciphertext)
    } else {
      prefs.getString(slot, null)?.also { if (migrate) upgradePlaintext(slot, it) }
    }
  }

  /**
   * Re-persists a pre-encryption plaintext value as ciphertext.
   *
   * Returns without touching storage when the value cannot be sealed. Removing the plaintext alongside a write that silently did nothing would cost
   * the device its identity whenever the Keystore is momentarily unreachable. The signing path reaches this for the key slot, and an ordinary
   * settings render reaches it for the token, so neither may assume a write succeeded.
   */
  private fun upgradePlaintext(slot: String, value: String) {
    val ciphertext = seal(slot, value) ?: return
    prefs.edit().putString(encSlot(slot), ciphertext).remove(slot).apply()
  }

  /** Whether a slot holds anything, in either form. A pure read: it does not decrypt, migrate, or write. */
  private fun hasSecret(slot: String): Boolean = prefs.contains(encSlot(slot)) || prefs.contains(slot)

  /**
   * Encrypts [value] into the slot and drops any pre-encryption plaintext, reporting whether it landed.
   *
   * Storage is left completely untouched when the value cannot be sealed, and false is returned. Callers must not clear any other copy of a secret
   * until this has returned true: an earlier version cleared both forms on failure, which turned a transient Keystore outage into permanent loss of
   * the signing identity. Falling back to plaintext is not an option either, as it would defeat the encryption entirely.
   */
  private fun writeSecret(slot: String, value: String): Boolean {
    val ciphertext = seal(slot, value) ?: return false
    prefs.edit().putString(encSlot(slot), ciphertext).remove(slot).apply()
    return true
  }

  /** Clears both forms, so deleting a secret can't leave a stale plaintext copy behind. */
  private fun SharedPreferences.Editor.removeSecret(slot: String): SharedPreferences.Editor = remove(encSlot(slot)).remove(slot)

  /**
   * Base64 AES-256-GCM ciphertext for [value], or null if the Keystore is absent or the operation fails. The slot name is the GCM associated data, so
   * a ciphertext copied from one slot into another fails to authenticate rather than decrypting into the wrong identity.
   */
  private fun seal(slot: String, value: String): String? {
    val aead = aead ?: return null.also { Log.w(TAG, "No Keystore AEAD; refusing to persist $slot in plaintext") }
    return runCatching { Base64.getEncoder().encodeToString(aead.encrypt(value.toByteArray(Charsets.UTF_8), slot.toByteArray(Charsets.UTF_8))) }
      .onFailure { Log.e(TAG, "Failed to encrypt $slot", it) }
      .getOrNull()
  }

  /** Inverse of [seal]; null when the Keystore is absent, the base64 is malformed, or the ciphertext does not authenticate under [slot]. */
  private fun unseal(slot: String, ciphertext: String): String? {
    val aead = aead ?: return null.also { Log.w(TAG, "No Keystore AEAD; cannot read $slot") }
    return runCatching { String(aead.decrypt(Base64.getDecoder().decode(ciphertext), slot.toByteArray(Charsets.UTF_8)), Charsets.UTF_8) }
      .onFailure { Log.e(TAG, "Failed to decrypt $slot", it) }
      .getOrNull()
  }

  companion object {
    private const val TAG = "Secrets"

    /** Where a slot's ciphertext lives. Distinct from the plaintext key so a half-finished migration can't be read as ciphertext or vice versa. */
    private fun encSlot(slot: String) = "${slot}_enc"

    /**
     * The Keystore-backed AEAD, or null if the Keystore can't be reached. Null rather than a throw so a Keystore failure degrades to "no stored
     * secrets" instead of making the app unconstructable.
     *
     * Only a SUCCESS is cached. Caching a null would latch the whole process into a broken state after one transient failure (a busy keystore daemon,
     * a touch before first unlock): signing would fail, token saves would no-op, and the settings panel would report the key missing until the
     * process was killed, all while the key sat intact on disk. Retrying per construction costs a binder round trip on a path that had already
     * failed.
     *
     * The lock does more than serialize that retry: `generateNewAes256GcmKey` overwrites an existing alias unconditionally, so an unsynchronized
     * check-then-generate could mint the key twice (main thread via `AppContainer` against a WorkManager thread via `DownloadWorker`) and orphan
     * whatever the first key had encrypted.
     */
    @Volatile private var cachedAead: Aead? = null

    private fun defaultAead(): Aead? =
      cachedAead
        ?: synchronized(KEYSTORE_LOCK) {
          cachedAead
            ?: runCatching {
                if (!AndroidKeystore.hasKey(KEYSTORE_ALIAS)) AndroidKeystore.generateNewAes256GcmKey(KEYSTORE_ALIAS)
                AndroidKeystore.getAead(KEYSTORE_ALIAS)
              }
              .onFailure { Log.e(TAG, "Android Keystore unavailable; secrets will not be readable", it) }
              .getOrNull()
              ?.also { cachedAead = it }
        }

    private val KEYSTORE_LOCK = Any()

    // Keystore alias for the AES-256-GCM key wrapping every slot. Not the secrets themselves: the key never leaves the Keystore, and is not included
    // in backup, which is why pipette_secrets is excluded from backup too (see res/xml/data_extraction_rules.xml, the file that governs at minSdk
    // 31).
    private const val KEYSTORE_ALIAS = "pipette_secrets_aead"

    private const val KEY_PRIVATE_HEX = "private_key_hex"
    private const val KEY_PENDING_PRIVATE_HEX = "pending_private_key_hex"
    // Written by the pre-Tink JCA implementation (base64 PKCS#8); see migrateLegacyPkcs8Key().
    private const val LEGACY_KEY_PRIVATE_PKCS8 = "private_key_pkcs8"
    private const val LEGACY_KEY_PENDING_PRIVATE_PKCS8 = "pending_private_key_pkcs8"
    private const val KEY_HF_TOKEN = "hf_token"
    private const val ED25519_PRIVATE_KEY_BYTES = 32

    // Serializes every read and write of the key slots. Several paths write while reading:
    // sign() converts a legacy blob, and any read of a slot still in pre-encryption
    // plaintext re-persists it encrypted, which is why the HF token accessors take the lock too.
    // without the lock a concurrent deletePrivateKey() can land mid-migration and be
    // undone by the write that follows, resurrecting an identity the user just cleared.
    //
    // Static so it still holds if a second Secrets is constructed (Secrets is cheap and
    // stateless beyond prefs, so nothing stops that). A blocking monitor rather than a
    // suspending Mutex because this is a non-suspend API reached from an already-
    // blocking HttpURLConnection path, and each critical section is an in-memory
    // SharedPreferences map operation plus an apply() that only queues the disk write.
    private val KEY_LOCK = Any()

    // Layout of an RFC 8410 Ed25519 PKCS#8; see seedFromPkcs8().
    //   v1 (48B): 30 2e 02 01 00 | 30 05 06 03 2b 65 70 04 22 04 20 | seed
    //   v2 (83B): 30 51 02 01 01 | 30 05 06 03 2b 65 70 04 22 04 20 | seed | 81 21 00 pub
    private const val DER_SEQUENCE_TAG: Byte = 0x30
    private const val PKCS8_PREFIX_OFFSET = 5
    private const val SEED_OFFSET = 16
    private val ED25519_PKCS8_PREFIX = byteArrayOf(0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20)
  }
}

fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }

/**
 * Inverse of [toHex]; null unless the receiver is a non-empty, even-length run of ASCII hex digits. Validates rather than coerces, because it parses
 * key material out of persisted state — the explicit ASCII range is narrower than both `Character.digit` (which accepts other Unicode digits) and
 * `toInt(16)` (which would read `"-1"` as a byte).
 */
internal fun String.hexToBytesOrNull(): ByteArray? {
  val isHex = isNotEmpty() && length % 2 == 0 && all { it in '0'..'9' || it in 'a'..'f' || it in 'A'..'F' }
  return if (isHex) chunked(2).map { it.toInt(HEX_RADIX).toByte() }.toByteArray() else null
}

private const val HEX_RADIX = 16
