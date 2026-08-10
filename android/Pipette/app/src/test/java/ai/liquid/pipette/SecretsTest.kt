package ai.liquid.pipette

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.google.crypto.tink.Aead
import com.google.crypto.tink.subtle.Ed25519Sign
import com.google.crypto.tink.subtle.Ed25519Verify
import java.security.GeneralSecurityException
import java.util.Base64
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

/**
 * Drives the real [Secrets] against Robolectric's SharedPreferences. The point of these is the *wire contract* — 64-hex public key, 128-hex signature
 * that verifies under the registered public key — plus the one-way migration off the pre-Tink PKCS#8 blobs.
 */
// A stock Application, not the manifest PipetteApp: Secrets needs only a Context, while PipetteApp.onCreate wires up WorkManager/Clerk/etc. that have
// no place in this unit test.
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = android.app.Application::class)
class SecretsTest {
  private val context: Context = ApplicationProvider.getApplicationContext()
  private val prefs = context.getSharedPreferences("pipette_secrets", Context.MODE_PRIVATE)
  private val fakeAead = FakeAead()
  private val secrets = Secrets(context, fakeAead)

  /**
   * Reads a slot the way [Secrets] stores it, so assertions go through the same encryption the app uses rather than peeking at a plaintext key that
   * no longer exists.
   */
  private fun storedSecret(slot: String): String? =
    prefs.getString("${slot}_enc", null)?.let { String(fakeAead.decrypt(Base64.getDecoder().decode(it), slot.toByteArray()), Charsets.UTF_8) }

  /** Writes a slot in the current (encrypted) format. */
  private fun storeSecret(slot: String, value: String) {
    prefs.edit().putString("${slot}_enc", Base64.getEncoder().encodeToString(fakeAead.encrypt(value.toByteArray(), slot.toByteArray()))).apply()
  }

  /** Writes a slot the way an install predating encryption did: plaintext, no ciphertext alongside. */
  private fun storePlaintextSecret(slot: String, value: String) {
    prefs.edit().putString(slot, value).apply()
  }

  /**
   * An RFC 8410 Ed25519 PKCS#8 for [seed], built rather than hardcoded so the layout the migration depends on is visible here. [withPublicKey] emits
   * the v2 `OneAsymmetricKey` form, which appends the public key — the case where reading the seed from the tail would silently yield the *public*
   * key instead.
   */
  private fun pkcs8(seed: ByteArray, withPublicKey: Boolean = false): ByteArray {
    val body =
      byteArrayOf(0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x04, 0x22, 0x04, 0x20) +
        seed +
        if (withPublicKey) byteArrayOf(0x81.toByte(), 0x21, 0x00) + publicKeyOf(seed) else ByteArray(0)
    val version: Byte = if (withPublicKey) 1 else 0
    return byteArrayOf(0x30, (body.size + 3).toByte(), 0x02, 0x01, version) + body
  }

  private fun publicKeyOf(seed: ByteArray): ByteArray = Ed25519Sign.KeyPair.newKeyPairFromSeed(seed).publicKey

  private fun storeLegacy(key: String, blob: ByteArray) {
    prefs.edit().putString(key, Base64.getEncoder().encodeToString(blob)).apply()
  }

  /** Asserts the signature produced by [Secrets] verifies under [publicKey] — the check the management server runs on `X-Signature`. */
  private fun assertSignsUnder(publicKey: ByteArray) {
    val signature = secrets.sign(PAYLOAD).hexToBytesOrNull()
    assertNotNull(signature)
    Ed25519Verify(publicKey).verify(signature, PAYLOAD.toByteArray())
  }

  /**
   * Asserts signing refuses because no usable key is stored. Pins the exception *type*: a regression in `seedFromPkcs8` or `hexToBytesOrNull` that
   * threw `IndexOutOfBounds` or NPE would otherwise satisfy a bare "something was thrown" check.
   */
  private fun assertSigningFails() {
    val error = runCatching { secrets.sign(PAYLOAD) }.exceptionOrNull()
    assertTrue("expected IllegalStateException, got $error", error is IllegalStateException)
  }

  @Test
  fun generatedPublicKeyIs32BytesOfHex() {
    val publicKeyHex = secrets.generatePendingSigningKeyPair()
    assertEquals(64, publicKeyHex.length)
    assertEquals(32, publicKeyHex.hexToBytesOrNull()?.size)
  }

  @Test
  fun signatureVerifiesUnderTheRegisteredPublicKey() {
    val publicKeyHex = secrets.generatePendingSigningKeyPair()
    assertTrue(secrets.promotePendingPrivateKey())

    assertEquals(128, secrets.sign(PAYLOAD).length)
    assertSignsUnder(publicKeyHex.hexToBytesOrNull()!!)
  }

  @Test
  fun signingWithoutAPromotedKeyFailsInsteadOfUsingThePendingOne() {
    secrets.generatePendingSigningKeyPair()
    assertFalse(secrets.hasPrivateKey())
    assertSigningFails()
  }

  @Test
  fun legacyKeyMigratesToTheSameIdentity() {
    val seed = ByteArray(32) { it.toByte() }
    storeLegacy("private_key_pkcs8", pkcs8(seed))

    // Same identity, so the registered client_id and the server's copy of the public key both survive the upgrade.
    assertSignsUnder(publicKeyOf(seed))
    assertEquals(seed.toHex(), storedSecret("private_key_hex"))
    assertNull("legacy blob should be consumed", prefs.getString("private_key_pkcs8", null))
  }

  @Test
  fun v2LegacyKeyMigratesToTheSeedNotTheAppendedPublicKey() {
    val seed = ByteArray(32) { it.toByte() }
    val v2 = pkcs8(seed, withPublicKey = true)
    // Guard the fixture itself: if the tail weren't the public key this test would prove nothing.
    assertEquals(83, v2.size)
    assertArrayEquals(publicKeyOf(seed), v2.copyOfRange(v2.size - 32, v2.size))
    storeLegacy("private_key_pkcs8", v2)

    assertSignsUnder(publicKeyOf(seed))
    assertEquals(seed.toHex(), storedSecret("private_key_hex"))
  }

  @Test
  fun legacyBlobShorterThanAnEd25519Pkcs8IsRejectedNotSliced() {
    // A truncated but otherwise well-formed encoding: header and AlgorithmIdentifier
    // match, so only the size guard stands between this and copyOfRange running off
    // the end and throwing out of what callers treat as a nullable read.
    val truncated = pkcs8(ByteArray(32) { 8 }).copyOfRange(0, 40)
    storeLegacy("private_key_pkcs8", truncated)

    assertFalse(secrets.hasPrivateKey())
    assertSigningFails()
    assertNull(storedSecret("private_key_hex"))
  }

  @Test
  fun legacyBlobPresentReadsAsPresentWithoutMigrating() {
    storeLegacy("private_key_pkcs8", pkcs8(ByteArray(32) { 2 }))

    // The whole reason legacyPrivateKey() exists apart from migrateLegacyPkcs8Key():
    // the debug panel can report the key without converting it.
    assertTrue(secrets.hasPrivateKey())
    assertNull("a pure read must not migrate", storedSecret("private_key_hex"))
    assertNotNull(prefs.getString("private_key_pkcs8", null))
  }

  @Test
  fun legacyBlobThatIsNotAnEd25519Pkcs8IsRejectedAndKept() {
    // Valid base64 and long enough, but not an Ed25519 PKCS#8: the AlgorithmIdentifier
    // prefix doesn't match, so there is no encoding here to trust an offset against.
    storeLegacy("private_key_pkcs8", ByteArray(64) { 0x41 })

    assertFalse(secrets.hasPrivateKey())
    assertSigningFails()
    assertNull("must not invent a key from an unparseable blob", storedSecret("private_key_hex"))
    assertNotNull("unparseable key material must be preserved", prefs.getString("private_key_pkcs8", null))
  }

  @Test
  fun undecodableLegacyKeyIsRejectedAndKept() {
    prefs.edit().putString("private_key_pkcs8", "not base64 at all !!").apply()

    assertFalse(secrets.hasPrivateKey())
    assertSigningFails()
    assertNotNull(prefs.getString("private_key_pkcs8", null))
  }

  @Test
  fun currentFormatKeyWinsOverALingeringLegacyBlob() {
    val publicKeyHex = secrets.generatePendingSigningKeyPair()
    secrets.promotePendingPrivateKey()
    val current = storedSecret("private_key_hex")
    storeLegacy("private_key_pkcs8", pkcs8(ByteArray(32) { 9 }))

    assertSignsUnder(publicKeyHex.hexToBytesOrNull()!!)
    assertEquals("migration must not clobber the current key", current, storedSecret("private_key_hex"))
  }

  @Test
  fun corruptCurrentFormatKeyDoesNotFallThroughToTheLegacyBlob() {
    storeSecret("private_key_hex", "not hex")
    storeLegacy("private_key_pkcs8", pkcs8(ByteArray(32) { 5 }))

    // Falling through would resurrect an identity this install already superseded.
    assertFalse(secrets.hasPrivateKey())
    assertSigningFails()
  }

  @Test
  fun currentFormatKeyOfTheWrongLengthIsRejected() {
    // Well-formed hex, wrong size. Without the length guard this reaches Ed25519Sign,
    // which is not something a persisted-state parser should let happen.
    storeSecret("private_key_hex", ByteArray(16) { 1 }.toHex())

    assertFalse(secrets.hasPrivateKey())
    assertSigningFails()
  }

  @Test
  fun promotingClearsBothLegacySlots() {
    storeLegacy("private_key_pkcs8", pkcs8(ByteArray(32) { 6 }))
    storeLegacy("pending_private_key_pkcs8", pkcs8(ByteArray(32) { 4 }))
    secrets.generatePendingSigningKeyPair()

    assertTrue(secrets.promotePendingPrivateKey())

    assertNull("stale key material must not survive promotion", prefs.getString("pending_private_key_pkcs8", null))
    assertNull("the promoted key supersedes the legacy one", prefs.getString("private_key_pkcs8", null))
  }

  @Test
  fun deletingClearsBothTheCurrentAndLegacyKeys() {
    secrets.generatePendingSigningKeyPair()
    secrets.promotePendingPrivateKey()
    storeLegacy("private_key_pkcs8", pkcs8(ByteArray(32) { 7 }))

    secrets.deletePrivateKey()

    assertFalse(secrets.hasPrivateKey())
    assertNull(storedSecret("private_key_hex"))
    assertNull(prefs.getString("private_key_pkcs8", null))
  }

  @Test
  fun deletingAPendingKeyClearsTheLegacyPendingBlobToo() {
    secrets.generatePendingSigningKeyPair()
    storeLegacy("pending_private_key_pkcs8", pkcs8(ByteArray(32) { 3 }))

    secrets.deletePendingPrivateKey()

    assertFalse(secrets.promotePendingPrivateKey())
    assertNull(prefs.getString("pending_private_key_pkcs8", null))
  }

  @Test
  fun hexRoundTripsAndRejectsMalformedInput() {
    val bytes = ByteArray(32) { (it * 7).toByte() }
    assertArrayEquals(bytes, bytes.toHex().hexToBytesOrNull())
    assertNull("odd length", "abc".hexToBytesOrNull())
    assertNull("non-hex digit", "zz".hexToBytesOrNull())
    assertNull("empty", "".hexToBytesOrNull())
    // "-1".toInt(16) is a valid -1, so a naive pairwise parse would accept this.
    assertNull("signed pair", "-1".hexToBytesOrNull())
  }

  @Test
  fun keyMaterialIsStoredOnlyInTheSealedSlot() {
    secrets.generatePendingSigningKeyPair()
    assertNull("pending key must not be stored in plaintext", prefs.getString("pending_private_key_hex", null))
    assertTrue(secrets.promotePendingPrivateKey())

    assertNull("promoted key must not be stored in plaintext", prefs.getString("private_key_hex", null))
    // Assert the stored blob is genuinely the sealed form, not the key under another encoding:
    // decoding it must reproduce exactly what the AEAD produced for this slot.
    val stored = prefs.getString("private_key_hex_enc", null)
    assertNotNull(stored)
    val key = storedSecret("private_key_hex")!!
    assertArrayEquals(fakeAead.encrypt(key.toByteArray(), "private_key_hex".toByteArray()), Base64.getDecoder().decode(stored))
  }

  @Test
  fun hfTokenRoundTripsAndIsStoredEncrypted() {
    assertTrue(secrets.saveHfToken("  hf_secret_value  "))

    assertEquals("hf_secret_value", secrets.loadHfToken())
    assertNull("token must not be stored in plaintext", prefs.getString("hf_token", null))
    assertEquals("hf_secret_value", storedSecret("hf_token"))

    secrets.deleteHfToken()
    assertNull(secrets.loadHfToken())
    assertNull(prefs.getString("hf_token_enc", null))
  }

  @Test
  fun plaintextSecretsWrittenBeforeEncryptionAreMigratedInPlace() {
    // The upgrade path that matters: an install already registered under the previous
    // build keeps its identity instead of silently losing it and re-registering.
    val seed = ByteArray(32) { (it + 3).toByte() }
    storePlaintextSecret("private_key_hex", seed.toHex())
    storePlaintextSecret("hf_token", "legacy_token")

    assertSignsUnder(publicKeyOf(seed))
    assertEquals("legacy_token", secrets.loadHfToken())

    assertNull("plaintext key must be removed once migrated", prefs.getString("private_key_hex", null))
    assertNull("plaintext token must be removed once migrated", prefs.getString("hf_token", null))
    assertEquals(seed.toHex(), storedSecret("private_key_hex"))
  }

  @Test
  fun aCiphertextMovedBetweenSlotsIsRejected() {
    // The slot name is the AEAD's associated data, so a token blob dropped into the key
    // slot must fail to authenticate rather than decrypt into a bogus identity.
    assertTrue(secrets.saveHfToken(ByteArray(32) { 9 }.toHex()))
    prefs.edit().putString("private_key_hex_enc", prefs.getString("hf_token_enc", null)).apply()

    assertFalse(secrets.hasPrivateKey())
    assertSigningFails()
  }

  @Test
  fun withoutAKeystoreNothingIsPersistedInPlaintext() {
    // Degrade to "no stored secrets" rather than quietly writing the key unencrypted.
    val keyless = Secrets(context, aead = null)
    assertFalse("a keyless save must report failure, not just decline to write", keyless.saveHfToken("should_not_land"))

    assertNull(prefs.getString("hf_token", null))
    assertNull(prefs.getString("hf_token_enc", null))
    assertNull(keyless.loadHfToken())
    assertFalse(keyless.promotePendingPrivateKey())
  }

  @Test
  fun promotionDestroysNothingWhenTheKeyCannotBeSealed() {
    // The exact shape of the original defect. promotePendingPrivateKey is the only multi-slot editor
    // transaction: an implementation that cleared the pending and legacy slots before sealing would wipe
    // every copy of the identity in one apply(). A null AEAD cannot reach this, because the pending key
    // would not be readable either, so encryption has to fail while decryption still works.
    val pending = ByteArray(32) { (it + 17).toByte() }
    storeSecret("pending_private_key_hex", pending.toHex())
    storeLegacy("private_key_pkcs8", pkcs8(ByteArray(32) { 1 }))

    val sealFails = Secrets(context, FakeAead(failEncrypt = true))
    assertFalse(sealFails.promotePendingPrivateKey())

    assertEquals("the pending key must survive", pending.toHex(), storedSecret("pending_private_key_hex"))
    assertNotNull("the legacy blob must survive", prefs.getString("private_key_pkcs8", null))
    assertNull(prefs.getString("private_key_hex_enc", null))

    // Once the Keystore is back, promotion still works and yields the same identity.
    assertTrue(secrets.promotePendingPrivateKey())
    assertSignsUnder(publicKeyOf(pending))
  }

  @Test
  fun aPlaintextSecretStaysReadableWhileTheKeystoreIsDown() {
    // Documented behaviour, and the right one: a not-yet-migrated install keeps working through an
    // outage rather than losing its identity to one. Pinned so nobody "fixes" the read path to refuse it.
    val seed = ByteArray(32) { (it + 23).toByte() }
    storePlaintextSecret("private_key_hex", seed.toHex())
    storePlaintextSecret("hf_token", "still_usable")

    val keyless = Secrets(context, aead = null)
    assertTrue(keyless.hasPrivateKey())
    assertEquals("still_usable", keyless.loadHfToken())
    assertEquals("plaintext must not be migrated with no Keystore", seed.toHex(), prefs.getString("private_key_hex", null))
  }

  @Test
  fun savingATokenReportsFailureRatherThanSilentlyDroppingIt() {
    storeSecret("hf_token", "original")
    val sealFails = Secrets(context, FakeAead(failEncrypt = true))

    assertFalse("a token that cannot be sealed must not report success", sealFails.saveHfToken("replacement"))
    assertEquals("the previous token must be left intact", "original", storedSecret("hf_token"))
  }

  @Test
  fun keyGenerationFailsLoudlyWhenItCannotBeStored() {
    // Returning a public key the caller would register, while storing no private key, would
    // leave the server holding an identity no device can ever sign for.
    val keyless = Secrets(context, aead = null)
    val error = runCatching { keyless.generatePendingSigningKeyPair() }.exceptionOrNull()

    assertTrue("expected IllegalStateException, got $error", error is IllegalStateException)
    assertNull(prefs.getString("pending_private_key_hex", null))
    assertNull(prefs.getString("pending_private_key_hex_enc", null))
  }

  @Test
  fun anUnavailableKeystoreDoesNotDestroyAnAlreadyStoredSecret() {
    // The regression that matters most: an earlier version cleared both the ciphertext and the
    // plaintext when encryption failed, so one transient Keystore outage erased the identity.
    val seed = ByteArray(32) { (it + 5).toByte() }
    storeSecret("private_key_hex", seed.toHex())
    storeSecret("hf_token", "keep_me")

    val keyless = Secrets(context, aead = null)
    assertFalse(keyless.saveHfToken("cannot_be_sealed"))
    assertFalse(keyless.hasPrivateKey())

    // Nothing was readable through the keyless instance, but nothing was destroyed either.
    assertEquals(seed.toHex(), storedSecret("private_key_hex"))
    assertEquals("keep_me", storedSecret("hf_token"))
    assertSignsUnder(publicKeyOf(seed))
  }

  @Test
  fun anUnavailableKeystoreLeavesALegacyBlobIntact() {
    // migrateLegacyPkcs8Key removes the blob; it must not do so when the replacement write is a no-op,
    // or an API-37+ install loses a recoverable identity on one unlucky boot.
    val seed = ByteArray(32) { (it + 11).toByte() }
    storeLegacy("private_key_pkcs8", pkcs8(seed))

    val keyless = Secrets(context, aead = null)
    runCatching { keyless.sign(PAYLOAD) }

    assertNotNull("legacy key material must survive a Keystore outage", prefs.getString("private_key_pkcs8", null))
    // The blob is still convertible once the Keystore is back.
    assertSignsUnder(publicKeyOf(seed))
    assertEquals(seed.toHex(), storedSecret("private_key_hex"))
  }

  @Test
  fun readingADiagnosticDoesNotMigratePlaintext() {
    // hasPrivateKey is documented as a pure read. It is reached from the settings panel, so a write
    // there would open the destructive window the previous tests pin shut.
    storePlaintextSecret("private_key_hex", ByteArray(32) { 4 }.toHex())

    assertTrue(secrets.hasPrivateKey())
    assertNotNull("a pure read must not migrate", prefs.getString("private_key_hex", null))
    assertNull(prefs.getString("private_key_hex_enc", null))
  }

  /**
   * Models the AEAD semantics [Secrets] actually depends on, which Robolectric's Keystore can't provide: the associated data is bound into the
   * ciphertext and a mismatch throws, so the slot-binding test exercises the real contract rather than a stub that ignores it.
   *
   * It deliberately does NOT model two properties of the real AES-256-GCM, and no test here may rely on them: it provides no confidentiality (the
   * plaintext sits verbatim after the AAD prefix), and it is deterministic (real GCM picks a fresh nonce per call, so re-encrypting would not
   * reproduce a stored blob byte for byte).
   *
   * [failEncrypt] models the case the fail-safe write paths exist for: a Keystore that resolved at construction and then failed, which is a different
   * branch of `seal` from a null AEAD and the only one that reaches the multi-slot editor transactions.
   */
  private class FakeAead(private val failEncrypt: Boolean = false) : Aead {
    override fun encrypt(plaintext: ByteArray, associatedData: ByteArray): ByteArray {
      if (failEncrypt) throw GeneralSecurityException("keystore unavailable")
      return byteArrayOf(associatedData.size.toByte()) + associatedData + plaintext
    }

    override fun decrypt(ciphertext: ByteArray, associatedData: ByteArray): ByteArray {
      // Every malformed input must surface as GeneralSecurityException, like a real AEAD: production code
      // funnels these through runCatching, so a fake that threw IndexOutOfBounds would hide the difference.
      val adSize = ciphertext.firstOrNull()?.toInt() ?: throw GeneralSecurityException("empty ciphertext")
      if (adSize < 0 || 1 + adSize > ciphertext.size) throw GeneralSecurityException("truncated ciphertext")
      val ad = ciphertext.copyOfRange(1, 1 + adSize)
      if (!ad.contentEquals(associatedData)) throw GeneralSecurityException("associated data mismatch")
      return ciphertext.copyOfRange(1 + adSize, ciphertext.size)
    }
  }

  private companion object {
    /** A `v1` signed payload, the shape `ManagementClient` hands [Secrets.sign]. Opaque bytes here — these tests are about key storage. */
    val PAYLOAD = signedPayload("GET", "/clients/me", "2026-07-27T12:00:00Z", "ev1_a3f8", "0f1e2d3c4b5a69788796a5b4c3d2e1f0")
  }
}
