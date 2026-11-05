package ai.liquid.pipette

import com.google.crypto.tink.Aead
import java.security.GeneralSecurityException

/**
 * Models the AEAD semantics [Secrets] actually depends on, which Robolectric's Keystore cannot provide: the associated data is bound into the
 * ciphertext and a mismatch throws, so a slot-binding test exercises the real contract rather than a stub that ignores it.
 *
 * It deliberately does NOT model two properties of the real AES-256-GCM, and no test may rely on them: it provides no confidentiality (the plaintext
 * sits verbatim after the AAD prefix), and it is deterministic (real GCM picks a fresh nonce per call, so re-encrypting would not reproduce a stored
 * blob byte for byte).
 *
 * [failEncrypt] models the case the fail-safe write paths exist for: a Keystore that resolved at construction and then failed, which is a different
 * branch of `seal` from a null AEAD and the only one that reaches the multi-slot editor transactions.
 *
 * Shared rather than private to `SecretsTest` because every test that drives a real [Secrets] needs the same fake, and two copies of it could
 * disagree about which of the properties above they model.
 */
internal class FakeAead(private val failEncrypt: Boolean = false) : Aead {
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
