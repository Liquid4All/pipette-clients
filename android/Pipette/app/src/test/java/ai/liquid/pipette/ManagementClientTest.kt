package ai.liquid.pipette

import java.net.URL
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/** Pins the `v1` request-signing inputs shared with the management server; see mgmt `docs/authentication.md` §2.1. */
class ManagementClientTest {
  /**
   * The payload is a byte-for-byte wire contract with the server, which rebuilds the same string from the request it received and verifies against
   * it. Field order, the `v1` tag, and the newline delimiters are all load-bearing: get any of them wrong and every authenticated request 401s.
   */
  @Test
  fun signedPayloadIsSixNewlineSeparatedFields() {
    assertEquals(
      "v1\nGET\n/clients/me?page=2\n2026-03-10T12:00:00Z\nev1_a3f8\n0f1e2d3c4b5a69788796a5b4c3d2e1f0",
      signedPayload("GET", "/clients/me?page=2", "2026-03-10T12:00:00Z", "ev1_a3f8", "0f1e2d3c4b5a69788796a5b4c3d2e1f0"),
    )
  }

  /**
   * The server rejects an empty or repeated nonce, and reads the payload as newline-delimited fields. Hex satisfies all three: never empty, never
   * carrying a newline that could forge a field boundary, and fresh per call.
   */
  @Test
  fun generateNonceIsFreshHexOf32Chars() {
    val first = generateNonce()

    assertEquals(32, first.length)
    assertTrue(first, first.all { it in '0'..'9' || it in 'a'..'f' })
    assertNotEquals(first, generateNonce())
  }

  /**
   * The signature must cover the target the *server* sees, so a configured server URL carrying a path prefix has to appear in the signed payload —
   * signing the bare endpoint path would 401 every request against such a deployment.
   */
  @Test
  fun requestTargetCarriesThePathPrefixAndTheQuery() {
    assertEquals("/api/clients/me", URL("https://mgmt.example.com/api/clients/me").requestTarget())
    assertEquals("/clients/me?page=2", URL("https://mgmt.example.com/clients/me?page=2").requestTarget())
  }

  @Test
  fun endpointKeepsAServerUrlThatReachesTheWireAsWritten() {
    assertEquals("https://mgmt.example.com/api/clients/me", endpoint("https://mgmt.example.com/api/", "/clients/me"))
    // Already percent-encoded stays untouched: `URL.getPath()` reports it raw, which is what OkHttp sends.
    assertEquals("https://mgmt.example.com/a%20b/clients/me", endpoint("https://mgmt.example.com/a%20b", "/clients/me"))
  }

  /**
   * A server URL OkHttp would rewrite has to be refused, not signed: the signature covers the request target, so signing the pre-canonicalization
   * form would 401 every authenticated request against that deployment while unsigned registration kept working.
   */
  @Test
  fun endpointRejectsAServerUrlOkHttpWouldCanonicalize() {
    val rewritten =
      listOf(
        "https://mgmt.example.com/a b",
        "https://mgmt.example.com/naïve",
        "https://mgmt.example.com/api/..",
        "https://mgmt.example.com/api/%2e%2e",
      )
    for (serverUrl in rewritten) {
      assertThrows(serverUrl, IllegalArgumentException::class.java) { endpoint(serverUrl, "/clients/me") }
    }
  }
}
