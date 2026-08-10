package ai.liquid.pipette

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AuthGateTest {
  private fun registration(clerkUserId: String? = null, clerkEmail: String? = null) =
    RegistrationData(
      clientId = "client-1",
      status = "active",
      serverUrl = "https://example.test",
      organization = "org",
      contactEmail = "device@example.test",
      registeredAt = "2026-01-01T00:00:00Z",
      clerkUserId = clerkUserId,
      clerkPrimaryEmail = clerkEmail,
    )

  private val signedIn = ClerkState.SignedIn(userId = "user_A", email = "a@example.test", sessionId = "sess_1")

  @Test
  fun bypassAlwaysReady() {
    assertEquals(AuthGate.Ready, reduceAuthGate(ClerkState.Loading, null, bypass = true))
    assertEquals(AuthGate.Ready, reduceAuthGate(ClerkState.SignedOut, null, bypass = true))
    // Even a mismatch is suppressed under bypass.
    assertEquals(AuthGate.Ready, reduceAuthGate(signedIn, registration(clerkUserId = "user_B"), bypass = true))
  }

  @Test
  fun loadingAndInitErrorPassThrough() {
    assertEquals(AuthGate.Loading, reduceAuthGate(ClerkState.Loading, null, bypass = false))
    assertEquals(AuthGate.InitError("boom"), reduceAuthGate(ClerkState.InitError("boom"), null, bypass = false))
  }

  @Test
  fun signedOutGatesToSignedOut() {
    assertEquals(AuthGate.SignedOut, reduceAuthGate(ClerkState.SignedOut, null, bypass = false))
  }

  @Test
  fun signedInWithoutRegistrationIsReady() {
    assertEquals(AuthGate.Ready, reduceAuthGate(signedIn, null, bypass = false))
  }

  @Test
  fun signedInWithUnlinkedRegistrationIsReady() {
    // Registered before Clerk linking (clerkUserId == null) → not a mismatch.
    assertEquals(AuthGate.Ready, reduceAuthGate(signedIn, registration(clerkUserId = null), bypass = false))
  }

  @Test
  fun signedInMatchingLinkedAccountIsReady() {
    assertEquals(AuthGate.Ready, reduceAuthGate(signedIn, registration(clerkUserId = "user_A"), bypass = false))
  }

  /**
   * What the sign-out fix buys, stated over the reducer: a device pinned to one account admits another once the link is dropped. The gate half only.
   * That signing out drops the link is `ShellViewModel.unlinkRegistration`, private to an AndroidViewModel whose sign-out needs the SDK, so it has no
   * unit coverage. Without the unlink the second account is stuck on [AuthGate.Mismatch], whose only other exit deletes the signing key and orphans
   * every result already submitted.
   */
  @Test
  fun unlinkedRegistrationAdmitsADifferentAccount() {
    val pinnedToB = registration(clerkUserId = "user_B", clerkEmail = "b@example.test")
    assertTrue(reduceAuthGate(signedIn, pinnedToB, bypass = false) is AuthGate.Mismatch)
    assertEquals(AuthGate.Ready, reduceAuthGate(signedIn, pinnedToB.withoutClerkLink(), bypass = false))
  }

  @Test
  fun withoutClerkLinkKeepsTheUploadIdentity() {
    val linked =
      registration(clerkUserId = "user_B", clerkEmail = "b@example.test").copy(clerkSessionId = "sess_B", clerkLinkedAt = "2026-01-02T00:00:00Z")
    val unlinked = linked.withoutClerkLink()

    assertNull(unlinked.clerkUserId)
    assertNull(unlinked.clerkSessionId)
    assertNull(unlinked.clerkPrimaryEmail)
    // Dates the current link, so it cannot outlive it: withClerkLink preserves an existing value, and the next account would inherit this timestamp.
    assertNull(unlinked.clerkLinkedAt)
    // The server-issued half is untouched, which is what makes unlinking safe where deleting the registration isn't.
    assertEquals("client-1", unlinked.clientId)
    assertEquals("https://example.test", unlinked.serverUrl)
    assertEquals("org", unlinked.organization)
    assertEquals("2026-01-01T00:00:00Z", unlinked.registeredAt)
  }

  @Test
  fun signedInWithDifferentLinkedAccountIsMismatch() {
    val gate = reduceAuthGate(signedIn, registration(clerkUserId = "user_B", clerkEmail = "b@example.test"), bypass = false)
    assertTrue(gate is AuthGate.Mismatch)
    gate as AuthGate.Mismatch
    assertEquals("b@example.test", gate.linkedEmail)
    assertEquals("a@example.test", gate.currentEmail)
  }
}
