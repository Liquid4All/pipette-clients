package ai.liquid.pipette

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * [isCompleteKey] is the switch that decides whether a build has auth at all: it gates whether `PipetteApp` initializes the SDK, and a false answer
 * leaves `AppContainer.clerkAuth` null and opens the gate.
 *
 * Historically a blank key was the PIP-391 fresh-install regression, and the remedy was to bake the production key into the build as a default. That
 * default is gone: no key is now a supported configuration meaning "this build has no sign-in", which is what lets a checkout with no Clerk instance
 * build and run. The distinction this predicate still has to make is between *unconfigured* and *broken*.
 */
class ClerkConfigurationTest {
  @Test
  fun aRealPublishableKeyIsUsable() {
    assertTrue(isCompleteKey("pk_live_abc123"))
    assertTrue(isCompleteKey("pk_test_abc123"))
  }

  @Test
  fun noKeyMeansNoAuthRatherThanABrokenBuild() {
    // Both answers are false, but read this as "auth is off", not "the gate should show an error". The caller that turns this into UI is the
    // authGate fallback in ShellViewModel, which resolves to Ready.
    assertFalse(isCompleteKey(""))
    assertFalse(isCompleteKey("   "))
  }

  @Test
  fun anUnsubstitutedPlaceholderIsNotAKey() {
    // Distinct from blank in meaning: a substitution was configured and did not run, so the build is broken rather than unconfigured. Treating it as
    // usable would hand the literal string to Clerk.initialize.
    assertFalse(isCompleteKey("\$(CLERK_PUBLISHABLE_KEY)"))
    assertFalse(isCompleteKey("pk_\$(SOMETHING)"))
  }

  @Test
  fun aShellOrGradleStylePlaceholderIsAlsoNotAKey() {
    // `$(...)` is Xcode's xcconfig spelling, which this predicate mirrors from iOS. A key that arrives through Gradle or a workflow's shell fails the
    // other way, so both spellings have to be rejected for the build-time guards to be able to enforce the same rule this predicate does.
    assertFalse(isCompleteKey("\${CLERK_PUBLISHABLE_KEY}"))
    assertFalse(isCompleteKey("pk_\${SOMETHING}"))
  }

  @Test
  fun aKeyContainingAnOrdinaryDollarSignIsStillUsable() {
    // Only the two placeholder openings are disqualifying. A bare `$` is not a substitution marker, and rejecting it would turn a working key into a
    // build failure.
    assertTrue(isCompleteKey("pk_live_abc\$123"))
  }
}
