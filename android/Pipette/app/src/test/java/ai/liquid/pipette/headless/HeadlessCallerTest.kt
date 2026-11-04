package ai.liquid.pipette.headless

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pins the guard on the headless surface (PIP-437).
 *
 * This is the security-relevant half of the feature. [HeadlessRunActivity] has to be exported for `adb shell am start` to reach it, and on Android
 * exported means reachable by any installed app; since a headless run submits by default, an unguarded surface would let an arbitrary app upload
 * under this device's identity.
 *
 * Which callers are admitted is decided by the framework against the manifest's `android:permission`, not by code, so it is not what these tests can
 * reach. What they pin is the half that *is* code: the value of that permission, and that the activity fails closed if the platform reports the
 * attribute missing or changed. The admit/refuse behaviour itself was verified on an API 36 device, where `adb shell am start` is admitted and an
 * ordinary app's `startActivity` on the same component dies with `SecurityException: ... requires android.permission.WRITE_SECURE_SETTINGS`.
 */
class HeadlessCallerTest {

  @Test
  fun theGuardIsAPermissionAdbHoldsAndAppsCannotGet() {
    // Pinned as a value because the choice is the design: it has to be a platform permission (a custom
    // `signature` one would exclude adb, which holds no app-defined permissions) that is not grantable
    // to a third-party app (`signature|privileged`). Changing it means re-checking both halves.
    assertEquals("android.permission.WRITE_SECURE_SETTINGS", HeadlessCaller.REQUIRED_CALLER_PERMISSION)
  }

  @Test
  fun theDeclaredGuardIsAccepted() {
    assertTrue(HeadlessCaller.check(HeadlessCaller.REQUIRED_CALLER_PERMISSION) is HeadlessCaller.Verdict.Allowed)
  }

  @Test
  fun anUnguardedComponentIsRefused() {
    // Deleting android:permission from the manifest is a one-line edit that would otherwise widen an
    // exported, submitting surface to every app on the device without failing anything.
    val verdict = HeadlessCaller.check(null)

    assertTrue("an exported activity with no permission must refuse, got $verdict", verdict is HeadlessCaller.Verdict.Refused)
    assertTrue((verdict as HeadlessCaller.Verdict.Refused).reason.contains(HeadlessCaller.REQUIRED_CALLER_PERMISSION))
  }

  @Test
  fun aWeakerGuardIsRefused() {
    // A permission an ordinary app can hold is not a guard: INTERNET is granted to anything that asks.
    val verdict = HeadlessCaller.check("android.permission.INTERNET")

    assertTrue("a normal-protection permission must not pass as the guard, got $verdict", verdict is HeadlessCaller.Verdict.Refused)
    assertTrue((verdict as HeadlessCaller.Verdict.Refused).reason.contains("android.permission.INTERNET"))
  }
}
