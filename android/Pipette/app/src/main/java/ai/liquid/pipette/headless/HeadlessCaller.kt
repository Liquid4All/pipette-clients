package ai.liquid.pipette.headless

/**
 * Who is allowed to start a headless run, and the check that the guard is actually in force.
 *
 * [HeadlessRunActivity] must be `exported` to be reachable by `adb shell am start`, and on Android exported means reachable by *any* app installed on
 * the device, not just by a host-side tool. That is a wider surface than iOS's equivalent, which is argv-gated behind `devicectl process launch` and
 * so needs developer mode and a host. Since a headless run defaults to submitting, an unguarded surface would let an arbitrary app trigger an upload
 * under this device's identity.
 *
 * **The guard is `android:permission` on the component, not code.** The manifest names [REQUIRED_CALLER_PERMISSION] and the framework enforces it
 * before this process is even asked to create the activity, which is the earliest and least forgeable place the check can live.
 *
 * Picking the permission is the whole trick. A *custom* `signature` permission does not work: `adb shell` runs as the `shell` uid, which is not an
 * app and holds no app-defined permissions, so the framework would reject the very automation path this feature exists for. A *platform* permission
 * that `shell` already holds does work. `WRITE_SECURE_SETTINGS` is one: `com.android.shell` holds it, which is what backs `adb shell settings put
 * secure`. AOSP declares it `signature|privileged|development`, so no third-party app *ships* holding it, though the `development` flag means one can
 * be granted it deliberately over adb, which is the residual surface below. Verified on an API 36 device: `adb shell am start` is admitted, and an
 * ordinary app calling `startActivity` on the same component is refused by the framework with `SecurityException: Permission Denial: ... requires
 * android.permission.WRITE_SECURE_SETTINGS`.
 *
 * **Why the caller's uid is not consulted.** The obvious runtime check, `Activity.getLaunchedFromUid()`, cannot see adb. It reports the launching uid
 * only when the launcher opted in via `ActivityOptions.setShareIdentityEnabled`, when the launcher shares this app's uid, or when this app is
 * platform-signed. `am start` does none of those and has no flag to ask for it, so it returns `Process.INVALID_UID` (confirmed on API 36) and a
 * uid-based guard refuses exactly the caller it was written to admit. `getLaunchedFromPackage` and `getCallingPackage` are empty for the same launch,
 * and `getReferrer` is caller-settable via `Intent.EXTRA_REFERRER`, so none of them is a boundary.
 *
 * What remains for code is making sure the manifest guard was not dropped: [check] reads the permission the platform actually has on record for this
 * component and refuses if it is not the expected one, so deleting the attribute turns the surface off rather than opening it.
 *
 * Residual surface, stated plainly: a privileged system app on the device image may hold `WRITE_SECURE_SETTINGS`, and anyone with adb can grant it to
 * an app through the permission's `development` flag. The first already implies authority over the device well beyond starting a benchmark. The
 * second does not, quite: a granted app reaches every verb here, including the deletes #1243 added, without being able to `pm uninstall` anything.
 * What it takes to get there is a deliberate `adb shell pm grant`, which is the same access that could install a different build of this app.
 */
object HeadlessCaller {

  /**
   * The permission a caller must hold, declared on the activity in `AndroidManifest.xml`.
   *
   * Not a permission this app requests or holds. It names what the *caller* must have, which is why it is a platform permission rather than one of
   * ours: the set of holders has to include `adb` and exclude every third-party app, and only the platform defines such a set.
   */
  const val REQUIRED_CALLER_PERMISSION = "android.permission.WRITE_SECURE_SETTINGS"

  sealed interface Verdict {
    data object Allowed : Verdict

    data class Refused(val reason: String) : Verdict
  }

  /**
   * Confirms the framework gate is in force, given [declaredPermission], the permission the `PackageManager` reports for this component
   * (`ActivityInfo.permission`), or null when the manifest declares none or the component could not be read.
   *
   * Reading it back rather than trusting the manifest is what makes this worth running: the attribute is one line in an XML file that no test would
   * otherwise miss, and losing it would silently widen an exported, submitting surface to every app on the device. Failing closed costs one
   * `PackageManager` call per headless invocation and turns that edit into a refusal with a reason.
   */
  fun check(declaredPermission: String?): Verdict =
    if (declaredPermission == REQUIRED_CALLER_PERMISSION) {
      Verdict.Allowed
    } else {
      Verdict.Refused(
        "this activity is exported but the platform reports it guarded by " +
          "${declaredPermission ?: "no permission"} rather than $REQUIRED_CALLER_PERMISSION; refusing rather than running an unguarded surface " +
          "that any installed app could reach"
      )
    }
}
