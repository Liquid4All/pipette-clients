# Preparing an App Store submission

What App Review needs from us, and why the reviewer credentials are shaped the
way they are. For building and archiving the app in general, see
[build.md](build.md); for the Clerk settings themselves see
[build.md § Clerk configuration](build.md#clerk-configuration).

## The two gates a reviewer hits

The app shows nothing until a reviewer gets through both of these, and each one
has to be passable with only what we put in App Store Connect:

1. **Clerk sign-in**: `ClerkAuthGateView` wraps everything in
   `EmailCodeSignInView`, with no dismiss and no guest path. Its main button
   says **Register**, but it does not only register: one `sendEmailCode` call
   signs an existing address in and registers an unknown one, and either way
   the next screen asks for an emailed 6-digit code. Only **Sign in with a
   password** avoids the mailbox, which is why the reviewer is sent there.
2. **Device registration**: `SetupView` then asks for an organization name and
   which collector to report to. **Liquid AI** is preselected and is the entire
   reviewer path; **Custom** is what reveals a URL field and an optional pre-auth
   key. A reviewer who leaves the picker alone registers against Liquid's
   collector with no key, so the registration they create is still **keyless**.
   But the control is on screen now, so a reviewer who does touch it can point
   the device somewhere that answers nothing.

The second gate is the one that actually causes rejections. Sign-in is easy to
remember to document; "the reviewer signed in and then sat on a pending screen"
reads to App Review as an incomplete app under Guideline 2.1.

## Reviewer sign-in: a password account

> [!IMPORTANT]
> Reviewers work the App Review Information fields. With "Sign-in required"
> checked they expect a username *and* a password, and they will not go find a
> self-serve path instead. An empty credential block reads as an incomplete
> submission. This has already happened to us on another submission that offered
> Sign in with Apple and passwordless email links: review still asked for
> credentials.

So whatever we do has to produce a literal username/password pair. That rules
out Sign in with Apple, email links, passkeys, and sign-in tokens (Clerk's
`signInWithTicket` is single-use, and there is nowhere in App Store Connect to
put a link).

### Why not Clerk's test mode

Clerk reserves `+clerk_test` email addresses that accept the fixed code
`424242`, which would fit the two fields exactly. We don't use it: test mode is
on by default only for *development* instances, and every iOS build configuration
points at the live instance, so it would have to be enabled on **production**,
which [Clerk's own docs](https://clerk.com/docs/guides/development/testing/test-emails-and-phones)
call "highly discouraged."

The disqualifying part is not the risk, it's the lifecycle. Test-mode
credentials work only while the toggle is on, and a live app can be re-reviewed
on any new build, resubmission, or appeal at a time we don't choose. Getting
that wrong costs a review cycle for a "we were unable to sign in" rejection, so
in practice the toggle would have to stay on permanently: a standing
verification bypass for the whole instance.

### What we do instead

Enable password as a first factor and keep one long-lived review account. A
password is an ordinary credential, it needs no per-submission ritual, and it
works on every future review.

In the Clerk Dashboard, on the `clerk.liquid.ai` production instance:

1. Enable **Password** as a first factor alongside email code.
2. Under **Users**, create the review account with an address we control, and
   set a password.
3. Confirm its email shows as **verified**. If it doesn't, the first password
   sign-in still demands an emailed code and the reviewer is stuck.
4. Confirm no MFA is enforced for the account.

Then verify it for real before submitting: on a **Release** build, from a clean
install, sign in with only the password; no code, no other device.

The reviewer reaches it through **Sign in with a password**, under the Register
button on the sign-in screen: type the address, tap that, type the password.
Nothing about the path is reviewer-specific; any account with a password can
use it, which is the same choice Android made for the Play Console's App access
form. Two consequences worth holding on to:

- **No address is special.** An earlier build revealed the password field only
  for one hard-coded address, which meant changing the review account silently
  broke review. It no longer does, so the credentials in App Store Connect can
  be any account that has a password.
- Password still has to stay enabled as a first factor on the Clerk instance.
  The client picks the strategy; the dashboard decides whether it is allowed.

**No MFA on the review account.** The gate can now answer a second factor
(email code, SMS, authenticator app, backup code), but every one of those needs
a device or mailbox the reviewer doesn't have. Enrolling the review account in
MFA locks review out.

**Client Trust is on, and the review account is somehow exempt from it.** Both
halves of that were measured, and the second half is not understood, which is why
this section prescribes a check rather than a conclusion.

Client Trust challenges password sign-ins from a device Clerk has not seen
before, which is the exact path a reviewer is sent down, on the exact kind of
install a reviewer uses. Answering it needs the code Clerk sends to the account's
own mailbox. The gate can now answer the challenge
(`EmailAuthModel.completeSignIn` admits the status and parks the second-factor
step), but that does not help a reviewer, who has the credentials and not the
mailbox. If the review account were ever challenged, review would be blocked.

Measured on 2026-08-05, against the production instance, on Android (the two
clients share this behavior because it is Clerk's, not ours):

- An ordinary `@liquid.ai` account, password sign-in: **challenged**, heading
  "Confirm this device", code emailed.
- The App Review account, password sign-in, **after wiping app data** so the
  Clerk client was brand new: **not challenged**, straight through.

The wipe matters, because Client Trust keys off the Clerk *client*, which lives
in app data and persists across a reinstall. A sign-in on a warm client proves
nothing. This one was cold, so the exemption is real and not an artifact.

On iOS the client is stored somewhere a reinstall reaches even less reliably: in
the system keychain, as a generic-password item
(`ClerkKit/Storage/Keychain/SystemKeychain.swift`, accessible
`AfterFirstUnlockThisDeviceOnly`). Whether deleting an app takes its keychain
items with it is Apple's call, has changed between releases, and is not something
this app controls, so "delete and reinstall" is not a way to get a cold client
here. Read the check below with that in mind: it names what does work.

What nobody has established is **why** that account is exempt, and until someone
does, there is no reason to expect it to stay that way. It is not an app-side
behavior and no code change here affects it.

Two traps worth naming:

- **`GET /v1/environment` cannot tell you whether Client Trust is on.** The
  setting is not in the payload. The `auth_config.native_settings.trusted_device_*`
  flags that look like it are Clerk's native trusted-device *enrollment* feature,
  and they all read `false` while Client Trust is demonstrably firing.
- **A sign-in on an install that has used that account before proves nothing**,
  for the client reason above.
- **Deleting the app is not that install becoming new**, on iOS specifically. The
  client is in the keychain, not in the app container this removes. A check run
  that way can pass on a warm client and read as an all-clear it did not earn,
  which is the one failure mode that makes this check worse than not running it.

**So run this before every submission, alongside step 4 above:** on a simulator
erased with `xcrun simctl erase <udid>` (Device > Erase All Content and Settings
does the same), or on a device that has never signed into Pipette with this
account, install a fresh build and sign in with the exact credentials going into
App Store Connect, through **Sign in with a password**. What either route has to
buy you is a keychain this account has never signed in against, since that is
where the client is: the erase clears it, and a device that has never run Pipette
never had one. Installing over a keychain that has is the case this check cannot
read.

- Straight to the setup screen: review can get in. This is the expected result
  today.
- A "Confirm this device" step: **review is blocked**, and the exemption has
  lapsed. You will be as stuck as the reviewer, since neither of you can read
  that mailbox, which is what makes this check trustworthy. Fix it Clerk-side
  before submitting, by restoring the exemption, by handing the mailbox to
  whoever prepares the submission, or by turning Client Trust off. The dashboard
  is the only place any of those can be done.

## Reviewer device registration

Provision this alongside the account, or the reviewer gets in and stalls:

- **Confirm a keyless registration comes up `approved`.** The app has no field
  for a pre-auth key on the Liquid collector: it appears only after selecting
  **Custom**, which the reviewer has no reason to touch. So the key still cannot
  be handed to a reviewer through the path they take, and approval
  has to come from the collector side. If a keyless registration lands `pending`,
  the reviewer sits on a broken-looking screen and this is a Guideline 2.1
  rejection. Verify it before submitting: register a clean install and read
  **Settings → Debugging → Status** (the collector's approval state; it was
  formerly on the account card as "Authentication Status", which read as though
  it described the Clerk session).

  > [!IMPORTANT]
  > That row lives in the Debugging card, which is compiled out of an App Store
  > archive, so **do this check on an Internal Testing build**, where
  > `PIPETTE_DEBUG_UI` is set. An App Store build cannot show it. See
  > [build.md](build.md#the-settings-debugging-card) for the flag.
- Walk the whole path on a clean install and confirm the registration lands
  `approved` and a benchmark actually completes.
- Point the reviewer at a **small** model. They won't wait out a multi-gigabyte
  download and a long run.

## App Store Connect: App Review Information

Check "Sign-in required" and fill in *both* fields: never just the notes.

```text
Sign-in:
  User name: <REVIEW_EMAIL>
  Password:  <REVIEW_PASSWORD>

Notes:
  1. Type the user name above into Email, then tap "Sign in with a password"
     (under the Register button) and enter the password. Do not tap Register:
     for an existing address it emails a 6-digit code and waits for it, and
     that mailbox isn't yours.
  2. On the setup screen, Organization name: <ORG>
  3. Tap Register.
  4. Jobs -> New Job -> <SMALL_MODEL> to run a benchmark on-device.

  The app measures LLM inference performance on the device itself; runs are
  local and take a few minutes.

  See the attached screen recording for the full flow.
```

Use the attachment slot in App Review Information for a 30–60 second screen
recording through to a completed benchmark. For a flow with a non-obvious
second gate, that converts "I couldn't get in" into "oh, that's what I do next."

## Pre-submission checklist

**No `CLERK_*` overrides in the target build settings.** Settings edited in
Xcode's UI land in `project.pbxproj` and override the xcconfig in *every*
configuration, Release included; a dev instance set this way ships in the
archive. Confirm the live values win:

```bash
xcodebuild -showBuildSettings -project ios/Pipette/Pipette.xcodeproj -scheme Pipette -configuration Release 2>/dev/null | grep CLERK
```

Expect `pk_live_Y2xlcmsubGlxdWlkLmFpJA` and `clerk.liquid.ai`.

**No private-thermal build.** `PIPETTE_PRIVATE_THERMAL` must be unset for the
archive. See [private-thermal-release-build.md](private-thermal-release-build.md).
The private API is resolved with `dlsym`, so there is no undefined symbol to
look for; the name appears as a string in `__TEXT` instead:

```bash
strings /tmp/Pipette.xcarchive/Products/Applications/Pipette.app/Pipette | grep -i IOHIDEventSystemClient
```

Expect no output.

**Account deletion is reachable.** Guideline 5.1.1(v) requires in-app deletion
for any app supporting account creation. Settings → Delete Account calls
ClerkKit's `user.delete()`.

**`PrivacyInfo.xcprivacy` matches reality**. The required-reason APIs listed
are the ones the shipping build actually uses.

**Sign in with Apple, if applicable.** We don't need it for the review account,
but if any social provider is ever enabled in Clerk, Guideline 4.8 requires an
equivalent login option alongside it. ClerkKit exposes
`clerk.auth.signInWithApple()` for that.

## After approval

Nothing to undo. That is the point of choosing a password account over test
mode. Keep the review account, its password, and its pre-auth key alive: every
future submission is reviewed again, and they need to keep working without
anyone remembering to flip a switch first.
