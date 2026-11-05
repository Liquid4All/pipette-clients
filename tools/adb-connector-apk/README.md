# Pipette ADB Connector

A lab utility for the Android benchmark fleet. It re-enables **wireless debugging** after a
reboot, so a phone that restarts comes back on adb with no person at the phone to turn the toggle
on in Developer options.

Deliberately a **standalone Gradle build**, not a module of `android/Pipette`: this app declares
`WRITE_SECURE_SETTINGS` and turns debugging on by itself, which must never be reachable from the
shipping Pipette artifact through a shared source set or a merged manifest.

## The problem it solves

Phones "lose wifi adb" without ever leaving wifi. Measured on the fleet: all five devices
answered ping at 0% packet loss while four of five refused their adb port. `Connection refused`
is a TCP RST: the network is fine, and the phone's adbd listener is simply gone.

### The actual cause: a boot race, not a missing setting

Android *does* try to restore wireless debugging at boot. It loses a race with wifi association
and then switches itself off. From a real boot on one phone:

```
06:58:29.072  SettingsProvider: PUT(/global/adb_wifi_enabled)  callingPackage:android
06:58:31.227  adbd: adbwifi started on port 39363
06:58:31.324  AdbDebuggingManager: Not connected to any wireless network. Not enabling adbwifi.
06:58:31.324  SettingsProvider: PUT(/global/adb_wifi_enabled)  callingPackage:android
06:58:31.448  adbd: persist.adb.tls_server.enable changed to 0
06:58:41.324  SettingsProvider: PUT(/global/adb_wifi_enabled)  callingPackage:ai.liquid.pipette.adbconnector
06:58:41.325  adbd: persist.adb.tls_server.enable changed to 1
```

`AdbDebuggingManager` restores the setting and adbd binds a port. Wifi is not up yet, so the
manager writes `adb_wifi_enabled` back to **0** and the listener is torn down. Wifi associates
seconds later and **nothing retries**.

That it is a race, not a policy, is why the fleet was inconsistent: a phone that associates fast
enough keeps wireless debugging, and one that does not loses it. `BOOT_COMPLETED` fires after
wifi is up, so this app's write always comes after wifi association. The app does not work
against the platform. It supplies the retry that the platform omits.

adbd also binds a **new random port** each time it starts listening, so after a reboot the
`IP:PORT` that the host saved is stale, even when debugging comes back. See `host-connect.sh`.

### Devices also lose adb without rebooting

A phone with 8 days of uptime was found refusing its adb port while still answering ping, having
never restarted. The boot log above explains that too: `AdbDebuggingManager` zeroes
`adb_wifi_enabled` itself whenever it finds no wireless network, so a short wifi drop or an
access point (AP) restart
can switch wireless debugging off on a device that is otherwise healthy, and again nothing
retries.

So the app also registers a **periodic check** (`AdbWatchdogJob`, JobScheduler, every 15 minutes,
which is the platform floor, persisted across reboot). It re-enables wireless debugging if it
finds the setting off.

This is safe to run on a timer precisely because it only ever writes when the setting reads `0`,
and a `0` means adbd is not listening, so there is no host connection to drop and no port to
move. It never forces a `0 -> 1` flip. That flip needs a moment when no host transport can be
live, which boot and a wifi rejoin supply and a timer does not.

### A roam needs an answer, not a retry

Wireless debugging is trusted **per BSSID** (basic service set identifier: the radio address of one
access point), not per SSID (the network name). The trusted set lives in
`/data/misc/adb`. The lab network uses three access points on two bands each, so a device can
associate with any of six BSSIDs, and a phone that has approved one of them has approved only
that one.

Write the setting to `1` on a BSSID that is not in the set and the manager does not enable
adbwifi. It puts up `WifiDebuggingActivity` ("Allow wireless debugging on this network?") and
writes the setting straight back to `0`. From one fleet phone:

```
00:11:26.786  SettingsProvider: PUT_ret(/global/adb_wifi_enabled)  callingPackage:ai.liquid.pipette.adbconnector
00:11:26.792  adbd: adb wifi started on port 45541
00:11:26.801  AdbDebuggingManager: startConfirmationForNetwork
00:11:26.809  SettingsProvider: PUT_ret(/global/adb_wifi_enabled)  callingPackage:android
00:11:26.812  adbd: adb wifi stopped
```

25ms, start to finish. **The periodic check cannot win this on its own**, and not because it is
too slow: the platform is not racing the app, it is asking a question and reading an unanswered
dialog as "no". That device sat in the loop for three days, re-enabled every period and revoked
every time, while nobody was standing next to it to tap the button.

`WifiDebuggingDialogService` answers the dialog: it ticks **always allow** and presses the
positive button. That ends the loop permanently, not for one cycle, because always-allow writes
the BSSID into the trusted set and the access point stops asking. Over time, a fleet device
approves each AP that it meets, and then the dialog stops.

`IAdbManager.allowWirelessDebugging` is the call the dialog makes. It is guarded by
MANAGE_DEBUGGING, which is signature-level and out of reach of `pm grant`, so tapping the dialog
the platform already drew is the only route open to an app. Hence an AccessibilityService, scoped
by `packageNames` to SystemUI alone so it never sees a benchmark run or anything typed into
another app.

The app registers the service itself with the WRITE_SECURE_SETTINGS grant it already holds,
rather than asking for a fourth manual install step. It is re-asserted on every launch and every
boot, since the enabled-services list is shared with any other accessibility service on the
device.

### A rejoin also has to re-announce the port

adbd owns the `_adb-tls-connect._tcp` multicast DNS (mDNS) record. It registers the record when its TLS listener
starts. A wifi drop stops that listener, and `AdbDebuggingManager` zeroes `adb_wifi_enabled` at the
same time. When wifi returns, nothing registers the record again, because the platform starts the
listener on a *change* of that setting.

`WifiRejoinMonitor` supplies the change. It holds a `ConnectivityManager.NetworkCallback` for wifi
transport and acts on the first `onLinkPropertiesChanged` that carries a routable IPv4 address. By
that point DHCP has answered, so a fresh registration carries a useful A record. The monitor then
forces the `0 -> 1` flip. The flip is safe here for the same reason it is safe at boot: every
wireless transport died with the link, so there is none left to drop.

Stopping the listener does not withdraw the record. adbd's socket to mdnsd has no `O_CLOEXEC`, so
every process adbd forks after registering (shells, the pipette worker, the runtimes it starts)
inherits a copy, and mdnsd withdraws the record only once every copy has closed. A re-arm while one
of them lives announces `<name> (2)` and leaves the old name on a dead port, where the host no
longer finds the phone. So every re-arm holds the setting at `0` for `HOLDER_GRACE_MS` (15s) before
it writes `1`, whether the setting already read `0` or the re-arm wrote it, which outlasts the worker
(it exits within ~10s of its session closing). The watchdog and the dialog accepter's own enable
leave the setting alone meanwhile. A streamed `adb install` leaves an `abb` holding a copy, hence
`--no-streaming` under Install.

The flip hands out a new port, which is the cost of the re-announcement, so the monitor debounces
on identity. One association emits several `onLinkPropertiesChanged` calls as the address, the
routes and the DNS servers arrive. A flip per callback would move the port several times, and the
host would then connect to a port that is already dead. The monitor re-arms only for a
`(network, IPv4)` pair it has not just re-armed for. A 30-second floor backstops that check. It is
longer than a whole re-arm: the 15s grace, plus the ~11s the OEM freezes the process over wifi,
where writing `0` drops the adb connection. A link the floor rejects is re-checked when the floor
elapses, because one association usually delivers a single qualifying callback. The monitor logs the skip and
the re-arm.

The first callback after registration is not a rejoin. `registerNetworkCallback` reports the
network that is already connected. A host transport can be live at that moment, because the service
rebinds whenever the app is reinstalled or the system re-enables it. That callback therefore takes
the non-destructive write, which arms a device whose setting reads `0` and leaves a live listener's
port alone. Only a link that differs from the one seen at registration gets the flip.

That holds only while wifi is up at registration. With wifi down there is no initial callback, so
the monitor records whether a wifi network already held a routable IPv4 address. When none did, the
first link to appear is a genuine rejoin and gets the flip.

**It depends on a bound accessibility service.** The callback needs a process that is already
alive, and that service is the only long-lived one in the app. A device whose accepter never bound
has no rejoin monitor either. The binding check under Install covers both.
`AdbWatchdogJob` stays the floor for that case: it recovers such a device at the 15-minute
platform period instead of within seconds of the rejoin.

## Keeping the screen on

`stay_on_while_plugged_in` stops the idle timeout, not a side-key press, and a handset that sleeps
suspends between samples and benchmarks slow. With `settings put global pipette_keep_screen_on 1`,
the same bound accessibility service turns the screen back on about 4s after it goes off (a 2s
settle, then the wake activity), while the device is plugged in, so the binding check under Install
covers it too. It is off by default. Set it to `1` on each phone that needs it. Set it to `0` to leave a phone dark on purpose. Its log lines are `screen keeper registered (enabled=...)` and,
per wake, `screen is back on`.

## What it does not solve

- **The port is still random.** The app cannot pin it, because that is adbd's choice. adbd does
  announce it over mDNS, and the re-arm above keeps the announcement current. The host still has
  to read the list and check each entry. Use `host-connect.sh`.
- **A holder that outlives the grace.** The pipette worker closes its copy of adbd's mDNS socket as
  it starts, so its runtimes never inherit one, but any other process adbd forked (a long-lived
  `adb shell`, a streamed install's `abb`) keeps the old record until it exits. A re-arm after the
  15s grace still announces `<name> (2)` while one lives.
- **A listener that dies while the setting still reads 1.** The periodic check above covers the
  common mid-session failure (the setting goes to 0), but not this one: re-writing `1` over `1`
  is a no-op in `SettingsProvider`, and only a `0 -> 1` transition re-arms. The app cannot tell
  this state apart from a healthy one, because Security-Enhanced Linux (SELinux) denies it both the `service.adb.tls.port`
  property and any view of adbd's sockets. Forcing a flip on a timer to cover it blindly would
  hand out a new port and drop in-flight benchmark connections on healthy devices, which is a
  worse trade. Recovery is manual: tap the button in the app, or toggle Developer options.
- **A wireless serial that a benchmark run saved.** `pipette-ctl`'s `wait_for_device` retries the
  same `IP:PORT`, for five minutes by default (`RejoinPolicy` in
  `crates/pipette-ctl/src/runner/probe.rs`; `--rejoin-max-retries` changes it). The drop that
  triggers a re-arm already puts the device out of reach of that serial, and adbd picks a fresh
  port whenever its listener restarts, so the saved port goes stale either way. The re-arm brings
  the device back, on a port that the saved serial does not know. Run the plan again against the
  port that `host-connect.sh` reports.
- **Bootstrap.** The one-time `pm grant` needs a working adb connection, so a phone that is
  already unreachable needs USB once.
- **Roaming itself.** The app makes a roam harmless; it does not stop one. Nothing on the device
  does: `cmd wifi` on the fleet's firmware exposes no roaming-aggressiveness control, and pinning
  the saved network to one BSSID trades a soft failure for a hard one, since the device then has
  nowhere to go when that AP restarts. Scoping the bench SSID to a single access point is a WLAN
  change, not an app change.
- **The first approval on a brand-new device.** The accepter needs the service registered, which
  needs the grant, which needs adb. USB once, as above.

## Install

Requires `ANDROID_HOME` and JDK 17 or newer (AGP 9.2.1 / Gradle 9.4.1, matching
`android/Pipette`; CI builds on 17).

The APK is signed with the repo's shared `android/Pipette/debug.keystore`, pinned so any build of
this app can update any other in place. This is important: without it, AGP signs with
a per-machine key, so a CI APK cannot update a locally built one, and the forced uninstall drops the
`WRITE_SECURE_SETTINGS` grant and returns the package to `FLAG_STOPPED`, silently disarming the boot
receiver. A missing keystore fails the build rather than falling back.

Published builds carry the release version in `versionName`; a local build reports `dev`, so
`dumpsys package ai.liquid.pipette.adbconnector | grep versionName` tells you which one a phone has.

```bash
./gradlew assembleDebug
# --no-streaming: a streamed install leaves an abb holding adbd's mDNS socket (see above).
adb -s <serial> install -r --no-streaming app/build/outputs/apk/debug/app-debug.apk

# Required. Without it the app installs and runs, but every write fails soft with a log line.
adb -s <serial> shell pm grant ai.liquid.pipette.adbconnector android.permission.WRITE_SECURE_SETTINGS

# Also required, and easy to miss. See below.
adb -s <serial> shell monkey -p ai.liquid.pipette.adbconnector 1
```

### Two install mistakes that stop the app without an error

**The app must be launched once.** A package that has been installed but never started carries
`FLAG_STOPPED` and is excluded from broadcasts, so `BootReceiver` never fires. The `monkey` line
above is enough. **A force-stop returns the package to that state**, so re-launch it if you ever
force-stop this app. Verify with:

```bash
adb -s <serial> shell dumpsys package ai.liquid.pipette.adbconnector | grep -E "stopped|notLaunched"
# want: stopped=false notLaunched=false
```

**A clean uninstall drops the grant.** `adb install -r` keeps it; a full uninstall/reinstall does
not. Re-run the `pm grant` after any reinstall, or the phone quietly loses its ability to come
back from a reboot.

Neither failure shows until the next reboot. Check both after you change the install.

**The accepter is registered, but check it bound.** The app writes itself into
`enabled_accessibility_services`, and that write succeeding is not the same as the system having
bound the service. The setting is what the app asked for; the log line is what happened. The app's
screen shows the same distinction: `registered but NOT RUNNING` means listed and not bound.

```bash
adb -s <serial> shell settings get secure enabled_accessibility_services
# want: ...ai.liquid.pipette.adbconnector/...WifiDebuggingDialogService...
adb -s <serial> shell settings get secure accessibility_enabled     # want: 1
adb -s <serial> logcat -d -s pipette-adb-connector | grep "accepter connected"
```

If the list has the entry and `dialog accepter connected` never appears, the system declined to
bind it. Enabling it by hand once (Settings > Accessibility > Installed apps) is the fallback;
on Android 13 and up a sideloaded app may need its restricted settings unblocked from the app info
screen first.

## Verifying without rebooting

`BOOT_COMPLETED` is a protected broadcast that only the system may send, which is why the
manifest also registers `QUICKBOOT_POWERON` (the OEM fast-boot broadcast some Samsung firmware
sends instead). That one is not protected, so it doubles as the test hook. Within 5 minutes of
boot the receiver forces the `0 -> 1` flip, which waits `HOLDER_GRACE_MS`; later it takes the
non-destructive write, so any app sending the broadcast cannot drop a live transport.

```bash
adb -s <serial> shell settings put global adb_wifi_enabled 0        # simulate post-reboot
adb -s <serial> shell am broadcast -a android.intent.action.QUICKBOOT_POWERON \
    -n ai.liquid.pipette.adbconnector/.BootReceiver
sleep 20                                                           # covers HOLDER_GRACE_MS near boot
adb -s <serial> shell 'settings get global adb_wifi_enabled; ss -ltn'
adb -s <serial> logcat -d -s pipette-adb-connector
```

Do this on a phone that also has `adb tcpip 5555` up, so port 5555 remains as a fallback channel
while wireless debugging is off. Otherwise a failed test leaves the device needing USB.

### Exercising the dialog accepter

The dialog only appears on a BSSID the device has not approved, so the test needs the device to
meet a new access point. Over USB, which keeps a channel while wireless debugging is off:

```bash
# What the device trusts now, against what it is associated with.
adb -s <serial> shell dumpsys adb | tr -c '[:print:]\n' ' ' | grep -oE "wifiAP.{0,60}"
adb -s <serial> shell dumpsys wifi | grep -oE 'SSID: "[^"]*", BSSID: [0-9a-f:]+' | head -1
```

Move the device onto an unapproved AP (walk it, or power-cycle the AP it is on), then write the
setting and watch. Accepted, the log carries `accepted wireless debugging for this network`
followed by `adbwifi started on port`; unaccepted, it carries `startConfirmationForNetwork` and a
`callingPackage:android` write back to 0.

```bash
adb -s <serial> shell settings put global adb_wifi_enabled 1
adb -s <serial> logcat -d -s pipette-adb-connector AdbDebuggingManager adbd | tail -20
```

## Finding the port from the host

```bash
./host-connect.sh 192.0.2.10 192.0.2.11
```

The script requires `adb`, `nmap` and `timeout`. On macOS, install the last two with
`brew install nmap coreutils`. The script pings each host first and skips a host that does not
answer: that is a network fault, not an adb one.

Three phases run per device, in order:

1. port 5555
2. the ports this host has heard announced over mDNS
3. an nmap sweep of the ephemeral range (~6s per device)

stderr reports which phase connected.

**mDNS is a candidate source, not a truth source.** The phone re-announces on every rejoin, so the
live entry is in the list. The list also holds records from earlier boots. A goodbye packet
withdraws a record, but adbd can send one only while the interface is still up, which is before the
app learns that the link will drop. Every entry therefore goes through an `adb connect` and a
`get-state` check. A closed port refuses the connection at once, so a stale candidate costs
milliseconds against the ~6s a sweep costs. A candidate that drops packets instead of refusing them
costs the operating system's connect timeout. `adb kill-server` clears the accumulated records when
a human wants a clean list.

**The list is also unstable, which is why the phase is a fast path and not a replacement for the
sweep.** It is the adb server's cache, and the server rebuilds it every few seconds. This host
asked four times a second for 60 seconds about a phone that was up and announcing. It returned the
record 69% of the time. The gaps where it did not lasted 1.35s, 1.62s, 1.90s, 5.99s and 7.64s.

The script asks five times, half a second apart. That covers the short gaps and gives up on the
long ones, because a wait long enough for a 7.6s gap costs more than the sweep it avoids. A device
that all five asks miss spends those 2 s and then the sweep, so `nmap` stays required.

The sweep stays as the fallback, so a device whose announcement never reached this host still
connects.

## Verified

Measured end-to-end on a Samsung SM-S948U1, across two real reboots.

| Step | Result |
| --- | --- |
| `pm grant WRITE_SECURE_SETTINGS` | `granted=true`, and **survives a reboot** |
| Launch once | `stopped=false notLaunched=false` |
| Real reboot | `BOOT_COMPLETED` reached the app **+21s** after boot, both times |
| App's write | `wrote adb_wifi_enabled=1: provider accepted=true` |
| Listener afterwards | `42637`, then `41687` (a fresh random port each boot) |
| `host-connect.sh` | found and connected to each new port unaided |
| Connection state | `device`, authorized, **no pairing code**; the paired host key survives |
| Watchdog registered | `PERIODIC: interval=+15m0s0ms`, `PERSISTED` in `dumpsys jobscheduler` |
| Watchdog recovery | setting forced to `0` (device unreachable, no 5555) → app wrote `1` and armed `45155` |

Two things the reboots settled:

**The setting really is 0 by the time the app runs.** `AdbDebuggingManager` zeroes it about ten
seconds before the app's write, when it loses the wifi race shown in the boot log above.

**Port 5555 does not survive a reboot.** After the first reboot the *only* listener was the
wireless-debugging port; `5555` was gone. Without this app, that phone would have needed a USB
connection to recover. Do not plan around `adb tcpip 5555` persisting.

The forced-flip path (`forceRearm` when the setting already reads `1`) was exercised separately
via the `QUICKBOOT_POWERON` hook: it retired port `42637` and armed `39487`. Its two writes sit at
least `HOLDER_GRACE_MS` (15s) apart, and a re-arm that finds the setting at `0` waits the same 15s.

This measurement of the rejoin monitor used an earlier build of the app, with a 750ms flip gap, a
15s floor and no wait after a rejoin. The code in this directory waits `HOLDER_GRACE_MS` (15s)
and uses a 30s floor, which this test did not cover. The test ran over USB on an SM-S948U1 in rig
condition (airplane mode on, wifi on under it), bouncing wifi with `cmd wifi set-wifi-enabled disabled`, then `enabled` plus
`settings put global wifi_on 2`; in airplane mode a plain write of `wifi_on` does not bring the
radio back. Without the monitor, the setting stayed `0` and nothing listened. With it, each bounce
logged one `wifi rejoined` and one re-arm, and `host-connect.sh` connected on the mDNS phase 12
runs out of 12, in ~1.8s against the ~6s sweep. Two flips started in the same millisecond both
passed the in-progress guard and both ran; each write is idempotent, so adbd still saw one
`0 -> 1` edge.

### The screen keeper

Measured on a Samsung SM-S948U1 on AC power, the debug build installed over the release in place.

| Step | Result |
| --- | --- |
| In-place install | grant kept, accessibility service rebound, `screen keeper registered (enabled=false)` |
| Switch on, side key, no adb traffic for 30s | `started the wake activity` 2.8s after the key, `screen is back on` 1s later, still `Awake` at 30s |
| Switch off, side key | stayed `Dozing` |
