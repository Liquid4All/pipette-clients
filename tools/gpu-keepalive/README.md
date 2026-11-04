# Windows GPU keepalive

A Windows service that periodically dispatches **one GPU thread to increment a four-byte buffer**, using Direct3D 11 on an explicitly selected hardware adapter. No model, inference server, desktop window, or third-party runtime is needed to run it.

This is a configurable diagnostic/workaround for GPU idle behavior. Whether it preserves another process's GPU allocations must be measured on the target driver; submitting GPU work is not a residency lock. The deployment target is the GMKtec EVO-X2 fleet (Ryzen AI MAX+ 395 / Radeon 8060S); no other hardware has been tested.

One binary, `gpu-keepalive.exe`, is three things: the CLI you type, the service that Windows starts at boot (`gpu-keepalive service`, which is not for interactive use), and a foreground pulse for experiments (`gpu-keepalive run`).

Why it exists and what it measurably does on the EVO-X2 is in [GMKtec GPU idle residency](../../docs/fleet-perf-troubleshooting/gmktec-evo-x2-gpu-idle.md), with the four-box validation in its §6. For the wider fleet diagnostic ladder, see [`docs/fleet-perf-troubleshooting/gmktec-evo-x2.md`](../../docs/fleet-perf-troubleshooting/gmktec-evo-x2.md).

## Install

Download `pipette-gpu-keepalive-x86_64-pc-windows-msvc.msi` from the release, then from an elevated shell:

```powershell
msiexec /i pipette-gpu-keepalive-x86_64-pc-windows-msvc.msi /qn /l*v C:\Windows\Temp\gk-install.log
gpu-keepalive status
```

That one command installs the binary, registers and starts the `PipetteGpuKeepalive` service, and puts the CLI on the system PATH (open a new shell to pick it up). The install **waits for the first verified GPU write**: on a box where the pulse cannot run, the install fails and rolls back rather than leaving a dead service behind.

Upgrading is the same command with a newer MSI. Uninstalling is `msiexec /x <the same msi> /qn`, or Settings > Apps, or `winget uninstall --name "Pipette GPU Keepalive"`. Uninstall leaves the configuration and logs behind, as Windows convention expects.

What lands where:

```
C:\Program Files\Pipette\GPU Keepalive\gpu-keepalive.exe   the only installed file
C:\ProgramData\Pipette\gpu-keepalive\gpu-keepalive.log  start, stop and failure lines
HKLM\SOFTWARE\Pipette\GpuKeepalive                         configuration (see below)
HKLM\SYSTEM\CurrentControlSet\Services\PipetteGpuKeepalive service registration
```

The service runs as LocalSystem in session 0 with no interactive login, starts delayed at boot. The Windows Service Control Manager (SCM) restarts it 60 seconds after each failure, with no limit on the number of restarts, and resets its failure count each day.

## Operate

`gpu-keepalive --help` lists the verbs. `enable`, `disable`, `start` and `stop` need an elevated shell.

On failure these print the reason to stderr and exit nonzero. When `start` fails because the service died on its way up, it reports the state and exit code that SCM has, and points at the log, which is where the reason is.

**`disable` is policy, `stop` is right now.** `disable` writes `Enabled=0`, which the MSI never touches, so it survives reboots *and* upgrades. `stop` leaves the configuration alone, so a reboot brings the service back: a forgotten experiment cannot leave the fleet degraded indefinitely.

For an A/B measurement use `stop`, not a pause: stopping tears down the D3D11 device, which is the off condition the validation measured. A paused process still holding a device would be varying something other than the pulse.

### Configuration

`HKLM\SOFTWARE\Pipette\GpuKeepalive`, read when the service starts. Stop and start it to apply a change.

| Value | Type | Default | Meaning |
|---|---|---|---|
| `Enabled` | DWORD | 1 | 0 stops the service reporting a clean exit, so no restart is attempted and the reason is visible in `status`. |
| `IntervalMs` | DWORD | 100 | Milliseconds between pulses, 1 to 60000. |
| `Adapter` | string | `vulkan` | `vulkan`, `software`, or a DXGI adapter index. |

`Adapter=vulkan` enumerates Vulkan in-process and matches the single reported GPU to a DXGI adapter by LUID. That is the same GPU llama.cpp's Vulkan backend uses, resolved at every start, which matters because **adapter indices and LUIDs are not stable across reboots**, and because on the EVO-X2 two DXGI adapters carry the identical "Radeon 8060S" name. `software` selects the Microsoft Basic Render Driver and keeps no real GPU awake; it exists so CI can exercise this code on a GPU-less runner.

To change the interval or adapter, use an elevated PowerShell shell. This example sets a 100 ms interval and Vulkan selection:

```powershell
gpu-keepalive stop
if ($LASTEXITCODE -ne 0) { throw 'Could not stop gpu-keepalive' }
$keepaliveKey = 'HKLM:\SOFTWARE\Pipette\GpuKeepalive'
if (-not (Test-Path $keepaliveKey)) { New-Item -Path $keepaliveKey -Force | Out-Null }
New-ItemProperty -Path $keepaliveKey -Name IntervalMs -PropertyType DWord -Value 100 -Force | Out-Null
New-ItemProperty -Path $keepaliveKey -Name Adapter -PropertyType String -Value 'vulkan' -Force | Out-Null
gpu-keepalive start
gpu-keepalive status
```

To pin a DXGI adapter, run `gpu-keepalive list` and use its index as a string in the `Adapter` command above, for example `-Value '1'`. To restore the defaults, remove the `IntervalMs` and `Adapter` values with `Remove-ItemProperty`, then stop and start the service. These edits preserve `Enabled`; if it is disabled, use `gpu-keepalive enable` when you are ready to resume pulsing. `status` reports malformed pulse settings even while disabled; disabled service startup exits cleanly without validating them.

## Foreground use

`run` pulses in the console, independent of the service, and is what experiments should use:

```powershell
gpu-keepalive list                                  # DXGI adapters, and what Vulkan reports
gpu-keepalive run --interval-ms 100 --duration-secs 60
gpu-keepalive run --adapter 1 --interval-ms 100     # pin a DXGI index
```

`--duration-secs 0` (the default) runs until Ctrl+C. It prints `READY ...` once the first GPU write is verified and `DONE ...` with pulse count, process CPU time, completion statistics and missed deadlines at the end. It warns on stderr when the service is pulsing at the same time.

## Behavior

- Creates the device, shader, buffer and completion query once.
- Verifies the first GPU write before reporting the service as running.
- Submits one dispatch per interval and explicitly flushes it. An event query confirms completion.
- Sleeps between pulses and while waiting for the GPU; does not spin or alter global timer resolution, power settings, or driver settings.
- Skips missed scheduling deadlines instead of issuing a burst of catch-up work.
- At shutdown, reads the buffer back and checks the counter against the pulses issued. A mismatch fails the service, and the log's `ERROR` line gives the expected and the read counts. A clean stop writes the count in the log's `DONE` line.
- Fails with a service-specific exit code (1) for a bad configuration, an adapter it cannot select, a shader/device failure, a five-second GPU timeout or a failed counter check. The service manager then restarts it; the reason is the `ERROR` line in the log.

`pulse_completion_*_ms` is **CPU-observed dispatch-to-completion time**, including thread scheduling and polling. It is not GPU execution time or GPU utilization. Four bytes describes the shader's logical buffer, not the driver's total memory allocation. Keeping a GPU awake can change idle power even with a very small dispatch; power draw is not measured by this program.

The service cannot interrupt an already-running GPU operation immediately; shutdown waits for that operation's completion or the bounded timeout.

## Build

Rust (stable) on Windows:

```powershell
$env:GPU_KEEPALIVE_VERSION = "26.9.11"; $env:GPU_KEEPALIVE_BUILD = "local"
cargo build --release --locked
```

Run these commands from `tools/gpu-keepalive` so Cargo loads this tool's `.cargo/config.toml`, which enables `+crt-static` for MSVC. The build rejects MSVC settings that would require the Visual C++ redistributable.

The executable is `target\release\gpu-keepalive.exe`, and it is the only file the MSI ships. Direct3D, DXGI and the shader compiler come from Windows; the Vulkan loader comes from the installed Vulkan-capable graphics driver. Rust and the C runtime are linked statically. Both variables have defaults (`0.0.0` and `dev`), so a plain `cargo build` works too.

This is a **standalone Cargo workspace**, deliberately outside the repository's main one, for the reason `tools/pdh-multiproc-exp` is: the crate is Windows-only, so `cargo build --workspace` on macOS or Linux must not try to compile it. The cost is that the `rust-check` job never sees it, so `cargo fmt --check`, `cargo clippy -- -D warnings` and `cargo test` run in this tool's own CI job instead.

`GPU_KEEPALIVE_VERSION` doubles as the MSI ProductVersion, whose fields cap at 255.255.65535; CI maps the release version `2026.09.11-<sha>` to `26.9.11` and passes the full string as `GPU_KEEPALIVE_BUILD`, which `gpu-keepalive --version` and the file's properties report.

The package is `package/Package.wxs`, built with WiX 5. The `gpu-keepalive` job in [`.github/workflows/build.yml`](../../.github/workflows/build.yml) builds the executable and the MSI. Then it runs `package/smoke-test.ps1`, which installs the MSI on the runner and does these steps in order: install, disable, upgrade, enable, stop, start and uninstall. The smoke test also checks a disabled startup with invalid settings. It also checks the automatic recovery after a reported configuration failure, and waits for SCM's 60-second restart for that.

The runner has no GPU. So the smoke test proves the packaging, the service protocol and the CLI. It does not prove that the workaround works, which only a real box can show.

## API references

[Device creation](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-d3d11createdevice), [dispatch](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-dispatch), [submission and completion](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-flush), [service control handler](https://learn.microsoft.com/en-us/windows/win32/api/winsvc/nf-winsvc-registerservicectrlhandlerexw), [service configuration in MSI](https://learn.microsoft.com/en-us/windows/win32/msi/msiserviceconfig-table).
