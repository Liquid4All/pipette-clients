# Perf troubleshooting: GMKtec EVO-X2 fleet

Diagnose nominally-identical benchmark boxes that report different llama-bench
speed. The worked examples are the GMKtec EVO-X2 (Ryzen AI MAX+ 395 /
Radeon 8060S) boxes. Steps 0–2 of the ladder apply to any pipette fleet; steps
3–6 are specific to Windows on AMD Strix Halo.

The first request after idle also loses time on these boxes. That is an
idle-only penalty, not a persistent per-box gap, and it has its own
report: [GMKtec GPU idle residency](gmktec-evo-x2-gpu-idle.md).

## Rules

1. **Measure, never assume**: every layer below produced a wrong first guess
   that only a measurement corrected.
2. **Split the metrics**: `pp` (prefill) is bound by compute and clock; `tg`
   (decode) is bound by memory bandwidth. They point to different subsystems.
3. **Reboot to a clean state before you conclude anything.** A box in the
   middle of an update, or with a wedged GPU, gives 2× swings and CPU fallbacks
   that look like hardware faults.
4. A 1–2 % gap needs interleaved rounds and 95 % confidence intervals (CIs):
   single runs cannot separate it from noise. See
   [Proving a small gap](#proving-a-small-gap-method).

## Diagnostic ladder

| # | Check | Command | Verdict rule |
|---|-------|---------|--------------|
| 0 | Reproduce | `llama-bench -m <gguf> -p 2048 -n 128 -ngl 99 -t 8 -r 3` per box | High stddev: the box is unstable *now*; fix that first |
| 1 | Inputs | `pipette models list` / `pipette runtimes list`; bench prints `backend=` | A missing model gives a failed cell that reads as "slow". `backend=CPU` with `-ngl 99` means a wedged Vulkan driver: **reboot** |
| 2 | Software | driver version (`Win32_VideoController`), OS update build revision (UBR), power plan, stale `llama-server` processes | Make them all the same; boxes **in the middle of an auto-update bench erratically** |
| 3 | Thermal | ACPI zone (`\Thermal Zone Information(*)\Temperature`, Kelvin). It *is* the die temperature here, and it matches LibreHardwareMonitor (LHM) `Tctl/Tdie` within 1–4 °C from 33 °C to 98 °C. **Ignore `ryzenadj -i` → `THM VALUE CORE`**: it read 42.7 °C while the die was at 98 °C under 73–103 W package power (the same broken Strix Halo support that makes its clocks `nan`) | A gap that grows at high power is thermal; a gap that is worst at *low* power is not thermal |
| 4 | SoC power limit (not the Windows power plan) | `ryzenadj -i` → `STAPM/PPT LIMIT` (the Skin Temperature Aware Power Management (STAPM) limit and the Package Power Tracking (PPT) limit). EVO-X2 BIOS presets: Perf 120/140 W, Balanced 85 W, Quiet 54 W (also a physical button) | A box on a lower preset: fix the preset. Confirm with a power-matched sweep, and check that the cap binds with `PPT VALUE FAST` |
| 5 | Tuner residue | third-party power tools (ParkControl, Process Lasso) write into the power scheme, and the edits **survive app removal** | `powercfg -restoredefaultschemes`, re-apply the plan, and bench again |
| 6 | Clock and silicon | the LHM DLL reads the `GPU Core` clock headless (ryzenadj clocks are `nan` on Strix Halo) | A lower clock at equal power, temperature and voltage is the silicon bin (chip-to-chip variation). The CPU and the integrated GPU (iGPU) **bin independently** (one box: fastest CPU, slowest iGPU) |

### Quick fleet check (~25 s, all boxes in parallel)

```
llama-bench -m <gguf> -p 2048 -n 0 -ngl 0,99 -t 8 -r 1
```

Run one invocation per box (one model load, CPU and GPU in one process), with
the boxes in parallel. The precision is ±3 %: enough for "is anything broken",
not for a ranking of near-identical units. `-t` applies to both backends, so
the CPU reads lower than at `-t 16` and under-states power-limit differences.
Before you compare boxes, use `-r 3` or more with `-ngl 0 -t 16` for the CPU and
`-ngl 99 -t 8` for the GPU, as in the [reference numbers](#reference-numbers-lfm25-12b-ud-q4_k_xl-b9659-2026-07-06-fresh-boot).

## Known root causes & fixes (this fleet)

| Symptom | Root cause | Fix |
|---|---|---|
| Box 735 t/s, `backend=CPU` | the Vulkan installable client driver (ICD) wedged after a driver auto-update without a reboot | reboot |
| Box slow at every context | BIOS Power Mode on Balanced (85 W) | set the Performance preset (BIOS or button) |
| CPU +2–3 % on one box, paradoxical clocks | ParkControl scheme residue | `powercfg -restoredefaultschemes` |
| iGPU −1.5 % at equal power/temp | weaker iGPU bin | either accept (stock, representative) or equalize: `ryzenadj --stapm-limit=130000 --fast-limit=150000 --slow-limit=130000` (+10 W ≈ +1.5 %; verified 156 W peak @ 68 °C vs 98 °C limit) |
| E2E slow only after idle, warm repeats fine | GPU evicts the server's allocations while idle (AMD driver) | install [`tools/gpu-keepalive`](../../tools/gpu-keepalive/README.md); threshold and measured effect in [GMKtec GPU idle residency](gmktec-evo-x2-gpu-idle.md) |

## CPU deltas: two independent sources (both identified)

1. **ParkControl scheme residue** (+2–3 %): Bitsum tools write into the
   Windows power scheme, and the edits **survive when you disable the app**.
   Proof by convergence: box A CPU 3364–3440 with residue → 3295–3331 after
   `powercfg -restoredefaultschemes` ≈ control box 3280–3299 (within ~1 %).
   The residue also confuses the Windows clock counters (more work at a lower
   reported clock).
2. **The +10 W equalize-up raise** (+3 %, deliberate): the package limit feeds
   the CPU and the iGPU alike, so GPU parity raises the CPU-only numbers too. See
   the trade-off below. The two effects cannot be separated.

Residual after both: ~1 %, which is normal bin and run variance.

## Equalize-up persistence (optional)

No fleet box keeps the +10 W raise across reboots. To keep it on one box,
register a startup task that runs ryzenadj about 45 s after boot as SYSTEM.
**ryzenadj exits with `0xC0000005` after it applies the limits. Check the limits
after boot (`ryzenadj -i`), not the exit code.** To return to stock,
unregister the task and reboot.

Create the task in an elevated PowerShell, with the path to your ryzenadj:

```powershell
$a = New-ScheduledTaskAction -Execute 'C:\tools\ryzenadj\ryzenadj.exe' `
  -Argument '--stapm-limit=130000 --fast-limit=150000 --slow-limit=130000'
$tr = New-ScheduledTaskTrigger -AtStartup; $tr.Delay = 'PT45S'
$p = New-ScheduledTaskPrincipal -UserId SYSTEM -LogonType ServiceAccount -RunLevel Highest
$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
Register-ScheduledTask PipetteRaiseSocPowerLimits -Action $a -Trigger $tr -Principal $p -Settings $s -Force
```

To remove it: `Unregister-ScheduledTask PipetteRaiseSocPowerLimits`, then reboot.

Trade-off: equalize-up gives a matched GPU, but one box runs ~8 % above the
vendor preset and its CPU-only numbers are +3 %. The package limit feeds both
domains. The APU sub-limit (`--apu-slow-limit`, 70 W) is **not** the binding
limit for GPU prefill, and a raise of it alone does nothing, so the effects
cannot be separated. Stock gives a ~1.5 % spread and fully represents the
shipped product. **Do not level down** (85 W preset): the bin spread *widens*
when power is short (−9.2 % @ 60 W vs −5.3 % @ 120 W), and the fleet loses ~5 %
absolute.

## Telemetry caveats (Strix Halo / Windows)

What `ryzenadj --info` power readings mean (time constants, socket scope) is in
[energy device capabilities §2](../methodology/energy/02-device-capabilities.md#2-windows-on-amd-apu-gmktec-evo-x2).

- System Management Unit (SMU) reads and writes (ryzenadj, LHM power) work
  **only** where the Ryzen Master SDK driver is resident; elsewhere they are
  blocked (`Unable to get os_access Obj`). The AMD Platform Management Framework
  (PMF) does not apply to the EVO-X2: it has no PMF ACPI device, so the service
  can never run. Do not install it.
- `% Processor Performance` reports the requested P-state, not the achieved
  clock: it contradicted measured work. Trust throughput (llama-bench t/s) over
  the Windows clock counters.
- Do not run `ryzenadj --set` and LHM at the same time: the cap silently fails
  to bind.
- `PendingFileRenameOperations` does not mean that a reboot is required. Trust
  the Component-Based Servicing (CBS) `RebootPending` and Windows Update (WU)
  `RebootRequired` keys.
- No live remote-desktop session during benches (an active RustDesk viewer
  uses ~10 % of the iGPU).

## Proving a small gap (method)

1. Interleave: rounds of box A→B→C→D, 10 or more rounds, one bench invocation
   each.
2. Per box: mean, stddev, 95 % CI (`1.96·sd/√n`).
3. CIs that do not overlap show a real difference. CIs that overlap show noise:
   stop.

## Reference numbers (LFM2.5-1.2B UD-Q4_K_XL, b9659, 2026-07-06, fresh boot)

| Box | GPU pp2048 (`-ngl 99 -t 8`) | CPU pp2048 (`-ngl 0 -t 16`) |
|---|---|---|
| box A (raised to 130/150 W) | ~6875–6935 | ~3410–3470 (+4 % from the raise) |
| box B | ~6930–6960 | ~3290–3345 |
| box C | ~6890–6900 | ~3270–3295 |
| box D | ~6865–6905 | ~3305–3320 |

In a healthy fleet, the GPU and the CPU numbers are each within ~1 %. For a
larger deviation, start at ladder step 0.
