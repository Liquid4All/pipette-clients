# Perf troubleshooting: gmktec EVO-X2 fleet

Diagnose nominally-identical benchmark boxes reporting different llama-bench
speed. Derived from the gmktec EVO-X2 (Ryzen AI MAX+ 395 / Radeon 8060S)
investigation, 2026-07. Applies to any pipette fleet.

## Rules

1. **Measure, never assume**: every layer below produced a wrong first guess
   that only a measurement corrected.
2. **Split the metrics**: `pp` (prefill) = compute/clock-bound; `tg` (decode) =
   memory-bandwidth-bound. They implicate different subsystems.
3. **Reboot to a clean state before concluding anything.** Mid-update or
   wedged-GPU boxes produce 2× swings and CPU fallbacks that look like
   hardware faults.
4. A 1–2 % gap needs **interleaved rounds (A→B→C→D ×10) + 95 % CIs**:
   single runs cannot separate it from noise.

## Diagnostic ladder

| # | Check | Command | Verdict rule |
|---|-------|---------|--------------|
| 0 | Reproduce | `llama-bench -m <gguf> -p 2048 -n 128 -ngl 99 -t 8 -r 3` per box | high stddev ⇒ box unstable *now*; fix that first |
| 1 | Inputs | `pipette models list` / `pipette runtimes list`; bench prints `backend=` | missing model ⇒ failed cell read as "slow"; `backend=CPU` with `-ngl 99` ⇒ Vulkan wedged → **reboot** |
| 2 | Software | driver ver (`Win32_VideoController`), OS UBR, power plan, stale `llama-server` procs | normalize all; boxes **mid-auto-update bench erratically** |
| 3 | Thermal | ACPI zone (`\Thermal Zone Information(*)\Temperature`, Kelvin). It *is* the die temp here, matching LHM `Tctl/Tdie` within 1–4 °C from 33 °C to 98 °C. **Ignore `ryzenadj -i` → `THM VALUE CORE`**: it read 42.7 °C while the die was at 98 °C under 73–103 W package power (same broken Strix Halo support that makes its clocks `nan`) | gap grows at high power ⇒ thermal; gap worst at *low* power ⇒ not thermal |
| 4 | SoC power limit (≠ Windows plan) | `ryzenadj -i` → `STAPM/PPT LIMIT`. EVO-X2 BIOS presets: Perf 120/140 W, Balanced 85 W, Quiet 54 W (also a physical button) | box on lower preset ⇒ fix preset. Confirm with a power-matched sweep, verifying the cap binds via `PPT VALUE FAST` |
| 5 | Tuner residue | third-party power tools (ParkControl/Process Lasso) write into the power scheme and **survive app removal** | `powercfg -restoredefaultschemes` + re-apply plan; re-bench |
| 6 | Clock/silicon | LHM DLL reads `GPU Core` clock headless (ryzenadj clocks are `nan` on Strix Halo) | lower clock at equal power+temp+voltage ⇒ bin. CPU and iGPU **bin independently** (one box: fastest CPU, slowest iGPU) |

### Quick fleet check (~25 s, all boxes in parallel)

```
llama-bench -m <gguf> -p 2048 -n 0 -ngl 0,99 -t 8 -r 1
```

One invocation per box (one model load, CPU + GPU in one process), boxes in
parallel. Precision ±3 %; fine for "is anything broken", not for ranking
near-identical units. Caveats: `-t` applies to both backends, so CPU reads
lower than the `-t 16` long form and under-states power-limit differences;
use `-r 3`+ at `-t 16` (the ladder's step 0) before comparing boxes.

## Known root causes & fixes (this fleet)

| Symptom | Root cause | Fix |
|---|---|---|
| Box 735 t/s, `backend=CPU` | Vulkan ICD wedged after driver auto-update without reboot | reboot |
| Box slow at every context | BIOS Power Mode on Balanced (85 W) | set Performance preset (BIOS or button) |
| CPU +2–3 % on one box, paradoxical clocks | ParkControl scheme residue | `powercfg -restoredefaultschemes` |
| iGPU −1.5 % at equal power/temp | weaker iGPU bin | either accept (stock, representative) or equalize: `ryzenadj --stapm-limit=130000 --fast-limit=150000 --slow-limit=130000` (+10 W ≈ +1.5 %; verified 156 W peak @ 68 °C vs 98 °C limit) |
| SSH dead, tailscale "active" but rx≈0 | tailnet path lost direct UDP and fell back to a relay that drops traffic. The **box is fine** | `tailscale status`: `relay` + stalled rx = path issue; direct = box issue. Self-heals in ~5–20 min when the direct path renegotiates; don't power-cycle. Avoid killing in-flight ssh+bench sessions (drops the direct path) |

## CPU deltas: two independent sources (both identified)

1. **ParkControl scheme residue** (+2–3 %): Bitsum tools write into the
   Windows power scheme; the edits **survive disabling the app**. Proof by
   convergence: gmktec CPU 3364–3440 with residue → 3295–3331 after
   `powercfg -restoredefaultschemes` ≈ control box 3280–3299 (within ~1 %).
   Side effect: the residue also confuses Windows clock counters
   (more work at lower reported clock).
2. **The +10 W equalize-up task** (+3 %, deliberate): the package limit feeds
   CPU and iGPU alike, so GPU parity raises CPU-only numbers too. See
   trade-off below. Not decouplable.

Residual after both: ~1 %; normal bin/run variance.

## Equalize-up persistence (gmktec only)

Scheduled task `PipetteRaiseSocPowerLimits`: at-startup +45 s, SYSTEM, runs
`C:\tools\ryzenadj\ryzenadj.exe --stapm-limit=130000 --fast-limit=150000
--slow-limit=130000`. **ryzenadj exits 0xC0000005 after applying; verify by
reading limits post-boot, never by exit code.** Do not delete
`C:\tools\ryzenadj\`. To revert to stock:
`Unregister-ScheduledTask PipetteRaiseSocPowerLimits` + reboot.

Recreate (elevated PowerShell):

```powershell
$a = New-ScheduledTaskAction -Execute 'C:\tools\ryzenadj\ryzenadj.exe' `
  -Argument '--stapm-limit=130000 --fast-limit=150000 --slow-limit=130000'
$tr = New-ScheduledTaskTrigger -AtStartup; $tr.Delay = 'PT45S'
$p = New-ScheduledTaskPrincipal -UserId SYSTEM -LogonType ServiceAccount -RunLevel Highest
$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
Register-ScheduledTask PipetteRaiseSocPowerLimits -Action $a -Trigger $tr -Principal $p -Settings $s -Force
```

Trade-off: equalize-up = matched GPU, one box ~8 % above vendor preset and
its CPU-only numbers +3 % (the package limit feeds both domains; the APU
sub-limit (`--apu-slow-limit`, 70 W) is **not** the binding limit for GPU
prefill, raising it alone does nothing, so the effects can't be decoupled).
Stock = ~1.5 % spread, fully representative of shipped product. **Do not
level down** (85 W preset): bin spread *widens* when power-starved
(−9.2 % @ 60 W vs −5.3 % @ 120 W) and the fleet loses ~5 % absolute.

## Telemetry caveats (Strix Halo / Windows)

- SMU reads/writes (ryzenadj, LHM power) work **only** where the Ryzen Master
  SDK driver is resident; blocked elsewhere (`Unable to get os_access Obj`).
  PMF is irrelevant on EVO-X2: no PMF ACPI device exists. The service can
  never run; don't install it.
- `% Processor Performance` reports requested P-state, not achieved clock:
  it contradicted measured work. Trust throughput (llama-bench t/s) over
  Windows clock counters.
- Don't run `ryzenadj --set` and LHM concurrently: the cap silently fails
  to bind.
- `PendingFileRenameOperations` ≠ reboot required; trust `CBS RebootPending` /
  `WU RebootRequired`.
- No live remote-desktop session during benches (an active RustDesk viewer
  eats ~10 % iGPU).

## Proving a small gap (method)

1. Interleave: rounds of box A→B→C→D, ≥10 rounds, one bench invocation each.
2. Per box: mean, stddev, 95 % CI (`1.96·sd/√n`).
3. Non-overlapping CIs ⇒ real difference; overlapping ⇒ noise, stop.

## Reference numbers (LFM2.5-1.2B UD-Q4_K_XL, b9659, 2026-07-06, fresh boot)

| Box | GPU pp2048 (`-ngl 99 -t 8`) | CPU pp2048 (`-ngl 0 -t 16`) |
|---|---|---|
| gmktec (130/150 W task) | ~6875–6935 | ~3410–3470 (+4 % from the raise) |
| gmktec-2 | ~6930–6960 | ~3290–3345 |
| gmktec-4 | ~6890–6900 | ~3270–3295 |
| gmktec-5 | ~6865–6905 | ~3305–3320 |

Healthy fleet ⇒ GPU within ~1 %, CPU within ~1 %. Deviations beyond that:
start at ladder step 0.
