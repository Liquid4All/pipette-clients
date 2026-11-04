# pdh-multiproc-exp

Reproducible harness for the per-process driver-state finding in
[Peak Memory: Windows](../../docs/methodology/peak-memory-windows.md#caveats).

Spawns N concurrent `llama-bench` instances on Windows, polls each PID's
Performance Data Helper (PDH) `\GPU Process Memory(pid_<PID>_*)\Total Committed` counter plus the
adapter-wide `\GPU Adapter Memory(*)\Total Committed`, and reports
whether per-PID peaks sum to the adapter delta from baseline (ratio
1.000 = strict per-process attribution with no cross-process sharing).

## Build / run

Windows only. Requires Rust and the MSVC linker (`link.exe`). The windows-sys
crate imports its functions through `raw-dylib`, so the build needs no Windows
SDK import library.

```powershell
cargo run --release -- <n_processes> <n_prompt>
```

The `llama-bench.exe` and LFM2-350M GGUF paths are constants at the top of
`src/main.rs`; edit them for your install. This is a one-off diagnostic, not a
maintained tool.

## Example output

```
=== n_processes=2 n_prompt=256 samples=83 ===
  pid  19916  Total Committed peak       394250240 bytes      375.96 MiB
  pid  20640  Total Committed peak       394250240 bytes      375.96 MiB
  Σ per-PID peaks                        788500480 bytes      751.91 MiB
  adapter Total Committed baseline      1165307904 bytes     1111.34 MiB
  adapter Total Committed peak (raw)    1953808384 bytes     1863.25 MiB
  adapter Δ (peak − baseline)            788500480 bytes      751.91 MiB
  Σ per-PID / adapter Δ                          1.000     (1.0 = perfect agreement; >1 = PDH double-counts)
```

Result: at a ratio of 1.000, the Windows Display Driver Model (WDDM) bills each process its full driver-state
overhead, with no double-counting and no cross-process sharing.

## Caveats

- At higher GPU memory pressure (~3 concurrent processes at n=256, or
  ~2 at n=2048 on the Radeon 890M's default GPU carve-out), the OS appears
  to serialize the workloads: only one PID is active in PDH at a time.
  The per-process attribution remains exact for whichever process is
  currently allocating, but the experiment cannot distinguish "the
  other process serialized" from "the other process completed before
  PDH registered its instance" without exit-code instrumentation.
- The exact `Total Committed` numbers are specific to the GPU + driver
  combination tested (AMDVLK 32.0.23002.1006 on a Ryzen AI 9 HX 370
  with a Radeon 890M). Discrete GPUs with shader-cache-aware drivers may amortize
  some overhead for processes loading identical shaders.
