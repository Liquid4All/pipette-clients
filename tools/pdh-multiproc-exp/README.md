# pdh-multiproc-exp

Reproducible harness for the multi-process attribution finding documented in
[`peak-memory-windows.md`](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/methodology/peak-memory-windows.md)
("Per-process attribution under concurrent GPU use").

Spawns N concurrent `llama-bench` instances on Windows, polls each PID's
PDH `\GPU Process Memory(pid_<PID>_*)\Total Committed` counter plus the
adapter-wide `\GPU Adapter Memory(*)\Total Committed`, and reports
whether per-PID peaks sum to the adapter delta from baseline (ratio
1.000 = strict per-process attribution with no cross-process sharing).

## Build / run

Windows only. Requires Rust + MSVC link.exe (the windows-sys crate
needs the link library).

```powershell
cargo run --release -- <n_processes> <n_prompt>
```

The binary expects `llama-bench.exe` and an LFM2 GGUF model at fixed
paths:

```rust
const LLAMA_BENCH: &str = r"C:\Users\yuri\mem-test\llama-b9058\llama-bench.exe";
const MODEL: &str = r"C:\Users\yuri\mem-test\LFM2-350M-Q4_K_M.gguf";
```

Edit `src/main.rs` if your install paths differ. This is a one-off
diagnostic, not a polished tool: kept here for reproducibility of the
peak-memory methodology's "33–44 MiB driver-state overhead is
per-process" claim.

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

Ratio 1.000 = WDDM attributes per-process exactly, no double-counting,
no sharing.

## Caveats

- At higher GPU memory pressure (~3 concurrent processes at n=256, or
  ~2 at n=2048 on Strix Halo's default GPU carve-out), the OS appears
  to serialize the workloads: only one PID is active in PDH at a time.
  The per-process attribution remains exact for whichever process is
  currently allocating, but the experiment can't distinguish "the
  other process serialized" from "the other process completed before
  PDH registered its instance" without exit-code instrumentation.
- The exact `Total Committed` numbers are specific to the GPU + driver
  combination tested (AMDVLK 32.0.23002.1006 on Radeon 890M / Strix
  Halo). Discrete GPUs with shader-cache-aware drivers may amortize
  some overhead for processes loading identical shaders.
