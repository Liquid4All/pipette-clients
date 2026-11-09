# Measuring OpenVINO cells

This page describes what the `uv_openvino` backend measures, and the decisions
behind it that a run on the device settled. Export, precision and NPU support
are in [openvino-ir.md](openvino-ir.md); this page is about the harness.
Operator setup is in [pipette-cli/openvino.md](pipette-cli/openvino.md).

Each *measured* result names its host. Unless stated, it ran on an Intel Core
Ultra 7 258V (Lunar Lake, Intel AI Boost NPU, Windows 11), `openvino-genai`
2026.2.1, `LFM2.5-350M/int4-sym-cw`.

Results not marked as taken on the server path came from a per-rep driver:
each rep was its own process, which timed itself with `perf_metrics`. Rows
whose `client_version` is dated before 2026-09-25 were measured that way. In
those rows:

- prefill was GenAI's time to first token (TTFT) inside `generate()`, with no
  tokenization, HTTP or server-sent events (SSE) framing. This read about 7 ms
  below the streamed reading of OVMS at a 1024-token prompt on an Arc B390.
- decode was `tpot × n`, where TPOT is GenAI's time per output token.
- end-to-end was computed as `ttft + tpot × (n − 1)`, not timed.

Do not compare those rows with server-path rows.

## What each cell samples

Each timing cell runs one in-house OpenAI-compatible server
(`pipette_openvino_server.py`) that compiles one pipeline before it listens.
The memory cell runs two such servers, a warm-up server and then a measured
server ([the memory cell and the cache](#the-memory-cell-and-the-cache)).
The timing cells read a stream on the client's clock, like every other
server-driven engine; the methodology pages define each figure.

| cell | sample | first token? | compile? |
|---|---|---|---|
| `prefill_throughput` | client-clock time to the first streamed token (`max_tokens = 1`) | n/a | no |
| `decode_throughput` | client-clock first-to-last-token interval, scaled from `n − 1` to `n` | **no** | no |
| `end_to_end_latency` | client-clock time of one streamed completion, send to finish chunk | **yes** | no |
| `max_memory_usage` | peak resident memory of the server process after one request; its GPU memory too on Windows and Linux, its NPU memory on Windows | n/a | no (see below) |

The readiness wait for the server covers the compile, so the compile is outside
every measured *time*, and
[`attempt_timeout_seconds`](pipette-cli/usage.md#5c-the-attempt-limit) does
not count the first server's wait, the load in it included. A peak-memory
cell's measured server loads what its warm-up compiled, and its wait counts.
Each server process compiles one pipeline, so no
process compiles twice (the constraint that [openvino-ir.md](openvino-ir.md#npu-constraints)
records). Sequential processes are safe on the NPU: five sequential compiles
caused no device loss, and throughput stayed at 153.8 to 154.0 tok/s.

The server speaks HTTP/1.1 and keeps a connection open between requests, so a
client that reuses its connection pays no new connect.

On every device but the NPU, custom device strings included, the cell pins
GenAI's SDPA (scaled dot-product attention) backend. GenAI's default there is
paged attention, which a long-lived server cannot use for these cells. *Measured* on an x86 Xeon 8358,
TinyLlama-1.1B int4, CPU:

| | paged attention (default) | SDPA |
|---|---|---|
| 512-token prompt repeated, TTFT | 8.9 ms (prefix cache hit) | 224 ms |
| 32 tokens, streamer detokenizing each token | ~1,500 ms | ~290 ms |
| 32 tokens, streamer doing nothing | ~270 ms | ~290 ms |

LFM2.5-350M reads the same on both (TTFT 79 ms at 512 tokens, TPOT 6.2 ms): it
already runs on SDPA by default. On the NPU (Core Ultra X7 358H) and on CPU
under SDPA, the streamed first-to-last interval equals GenAI's `tpot × (n − 1)`
to the millisecond, with or without the per-token detokenization.

Per-rep driver prefill rows on CPU and GPU for a model that runs on paged
attention are prefix-cache hits: each rep ran its warm-up on the same seed ids
in the same process before the measured generate. LFM2 rows are not affected,
since LFM2 already runs on SDPA.

`max_memory_usage` is the server process's peak resident memory, read from the
parent by the per-OS sampler that every engine uses, and watched from the
launch. It therefore covers the blob load as well as the request. A warm-up
server runs the same request first, so the measured server loads from the
cache; see [the memory cell and the cache](#the-memory-cell-and-the-cache).

Off the CPU, the device's memory comes from the OS, as it does for OVMS. How
each OS reads it, and which memory cells pipette refuses on Linux, are in
[peak-memory-windows.md](methodology/peak-memory-windows.md#gpu-and-npu-counter)
and
[peak-memory-linux.md](methodology/peak-memory-linux.md#ovms-openvino-and-onnx-genai).
On macOS, pipette refuses every OpenVINO cell before anything installs.

Every successful cell also records the `observation_max_*` fields
([peak-memory-usage.md](methodology/peak-memory-usage.md#observations-vs-metrics)).
For `uv_openvino` they cover a cold compile: a timing cell's compile in its one
server, and the memory cell's compile in its warm-up server.

On Windows the client starts the venv's base interpreter directly, not the
venv's `python.exe` launcher; [ONNX GenAI measurement](onnx-genai-measurement.md#host-memory)
explains why, and how to read Windows memory rows from older clients.

`decode_throughput` excluding the first token is deliberate and matches the
other backends: llama.cpp runs `llama-bench --n-prompt 0`, MLX reads `mlx_lm`'s
`generation_tps`. A change that moves only the first token will show in
`end_to_end_latency` and not in `decode_throughput`, which is exactly what the
warm-up does.

### No chat template

Prefill and decode are raw-continuation measurements in every pipette backend.
The client sends the prompt as text. The server encodes that text itself
(`add_special_tokens=True`) and passes the token ids to `generate` as
`TokenizedInputs`. If `generate` gets a string, OpenVINO applies the model's
chat template, which adds 9 tokens for LFM2. *Measured* 512/100 decode on CPU
with the per-rep driver: 782.78 ms raw against 794.78 ms templated.

## The warm-up

Every timing cell runs one untimed pass before its measured reps, at the cell's
own prefill and decode counts, on the NPU too. A warm-up is a second
*generate* on an already-compiled pipeline. On the NPU it compiles nothing, so it
does not add to the compiles that took the NPU down
([openvino-ir.md](openvino-ir.md)). On a GPU the first request at a prompt shape
can still compile kernels for it (9 oneDNN kernels on an Intel Arc B390 iGPU,
OpenVINO 2026.4.0), and the warm-up pays them instead of the first rep.

*Measured* through the harness; `decode_throughput_512_100`, 350M on NPU, 5
reps each, with and without the warm-up (CV is the coefficient of variation,
stddev divided by the mean):

| | `decode_time_ms` | stddev | CV |
|---|---|---|---|
| no warm-up | 647.9 | 36.0 | **5.6%** |
| warm-up | 623.4 | 15.2 | **2.4%** |

The warm-up more than halves the deviation (stddev −58%) and moves the mean
−3.8%. The un-warmed number was inflated, not merely noisy. Per-rep the
un-warmed run wanders (166.4, 155.3, 154.2, 142.4, 155.3 tok/s) where the warmed
one holds (163.1, 156.8, 163.7, 163.0, 155.8).

To reproduce these figures outside the harness:

- Build the pipeline with the properties that the harness derives. For the NPU
  these include `GENERATE_HINT=BEST_PERF`. A bare `LLMPipeline` compiles a
  different graph: 12.3 s and about 89 tok/s, against 17.2 s and about
  155 tok/s for the harness.
- Time what the cell times. A wall-clock `generate()` includes the prefill
  that `decode_throughput` excludes. Read that way, the warm-up looked worth
  +18.5%, with CV falling from 8.6% to 0.3%.

### The n=5 stddev is itself unstable

Two runs of the *same* configuration reported `598.1 ± 1.8` and `615.3 ± 22.8`;
a 12x spread in the stddev estimate. So a single 5-rep run cannot settle a
question about variance, and every comparison here should be read as one sample.
The warm-up result above survives that caveat only because the per-rep values
back it: the un-warmed run wanders 166→142 tok/s where the warmed one holds
inside 156-164. Anything smaller than that deserves repeated runs before it is
believed.

## Compile time

The compile is `LLMPipeline(...)` construction, which the server runs before it
listens. It is outside every measured window and is not recorded. *Measured*:

| model | device | compile | measured decode |
|---|---|---|---|
| 350M int4-sym-cw | NPU | 12.4s | 1.1s |
| 1.2B int4-sym-cw | NPU | 32.5s | 1.7s |

A warm-up cannot help this: it runs after the compile.

### `CACHE_DIR`

*Measured.* Setting the `CACHE_DIR` pipeline property makes OpenVINO persist the
compiled blob and load it on the next construction:

| | compile |
|---|---|
| cold (writes the blob) | 12.4s |
| warm (loads it) | **0.58s** |

The cache keys on the pipeline properties, not just the model; verified by
compiling two static shapes into one directory:

```
MAX_PROMPT_LEN=1024   12.87s   cold
MAX_PROMPT_LEN=1024    0.84s   hit
MAX_PROMPT_LEN=2048   21.41s   cold, not a false hit
MAX_PROMPT_LEN=2048    0.87s   hit
MAX_PROMPT_LEN=1024    0.79s   still hit, both blobs coexist
```

So two cells differing only in `MIN_RESPONSE_LEN` cannot silently share a graph.
(The 1024 → 2048 compile cost also confirms the superlinear growth
[openvino-ir.md](openvino-ir.md) records.) The blob is 94 MB for a 350M IR.

The cache is the workspace's per-runtime `cache/<runtime-key>/`, shared across
cells and runs; [architecture.md](architecture.md#runner-cli-layout) says where
it lives and why it is outside the quota. The runtime key digests the
requirements, so an `openvino-genai` bump compiles into a new directory rather
than loading a stale blob. `CACHE_DIR` is added where the
server's command is built (`server.rs`), not by `flags.rs`: it is this host's
scratch path, not something the cell declared, and it must not reach the
recorded flags.

The cache buys wall clock; its correctness value is that every run of a cell
starts from the same place.

#### The memory cell and the cache

The three timing cells time requests to a server that compiled before it
listened, so a shorter compile cannot reach them. `max_memory_usage` samples
the peak resident memory of the process the compile runs in, so it can and does. That is why
the memory cell warms up: a first server loads the model, which compiles the
blob, and runs the cell's request, which on a GPU compiles the kernels for the
cell's prompt shape. The measured server then loads both from the cache.
*Measured* on an NVIDIA A10 through OpenCL: when the warm-up only loaded the
model, the measured server compiled 12 to 16 `.cl_cache` kernels during its
request; once the warm-up sent the request too, it wrote all 39 and the
measured server none, with no change to the measured figures.

On a macOS CPU (`LFM2.5-350M-int8-ov`, 128-token prompt), the server path read
the same peak as the per-rep driver from the same cache state: 3,359,490,048
against 3,359,719,424 bytes warm, 2,568,470,528 against 2,568,503,296 cold.
That run used a release that served OpenVINO on macOS. Current releases refuse
OpenVINO cells on macOS.

*Measured*, NPU, `max_memory_usage_512`:

| | compile in the measured process | `max_host_bytes` |
|---|---|---|
| no cache | 17.7s | 927,793,152 |
| cache | 0.6s (blob load) | 817,033,216 |

−110 MB, −11.9%: on the NPU, compiling costs transient memory that loading a
blob does not.

The cached figure is the better one. It is what running the model costs, while
the uncached figure mixes inference with a one-time compile. But it is a
**discontinuity**: an NPU `max_memory_usage` row measured without the compile
cache does not compare with one measured with it. The gap is wide enough to
read as a regression in a trend. The compile cache is in every client whose
`client_version` is dated 2026-08-03 or later. A comparison across that boundary must say which
side each row is on.

*Measured* on the server path, on an Intel Core Ultra X7 358H (Arc B390 iGPU,
NPU, Windows 11), OpenVINO 2026.4.0, `LFM2.5-350M-int8-ov`,
`max_memory_usage_512`:

- **GPU kernels.** When the warm-up did not send the cell's request, the
  measured request compiled 9 oneDNN kernels for the prompt shape
  (`*.onednn.cl_cache`, 924,204 bytes). Once the warm-up sent the request
  too, it wrote all 34 cache files and the measured server none. On an empty
  cache that raised `max_gpu_bytes` by 0.9 MB (0.2%), from 447.5 to 448.4 MB,
  the filled-cache figure. Host memory did not move. On the CPU and the NPU
  no request writes to the cache: the blob is written during the load.
- **Compile against load.** On the NPU, as on the Core Ultra 7 258V above,
  the server that compiled peaked higher: 1.26 to 1.29 GB of host memory
  against the measured server's 1.10 to 1.17 GB. On the Arc iGPU it is the
  other way round: loading the blob costs more host memory than compiling
  from the IR, so on an empty cache the warm-up read 780 to 783 MB against
  the measured server's 1.06 GB.
- **NPU host noise.** The NPU's host figure moves between runs with no change
  in code or cache. Of 14 runs, 8 read 1.16 to 1.18 GB and the other 6 read
  1.09 to 1.12 GB, while `max_npu_bytes` read 779,628,544 bytes in every
  run. Two NPU rows can therefore differ by up to 89 MB of host memory.
- **Cost.** Sending the request in the warm-up adds about 0.4 s to a CPU
  cell, 0.15 s to a GPU cell and 1.3 s to an NPU cell.

## What the readiness gate cannot see

The Windows gate
([end-to-end-latency.md](methodology/end-to-end-latency.md#system-readiness-control))
holds on CPU load, GPU compute and the throttle-reason flags, and on
temperature only through a configured
ceiling or the critical trip. None of those see NPU work
(`gpu=compute:0%` throughout an NPU cell), as `windows.rs` records. The gate
runs before the server starts and again before every rep, after the compile,
but a rep can still start on an NPU that has not settled.

Gating the NPU properly would mean a signal that sees NPU work. Unmeasured; the
`\Energy Meter(*)\Power` counter that does see NPU draw (6.6W against a 3.3W
floor) is verified on one laptop and reads as idle when absent.

## Per-rep driver rows

`LFM2.5-350M` int4-sym-cw on the Core Ultra 7 258V, 5 reps per cell, measured
with the per-rep driver and no compile cache:

| Cell | CPU | NPU |
|---|---|---|
| `prefill_throughput_512` | — | 173.12 +/- 14.41 ms |
| `decode_throughput_512_100` | 782.78 +/- 3.74 ms | 660.37 +/- 24.61 ms |
| `end_to_end_latency_512_256` | — | 1841.44 +/- 50.37 ms |
| `max_memory_usage_512` | — | 923,475,968 B |

The NPU decodes faster: 151 against 128 tok/s for these 100 tokens. Each rep
paid about 17 s of NPU compile, because these rows have no compile cache
([`CACHE_DIR`](#cache_dir)). Do not compare them with server-path rows.

## Long generations on 2026.2.1

*Measured* with the per-rep driver on a Core Ultra X7 358H, Arc B390
GPU, `OpenVINO/LFM2.5-350M-int8-ov`: in-process `openvino-genai` 2026.2.1
slowed from 6.9 to 10.6 ms per token once a generation reached 256 tokens, and
stayed slow for the rest of the process. `end_to_end_latency_1024_256` read
2655 ± 180 ms against OVMS v2026.4.0's 1534 ± 14 ms (41.5 ms prefill plus
5.85 ms per later token). OpenVINO 2026.4, as bundled in OVMS, held its rate.
Not rechecked on the server path.

## Open

- `pipette storage` does not see the compile cache. `runtimes remove` reclaims
  a runtime's blobs, but nothing evicts them under quota pressure. They should
  be a storage kind that is always the first candidate, being pure derived
  bytes.
- The NPU has no readiness signal: the gate runs before every rep but cannot
  see NPU work.
- Whether the warm-up's 2.4% residual CV can go lower, and whether it holds on
  larger models. The 1.2B showed 7.7% in a run without the warm-up, and no run
  has measured it with the warm-up.
