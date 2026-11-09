# OVMS: OpenVINO Model Server

`ovms` serves a `Model::Openvino` directory through Intel's OpenVINO Model
Server (OVMS), a native C++ binary from an upstream release archive, and
measures it over HTTP. `uv_openvino` runs the same OpenVINO IR (intermediate
representation) model through an `openvino-genai` server of its own
([openvino-measurement.md](openvino-measurement.md)).
Both use the same model type, the same per-cell `device`, and the same NPU
static-shape rules ([openvino-ir.md](openvino-ir.md)).

The backend is `crates/pipette-ovms`. It needs no Python and no venv.

OVMS runs on Linux and Windows x86_64. On any other OS, macOS included, pipette
refuses the cell before anything downloads. On Linux and Windows the install
also refuses a release archive built for the other one.

## Runtime

Name the runtime with an `ovms://` URI, for example
`ovms://version=v2026.4.1&flavor=windows-x64`. The keys and the flavors are in
[models-and-runtimes.md](pipette-cli/models-and-runtimes.md#runtime-schemes).
`pipette runtimes catalog ovms` lists the bundled releases, and
[usage.md](pipette-cli/usage.md#3-install-a-runtime) shows how to install one.

## A cell

Each cell launches its own server, waits for the model to load, measures, then
kills and reaps the server. The argv is built in
[`server.rs`](../crates/pipette-ovms/src/server.rs) and
[`flags.rs`](../crates/pipette-ovms/src/flags.rs). The flags that shape a
reading:

- `--model_path`, not `--source_model` (gotcha 5).
- `--enable_prefix_caching`, always passed, `false` unless the cell sets it
  (gotcha 1).
- `--max_prompt_len` and `--plugin_config` carry the NPU settings ([NPU](#npu)).
- `--pipeline_type LM_CB --max_num_seqs <batch>`, only on a batched entry
  ([Batched entries](#batched-entries)).
- `--cache_dir` is the runtime's compile cache (`cache/<key>/`, laid out as
  `uv_openvino`'s but keyed to this runtime).

Every timing cell starts two servers with that argv. The first server sizes
the prompt with `/v3/tokenize`, and pipette then stops it. The second server
measures. A tokenize call breaks the pooled graph it ran on for the next
completion (gotcha 8), so the server that measures never tokenizes. The second
server loads from the compile cache that the first one filled: about a second
on the Arc GPU. The memory cell also starts a warm-up server between the two
([`max_memory_usage`](#max_memory_usage)).

- The server runs with the release directory at the front of its library path
  (`PATH` on Windows, `ovms/lib` on `LD_LIBRARY_PATH` on Linux), plus the
  client configuration's `envs`.
- Readiness: `/v2/models/pipette/ready` returns 404 until the model has loaded.
  OVMS exits on a bad device or model path, and the wait reports that exit with
  the log tail.
- The executable and argv are recorded as the result's `command`. The server's
  log is appended to the result's stdout.
- The child is registered with `pipette_subprocess::cleanup`, so an interrupt
  (Ctrl+C) does not leave `ovms.exe` holding the port or the device.
- If the cell does not set `http_timeout_seconds`, each wait of a timing
  request is at most 3600 s for each request in flight. The bound applies to
  the wait for the headers, then to the wait for each chunk, and not to the
  whole exchange. The longest wait is the prefill before the first chunk.
  `attempt_timeout_seconds` bounds each whole repetition, 300 s by default
  ([the attempt limit](pipette-cli/usage.md#5c-the-attempt-limit)). The first
  server's whole start, where a cold cache compiles the model, does not count
  against it; the starts of the servers after it do.

## Batched entries

The batch rules shared by every engine are in
[end-to-end-latency.md](methodology/end-to-end-latency.md#batched). OVMS takes
one streamed request per slot, each timed on its own clock. The server argv adds
two flags, and only for a batch, so the recorded command of a batch-1 cell has
no batch flags:

- `--pipeline_type LM_CB` asks for OVMS's continuous-batching servable. Without
  it, a device that cannot batch continuously falls back to the legacy
  servable, which runs requests one after another. With it, OVMS refuses such
  a device at load (the NPU: `The LLM Node resource initialization failed`,
  after 2.5 s).
- `--max_num_seqs <batch>` caps the scheduler at the batch.

Two batches are refused before anything launches, because OVMS would run them
serially and the row would name a concurrency it did not have:

- **On the NPU.** Its stateful pipeline does not batch continuously.
- **Above one request per hardware thread.** OVMS pools one MediaPipe graph
  per hardware thread, and starts as many streaming REST (HTTP API) threads. A request
  that finds no idle graph waits for one: at a batch of 32 on the 16-thread
  box, with 16-token prompts and 512 new tokens, the 17th request's first
  token came after the first 16 had finished. `--rest_workers` does not lift
  this, since the pool is the limit.

Measured through the live test on a Windows test box (Core Ultra X7 358H), Arc
B390 GPU, `LFM2.5-350M-int8-ov`, 1024-token prompts, 128 new tokens, 2026-09-24:

| cell | batch 1 | batch 4 |
|---|---|---|
| `prefill_throughput` (time to first token, TTFT, per request) | 32.9 ms | 81.6 ms |
| `decode_throughput` (per sequence) | 179 tok/s | 158 tok/s, 634 tok/s in all |
| `end_to_end_latency` | 730 ms | 933 ms |

A slot that fails fails the rep. A retried slot would run after its siblings
had finished, at a concurrency the row does not name.

## API contract and its gotchas

Probed against OVMS 2026.4.0 on the same test box:

1. **Prefix caching is on by default.** On identical prompts it lowers
   later reps' TTFT from about 90 ms to 66 ms. The cell always passes
   `--enable_prefix_caching`, and when the plan leaves it unset the resolved
   flags record the `false` this client supplied. Without that, a record
   could not tell a cache-on run from an unconfigured one.
2. **Token-id prompts are refused** (`prompt is not a string`, for `[ids]` and
   `[[ids]]`). Prompts are text, sized with `/v3/tokenize`
   (`{"model", "text"}`, which includes the beginning-of-sequence (BOS) token
   in its count). `usage` checks the
   shape on every rep, and `finish_reason` must be `"length"`.
3. **The first stream chunk has empty text, but it is the first token.** On
   the Arc GPU with a 1022-token prompt, a two-token stream's opening chunk
   arrived at 46.1 ms, against 47.0 ms for a whole one-token request, and the
   second token at 59.6 ms. A one-token stream sends only its finish chunk.
   TTFT is stamped on the first chunk. Before the warm-up, a
   `prefill_throughput` or `decode_throughput` cell sends one untimed
   one-token request. The cell fails if the gap from the opening chunk to the
   finish chunk is longer than the gap from the send to the opening chunk.
   That pattern means that the OVMS release sends the opening chunk before the
   prefill.
4. **Text trails the tokens by about five.** In a 100-token stream the first
   text came 22 to 35 ms after the opening chunk, and up to ten tokens late
   in some runs; after that, chunks come about one per token (94 text chunks
   for 100 tokens). Stamping the first text started the decode interval late
   and read decode about 6% fast. The last-token stamp is taken on the
   `finish_reason` chunk. Usage comes in a trailing chunk with `choices: []`.
5. **`--source_model` writes `graph.pbtxt` into the model directory.** The cell
   uses `--model_path`, which leaves the stored directory untouched.
6. **The NPU does not use continuous batching.** It uses the stateful
   static-shape pipeline. `--max_prompt_len` and `--plugin_config` are merged
   into one plugin config, alongside the `CACHE_DIR` that `--cache_dir` sets.
7. **Requests share a pool of graphs.** The `--task` graph OVMS generates
   pre-builds one MediaPipe graph per hardware thread (16 on
   the test box) and hands them out in turn. A graph file without the
   pool directive, or `OVMS_GRAPH_QUEUE_OFF=1`, runs each request on a fresh
   graph instead, but then the first streamed request crashes the server
   (an access violation in `openvino_genai.dll`). The pool is not optional.
8. **`/v3/tokenize` breaks the graph it ran on.** The next completion on that
   graph fails with HTTP 400, `Packet timestamp mismatch ... Current minimum
   expected timestamp is 1 but received 0`, and the graph works again after
   that. It is deterministic: after K tokenize calls on a fresh 16-graph pool,
   completions 17 − K to 16 fail and the rest succeed. Every cell therefore
   sizes the prompt on a server it discards.
9. **A request can hang.** Under OVMS 2026.4.0 a request sometimes gets no
   response while the server keeps serving others: one `end_to_end_latency`
   rep sent on its own, and a few streams in the first concurrent round after
   some launches (up to 6 of 32, 2 of 4, 1 of 8), but none in 15 fresh
   launches at a batch of 8. The client waits until its HTTP timeout. No flag
   was found that avoids it.

## Readings

Requests use `ignore_eos: true` and `temperature: 0`, one at a time or a batch
at a time. Readings are taken on the client's clock, as the
`pipette_ops::measurement` entries define them:

| cell | request | reading |
|---|---|---|
| `prefill_throughput` | streamed, 1 token | send → first chunk |
| `decode_throughput` | streamed, *n* tokens | (finish chunk − first chunk), scaled from *n − 1* to *n* tokens |
| `end_to_end_latency` | streamed, *n* tokens | send → finish chunk |
| `max_memory_usage` | not streamed, 2 tokens | the server's peak resident memory; its GPU memory too on Windows and Linux, its NPU memory on Windows |

Before the reps, the entry runs one untimed warm-up at the cell's shape,
one request per slot. The server stays up across reps, so this warms the
server rather than a single process.

A cell sends every request through one HTTP client, so the reps reuse one
pooled connection. A fresh loopback connection costs about 15 ms on
the test box, and a client per request put that cost inside every
reading.

## `max_memory_usage`

The reading is the server process's lifetime peak resident memory, read from
the kernel after one request at the cell's prompt length: `PeakWorkingSetSize`
on Windows, `VmHWM` on Linux. The request generates two tokens. The second
token runs one decode pass, so the peak includes the decode buffers. The
`uv_openvino` memory cell generates one token, so its peak has no decode pass:
do not compare the two runtimes' `max_memory_usage` rows as one measurement.

A launch that compiles reads differently from one that loads from the
compile cache. The sizing server compiles a cold `(model, device)` first, but
it sends no completion, so on a GPU the kernels for the prompt's shape are
still to compile. A warm-up server then sends the cell's request, its reading
checked and dropped, and the measured server loads both from the cache. The
sizing calls run on the discarded server, so they are not in the peak.

On an Intel Arc B390 iGPU (Core Ultra X7 358H, Windows 11, OVMS v2026.4.0,
`LFM2.5-350M-int8-ov`, a 512-token prompt) with an empty cache, the sizing
server writes the blob and 13 `.cl_cache` files and the request 15
`*.onednn.cl_cache` files (1.3 MB). Without the warm-up, the measured request
wrote those 15, and its `max_gpu_bytes` read 1.3 MB (0.3%) below a filled
cache's, 440.57 against 441.89 MB, with the same host figure. With it, the
measured server writes nothing and reads 441.89 to 441.90 MB on either cache.
On the CPU and the NPU the request writes nothing, so there the warm-up only
adds time. It adds about 1 s to a GPU cell, 1.8 s to a CPU cell and 2.6 to
3.4 s to an NPU cell.

**Off the CPU, the device's memory has its own field:** `max_gpu_bytes` for a
GPU and `max_npu_bytes` for an NPU. How each OS reads these fields is in
[peak-memory-windows.md](methodology/peak-memory-windows.md#gpu-and-npu-counter)
and
[peak-memory-linux.md](methodology/peak-memory-linux.md#ovms-openvino-and-onnx-genai).
The device that the cell names must show memory, or the cell fails. A custom
device records what it held. On Windows, the bytes on the Arc iGPU and the NPU
are also inside the working set. During each run the free memory of the host
fell by no more than the working set, so the fields overlap and do not add up
(LFM2.5-350M int8, 512-token prompt):

| device | `max_host_bytes` | `max_gpu_bytes` | `max_npu_bytes` | free memory fell by |
|---|---|---|---|---|
| CPU | 970 MiB | `null` | `null` | 975 MiB |
| GPU | 686 MiB | 421 MiB | `null` | 673 MiB |
| NPU | 1,190 MiB | 22 MiB | 757 MiB | 963 MiB |

OVMS keeps about 22 MiB on the iGPU whichever device it serves from; a CPU cell
does not read it.

The NPU's host figure moves between runs with no change in code or cache:
eight NPU runs read 1,126 to 1,205 MiB, while `max_npu_bytes` read 757 MiB
in every one.

On Linux, `max_host_bytes` and `max_gpu_bytes` add up. On Linux, pipette also
refuses a memory cell on the NPU or on a custom device. Both rules are in
[peak-memory-linux.md](methodology/peak-memory-linux.md#ovms-openvino-and-onnx-genai).

Every cell also records the `observation_max_*` fields
([peak-memory-usage.md](methodology/peak-memory-usage.md#observations-vs-metrics)).
For OVMS, the memory cell's observation also covers its warm-up server, and
with it the kernel compiles of a GPU. No cell watches the sizing server, so on
an empty cache no observation includes the model's compile. On the NPU that
compile took about 37 s, and the observation of the memory cell on an empty
cache read within 4 MB of the observation on a filled cache.

## NPU

The NPU rules are `uv_openvino`'s
([pipette-cli/openvino.md](pipette-cli/openvino.md#runtime-flags)). OVMS keeps
its own copy of them in `crates/pipette-ovms/src/flags.rs`. That copy passes
`--max_prompt_len` only when the cell sets `max_prompt_len`. It passes
`min_response_len` and `generate_hint` in `--plugin_config`. On the test
box, the NPU (driver `32.0.100.4509`) runs at about 1 tok/s both through OVMS
and through `openvino-genai`, so NPU numbers need a box with a healthy driver.
