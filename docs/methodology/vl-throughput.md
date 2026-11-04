# Vision-Language Throughput and Latency Methodology

## Measured Quantity

`vl_throughput` measures how fast a vision-language model turns an
image-plus-text prompt into generated tokens; the image passes through the
vision encoder and multimodal projector, then the language model prefills and
decodes.

**Why we test it.** Vision-language prompts are dominated by image tokens
(hundreds to thousands per image; see [vl-image-tokens.md](vl-image-tokens.md)),
so their prefill and time-to-first-token behave very differently from text. This
benchmark captures per-model vision-prefill and decode speed. Peak memory for
the same workload is covered by [vl-max-memory.md](vl-max-memory.md).

The submitted result reports elapsed times and token counts, not rates:

- `prompt_tokens`: the whole prefill (image, text and template).
- `image_tokens`: the image-only share, when counted.
- `prompt_ms`.
- `predicted_ms`: every `parameter_decode_tokens` decode step (see
  [Decode time](#decode-time)).
- `total_time_ms`, where the client times it (see
  [End-to-end latency](#end-to-end-latency)).

Each timing comes with its sample standard deviation. The fields, the benchmark
parameters and the derived metrics (`ttft`, `prefill_throughput`, `decode_throughput`,
`end_to_end_latency`, with the token counts as observations) are defined in
pipette-mgmt
[docs/benchmarks.md §2.6](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#26-vl-throughput).

On `llama-server`, `ttft` is the server-reported prefill time, not a client
wall-clock time-to-first-token; MLX and the in-process clients (Android, iOS)
time it on their own clock (see [Runtime Details](#runtime-details)). Runtime install,
model/projector load, server startup, warmup, readiness waits, and result sync
are outside the timed windows.

## Measurement Challenges and Controls

Prefill cost scales with image tokens (resolution, frame count, per-model
tiling), text-prompt length, batch sizing, KV-cache format, and, separately
from the language weights, the vision encoder and projector precision. A result
is comparable only for the same benchmark shape, model, projector, runtime, and
relevant runtime flags.

Image-token count is model-specific, not a function of pixel dimensions; the
per-model counts and the runtime flags that change them are in
[vl-image-tokens.md](vl-image-tokens.md). This benchmark records
`prompt_tokens` and `image_tokens` so the workload each model ran is visible.
A model that represents an image in fewer tokens does less work. How to compare
rows across models is in
[Result Validity and Comparability](#result-validity-and-comparability).

Warmup and thermal state can dominate timings, especially on small devices and
heavy cells (large images or many frames). The runner separates a warmup run
from the measured samples and gates every measured repetition (see
[System Readiness Control](#system-readiness-control)). Individual samples are
still subject to scheduler noise, so
the benchmark measures five repetitions and reports mean and sample standard
deviation for prefill, decode and, where the client times it, the whole
request.

## Measurement Protocol

1. Resolve the shape (`parameter_image_width`, `parameter_image_height`,
   `parameter_text_tokens`, `parameter_decode_tokens`, `parameter_num_images`).
2. Require a multimodal projector (`--mmproj`); fail early if absent.
3. Before the server starts, count the image tokens of one frame with the
   calibration helper (see [vl-calibration.md](vl-calibration.md)). The count
   sizes the context. A multi-image cell runs the helper once more, on all of
   its frames, for the `image_tokens` it reports.
4. Start `llama-server` with the context sized from that count, wait until
   ready, and build the synthetic image and exact-length text prompt.
5. Run one warmup request and validate its shape.
6. Run five measured requests; before each, run the platform readiness gate
   (outside the timed window).
7. Validate each response: generated tokens equal `parameter_decode_tokens` and
   `finish_reason == "length"`.
8. Check the calibrated `image_tokens` against the measured `prompt_tokens`.
   The row reports no `image_tokens` if calibration is unavailable or the count
   does not fit in `prompt_tokens`.
9. Report `prompt_tokens`, `image_tokens`, and the mean and sample standard
   deviation of `prompt_ms`, `predicted_ms` and, where the client times it,
   `total_time_ms`.

The timed windows are the server's own `prompt_ms` and `predicted_ms` counters,
the latter scaled to every decode step (see [Decode time](#decode-time)), and,
for `total_time_ms`, this client's clock from the request's send to its reply
(see [End-to-end latency](#end-to-end-latency)). Image generation, token
construction, readiness waits, and the `image_tokens` calibration are outside
all three on the desktop clients; the iOS `total_time_ms` includes building and
tokenizing the frames.

### Decode time

The first generated token comes from the prefill's last logits, so a clock
that starts at it times `parameter_decode_tokens - 1` decode steps. Both desktop
clocks start there: `llama-server` starts `predicted_ms` when it samples the
first token, and the MLX stream's interval runs from the first token to the
last. The desktop clients scale each rep by
`parameter_decode_tokens / (parameter_decode_tokens - 1)` before taking the mean
and standard deviation, so `predicted_ms` covers every step, as llama-bench's
decode does for the text benchmarks. A cell of fewer than 2 decode tokens has
no step to time and is refused before the model loads. The Android and iOS
clients time every step themselves and are not scaled.

**In-process clients (Android, iOS)** follow the same protocol. Steps 2-4 and
8 differ, because there is no server to start or ask. The clients call mtmd,
llama.cpp's multimodal library, directly:

- The projector loads with `mtmd_init_from_file` against the already-loaded
  model. Its own warmup pass is off, so the benchmark's untimed warm-up rep is
  the first work the encoder does.
- The model's own tokenizer sizes the text prompt, in place of `/tokenize`.
- The two windows are measured around `mtmd_helper_eval_chunks` and the
  generation loop. These are the same two phases that the server's counters
  separate.
- `image_tokens` needs no calibration. The client reads mtmd's per-chunk
  accounting directly, so it is the same marker-free count that the desktop
  helper reports on the same llama.cpp build, frame merging included.

Android runs single-image cells only and refuses `parameter_num_images > 1`.

The iOS client also times `total_time_ms` as one window from building the
synthetic frames and tokenizing them with the prompt (`mtmd_tokenize`, which
also preprocesses the image) to the sample of the last token. That is the scope
of its text end-to-end latency. Filling the gray bitmap stands in for the
server's PNG decode. Its projector loads before the context and counts one
frame, and the context is sized from that count as a calibrated desktop run's
is, with no 8192 floor (see
[vl-calibration.md](vl-calibration.md#deriving-n_ctx)).

The iOS client rebuilds the desktop request rather than approximating it, so
`prompt_tokens` counts the same scaffolding on both paths:

- The text is sized with specials counted, as `/tokenize` counts them.
- The content is the text followed directly by one image marker per frame. This
  is the join that the server applies to a text part followed by image parts.
- That single user message is rendered through the model's chat template with
  the assistant turn opened, through llama.cpp's built-in template path.

A template that path does not recognize
(Gemma-4's) falls back to the Jinja renderer the server itself uses, with the
server's default thinking setting; a template it does recognize can still
render slightly differently from the server's Jinja render. The projector
is placed on the GPU regardless of `n_gpu_layers`, as `llama-server` places it
by default. Android sends the bare marker followed by the text with no
template, sizes the text without specials, and derives the projector's
placement from `n_gpu_layers`; its `prompt_tokens` is therefore smaller than
the desktop's by the template's tokens for the same cell.

### End-to-end latency

`total_time_ms` is one request on this client's clock, the kind of window the
text benchmark's field of the same name times (see
[end-to-end-latency.md](end-to-end-latency.md#measurement-protocol)). On
`llama-server` it runs from the send, once the request's body is encoded, to
the arrival of the whole reply, which is not streamed. On MLX it runs from the
send to the chunk carrying `finish_reason`. On iOS it runs from building and
tokenizing the prompt's frames to the sample of the last token, where
`llama-server` ends a request. Android does not time it. It is measured, not
`prompt_ms + predicted_ms`.

- `llama-server` reads the request and decodes and preprocesses the image on
  its HTTP thread before the slot's prefill timer starts, and sends the reply
  after its decode timer stops. At 512x512 with 1024 text tokens, across two
  models, that took 4 to 9 ms a request on Apple silicon on hand-started
  servers and 7 to 11 ms through the client's default server, and 12.5 to
  18.7 ms on a CUDA GPU, depending on the server's flags (threads, offload,
  flash attention). That is up to 7.5% of the request, and 25 to 35 ms at
  1280x1280.
- `predicted_ms` covers every decode step (see [Decode time](#decode-time)),
  but a request runs only the `parameter_decode_tokens - 1` steps after the
  first token, and the iOS window ends at the same point. So `total_time_ms` can
  read below `prompt_ms + predicted_ms` by up to one decode step.

The request goes out on the client's pooled connection, while the text
benchmark's `llama-server` requests ask for a new connection each. Both
`llama-server` and the uvicorn server behind `mlx_vlm.server` close a
connection left idle for 5 s, so a rep that the readiness gate held longer
than that opens a new connection inside its window. A new connection costs
about 0.1 to 0.25 ms on Linux and macOS and 13 to 18 ms on Windows, so on
Windows `total_time_ms` and its standard deviation depend on how long the gate
held each rep. `mlx_lm.server`, which serves the MLX text benchmark, answers in
HTTP/1.0 and closes every connection, so each of its text requests connects
anew. The desktop window also covers the server rendering the chat template,
which the text benchmark's `/completion` and `/v1/completions` requests do not
run; the iOS client renders the template before its window.

## Benchmark Shape

The five `parameter_*` fields are defined in pipette-mgmt
[docs/benchmarks.md §2.6](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#26-vl-throughput).
The CLI's smoke benchmark uses one 224x224 image with 8 text and 8 decode
tokens.

## Token Control

The prompt is built to the exact requested shape:

- The image is a solid-gray PNG generated in the runner at exactly the requested
  dimensions; not stored in the definition and not downloaded. Content is
  irrelevant to token count and encode cost, so a flat field gives a
  deterministic workload; `vl_throughput` is therefore a performance benchmark
  only, with image understanding covered by the eval benchmarks.
- The text portion is built to exactly `parameter_text_tokens` tokens via the
  server's `/tokenize` endpoint; `0` sends no text part.
- **Multiple images:** `parameter_num_images` copies of the (same synthetic)
  image are attached as separate parts of the one user message, so the prompt is
  `[optional text] + N × [image]`. Each image expands into its own block of image
  tokens. The context is sized from the one-frame count times N
  ([vl-calibration.md](vl-calibration.md#deriving-n_ctx)). The reported
  `image_tokens` is one count over all the frames together, because some
  llama.cpp builds merge frames
  ([vl-image-tokens.md](vl-image-tokens.md#multiple-images-num_images-and-the-prompt)).

The invariant is the generated-token count: a run is invalid if the server does
not generate exactly `parameter_decode_tokens` tokens or does not stop at the
length limit. The image-token count is a model-specific output of preprocessing,
recorded rather than asserted. It is the calibrated marker-free count for the
cell's frames on every path, and none when calibration is unavailable: a
marker-inclusive difference would not compare with the other rows.

## System Readiness Control

The runner runs the platform readiness gate before each measured repetition,
outside the timed window, and fails the run if the device does not reach
readiness before the platform deadline. The per-platform criteria are in
[end-to-end-latency.md](end-to-end-latency.md#system-readiness-control).

## Runtime Details

`vl_throughput` runs on the **llama.cpp** backend, and on desktop **MLX**
(below). On desktop platforms llama.cpp drives `llama-server` over its chat
endpoint; on Android and iOS it runs in-process against `libmtmd` (mtmd chunks,
no server, no `/tokenize`). The chat endpoint is required on desktop because
llama.cpp's `/completion` + `multimodal_data` path does **not** run the image
through the vision tower (prefill ignores it and `prompt_n` stays constant with
image size); `/v1/chat/completions` encodes it, so the reported prompt length
scales with resolution. The in-process clients call `mtmd_tokenize` +
`mtmd_helper_eval_chunks` directly, which always encodes.

- Workload: `POST /v1/chat/completions` with one user message containing the
  optional text part and `parameter_num_images` `image_url` parts,
  `temperature` 0, `max_tokens = parameter_decode_tokens`, `ignore_eos: true`
  (under which the server never samples an end-of-generation token, so the
  decode count is exact), `cache_prompt: false`. Builds b7508 to b8717 ignore
  the flag (fixed upstream in PR #21203); on those, a model that stops early
  fails the `finish_reason == "length"` check rather than reporting a short run.
- Timing source: the server's `timings.prompt_ms` and `timings.predicted_ms`,
  the latter scaled to every decode step, and, for `total_time_ms`, this
  client's clock from the send to the reply.
- Server launch: the client adds no checkpoint or slot flag
  ([Context checkpoints](#context-checkpoints)).
- The projector is passed with `--mmproj`; its quantization is part of the
  runtime condition and must match across compared runs.

Runtime flags that shape the server (context, batching, KV cache, mmap,
threads, accelerator placement) are part of the benchmark condition and share
the reserved-flag policy of the other `llama-server` benchmarks.

**TensorRT Edge-LLM** (`trt_edgellm`, a Jetson) runs the `llama-server` path
above request for request, through the same client
([`pipette_completions::llama_server`](../../crates/pipette-completions/src/llama_server.rs)):
the same filler, gray PNGs and chat body, the server's `timings` and the same
reduction. It differs in four ways:

- The engines are built first, by the server's `--build-only` into the
  runtime's compile cache, before the server starts and outside the memory
  observation. A warm cache makes that a check of a few seconds
  ([TensorRT Edge-LLM](../pipette-cli/trt-edgellm.md)).
- `image_tokens` is the server's own count, `timings.image_n`, which applies
  the vision engine's resize bounds as the archive's `llama-mtmd-cli` shim
  does, so no calibration helper runs. Each rep's count is checked like
  `prompt_tokens`.
- `prompt_ms` covers the on-GPU resize and normalization of the images, which
  `llama-server` does on the CPU before its timer starts: about 1.3 ms on two
  256x256 images and 7 ms on three 512x512 ones on an AGX Orin.
- The row's `client_extras.trt_edgellm` is the engine stack the server's
  `/props` reports (Edge-LLM commit, TensorRT, CUDA, L4T, SoC).

**MLX (macOS)** drives `mlx_vlm.server`, the runtime's own OpenAI-compatible
server, from a runtime row that ships `mlx-vlm` (for example `0.7.4`), with
its prefix cache off (`APC_ENABLED=0`). It
differs from the `llama-server` path in these ways:

- Timing source: this client's clock around a streamed request. `prompt_ms` is
  the send to the first token, `predicted_ms` the first token to the last,
  scaled to every decode step, and `total_time_ms` the send to the chunk
  carrying `finish_reason`. The
  server's own `prompt_ms` is not the same quantity on every model: the
  families that cache image features (Qwen3.5, Gemma 4, DeepSeek V4) encode the
  image before that timer starts, the others (LFM2-VL) inside it. On one
  768x768 image, Qwen3.5-0.8B read 134 to 174 ms on the server's timer and 305
  to 396 ms on the client's clock; LFM2-VL-450M read 265 to 285 ms and 310 to
  328 ms. The client's clock also counts the upload and the server's image
  decoding and preprocessing, which `llama-server`'s `prompt_ms` leaves out, so
  an MLX `ttft` reads higher than a llama.cpp one for the same work.
- Each request's image URL carries its own `attempt=` parameter. The server
  caches image features by URL, so a repeated URL would drop the image encode
  from every rep after the warm-up.
- The server has no `ignore_eos`, so `logit_bias` alone keeps the EOS ids (the
  tokenizer's and the model config's) from being sampled. The text is sized
  with the same tokenizer, loaded the way the server loads it (without the
  model repo's own code).
- `image_tokens` comes from the model's own processor, which a helper runs on
  the cell's frames before the server starts. How it counts, and when the row
  reports none or the cell fails, is in
  [vl-image-tokens.md](vl-image-tokens.md).

### Context checkpoints

`no_cache = true` makes the server comparable with `llama-bench`; what it
renders and the build it needs are in
[llamacpp.md](../pipette-cli/llamacpp.md#execution-notes). On a hybrid or
sliding-window model llama-server saves the state it cannot roll back while it
prefills, decoding the prompt's last tokens in batches of their own to do so,
and reads a saved state back only to reuse a cached prompt, which
`cache_prompt: false` never does. `llama-bench` saves none.

On llama.cpp b10216 (Metal) at 1024 text tokens and one 512x512 image, turning
them off lowers `prompt_ms` by 28.5 ms (7.4%) on gemma-4-E2B and 3.5 ms (1.2%)
on Qwen3.5-2B, each percentage of the reading with them on, and by less than
run-to-run noise on LFM2.5-VL-450M, so `ttft` and `prefill_throughput` move
with it, as does `end_to_end_latency` on rows that send `total_time_ms`.
`predicted_ms` and the dense InternVL3.5-2B do not move. The recorded memory
drops by the checkpoints that the server does not hold, about 19 MiB each on
Qwen3.5-2B.

One slot instead of the default four lowers the recorded memory by about 60 MiB
on Qwen3.5-2B and 9 MiB on InternVL3.5-2B and moves the timings by at most
3 ms; the one slower reading is gemma-4-E2B `prompt_ms`, 1.2 ms (0.3%). How
to tell rows that ran four slots under `no_cache` is in
[llamacpp.md](../pipette-cli/llamacpp.md#execution-notes) (No cache).

## Unsupported Runtime Paths

These runtimes refuse a `vl_throughput` cell with an error, so a plan targeting
them fails fast instead of silently skipping:

- `pipette-torch-oai` refuses `vl_throughput` and `vl_max_memory`; it runs
  `vl_spec_decode` and `vlm_eval`.
- Core AI, OpenVINO, OVMS and ONNX GenAI on the CLI refuse every VL benchmark.
- The iOS client's ONNX GenAI, MLX and Apple Foundation engines refuse every VL
  benchmark.

## Result Validity and Comparability

A run is invalid if the projector is missing, the server fails or times out, the
readiness gate fails, the generated-token count differs from
`parameter_decode_tokens`, or the response does not stop at the length limit.

To compare `vl_throughput` results:

- Use the same benchmark ID, model, projector and its quantization, runtime,
  version, and relevant runtime flags.
- Read `prompt_tokens` / `image_tokens` alongside the timings: at a fixed
  resolution different models process different token counts, so `ttft` and
  `end_to_end_latency` are comparable head-to-head only at matched token counts
  (the iso-token view in
  [vl-image-tokens.md](vl-image-tokens.md#comparing-models)).
- Prefer `decode_throughput` and per-token rates when comparing across models at
  a fixed image, since they normalize out the differing image-token counts.
- Use the mean and sample standard deviation together; a large deviation means a
  noisy run even when the mean looks plausible.
- Read `end_to_end_latency` with its client's window in mind (see
  [End-to-end latency](#end-to-end-latency)), and do not compare it with an
  earlier row's `e2e_latency`, which sums its prefill and decode times.
- Desktop rows from builds that did not scale `predicted_ms` read
  `decode_throughput` high (1.6% at 64 decode tokens, 6.7% at 16); compare
  desktop rows only on one side of that change, by `client_version`.
- On the MLX `0.7.1+lfm2-vl-speculative` row LFM2-VL's images precede the text,
  so its `prompt_tokens` can differ by a token from `0.7.4` and `llama-server`.

## Code References

Benchmark definition and result fields:
[`pipette_plan_types::benchmark`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/benchmark/mod.rs) and
[`pipette_cli::benchmarks::standard`](../../crates/pipette-cli/src/benchmarks/standard.rs).

The llama.cpp runner (request, synthetic image, text-prompt construction,
validation) is
[`pipette-llamacpp vl_throughput`](../../crates/pipette-llamacpp/src/execute/vl_throughput.rs),
with server startup in [`server`](../../crates/pipette-llamacpp/src/server.rs),
the chat request, its timings and the text filler in
[`pipette_completions::llama_server`](../../crates/pipette-completions/src/llama_server.rs),
and context sizing and the `image_tokens` calibration in
[`mtmd`](../../crates/pipette-llamacpp/src/mtmd.rs).
The MLX runner (request, synthetic image, stream timing) is
[`pipette-mlx vl_throughput`](../../crates/pipette-mlx/src/execute/vl_throughput.rs),
with the server launch in [`server`](../../crates/pipette-mlx/src/server.rs) and
the `image_tokens` helper in
[`image_tokens`](../../crates/pipette-mlx/src/image_tokens.rs).
The in-process runners are
[`LlamaBenchmark.runVlThroughputImpl`](../../android/Pipette/app/src/kotlinEngine/java/ai/liquid/pipette/llama/LlamaBenchmark.kt)
on Android and
[`LlamaExecute.vlThroughput`](../../ios/Pipette/Pipette/Runtimes/Llama/Execute/LlamaVlThroughput.swift)
on iOS, the latter over the mtmd primitives in
[`LlamaMtmd`](../../ios/Pipette/Pipette/Runtimes/Llama/LlamaMtmd.swift).
