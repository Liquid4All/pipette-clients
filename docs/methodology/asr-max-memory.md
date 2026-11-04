# Speech Recognition Peak Memory Methodology

## Measured Quantity

`asr_max_memory` measures the **peak memory** used while running one clip of
audio, and any text beside it, through an audio-capable language model. It
reports the fields of the text [peak memory benchmark](peak-memory-usage.md),
`max_host_bytes`, `max_gpu_bytes` and `max_npu_bytes`, in the ordinary
`MaxMemoryUsage` result shape; the benchmark's type is what tells it from
`vl_max_memory`. The definition, parameters and derived metrics are
management's
([pipette-mgmt docs/benchmarks.md §2.11](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#211-asr-peak-memory)).

The measurement is `vl_max_memory`'s and shares its code path: the per-flavor
peak sampler, the counter semantics and unified-versus-split rule, the
[warm-up server](vl-max-memory.md#warm-up-server) whose reading is dropped,
the swap-aware figure and its `client_extras` on Linux and Android, and one
measured run with no thermal loop. The metric is the measured server's peak;
the payload's memory observation covers the warm-up server too, so it can read
a little above the metric, as it can for `vl_max_memory`. This doc covers only
what differs for audio.

## Purpose

A speech model adds two memory costs on top of its language weights: the
**audio encoder and projector** (a separate file, resident alongside the
weights) and a **prefill the clip decides**. A 30 s clip is several hundred
audio tokens (see [asr-throughput.md](asr-throughput.md#counting-audio-tokens)),
and those tokens occupy the KV cache and the prefill buffers. The number that
says whether a model fits a device is how its footprint grows with clip length.

## Workload

The cell runs the workload of the `asr_throughput` cell with the same
parameters, on the same ladder
([The ladder](asr-throughput.md#the-ladder)): the same clip, the text beside
it built and placed as `asr_throughput` builds and places it
([The text beside the clip](asr-throughput.md#the-text-beside-the-clip)) on
each server before the clip's request, and the same ask (the recipe in
[asr-throughput.md](asr-throughput.md#the-transcription-prompt)). The clip goes
through `llama-server`'s `/v1/chat/completions` as an `input_audio` part, and
**`parameter_decode_tokens` tokens are decoded** (`max_tokens: N`,
`ignore_eos`). The peak therefore captures the audio encode, the KV cache of
the prompt and of the output, and the decode buffers. The reply must stop at
the limit with all N tokens decoded, or the cell fails: a peak that did not
decode the output is not the workload's. Its times are not read.

## The Context Is Sized From The Exact Audio Token Count

The KV cache is allocated for the whole context up front, so a context larger
than the work inflates the peak with memory nothing uses. The context uses the
`asr_throughput` formula
([The text beside the clip](asr-throughput.md#the-text-beside-the-clip)), where
`audio_tokens` is the **exact count that libmtmd produces for the clip**. The
runtime's own `llama-mtmd-cli` reads the count, and the client memoizes it (see
[Counting audio tokens](asr-throughput.md#counting-audio-tokens)). The count
depends on the model's encoder, so each model gets the context that its own
audio needs. A fixed "safe" size would have to cover the worst case of the
largest encoder (750 tokens per 30 s window) for every model.

Rules:

- A cell that pins `ctx_size` in its runtime flags keeps it, and the calibration
  does not run (its cache miss would be a model load paid for a number never
  used).
- When the helper is absent or fails, the context is sized from the upper bound
  `max(25 tokens/s, 750 per 30 s window)` and a warning says the peak carries KV
  for up to that many audio tokens. This is the only path on which the KV cache
  is larger than the clip.
- The warm-up server (whose reading is dropped) checks the prompt on every
  path, as `asr_throughput` checks its own after the reps, whether the context
  was sized from the count, from the upper bound or pinned. It counts the
  whole prompt with the clip, then sends the same turns with no clip, which
  must prefill fewer tokens; a cell with text also sends the same turns
  without the text, which must be exactly `parameter_text_tokens` shorter than
  the text-only request
  ([The text beside the clip](asr-throughput.md#the-text-beside-the-clip)).
  Every later server must then prefill the same prompt as the warm-up
  server, so the measured server's window holds the clip's request alone and
  its prompt is the one checked.
- Where the count sized the context, the server must also confirm it: the
  prompt with the clip less the prompt without it must be exactly the count
  plus the recipe's audio markers ([Counting audio
  tokens](asr-throughput.md#counting-audio-tokens)). This is the one check
  that needs a count, so it is the only one an uncalibrated or pinned cell
  skips.
- A reply without token counts, a count the server does not confirm (an
  undercount the 64-token margin would hide included), a text the server does
  not confirm or a measured server with another prompt fails the cell rather
  than publish a peak that is not the clip's own; an unconfirmed count also
  drops its row for the next run to count again.
- A cell that fails after the count sized its context, other than by a
  refusal (a stale undercount the server refuses with its context-overflow
  400, say), drops that row too, and says so, so the next cell counts again
  rather than fail on the same row.
- A pinned `ctx_size` smaller than the clip's prompt is refused by
  llama-server (`exceed_context_size_error`, HTTP 400, as measured on pipette's
  ASR build of llama.cpp with a prompt longer than the pinned context), and the
  cell fails. It never measures a shorter clip. Pin a context only to measure a
  larger one, as a comparison against the derived size.
- A cell is refused before anything loads for what refuses an `asr_throughput`
  cell ([Runtime details](asr-throughput.md#runtime-details)): a `raw` slot
  count, since llama-server would divide the context among its slots, and a
  reasoning or template knob in `raw`, an env or a llama.cpp config file.
- Each row's `client_extras.asr_context_sizing` is `exact` only where the
  counted audio tokens sized the context and the server confirmed them; only
  those rows are the clip's own peak. `upper_bound` rows carry KV for audio the
  clip lacks.
- Each row's `client_extras.asr_beyond_model_audio_limit` is set as on
  `asr_throughput`: `true` for Gemma 4 on the 119.9 s rung
  ([Clips longer than 30 s](asr-throughput.md#clips-longer-than-30-s)).
- The resolved `ctx_size` is reported in the row's `runtime_flags`, so the
  stored result records the context that the server ran with.

Compare only across the same model, **projector and its quantization**, runtime,
clip length, text length and weight load mode. A cell that maps the weights
(`load_mode = "mmap"`) does not count the mapped file pages in the macOS
footprint the sampler reads, so its peak reads far lower than the same cell
with the weights read into memory (the default here, `load_mode = "none"`).

## Platform status

The runner dispatches on `LlamaCppFlavor` through the same peak sampler as
`vl_max_memory`, so its coverage, and the flavors refused before any model
loads, are [`vl_max_memory`'s](vl-max-memory.md#platform-status).

The benchmark also runs through the Android CLI's CPU build
(`android-arm64-v8a`), on upstream llama.cpp and on pipette's ASR build alike.

The iOS app measures it in process (below). The Android app does not run this
cell ([client-parity.md](../client-parity.md), gap 16), and other engines refuse
it.

## iOS

The phone has no server to drive, so the app runs the workload in its own
process, through the same path as
[`asr_throughput` on iOS](asr-throughput.md#ios), and reads the
[iOS peak-memory](peak-memory-ios.md) counter around it, as
[`vl_max_memory`](vl-max-memory.md#ios) does. It runs no warm-up first, since a
warm-up in the app's own process would raise the measured peak:

1. Pick the recipe from the projector's metadata and write the clip to a temp
   file, both before the bracket, with `asr_throughput`'s refusals.
2. Settle the footprint to a floor, then start the `phys_footprint` sampler. An
   [private-API build](../pipette-ios/private-thermal-release-build.md) also resets the kernel's interval peak.
3. Load the weights, then the projector against them.
4. Count the clip's audio tokens with the projector, as `asr_throughput` on iOS
   counts them, and create the context from that count, with no floor; a cell
   that pins `ctx_size` keeps it and counts nothing, as the desktop skips its
   helper.
5. Send the clip's request: the recipe's turns with the text beside the clip,
   the audio encode, the prefill, and `parameter_decode_tokens` samples with a
   decode between each, ending at the last sample as the server's request does.
6. Free the projector and the model, stop the sampler, and report the peak as
   `max_host_bytes`. `max_gpu_bytes` is null: the memory is unified.

The row carries `client_extras.asr_context_sizing` and
`asr_beyond_model_audio_limit`, with the desktop's keys and values, and the
resolved `ctx_size` in its `runtime_flags`.

What differs from the desktop runner:

- **The checks run after the request, not on a warm-up server.** The desktop
  confirms the text and the count on the warm-up server and keeps them out of
  the measured one. The app has no warm-up, so it confirms them once the clip's
  request is done, as text-only tokenizations of the same renders
  ([asr-throughput.md](asr-throughput.md#ios)): no clip and no prefill. A text
  or count the tokenizations do not confirm fails the cell, as on the desktop,
  and behind a pinned context the count, which sized nothing, is not checked.
- **The count is inside the window.** The desktop counts the clip in a separate
  `llama-mtmd-cli` process, so its mel spectrogram is outside the measured
  server. The app counts it with the projector it measures, so that mel
  spectrogram is inside the bracket, as `vl_max_memory`'s image count is.
