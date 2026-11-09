# VL Token Calibration

How pipette gets **exact image-token counts** for vision-language (VL) models on
the `llama-server` path (desktop hosts and the CLI on Android and Linux edge
devices). The count reports the `image_tokens` metric and sizes `n_ctx` for the
`vl_throughput`, `vl_max_memory` and [`vlm_eval`](vlm-eval.md) benchmarks. The count is exact and tiling-aware, computed once and cached. The
synthetic-image cells take the resolution from their parameters; `vlm_eval`
calibrates up to four worst-case shapes scanned from its dataset (see
[its context sizing](vlm-eval.md#context-sizing-no-overflow-single-calibration)).

## Why calibrate
The image-token count is model-specific (patch size, pooling/merge, and, for
tilers, a dynamic tile grid) and moves with the llama.cpp build. The per-model counts are in
[vl-image-tokens.md](vl-image-tokens.md). No single divisor fits every model, so
pipette **measures** the count with `libmtmd`, llama.cpp's multimodal library,
which is exact and tiling-aware.

## How: upstream `llama-mtmd-cli`
The count comes from **upstream's own `llama-mtmd-cli`**. Upstream ships it in
the llama.cpp release archive next to `llama-server`, so it matches the server's
version. A runtime without it reports no `image_tokens`
(see [Fallback & safety](#fallback--safety)).

On a calibration miss the runner runs, in the resolved runtime dir:
```
llama-mtmd-cli --model <model> --mmproj <mmproj> --prompt hi --n-predict 1 \
  [placement flags] [--image-min-tokens N --image-max-tokens M] \
  [--batch-size N --ubatch-size N] --ctx-size C --jinja --verbose \
  --image <gray W×H png> [--image <the same png> ...]
```
(each bracketed flag appears when the cell sets it, in the server's flag order;
the placement flags are `number_gpu_layers` and the `raw` offload flags listed
below; the cell's env forwards reach the helper too; a multi-image cell repeats
`--image` once per frame)
and sizes the context for the helper's own prompt, `hi` and the frames, rather
than leaving it at llama.cpp's default, the model's full training context: about
5 GB of peak against 2 GB on a phone. A one-frame run gets the frame's `(w/14)²`
estimate, at most 16384, or the cell's `--image-min-tokens` with room for
alignment if higher, plus the margins, at least 4096. A multi-image run gets the
one-frame count times the frames, plus the margins. A `vlm_eval` candidate
gets the same estimate without the 16384 cap, capped instead by the cell's
`--image-max-tokens` (retried once without it if the helper fails), plus 1024,
between 4096 and the vlm_eval ceiling of 131072: a dataset's own images tile
past 16384 (SmolVLM2 counts a 35-megapixel OCRBench page at about 16k tokens).
Each run is bounded at 600 s and does not count against the cell's
`attempt_timeout_seconds`. The runner sums the `n_tokens_batch = N` values
from the `decoding image batch …` lines the mtmd layer logs on stderr, and
trusts them only after a clean exit. That is the **marker-free image content** count (the
raw embedding grid; the image markers are logged as separate text chunks and
are not included). Tiled models emit one batch line per tile-batch; the sum is the
full tiled count. The synthetic image is a solid gray PNG; token count is
set by the image geometry alone, so the pixel values do not matter.

- **`--jinja` is required.** Some models (e.g. gemma-4) ship a chat template
  `llama-mtmd-cli`'s default path rejects (`this custom template is not
  supported, try using --jinja`), aborting *before* the image is processed;
  `--jinja` uses the model's embedded template and fixes it. It only affects the
  *text* template (image preprocessing is untouched) so it is a **no-op for the
  count** on models that don't need it (verified: LFM/Qwen/InternVL identical
  with and without).
- **The helper gets the server's preprocessing overrides.** When the cell's
  runtime flags carry `--image-min-tokens` / `--image-max-tokens`, the helper
  runs **under the same overrides the server gets**, and the cache keys by them: measuring the
  default count while the server prefills an overridden one would report wrong
  `image_tokens` and mis-size `n_ctx`. Without overrides (the common case) the
  count reflects the mmproj's shipped default min/max-pixel bounds. The cell's
  `--batch-size` and `--ubatch-size` also reach the helper and key the cache:
  the smaller of the two (the effective micro-batch) caps the image tokens of a
  projector the text model attends to non-causally.
- **Placed like the server.** The helper loads the same model before the server
  starts, so it takes the cell's placement: `number_gpu_layers`, and from `raw`
  the device flags (`--device`, `--mmproj-device`, `--[no-]mmproj-offload`) and
  the offload flags a pinned layer count relies on once llama.cpp's fit is off
  (`--override-tensor`, `--n-cpu-moe`, `--cpu-moe`, `--tensor-split`,
  `--main-gpu`, `--split-mode`, `--no-kv-offload`, `--no-op-offload`). A cell
  that sets none of them leaves the helper on llama.cpp's default placement, as
  the server is. Placement is not in the cache key, because it changes the
  encode's speed, not the count. It does decide whether the helper survives: on
  the Snapdragon build the helper was killed under the default placement and
  ran under `--device HTP0`, an NPU cell's placement; a cell placed otherwise
  there may still report no `image_tokens`.

Implementation: `crates/pipette-llamacpp/src/mtmd.rs`.

The iPhone app counts the same way without the helper. Its projector loads against
the weights before any context exists, so it tokenizes one gray frame at the cell's
size, sums mtmd's IMAGE chunks (the marker-free count above) and sizes `n_ctx` with
the same formula. Nothing is cached: the count costs one image preprocess per cell
(`LlamaMtmd.perImageTokens`, `LlamaExecute.withCalibratedVision`).

## The count is platform-independent
mtmd's tiling/patch/merge math is deterministic integer geometry on
`(image dims, mmproj metadata)`; the GPU backend only affects encode *speed*,
not the token *count*. So a value calibrated on any one box is valid for the same
build everywhere (macOS / Linux / Windows / Android), and a cache produced on one
host may be seeded onto another running the same build.

## The cache (`<runtime_dir>/vl_calibration.json`)
The result is **memoized**. It lives next to the runtime binaries, so each
llama.cpp build has its own file:
```json
{
  "entries": {
    "<mmproj_fingerprint>|512x512": 256,
    "<mmproj_fingerprint>|512x512|frames:3": 768
  }
}
```
- **Value** = marker-free image content (tiling-aware) for one image, or for a
  multi-image cell's frames together. A multi-image cell's `image_tokens` is its
  own helper run, with one `--image` per frame and a `--ctx-size` sized for all
  of them, **not** the one-frame value multiplied: some llama.cpp builds merge
  consecutive same-size frames (see
  [vl-image-tokens.md](vl-image-tokens.md#multiple-images-num_images-and-the-prompt)).
  Context sizing still uses the one-frame `value × num_images` as a safe upper
  bound, so the multi-frame run is one more helper run per new frame count,
  before the server starts. It is skipped when the one-frame run found no count:
  the same helper would not find one for more frames, and a wedged one would
  hold the cell for a second deadline.
- **Key = `(mmproj_fingerprint, W×H[, image-token overrides and batch sizes][, env forwards][, frames][, count envs])`**;
  the override suffix appears only when the cell sets `--image-min/max-tokens`,
  `--batch-size` or `--ubatch-size`, the `env:` suffix only when the cell
  forwards envs, the `frames:N` suffix only for more than one frame, and the
  `count-env:` suffix only when the helper's environment (pipette's own, the
  client configuration's or the cell's) sets `LLAMA_ARG_IMAGE_MIN_TOKENS`,
  `LLAMA_ARG_IMAGE_MAX_TOKENS`, `LLAMA_ARG_BATCH` or `LLAMA_ARG_UBATCH`. The
  file carries no build field; its location scopes it to a build (it sits in
  the build's runtime dir, so a different build reads a different file;
  tiling can change across builds). A store-managed runtime installs under
  `runtimes/<key>/`, keyed by its source (a release and its version, or an
  archive's URL) and flavor, so two builds never share a file; a local runtime
  dir (a `RelativeDir` or `AbsoluteDir` source) whose binaries are swapped in
  place keeps the old binaries' counts.
  Fingerprint is a **content id, not the filename**:
  `<size_bytes>:<sha256(first 1 MiB), first 8 bytes as hex>`: five different
  models ship an `mmproj-F16.gguf` with different vision encoders
  (819/668/985 MB …), so the name collides but the fingerprint does not.
- **First run computes, later runs reuse.** The expensive step (a model+mmproj
  load) happens once per key and build; every later benchmark on that host and
  build reads the cached integer.

## Deriving `n_ctx`
```
n_ctx = value[fp, W×H] × num_images + text_tokens + decode_tokens
        + 64 + 8 × num_images
```
The margin covers the chat-template / media-marker scaffolding the server
wraps the prompt in (the image count is exact and has no built-in slack): a
64-token base plus 8 per image, because the template emits markers/separators
**per image** (a flat margin overflowed at 80 frames; the code comment
estimates that the LFM template adds about 2.5 tokens per image, about 200 past
a flat 64). This is
far tighter than the `(w/14)²` estimate, which is 5× or more above the count
for Qwen and LFM, and it stays exact for tilers. A cell that pins `ctx_size` keeps it;
`vl_max_memory` then skips the calibration, since it reports no `image_tokens`.

llama.cpp preallocates the KV cache to the full `n_ctx`, so a tight `n_ctx`
lowers the peak that `vl_max_memory` measures; see
[vl-max-memory.md](vl-max-memory.md#why-the-context-size-moves-the-peak).

## Fallback & safety
Calibration is **best-effort**. After a failure the row reports no count, and
it never reports an estimate as a measured count:
- **`llama-mtmd-cli` absent / non-zero exit / timeout / unparseable log** → the
  row reports no `image_tokens` (with a warning) rather than the server's
  marker-*inclusive* difference, which would not compare with every other row
  (see [vl-image-tokens.md](vl-image-tokens.md)); a count larger than the measured
  `prompt_tokens` (a stale cache row) is dropped the same way, as is a frame the
  helper's bounded context cannot hold.
- `n_ctx` falls back to the `(w/14)²` patch-grid estimate floored at 256. On
  small images the raw grid falls below InternVL's one 256-token tile and
  gemma-4's flat 256 on `b8814`, so the floor keeps the estimate above them.
  The estimate is raised to the cell's
  `--image-min-tokens` plus alignment room when that is higher. `vl_throughput`
  then floors the whole context at 8192; `vl_max_memory` has no floor.
