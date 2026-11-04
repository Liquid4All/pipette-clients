# Vision-Language Image-Token Reference

This is the shared reference for **how image tokens are counted, why they vary
across models, and how to control them**. It backs the throughput and memory
methodologies so those stay runtime-agnostic.

Image tokens are a measured metric. Every vision-language (VL) run records
`image_tokens` and `prompt_tokens`, so the token count of a model at a given
resolution comes from the stored results. A new model needs no change to this
page. This page holds the method: how the count is made, why models differ, and
the flags that change it. Each numeric table below names the llama.cpp build it
was measured on, and is an illustration. The llama.cpp that pipette vendors
(`vendor/llama.cpp`) is `b11988`.

Image tokens are **not** a function of pixel dimensions alone; each model's
vision encoder has its own patch size, pooling, and tiling policy. `image_tokens`
is measured, model- and runtime-agnostic (works for any VLM the runtime can
load), and it means the same thing on every path and at every frame count: **the
image embeddings alone, without the image markers around them** (see
[`image_tokens` and the model's image markers](#image_tokens-and-the-models-image-markers)).
Each path reads it from its own engine:

- **Native (mtmd, Android and iOS): summed directly from mtmd's IMAGE-type chunks**
  (`mtmd_input_chunk_get_n_tokens` over `MTMD_INPUT_CHUNK_TYPE_IMAGE`). This is
  mtmd's own accounting. It counts the image content only: the image markers
  are separate text chunks.
- **Desktop (llama-server): the same mtmd chunk count**, from a calibration run
  of upstream's `llama-mtmd-cli` on the cell's frames; see
  [vl-calibration.md](vl-calibration.md). The server's HTTP API exposes only
  token *totals*, so when the helper is unavailable or fails the row reports
  **no** `image_tokens` rather than a marker-inclusive difference.
- **MLX (`mlx_vlm.server`): the model's own processor**, run on the same frames
  the server gets, with the image placeholder ids counted in the `input_ids` it
  builds. That expansion is what the server prefills: for the same prompt, the
  processor's `input_ids` matched the server's `prompt_tokens` on every model
  checked. No weights load. A failure reports no `image_tokens`, as on
  llama-server. A count the measured `prompt_tokens` cannot hold fails the cell
  instead: the helper runs the server's own processor, so the images did not
  reach the server's prompt.

`prompt_tokens` always carries the full prompt cost, markers included, so no cost
is lost; `image_tokens` is the image-embedding share of it.

## Per-model token formula (source of truth: `clip.cpp`)

Each model's count comes from `clip_n_output_tokens` in
`vendor/llama.cpp/tools/mtmd/clip.cpp` (`b11988`), driven by constants in the
mmproj GGUF metadata:

| Model | projector | patch | merge / scale | tiling | tokens (per tile) |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL | `qwen3vl_merger` | 16 | 2 | none (continuous) | `(W/32)·(H/32)` |
| InternVL3.5 | `internvl` | 14 | 2 | 448-px tiles + thumbnail (max 12) | `(448/14)²/2² = 256` |
| Gemma-4 | `gemma4v` | 16 | 3×3 pool | none (single) | about `(W/48)·(H/48)`, 70 to 1120 on `b11988` (40 to 280 on `b10216`; a flat 256 on `b8814`) |
| LFM2-VL | `lfm2` | 16 | 2 | single tile ≤~1024 px, then multi-tile | `⌈W/32⌉·⌈H/32⌉`, 64 to 256 per tile (measured: 64 at ≤256²) |

- **Continuous** (Qwen, LFM up to one tile, gemma-4 on `b10216` and later): the count
  scales with resolution between a minimum and a maximum pixel bound.
- **Tiled** (InternVL, LFM above about 1024 px): the image is split into a grid
  of tiles plus a downscaled thumbnail; count = `tiles × per-tile`.
- **Gemma-4 by build:** on `b8814` its default bounds put every resolution at
  256 tokens. On `b10216` it counts by resolution between 40 and 280 tokens
  (121 at 512², 256 at 1280²). On the vendored `b11988` the bounds are 70 and
  1120 tokens; `b11706` measured 729 at 1280². It never tiles.

## `image_tokens` and the model's image markers

Every VL model wraps its image embeddings in **image markers** (e.g.
`<image_start>` … `<image_end>`). These are real prefill tokens: they occupy the
KV cache and cost prefill compute, so they are counted in **`prompt_tokens`** on
every path. **`image_tokens` excludes them**, everywhere: it is the raw embedding
grid (`clip_n_output_tokens` column below). The count itself is specific to the
engine and the llama.cpp build: gemma-4-E2B at 512² counts 121 on llama.cpp
`b10216` and `b11706` (256 on `b8814`) and 256 on MLX, and multi-frame Qwen3.5
counts depend on the build (see
[Multiple images](#multiple-images-num_images-and-the-prompt)).

**Reading older rows.** Rows from a `client_version` dated before 2026-10-17, and
rows from every `2026.08.1` public release, count differently:
their multi-image llama-server rows and their MLX rows include the markers, a
few tokens high per image. Their single-image llama-server rows include the
markers only when the helper was unavailable, and the row does not record that.

**The marker scheme differs per model**, and the count per image can vary.
For this reason, a marker-free count cannot be derived from a marker-inclusive
difference. Each vision model wraps the image embeddings in its own image
markers:

| Model | image markers (example) | markers / image | marker-inclusive difference | `clip_n_output_tokens` (marker-free) |
| --- | --- | --- | --- | --- |
| Qwen3-VL | `<\|vision_start\|>` … `<\|vision_end\|>` | 2 | 258 @ 512² | 256 |
| LFM2-VL | model-specific image start/end markers | 2 | 258 @ 512² | 256 |
| Gemma-4 | `<start_of_image>` … `<end_of_image>` | 2 | 258 @ any res on `b8814` (floor 256) | 256 |
| InternVL3.5 | `<img>` … `</img>` | 2 | 1282 @ 512² (5 tiles) | 1280 |

Those are llama-server figures on `b8814`. `b11706` counts gemma-4 by
resolution (729 at 1280², where `b10216` still counted 256). Under
`mlx_vlm.server` (the pinned
`0.7.1+lfm2-vl-speculative` runtime) the same comparison at 512² measured one
marker for one LFM2-VL-450M image but five for three, and two per Qwen3.5-0.8B
image; the counted placeholders were 256 per frame for both, matching mtmd.

## Measured image tokens by resolution (default preprocessing)

**Marker-free, exact, measured on llama.cpp build `b8814`** with `llama-mtmd-cli`
(the calibration path, see [vl-calibration.md](vl-calibration.md)). Square input,
no text (`t0`), single frame, default preprocessing (no `--image-min/max-tokens`
override). Counts are **per vision encoder**, not per weight quant, verified
identical across a family's members (LFM 3B/1.6B/450M; Qwen3-VL-2B / Qwen3.5-2B /
Qwen3.5-4B; gemma-4 E2B/E4B) and across mmproj quant (F16 ≡ Q8):

| px | LFM2-VL | Qwen3-VL / Qwen3.5 | InternVL3.5 | gemma-4 |
| --- | --- | --- | --- | --- |
| 224 | 64 | 49 | 256 | 256 |
| 256 | 64 | 64 | 256 | 256 |
| 384 | 144 | 144 | 256 | 256 |
| 448 | 196 | 196 | 256 | 256 |
| 512 | 256 | 256 | **1280** | 256 |
| 768 | 256 | 576 | 1280 | 256 |
| 1024 | 256 | 1024 | **2560** | 256 |
| 1280 | **2560** | 1600 | 2560 | 256 |
| 2048 | 2560 | 4096 | 2560 | 256 |

These are **marker-free** (image content only), which is what `image_tokens`
reports. The count depends on the build. The gemma-4 column is `b8814` only: on
`b10216` and later, gemma-4 counts by resolution (see
[Per-model token formula](#per-model-token-formula-source-of-truth-clipcpp)).

Inflection points: **InternVL** tiles above 448 px (256 → 1280 → 2560 crops incl.
thumbnail); **LFM** is continuous `⌈W/32⌉²` capped at 256 (512–1024²), then tiles
to 2560 (≥1152²); **Qwen** scales continuously `(W/32)²` to its `max_pixels`
(→4096 @2048²) with no tiling; **gemma** is flat 256 at every resolution on this
build (see below).

> **gemma-4 on `b8814`.** With default preprocessing, every resolution measures
> 256 on `b8814` (checked down to 64²). It responds to `--image-min-tokens`
> (`64` gives 64), so the encoder is dynamic, and 256 is the default bound of
> that build. Three measurements agree: the server's marker-inclusive
> difference (258 = 256 + 2 markers), the `llama-mtmd-cli` marker-free count
> (256), and the calibration cache. The `vl_throughput` and `vl_max_memory`
> catalog cells set no `--image-min-tokens`; `vlm_eval` sets it for Gemma 4
> ([vlm-eval.md](vlm-eval.md#model-recipes)).

## Runtime flags that change the count

Two runtime knobs resize the image (`common/arg.cpp`). Each sets one resize
bound, `N × (patch·merge)²` pixels: `--image-min-tokens` the minimum and
`--image-max-tokens` the maximum (`set_limit_image_tokens` in
`tools/mtmd/clip-model.h`):

| Flag | env | effect |
| --- | --- | --- |
| `--image-min-tokens N` | `LLAMA_ARG_IMAGE_MIN_TOKENS` | floor: upscales small images to ≥ N tokens |
| `--image-max-tokens N` | `LLAMA_ARG_IMAGE_MAX_TOKENS` | cap: downscales large images to ≤ N tokens (per tile) |

The batch sizes matter too, on one kind of model: llama.cpp caps the image
tokens of a projector the text model attends to non-causally at the effective
micro-batch, the smaller of `--batch-size` and `--ubatch-size`.

**They only affect dynamic-resolution models.** Measured with
`--image-min-tokens 256 --image-max-tokens 256` (pin to ~256), as the
marker-inclusive difference the server reported, so each figure includes the
model's image markers (about 2); `image_tokens` excludes them:

| Model | pinned @256 px | pinned @512 px | pinnable? |
| --- | --- | --- | --- |
| Qwen3-VL-2B | 258 | 258 | ✅ any resolution |
| LFM2-VL-1.6B | 259 | 259 | ✅ any resolution |
| Gemma-4-E2B | 258 | 258 | ✅ any resolution |
| InternVL3.5-2B | 258 | **1282** | ❌ caps per-tile, still tiles |

There is **no runtime flag to disable tiling or the thumbnail**: the tile count
(`preproc_min_tiles`/`preproc_max_tiles`) and thumbnail are baked into the mmproj
GGUF metadata and have no CLI override (unlike vLLM, which forwards HF
`mm_processor_kwargs` such as `max_dynamic_patch`/`use_thumbnail` at runtime).
To keep a tiling model at one tile, cap its **input resolution** below its tile
size (InternVL: ≤ 448 px → one tile = 256).

## Comparing models

There is no single resolution at which all models prefill the same tokens (at
512² InternVL tiles to 1280 while Qwen and LFM produce 256), so every row
reports `image_tokens` and `prompt_tokens` beside its timings. Two views follow:

- **Native view:** the same image for every model at default preprocessing,
  with the token difference shown as part of the result.
- **Iso-token view:** one token target per suite, reached by each model's lever
  above (the min/max-token flags for dynamic models, an input at or below the
  tile size for tilers), with the achieved `image_tokens` reported rather than
  assumed. A model that cannot reach the target is footnoted or left out.
  It needs no new benchmark definitions: the same `vl_throughput` and
  `vl_max_memory` definitions serve both views, and the cell's recorded runtime
  flags and resolution tell an iso-token row from a native one.

How to read the rows across models:
[vl-throughput.md](vl-throughput.md#result-validity-and-comparability).

## Multiple images (`num_images`) and the prompt

`num_images` packs **N images into a single prompt** to emulate multi-frame /
video inputs. Each image is inserted as its own image marker, so the
prompt is `[optional text] + N × [image]` and the image-token contribution is
**≈ N × (single-image tokens)** for independently-encoded frames: e.g. an
80-frame 256² LFM prompt is 80 × 64. Some llama.cpp builds break this
linearity: before upstream #27348 they merged consecutive same-size images in
pairs for Qwen-VL models (measured: Qwen3.5 at 3 × 512² takes 512 image tokens
on `b10216`, 768 on builds since #27348 such as `b11349` and `b11706`, which
merge only video frames).
That is why no path multiplies a one-frame count: each counts the cell's frames
together (mtmd's chunks natively, a helper run with all the frames on
llama-server, the processor on MLX). `mlx_vlm` does not merge frames
(3 × 512² counts 768), so the same model can report different `image_tokens` on
an older llama.cpp build and on MLX; that is the engines' difference, not the
count's.

`num_images = 1` is the ordinary single-image case. The context window is sized
from the per-image count times `num_images`; see
[vl-calibration.md](vl-calibration.md#deriving-n_ctx).

## On `t0` (zero text tokens)

Most sweep cells use `t0` (no text) deliberately: it isolates the **image's**
prefill and encode cost from text tokenization, so the resolution/frame axes
map cleanly to image tokens. Text-bearing cells (e.g. `t1024`) exist to measure
the image + text-prefill combination; real multimodal prompts always carry some
text, so both are represented, but `t0` is the controlled point for the
image-token characterization above.
