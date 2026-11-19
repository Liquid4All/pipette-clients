# Vision-Language Eval Methodology

## Measured Quantity

Vision-language eval benchmarks measure model-output correctness on a fixed
dataset of **image + text** prompts. As with text [evals](evals.md), the client
records the model's raw completion for every sample and the score is computed
later, from the hidden answers, outside the client. The client-side result is
the eval completions payload (`completions[].{id, completion, failed,
failed_reason, stop_reason, stop_detail, completion_tokens}`), so the same
score pipeline and warehouse columns apply.

In pipette this is the `vlm_eval` benchmark family: one catalog entry per
benchmark and model family, `vlm_eval_<benchmark>_<family>`
(`vlm_eval_mmstar_qwen3_5_small`, `vlm_eval_pope_lfm2_5_vl_3b`, ...), each
holding that family's evaluation recipe ([Model recipes](#model-recipes)). The
entry's parameters, its corpus pin, the recipe contract, the completeness rule
and the headline metrics are mgmt's: pipette-mgmt
[docs/benchmarks.md §2.9](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#29-vlm-eval-accuracy).
This page is how the client runs one, on `llama-server`, and why.

## Datasets

Seven benchmarks, each run in full (no subset), at dataset version
`2026.10.1`:

| Benchmark | Samples |
| --- | --- |
| MME | 2,374 |
| MMStar | 1,500 |
| OCRBench | 1,000 |
| MMMU (val) | 900 |
| RealWorldQA | 765 |
| MMBench (EN dev v1.1, every circular rotation) | 4,876 |
| POPE | 5,127 |

The evals server (pipette-scores) builds each from the TSV VLMEvalKit
publishes, checked by VLMEvalKit's MD5, with each prompt as VLMEvalKit's dataset
code at commit
[`8eca8e5`](https://github.com/open-compass/VLMEvalKit/tree/8eca8e5a1e27ba10c4e3760d09f912f11dca2389)
builds it. Several datasets allow research use but not
redistribution, so the entries live in an organization's private catalog and a
corpus, images included, goes only to that organization's workers.

A family evaluated with its own prompts (its VLMEvalKit wrapper's, or its model
card's) gets them through a **prompt profile**: the evals server rewrites only
the prompt text, the way that wrapper or card does, with every image, the
dataset and the scoring unchanged.

## Relationship to the text eval and VL throughput benchmarks

`vlm_eval` is the eval harness applied to a vision model:

- from the text [`eval`](evals.md) runner: the resumable per-sample checkpoint,
  bounded retries of failed samples, crash-vs-transport recovery, and
  `finish_reason` → `stop_reason` classification;
- from [`vl_throughput`](vl-throughput.md): the vision plumbing. The model and
  its projector both come from the vision model descriptor (the cell sets no
  projector flag; the client passes `--mmproj` from the descriptor), and the
  context is sized from exact image-token counts
  ([VL Token Calibration](vl-calibration.md)).

Two things differ from text eval. The runner talks to `/v1/chat/completions`
directly, not `/apply-template` then `/completion`, so the server renders the
multimodal chat template and runs the vision tower; the samples' `messages`,
which already carry the `image_url` blocks, are forwarded verbatim. And there
is no per-benchmark temperature policy: a cell decodes greedily unless its
entry pins sampling ([Sampling](#sampling)).

## Prompt and generation

Prompts reach the device exactly as the scorer grades them, so parity with the
reference harness is a property of the served dataset, not of the runner. The
client downloads the entry's `vlm-eval-corpus/1` tar from
`GET /datasets/{benchmark_id}` (each image stored once rather than in every
question that shows it: POPE is 5,127 questions over 500 images), rebuilds each
sample's `data:` URI, and checks the tar against the entry's `digest` and the
rebuilt samples against `upstream.identity`. Either mismatch fails the run:
unlike a throughput benchmark, answers to other prompts would be graded
against questions the model was never asked. The tar's digest is reported as
`observation_dataset_digest`. A local definition (the seeded `vlm_eval_smoke`)
carries its samples inline instead.

`parameter_max_tokens` caps generation. With `parameter_mcq_choices` the
request is grammar-constrained to those alternatives and capped at one token,
as on a text eval; otherwise generation is free-form and streamed, and a
repetition that never stops runs to the cap (`stop_reason = truncated`).

The two arms are bounded differently. An MCQ request is one JSON response
bounded end to end by the cell's `http_timeout_seconds` (default 120 s), and
that budget must cover the **full image prefill** before the one token comes
back: phone CPUs measured 65 s TTFT at 512², so an edge cell wants a generous
value. A stream can legitimately outlive any bounded timeout over a large image
prefill, so it is bounded by the silence between events instead: 1,800 s, which
`http_timeout_seconds` can raise but not lower. Its TCP connect is bounded at
60 s, which only catches a server that died between the readiness gate and the
call.

### Sampling

An entry may pin `parameter_temperature`, `parameter_top_p`, `parameter_top_k`,
`parameter_min_p`, `parameter_presence_penalty`, `parameter_repeat_penalty` and
`parameter_seed`; every sampler it leaves unset is sent at its neutral value
(top_k 0, top_p 1.0, min_p 0, no penalty, and DRY, XTC and mirostat off), so a
GGUF's embedded sampling defaults never decide a decode. An entry that pins
nothing decodes greedily. A pinned seed is a base, from which the entry's
`parameter_seed_rule` derives each sample's (`VlmEvalBenchmark::sample_seed`).
Under `per_sample_sha256` each sample is sent the first four bytes, read
big-endian, of the SHA-256 of `<seed>|<sample id>` in UTF-8, so every sample
draws with a seed of its own and a re-run draws each one again exactly. An
entry that names no rule sends the base seed offset per repeat, as the text
eval does, so every sample of an attempt shares one. That is not neutral on
vLLM, which turns a request's seed into a noise vector over the vocabulary: a
shared seed favors the same token ids across the whole dataset. MiniCPM-V
4.5's POPE F1 on vLLM rose from 55.8 under the shared seed to 87.0 under the
per-sample rule, against 87.6 to 87.8 on llama.cpp under either. The rule is
opt-in per entry: a sampled entry that names no rule sends one seed to all the
samples of an attempt, so every seeded entry should name `per_sample_sha256`.

The pinned numbers are tuned under Hugging Face and vLLM, so the request
applies them the way those do rather than llama.cpp's default way:

- **Order.** Penalties, then temperature, then the top-k, top-p and min-p cuts
  (`samplers` in the request). llama.cpp's default chain divides by the
  temperature last, which keeps a different set of tokens for the same `top_p`.
- **Penalty window.** The presence and repeat penalties count every token of
  the answer (`repeat_last_n` = `parameter_max_tokens`). llama.cpp otherwise
  counts only the last 64, so a reasoning model that repeats a passage longer
  than that is never penalized and loops until the token cap. One difference
  remains: llama-server also counts the prompt's text in the window, as Hugging
  Face's repetition penalty does; vLLM's presence penalty counts only the
  answer.

Measured on an AMD MI325X host (Qwen3.5-2B thinking, MMBench, its card's presence
penalty 1.5): answers cut at the 32K cap fell from 18 to 5 of the same 1,127
questions with the whole-answer window, against 61 with no penalty.

## Reasoning models

Whether a reasoning model thinks is the cell's model flags' `enable_thinking`,
as on a text eval (`model_flags = [{ benchmark_type = "vlm_eval", model_type =
"gguf_vision", enable_thinking = true }]`), applied through the model's own chat
template. The run reports it as `model_flags` (`enable_thinking=true`), so
every row of a reasoning model says which mode it scored; a model with no
reasoning mode declares none.

The server starts with no reasoning flag, and the client renders two probe
conversations through `/apply-template`, a user's "Hi" alone and behind a
system prompt, each with `enable_thinking` on, off, then on again. The
template has a thinking switch when an `off` render differs from its `on`
render, the test llama.cpp itself uses. The two `on` renders must agree: a
template that prints something that changes by itself (gpt-oss prints the
date) is probed once more, and a cell whose renders still disagree fails as
inconclusive. A template that reads `enable_thinking` only beside tools, an
assistant turn or an image passes as one without a switch.

| `enable_thinking` | Template with a thinking switch | Template without one |
| --- | --- | --- |
| `true` | `--reasoning on` | the cell fails, unless the prompt profile thinks (`internvl_r1`): then no reasoning flag |
| `false` | `--reasoning off --reasoning-budget 0` | the cell fails: the model has no mode to turn off |
| unset | the cell fails: it must say which mode it scores | no reasoning flag |

- A model whose template has no switch gets no reasoning control at all and
  runs exactly as its template renders.
- A model whose template has a switch is switched explicitly, on a server
  restarted with the mode's flags: one extra model load. `--reasoning-budget 0`
  beside `false` is the sampler-level hard stop. Every restart after a crash
  carries the same flags. The mode must be stated because llama-server's
  default (`--reasoning auto`) enables thinking per model (Qwen3.5 thinks by
  default, and Gemma 4's GGUF template is probe-enabled though the upstream HF
  template defaults it off), so an unstated mode would let the server choose.
- InternVL3.5 thinks under its R1 system prompt, not a template switch: the
  cell runs the `internvl_r1` prompt profile (`parameter_prompt_profile`) and
  declares `enable_thinking = true` so its rows say it thought. The profile
  without that declaration fails before any server starts.

The request's kwarg overrides the server's default, but only the Jinja
renderer reads it, so nothing may turn Jinja off or set thinking where the
recorded command would not show it. On top of the raw flags mgmt reserves, the
client refuses, before it calibrates anything (so a refusal costs no model
load):

- in `raw`: `--no-jinja`, the reasoning flags (`--reasoning`, `-rea`,
  `--reasoning-budget`, `--reasoning-format`, `--chat-template-kwargs`, ...),
  and every flag that takes the model or projector from anywhere but the
  `--model` and `--mmproj` the client launches (`-hf`, `--hf-file`,
  `--model-url`, `-mm`, `--mmproj-url`, `--models-dir`, `--models-preset`, and
  the built-in presets such as `--vision-gemma-4b-default`, which set
  `--hf-repo` themselves), in any spelling llama.cpp parses. A Hugging Face repo
  with a `preset.ini` at its root turns llama-server into a router running that
  file's options. The projector's offload flags stay allowed: they change
  where it runs, not which one serves;
- any `LLAMA_ARG_*` variable, in any case, in the cell's or the client's
  `envs`; the ones pipette itself inherited are removed from what llama-server
  and the calibration see;
- a key setting a reserved option in the config files llama.cpp applies to
  every launch, which nothing turns off: `/etc/llama.cpp/config.ini` and
  `${XDG_CONFIG_HOME:-~/.config}/llama.cpp/config.ini`, or
  `%PROGRAMDATA%\llama.cpp\config.ini` and `%APPDATA%\llama.cpp\config.ini` on
  Windows. Keys count in any section, deliberately broader than llama.cpp,
  which applies only the unheaded, `[default]` and `[*]` sections. The error
  names the file and the key.

The recipe policy: **every model with reasoning runs with it on**, in
the sampling and prompt its vendor recommends, and a model without reasoning
runs as it is. Thinking moves scores (as a spot check, the same Qwen checkpoint
answered 17×23 as 391, correct, with thinking on and 491 with it off), so
`model_flags` is part of row identity.

## Model recipes

Each family runs a recipe taken from its vendor: the prompt, decoding and
thinking mode its model card recommends for the published numbers, or that the
VLMEvalKit wrapper the vendor contributed uses. A third-party VLMEvalKit
configuration is corroboration, never the source. A setting no public source
gives is ours, and every departure from the vendor's own run is listed below
the table, so a score here is comparable to the vendor's only as far as that
list allows.

| Family (Hugging Face repos) | Prompt | Decoding | Output budget | Thinking | Sources |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5 0.8B / 2B (`unsloth/Qwen3.5-0.8B-GGUF`, `unsloth/Qwen3.5-2B-GGUF`) | `qwen3_5`: the dataset's, plus the card's `"answer": "C"` line on multiple choice | T 0.6, top_p 0.95, top_k 20, presence 1.5, seed 0 | 32,768 (card) | on | [0.8B card](https://huggingface.co/Qwen/Qwen3.5-0.8B), [2B card](https://huggingface.co/Qwen/Qwen3.5-2B): the vision-language results footnote and Best Practices |
| Qwen3.5 4B (`unsloth/Qwen3.5-4B-GGUF`) | `qwen3_5` | T 0.6, top_p 0.95, top_k 20, presence 1.5, seed 0 | 32,768 (card) | on | [2B card](https://huggingface.co/Qwen/Qwen3.5-2B): the settings of the Qwen3.5 small models' published vision-language numbers in thinking mode (the [4B card](https://huggingface.co/Qwen/Qwen3.5-4B) gives no vision-language setting of its own) |
| Qwen3.5 9B / 35B-A3B, Qwen3.6 35B-A3B (`unsloth/Qwen3.5-9B-GGUF`, `bartowski/Qwen_Qwen3.5-35B-A3B-GGUF`, `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF`) | `qwen3_5` | T 0.6, top_p 0.95, top_k 20, presence 1.5, seed 0 | 32,768 (cards) | on | [2B card](https://huggingface.co/Qwen/Qwen3.5-2B): the small models' vision-language setting, as for the 4B (the [9B](https://huggingface.co/Qwen/Qwen3.5-9B), [35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) and [Qwen3.6 35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) cards give no vision-language setting of their own) |
| Gemma 4 E2B / E4B / 12B / 26B-A4B (`ggml-org/gemma-4-E{2,4}B-it-GGUF`, `unsloth/gemma-4-E{2,4}B-it-GGUF`, `unsloth/gemma-4-12b-it-GGUF`, `bartowski/google_gemma-4-26B-A4B-it-GGUF`) | the dataset's | T 1.0, top_p 0.95, top_k 64, seed 0; every image at 1,120 visual tokens | 32,768 (ours) | on | [E2B](https://huggingface.co/google/gemma-4-E2B-it), [E4B](https://huggingface.co/google/gemma-4-E4B-it), [12B](https://huggingface.co/google/gemma-4-12B-it) and [26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) cards: Best Practices; Variable Image Resolution (the highest budget, which the cards recommend for fine detail and text) |
| InternVL3.5 1B / 2B / 4B / 8B / 30B-A3B (`bartowski/OpenGVLab_InternVL3_5-{1B,2B,4B,8B,30B-A3B}-GGUF`) | `internvl_r1`: the R1 system prompt, then InternVL's own prompts | T 0.6, top_p 0.95, top_k 50, seed 0 | 32,768 (ours) | on, through the R1 system prompt | [2B card](https://huggingface.co/OpenGVLab/InternVL3_5-2B); VLMEvalKit [config.py L1336-L1363](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/config.py#L1336-L1363) (the `-Thinking` entries), [internvl_chat.py L126-L127, L416-L417](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/vlm/internvl/internvl_chat.py#L416-L417) (OpenGVLab's) |
| MiniCPM-V 4.5 (`openbmb/MiniCPM-V-4_5-gguf`) | `minicpm_v_cot` (OpenBMB's step-by-step prompts to an `Answer:` line) on MMMU, MMBench, MMStar and OCRBench; `minicpm_v` (its direct prompts) on RealWorldQA, MME and POPE | T 0.7, seed 0 (top_p 1.0, top_k 0, repetition 1.0: neutral) | 32,768 (ours; OpenBMB's 8,192 cuts long reasoning) | on | VLMEvalKit [minicpm_v.py L1063-L1085](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/vlm/minicpm_v.py#L1063-L1085), [L1207-L1218](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/vlm/minicpm_v.py#L1207-L1218) (OpenBMB's) |
| MiniCPM-o 4.5, vision only (`openbmb/MiniCPM-o-4_5-gguf`, projector `vision/MiniCPM-o-4_5-vision-F16.gguf`) | as MiniCPM-V 4.5: `minicpm_v_cot` on MMMU, MMBench, MMStar and OCRBench; `minicpm_v` on RealWorldQA, MME and POPE | T 0.7, seed 0 (top_p 1.0, top_k 0, repetition 1.0: neutral) | 32,768 (ours; OpenBMB's 8,192 cuts long reasoning) | on | [card](https://huggingface.co/openbmb/MiniCPM-o-4_5); VLMEvalKit [minicpm_v.py L1320-L1349](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/vlm/minicpm_v.py#L1320-L1349), [L1366-L1384](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/vlm/minicpm_v.py#L1366-L1384) (OpenBMB's) |
| MiniCPM-V 4.6-Thinking (`openbmb/MiniCPM-V-4.6-Thinking-gguf`) | `minicpm_v_cot` on every benchmark | T 0.7, seed 0 (top_p 1.0, top_k 0, repetition 1.0: neutral) | 32,768 (ours; OpenBMB's 8,192 cuts long reasoning) | on | VLMEvalKit [minicpm_v.py L1566-L1576](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/vlm/minicpm_v.py#L1566-L1576), [L1611-L1617](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/vlm/minicpm_v.py#L1611-L1617) (OpenBMB's) |
| MiniCPM-V 4.7 1B / 35B-A3B (`openbmb/MiniCPM-V-4.7-1B-gguf`, which a plan pins at revision `75785938`; `AtomicChat/MiniCPM-V-4.7-1B-GGUF`, the same weights with a real F16 16x projector, where OpenBMB's `f16` projector files hold BF16; `openbmb/MiniCPM-V-4.7-35B-A3B-gguf`, its `thinking/` files. The recipe names no projector: the default 16x or the 4x) | `minicpm_v_cot` on every benchmark, as MiniCPM-V 4.6-Thinking | T 0.7, seed 0 (top_p 1.0, top_k 0, repetition 1.0: neutral) | 32,768 (ours) | on | [1B card](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B): the 16x and 4x projectors (the 4x for finer detail, four times the visual tokens); [35B-A3B card](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B); [gguf at 75785938](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf/tree/75785938dd7ed3efaf2dde273407ce8805b43498); [AtomicChat gguf at b0efd525](https://huggingface.co/AtomicChat/MiniCPM-V-4.7-1B-GGUF/tree/b0efd52562b5115d80f81eba7c2b250b5f63bcf1); VLMEvalKit [minicpm_v.py L1566-L1576](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/vlm/minicpm_v.py#L1566-L1576), [L1611-L1617](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/vlm/minicpm_v.py#L1611-L1617) (its 4.6-Thinking sibling's) |
| LFM2.5-VL 450M / 1.6B (`LiquidAI/LFM2.5-VL-450M-GGUF`, `LiquidAI/LFM2.5-VL-1.6B-GGUF`) | the dataset's | greedy | 8,192 | none | VLMEvalKit [config.py L2434](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/config.py#L2434) (the 1.6B's; there is no 450M entry), [liquid.py L125](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/vlm/liquid.py#L125) (Liquid AI's) |
| LFM2.5-VL-3B (`LiquidAI/LFM2.5-VL-3B-GGUF`) | `lfm2_5_vl_3b`: Liquid AI's VLMEvalKit prompts | greedy | 8,192 | none | VLMEvalKit [config.py L2437](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/config.py#L2437), [liquid.py L125](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/vlm/liquid.py#L125) (Liquid AI's) |
| LFM3-VL-2B (`LiquidAI/LFM3-VL-2B-GGUF`) | `lfm2_5_vl_3b`, as LFM2.5-VL-3B | T 0.1, top_k 50, repetition 1.1, seed 0 | 16,384 (card) | on | [card](https://huggingface.co/LiquidAI/LFM3-VL-2B): Sampling parameters, and the 16,384 of its Transformers example; VLMEvalKit [liquid.py L125](https://github.com/open-compass/VLMEvalKit/blob/8eca8e5a1e27ba10c4e3760d09f912f11dca2389/vlmeval/vlm/liquid.py#L125) (LFM2.5-VL-3B's wrapper, Liquid AI's) |
| Ministral 3 3B Reasoning (`unsloth/Ministral-3-3B-Reasoning-2512-GGUF`) | the dataset's, with no system message, so the template's reasoning system prompt stays | T 0.7, top_p 0.95, seed 0 | 32,768 (ours) | unset; the model always reasons | [card](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512): System Prompt, Sampling Parameters, and the vision example's `top_p` |
| MiMo-VL 7B RL 2508 (`bartowski/XiaomiMiMo_MiMo-VL-7B-RL-2508-GGUF`) | the dataset's | T 0.3, top_p 0.95, seed 0 | 16,384 (Xiaomi's) | unset; the model reasons by default | [card](https://huggingface.co/XiaomiMiMo/MiMo-VL-7B-RL-2508): Deployment Parameters, Thinking Control; Xiaomi's evaluation harness, [mivllm.py L62-L63](https://github.com/XiaomiMiMo/lmms-eval/blob/1ba1684caaf8c9f62433ec60b46120041dbad463/lmms_eval/models/mivllm.py#L62-L63) |
| Step3-VL 10B (`JamePeng2023/Step3-VL-10B-GGUF`, a community conversion) | the dataset's | T 1.0, seed 0 (top_p 1.0, top_k 0: neutral) | 32,768 (ours) | unset; the model always reasons | [card](https://huggingface.co/stepfun-ai/Step3-VL-10B) and its `generation_config.json`; [report](https://arxiv.org/abs/2601.09668): Inference Settings |

Where we depart from the vendor's own run, and why:

- **Output budgets.** No public source gives an evaluation budget for Gemma 4
  (any size), MiniCPM-V 4.7 or Ministral 3 Reasoning: 32,768 is ours for each.
  VLMEvalKit gives InternVL3.5 thinking 65,536 tokens, more than the models'
  40,960-token context: 32,768 here. OpenBMB's VLMEvalKit configs stop
  MiniCPM-V 4.5, MiniCPM-o 4.5 and 4.6-Thinking at 8,192 tokens; they get
  32,768 here, so a long reasoning chain is not cut and scored wrong.
  Step3-VL's report allows 65,536 tokens, the model's whole context: 32,768
  here. MiMo-VL keeps the 16,384 that Xiaomi's own harness caps it at;
  VLMEvalKit lists only the earlier MiMo-VL-7B-RL, at its wrapper's default of
  2,048.
- **MiniCPM-V 4.5 and MiniCPM-o 4.5 think on all seven benchmarks.** OpenBMB's
  wrappers think on MMMU, MMBench and MMStar (MiniCPM-o's on OCRBench too) and
  answer the others without thinking (with their own prompts, as here); the
  recipe policy runs every reasoning model with its reasoning on. MiniCPM-V
  4.5's wrapper's `<think>\nI` reply prefix is not reproduced: a chat template
  kwarg cannot write into the reply.
- **MiniCPM-V 4.7 thinks.** Its cards evaluate only the instruct mode, and
  the 35B-A3B's GGUF notes say it is not trained as a reasoning model.
  OpenBMB's 1B gguf commits of 2026-10-20 hard-disable thinking in the chat
  template: a plan pins that gguf at `75785938`, the last revision whose
  template carries the `enable_thinking` switch (the weights are the same),
  and AtomicChat's GGUFs carry the same template. The 35B-A3B's root GGUFs
  cannot think either; it runs the identical weights in the repo's `thinking/`
  folder, whose template has the switch. The recipe and prompts are the
  4.6-Thinking sibling's. On a template without the switch the probe finds
  none and the cell fails rather than run without thinking.
- **Gemma 4 thinks.** VLMEvalKit's Gemma 4 wrapper (not Google's) runs it
  without thinking at 4,096 tokens; Google's card documents the thinking
  switch, and the recipe policy turns it on.
- **The larger Qwen models run the small models' setting.** The Qwen3.5 9B
  and 35B-A3B and Qwen3.6 35B-A3B cards give no vision-language setting, and
  their general thinking setting is T 1.0. They run the T 0.6 of the Qwen3.5
  small models' vision-language numbers, so the sizes stay comparable.
- **MiMo-VL runs its card's deployment setting.** Xiaomi's own harness decodes
  greedily, sends the system prompt "You are a helpful assistant." and asks
  for a `\boxed{}` answer. Here it runs the card's T 0.3 and top_p 0.95; the
  datasets' prompts, as VLMEvalKit runs the earlier MiMo-VL-7B-RL; and no
  system message of its own, so the chat template's applies, as the card
  advises.
- **Step3-VL gets the datasets' prompts.** Its report asks MMMU, MMBench,
  MMStar and RealWorldQA questions for a `Final answer:` line; VLMEvalKit's
  Step3-VL entry sends the datasets' prompts, as here.
- **LFM2.5-VL 450M and 1.6B get the datasets' prompts alone.** Liquid AI's
  wrapper runs the 1.6B on the datasets' prompts but appends "Please answer
  directly with only the final answer, do not give any explanation." to each
  of these seven; the entries send the prompts without it.
- **LFM3-VL-2B runs LFM2.5-VL-3B's prompts.** Its card names no benchmark
  prompts, so it gets Liquid AI's VLMEvalKit prompts for LFM2.5-VL-3B.
- **One seed.** A sampled recipe runs once, from base seed 0 (ours), each
  sample at the seed its entry's rule derives ([Sampling](#sampling)).
  llama.cpp's random numbers are not
  Hugging Face's or vLLM's, so a sampled answer is not the same draw as theirs
  at any seed; OpenBMB reports MiniCPM-V 4.5's thinking scores as the mean of
  three runs.
- **Sampling order and penalty window.** As Hugging Face and vLLM apply them
  ([Sampling](#sampling)), with the one remaining difference stated there.
- **Image preprocessing.** Each image goes through the model's own projector
  in llama.cpp, at its own resolution handling, with one exception where
  llama.cpp's default would hold the model below its own processor. Gemma 4's
  processor sizes every image to a visual-token budget, and llama.cpp's
  default gives a small image a fraction of it (a POPE image got about 150
  tokens, and the model would answer that it could not see the image): its
  entries fix every image at 1,120 tokens (`parameter_image_tokens`).
  MiniCPM-V 4.7 runs either projector its GGUF ships, the default 16x or the
  4x; the recipe names neither, and a result names the one its plan loaded.
  VLMEvalKit's per-dataset image handling (InternVL's tile counts per dataset,
  the MiniCPM wrappers' upsizing and slicing options) is not reproduced.
- **Answer extraction** has no LLM judge ([Scoring](#scoring)).

**A cell runs only its entry's recipe.** mgmt refuses to queue a cell whose
model, projector or `enable_thinking` is off the recipe, and reserves the raw
flags that would reach a recipe's settings ([pipette-mgmt docs/benchmarks.md §2.9](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#29-vlm-eval-accuracy)). The client
applies the same recipe check before it downloads anything, so a run nothing
queued (`pipette benchmarks run`) is held to it too, and refuses a config-file
key that sets a reserved option, as above. A recipe covers every quantization
in a repo, so a quantization grid is one plan with a variant per file.

## Context sizing (no overflow, single calibration)

`n_ctx` is fixed when `llama-server` starts; resizing it per sample would mean
thousands of restarts. So `vlm_eval` sizes one context meant as an **upper
bound over every sample in the dataset**. An oversized request fails at the
server (an HTTP 400 on the MCQ arm, an in-band SSE `error` event on the
streaming arm), and the runner records either as a failed sample with the
server's message in `failed_reason`, never as an empty answer. A failed sample
keeps the cell from being a result ([Robustness](#robustness)), hence the
upper bound:

```text
  n_ctx = max_over_candidates( image_tokens(candidate dims) )
        × max_over_samples( images per sample )
        + max_over_samples( text tokens )      (floored at 256)
        + max output tokens
        + margins (64 base + 8 per image),
  floored at 8192 + max_output_tokens,          (one slot's budget)
  × parallel slots,
  and a total past 131072 is a hard error (lower parallel, or pin ctx_size).
```

**Image term.** The scan reads only each image's PNG/JPEG header (no pixel
decode) for its width, height and per-sample image count. Maximizing width and
height independently gives a synthetic `max_w × max_h` that bounds a continuous
encoder, but token count is **not coordinate-wise monotonic** for aspect-ratio
tilers: InternVL picks its tile grid by aspect-ratio fit, so a 6000×500
panorama can need *more* tokens than 6000×6000. The candidate set therefore
also carries the dataset's own extreme shapes, the largest-area, widest and
tallest images, deduped (at most 4). Each candidate's exact count comes from
`llama-mtmd-cli` ([VL Token Calibration](vl-calibration.md), cached), and the
maximum wins. A candidate past the calibration helper's 64 MP synthetic-image
bound (OCRBench's A4 scans and receipt strip combine into a 72 MP point) is
calibrated at an aspect-preserving downscale, which is safe because every VLM
preprocessor is resolution-capped and the count saturates at the cap. This is
an empirical bound over the shapes the dataset contains, not a proof of a
global maximum; the margins are the backstop, and a sample that overflows
anyway fails visibly and keeps its cell from being published, never scores as
a wrong answer.

If any candidate fails to calibrate, the image term falls back to the
`(w/14)·(h/14)` patch grid at the synthetic point (the grid *is* monotonic, so
that point bounds it), floored at 256 tokens per image
([Fallback & safety](vl-calibration.md#fallback--safety)).

**Non-causal images.** llama.cpp decodes the image tokens of Gemma 4 12B and
26B-A4B with non-causal attention, which needs a whole image in one
micro-batch, and aborts when an image's tokens exceed it (the default is 512;
the 12B's images come to 1,166 tokens at the 1,120 budget, measured on
b11551). Their cells set raw `-ub 2048 -b 2048`, which the calibration runs
under too.

**Text term.** Scanned from the dataset, not guessed: the largest sample's
text at one token per UTF-8 byte, floored at 256. That is an upper bound even
for digit-dense OCR text, where BPE emits about one token per digit, and about
4× for Latin prose; the text is small next to the images, so over-estimating
is cheap, and it needs no server round-trip.

The `8192 + max_output_tokens` floor keeps small-image datasets at the context
validated end to end. A requirement past `131072` (128K) is a **hard error**
naming the value and the ceiling: it usually means the estimate exploded (a
mis-read header feeding the patch-grid fallback), but it can be genuine, and
clamping would undersize the genuine case into per-sample overflows deep into
the run. A `parameter_max_tokens` whose floor alone passes the ceiling fails at
run start, before any calibration. An operator's `ctx_size` wins over the
automatic sizing, so on a ctx-pinned cell an over-ceiling estimate only warns,
and the calibration is skipped unless the check below needs it. No other
runtime flag is derived from token counts.

### Concurrent samples (`parallel`)

`parallel` means what it does on the text eval, on the same sample pool (crash
arbitration, serial retry of crash collateral, heartbeat, token tee off above
one slot):
[Concurrent samples](evals.md#concurrent-samples-parallel). What differs is the
context arithmetic, because `llama-server` divides `--ctx-size` across its
slots and this cell's default is calibrated:

- The **calibrated** value is one slot's budget, since every sample needs the
  same context however the run is split, and is multiplied by the slot count
  for `--ctx-size`. The floor bounds the per-slot figure.
- An **explicit** `ctx_size` is passed through verbatim: a stated total is
  already the final argument. It must still hold the largest sample (its
  image, prompt and whole output budget) in every slot, or the run is refused
  before it starts. A slot that holds the patch-grid over-estimate passes
  without measuring; one below it is calibrated first, since a budgeted
  projector can give far fewer tokens than the grid. Every sample fitting is
  what lets the resume identity leave `ctx_size` out.
- A raw `-np` above 1 would split a one-slot context: it is refused beside
  the typed `parallel`, and without it unless `ctx_size` is pinned. Use the
  typed `parallel` instead.
- The `131072` **ceiling** bounds the total, because it stands for a KV cache
  the device allocates in one piece. A slot count a small-image dataset runs
  comfortably can be refused on an image-heavy one; the error names the
  per-slot budget, the slot count and the product.

The thermal gate runs once, before the server starts, at every slot count
([Robustness](#robustness)).

### Prompt caches

The server runs with llama.cpp's prompt caches off: `--no-cache-prompt`,
`--ctx-checkpoints 0` and `--cache-ram 0`, each unless the cell's `raw` sets
it. Every sample is a new prompt, so the caches can only cost. With llama.cpp's
defaults, a sample that lands on a slot first copies that slot's previous
state into an 8 GiB host-memory cache, and each new sample copies every idle
slot's, while the server loop, and every other slot, waits. A hybrid (Qwen3.5,
LFM2) or sliding-window (Gemma 4) model also copies its state into checkpoints
as it reads each prompt.

On one MI325X (llama.cpp b11551), 24 distinct one-image samples, four at a
time, 32 output tokens each, median of three runs:

| Model | flash attention off, defaults | off, caches off | on, defaults | on, caches off |
|---|---|---|---|---|
| LFM2.5-VL-1.6B BF16 | 46 s | 1.4 s | 1.6 s | 1.4 s |
| Qwen3.5-4B BF16 | 122 s | 2.6 s | 4.4 s | 2.5 s |
| Gemma 4 E2B BF16 | 79 s | 2.8 s | 3.5 s | 2.8 s |
| MiniCPM-V 4.5 Q4_K_M | 503 s | 2.9 s | 3.7 s | 2.7 s |

With flash attention off, as the ROCm runs here use it, llama.cpp stores the V
cache transposed and copies a slot's state out a row at a time, about 2 s for
a 250-token LFM2.5 slot and 24 s for a 120-token MiniCPM-V 4.5 one. On an
Apple M4 Max (Metal, flash attention on) the copies are cheap, but the host
cache still held about 1.9 GB beside Qwen3.5-0.8B; on a phone that is memory
the run may not have.

No prompt changes; what changes is that no sample's computation depends on
which samples ran before it. A request's `cache_prompt: false` stops none of
the copying, which is why the server flags are needed. The timing cells keep
llama.cpp's defaults: `vl_throughput` reads the server's own clocks, which
start after a slot is chosen, so a copy made while choosing one is not in its
numbers.

## Robustness

The runner shares the text eval's sample loop and its hardening: the
resumable checkpoint ([Eval checkpoint & resume](../pipette-cli/eval-checkpoint.md)),
whose identity takes the samples through the corpus pin, or for a local
definition a hash of its samples, so the images are never re-hashed on a
catalog run; crash-vs-transport recovery; and up to three attempts per sample
across runs. `parallel` and `ctx_size` stay out of the identity, so an
operator who lowers either after running out of memory resumes rather than
re-runs.

**A result answers every sample, or it is not a result.** A failed or missing
row is never scored as a wrong answer, and a reply the model gave, however
wrong, empty or cut off, is an answer and is graded. The client refuses to
save or submit an incomplete cell, so the checkpoint stays and the next run
retries the failed rows; mgmt and the evals server hold the same rule
([pipette-mgmt docs/benchmarks.md §2.9](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#29-vlm-eval-accuracy)). `PIPETTE_ALLOW_INCOMPLETE_EVAL=1` lets the client finish
an incomplete cell for inspecting failures, but mgmt still refuses the
submission.

Accuracy is thermal-insensitive: an answer depends on the prompt, the decoding
and its seed, not on how hot the device is. The readiness gate therefore runs
**best-effort**, once before the server starts, and a device that cannot cool
within the deadline proceeds warm with a warning rather than failing an
otherwise complete run: it protects the hardware, not the score. It does not
re-run between samples, because the shared loop's workers must be `Sync` and
the gate is not.

## vLLM

Everything above is the llama.cpp cell, a GGUF and its projector. The same
entries also run on vLLM, with a Hugging Face repo's own weights: the
`uv_vllm` × `torch` cell (`pipette-torch-oai`).

```toml
models = [{ type = "torch", source = "huggingface", org = "Qwen", repo_name = "Qwen3.5-2B" }]
runtime_flags = [
  { benchmark_type = "vlm_eval", runtime_type = "uv_vllm", model_type = "torch", parallel = 64 },
]
model_flags = [{ benchmark_type = "vlm_eval", model_type = "torch", enable_thinking = true }]
```

What is the same: the corpus and its two digest checks, the recipe (the repo
must be one of the entry's `models` lines; a line that names an `mmproj`
covers only that projector's GGUF, and a subdirectory of a repo is refused),
the thinking rule, the checkpoint, and the refusal to submit without an answer
to every sample. What vLLM does itself is not repeated: there is no
calibration and no slot arithmetic.

- **Sampling.** Each request sends every knob, the entry's or the neutral
  value: `temperature`, `top_p` 1.0, `top_k` -1 (vLLM's "no cut"), `min_p` 0,
  `presence_penalty` and `frequency_penalty` 0, `repetition_penalty` 1.0, and
  the sample's seed ([Sampling](#sampling)) when the entry pins one. The
  server starts with `--generation-config vllm`, so a repo's
  `generation_config.json` sets no default behind the recipe (its stop tokens
  still apply). vLLM applies the
  penalties and the temperature as Hugging Face does; its presence penalty
  counts the answer only, its repetition penalty the prompt too.
- **Thinking.** The client probes the template through vLLM's `/tokenize`
  with the same two conversations, `enable_thinking` on, off, then on, and
  applies the table above. A template with a switch gets the cell's mode in
  each request's `chat_template_kwargs`; one without gets nothing, and the
  `internvl_r1` profile thinks through its system prompt as on llama.cpp.
  There is no restart: the kwarg is per request.
- **The reply** is the model's text as written, as llama.cpp returns it: no
  reasoning parser takes it apart (`--reasoning-parser` is reserved; reasoning
  a server streams apart anyway goes back in front of the answer, closed by
  `</think>`), and the special tokens the model writes stay in it
  (`skip_special_tokens: false`).
  Gemma 4's thought channel is marked by special tokens, and the scorer finds
  the reasoning by them.
- **Image budget.** `parameter_image_tokens` is sent as each request's
  `mm_processor_kwargs` `max_soft_tokens`, the budget Gemma 4's processor
  sizes every image to, on a server started with `--trust-request-mm-kwargs`
  (vLLM refuses a request's processor kwargs otherwise). Before any sample, two
  gray images, 1024×1024 and 256×512, are tokenized at the budget and at half
  of it (twice it, under a budget of 140): at the budget each must come to
  between 80% of it and the budget plus 16 marker tokens, and at the other to
  a different count, or the cell is refused
  rather than score images sized off its recipe. A processor that sizes images
  by their pixels, or to a fixed count, fails one or the other.
- **Context.** `max_model_len` is left unset, the model's own context. A cell
  that pins it at or under `parameter_max_tokens` is refused before the
  server starts, as are constrained answers and a sample count other than the
  entry's. A sample whose prompt leaves less room than the output budget gets
  no answer: vLLM up to 0.31 refuses the request, and from 0.32 it generates
  to the context's end, so a reply cut short of `parameter_max_tokens` at a
  length bound is failed rather than scored. A cut answer is one that ran the
  whole budget, as on llama.cpp.
- **`parallel`** is the number of samples in flight; vLLM batches them.
  `parallel` and `max_model_len` stay out of the resume identity, as
  `parallel` and `ctx_size` do on llama.cpp: neither changes an answer, so a
  cell retuned after running out of memory resumes its checkpoint.
- **Failures.** A request the server refuses (a 4xx, whose body says why) is
  the sample's failure. A request that broke on the way (a reset, a stream
  without its finish chunk, an overloaded server's 429 or 5xx) is lost: sent
  again alone, then failed. Either way a failed sample is retried on the next
  run of the cell. Unlike llama.cpp, the client does not restart the server:
  a vLLM that goes down fails every sample still to run, one attempt each,
  and the cell is refused. A retuned run keeps the checkpoint's attempt
  counts, so a sample that has spent all three is sent again only once its
  failed lines are dropped ([eval-checkpoint](../pipette-cli/eval-checkpoint.md)).
- **Not on vLLM:** `parameter_mcq_choices` (constrained answers).

A recipe's settings are closed to `raw` here by an allowlist: the cell's
`raw` may carry only flags that move speed, memory, device placement or
logging (`--gpu-memory-utilization`, `--max-num-seqs`, `--enforce-eager`,
`--trust-remote-code`, …; not `--limit-mm-per-prompt`, which also takes
per-image sizes), by their full names. Every other flag is refused: the
generation config, the chat template and its kwargs, the processor's kwargs,
quantization, the model implementation, a quantized KV cache, the reply
parsers and a `--config` file among them. Its `envs` are an allowlist too:
device selection, threads, cache locations, logging and a few NCCL settings,
never a `VLLM_*` variable that redirects the weights or picks kernels. The
management service refuses a torch model's vlm_eval on any runtime but
`uv_vllm`.

A vLLM row is not a llama.cpp row: the weights are the repo's own (BF16 as
published), the vision preprocessing is the repo's processor rather than
llama.cpp's port of it, and a sampled answer is a different random draw.

A vLLM cell runs only a repo its entry's recipe lists, and the recipes list
one only for Qwen3.5 0.8B, 2B and 4B, Gemma 4 E2B, E4B and 12B, InternVL3.5
2B and 4B, MiniCPM-V 4.5, 4.6-Thinking and 4.7 1B, LFM2.5-VL 1.6B and 3B, and
LFM3-VL-2B. The other models run on llama.cpp only. Which architectures run
depends on the vLLM version: vLLM 0.32, the first with MiniCPM-V 4.7, runs all
of these, and the bundled runtime catalog carries it for ROCm 7.2.3
(`vllm@0.32.0+rocm723`).
Gemma 4 12B has no vLLM rows: vLLM 0.32 on ROCm misreads its 1,120-token
images, an image span longer than the model's 1,024-token sliding window, so
its full-precision reference is the BF16 GGUF on llama.cpp.

## Scoring

Scoring is outside the client: mgmt submits the completions to pipette-scores,
which computes each headline over the full dataset with VLMEvalKit's own
scoring code at commit `8eca8e5`, and only when the reported corpus digest is
the entry's pin. The headline per benchmark and the extraction rules are in
[pipette-mgmt docs/benchmarks.md §2.9](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/benchmarks.md#29-vlm-eval-accuracy).
Before matching, the reasoning is removed: a closed block (`<think>…</think>`,
`[THINK]…[/THINK]`, Gemma 4's thought channel), the text up to a close whose
opener the chat template wrote, and an opener never closed, with everything
after it. A reply with no reasoning markup is graded whole. Answer extraction
is deterministic, with no LLM judge and no random choice: VLMEvalKit's matcher
reads the answer first, and only an answer it cannot read, where VLMEvalKit
would ask a judge, goes to fixed fallback readers, each counted in the score
context's `fallback_answers`. A score can therefore read below a published
VLMEvalKit number that a judge helped.

Where a family's wrapper cuts its reply down to the stated answer, so does the
scorer: under `minicpm_v_cot` it reads OpenBMB's `Answer:` line as MiniCPM-V
4.6's `extract_answer` does, even from a reply that ends inside its reasoning,
on every benchmark but OCRBench.
