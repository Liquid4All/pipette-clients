# Naming models, runtimes, and per-cell flags

**Scope:** the notation `pipette` takes for `--model`, `--runtime`,
`--runtime-flags`, and `--model-flags`. Operator workflow is
[usage.md](usage.md); backend specifics are [llama.cpp](llamacpp.md) ·
[MLX](mlx.md) · [torch-oai](torch-oai.md) · [OpenVINO](openvino.md).

Every `pipette` run is one **cell**: a benchmark, a runtime, and a model. This
page is how you spell the runtime and the model, and how you tune them.

## Three ways to name one artifact

| Form | Looks like | Use it when |
|------|------------|-------------|
| **URI** | `gguf-text://repo=org/name&path=Q4_K_M.gguf` | almost always: it is the short, typeable form |
| **JSON** | `{"type":"gguf_text","source":"huggingface",…}` | the URI cannot express it (see [below](#when-you-must-use-json)) |
| **Digest** | `model://sha256=794e8d3d` | re-running against something already in the local store |

All three work anywhere a `--model` / `--runtime` is taken: `benchmarks run`,
`models pull` / `models delete`, `runtimes pull` / `runtimes remove`.

## The URI grammar

```text
uri    ::= scheme "://" body        ; split on the FIRST "://"
body   ::= "" | pair ("&" pair)*    ; keys unordered, each at most once
pair   ::= key "=" value            ; the FIRST "=" splits; the value may contain more
```

Rules that bite:

- **No `?`.** Pairs follow `://` directly, as in `mlx://repo=org/name`, never
  `mlx://?repo=…`.
- **No escaping exists.** No value may contain `&`, and a `url=` value may not
  contain a query string. Either forces the JSON form.
- **Hyphens on the wire, underscores in the object.** The scheme is the JSON
  `type` with `-` for `_`: `gguf-text://` ↔ `"type": "gguf_text"`. Multi-word
  *keys* stay snake_case in both (`model_sha256`).
- **Case-sensitive**, schemes and keys alike.
- The first `=` splitting means a value may contain `=`, and splitting on the
  first `://` means `gguf-text://url=file:///models/m.gguf` works.

## Model schemes

| Scheme | Shape | Runtimes it can run on |
|--------|-------|------------------------|
| `gguf-text` | one GGUF file | llama.cpp |
| `gguf-vision` | GGUF weights + mmproj projector | llama.cpp |
| `mlx` | HF repo snapshot (quantized safetensors) | MLX |
| `torch` | HF repo snapshot (safetensors) | vLLM / SGLang, docker or uv |
| `openvino` | compiled OpenVINO IR bundle | OpenVINO |

`apple_foundation_text` exists as a JSON type but has no URI and is not
pullable: the weights ship with the OS.

### Keys

`gguf-text` takes exactly one of `repo` or `url`:

| Key | Required | Meaning |
|-----|----------|---------|
| `repo` | yes (HF arm) | `org/repo_name` |
| `path` | yes (HF arm) | repo-relative path to the `.gguf`; may be nested |
| `url` | yes (URL arm) | `https://`, `http://`, or `file://` |
| `rev` | no | tag, branch, or commit. HF arm only, since a raw URL has no revision |
| `sha256` | no | 64 lowercase hex, verified on download |

`gguf-vision` selects the HF arm when `repo` is present; without it,
`model`/`mmproj` become URLs:

| Key | Required | Meaning |
|-----|----------|---------|
| `repo` | HF arm | one repo hosting both files |
| `model` | yes | weights (repo-relative path, or URL on the URL arm) |
| `mmproj` | yes | multimodal projector, same |
| `rev` | no | HF arm only; applies to both files |
| `model_sha256`, `mmproj_sha256` | no | per-file digests |

`mlx`, `torch`, and `openvino` share one HF-only grammar:

| Key | Required | Meaning |
|-----|----------|---------|
| `repo` | yes | `org/repo_name` |
| `prefix` | no | subdirectory, when one repo bundles several variants |
| `rev` | no | revision pin |

There is no `sha256` on these three: they name a directory, not a file.
`prefix` is how you pick one precision out of a multi-variant OpenVINO repo
(`prefix=int4-sym-cw`), and one quant out of a multi-quant MLX repo.

```bash
--model 'gguf-text://repo=unsloth/gemma-3-270m-it-GGUF&path=gemma-3-270m-it-Q4_K_M.gguf'
--model 'gguf-vision://repo=ggml-org/gemma-3-4b-it-GGUF&model=gemma-3-4b-it-Q4_K_M.gguf&mmproj=mmproj-model-f16.gguf'
--model 'mlx://repo=mlx-community/Qwen3.5-0.8B-4bit'
--model 'torch://repo=Qwen/Qwen2.5-0.5B-Instruct'
--model 'openvino://repo=LiquidAI/LFM2.5-350M-ov&prefix=int4-sym-cw'
--model 'gguf-text://url=https://example.com/model-Q4_K_M.gguf'
```

## Runtime schemes

| Scheme | Keys | Notes |
|--------|------|-------|
| `llamacpp-cli-stock-tools` | `version` (+ optional `repo`) **xor** `url`; `flavor` | `repo` defaults to `github.com/ggml-org/llama.cpp` |
| `mlx-macos-pipette` | `version`; `flavor` | `flavor` defaults to `macos-arm64` |
| `docker-vllm`, `docker-sglang` | `image`, `tag`; `flavor` | `flavor` defaults to `nvidia_gpu`; pulled into the docker daemon |
| `uv-vllm`, `uv-sglang` | `server`, `build`, `python` | catalog-backed venv |
| `uv-openvino` | `version` | one venv serves cpu/gpu/npu, since the device is per cell |

```bash
--runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64'
--runtime 'llamacpp-cli-stock-tools://repo=github.com/acme/llama.cpp&version=b1&flavor=linux-x64-cpu'
--runtime 'llamacpp-cli-stock-tools://url=https://example.com/llama-b1.tar.gz&flavor=macos-arm64'
--runtime 'mlx-macos-pipette://version=0.31.3'
--runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.20.2&flavor=nvidia_gpu'
--runtime 'uv-vllm://server=0.21.0&build=cu121&python=3.12'
--runtime 'uv-openvino://version=2026.2.1'
```

`flavor` is not decorative: it picks the real upstream asset, and the wrong one
is the wrong binary for the host. `pipette runtimes flavors` lists the whole
vocabulary with the asset each name resolves to; it needs no workspace and no
network.

The four in-app runtimes (`apple_foundation`, `llamacpp_apk_pipette`,
`llamacpp_ios_pipette`, `mlx_ios_pipette`) have **no URI form** and cannot be
pulled or run by the desktop CLI. They appear in plan files because the
scheduler dispatches those cells to a phone or a Mac running the pipette app.
Naming one gets you:

```text
runtime `llamacpp-ios-pipette` is not representable as a URI; pass a JSON `--runtime` object instead
```

…and the JSON does parse, but then fails with `is not a desktop CLI runtime`.

## Model ↔ runtime compatibility

Checked before anything is downloaded.

| Model type | Desktop runtimes |
|------------|------------------|
| `gguf_text`, `gguf_vision` | `llamacpp_cli_stock_tools` |
| `mlx` | `mlx_macos_pipette` |
| `torch` | `docker_vllm`, `docker_sglang`, `uv_vllm`, `uv_sglang` |
| `openvino` | `uv_openvino` |

A mismatch fails with `model … is not compatible with runtime …`.

## The JSON form

The URI infers the source from which keys are present; JSON states it outright.
For a **model**, `type` and `source` are both required (except
`apple_foundation_text`, which has neither a source nor a URI). **Runtimes**
vary: `llamacpp_cli_stock_tools`, `mlx_macos_pipette` and the `uv_*` runtimes
carry a `source`, while `docker_vllm` and `docker_sglang` have no such field at
all, since an image name and tag are the whole coordinate.

The one field-name difference: the URI's single `repo=org/name` becomes **two**
JSON fields, `org` and `repo_name`. `rev` becomes `revision`.

| URI key (model) | JSON field |
|---------|------------|
| `repo=<org>/<name>` | `"org": "<org>"`, `"repo_name": "<name>"` |
| `rev` | `"revision"` |
| *(none)* | `"auth_token"` |
| everything else | same name |

On the **runtime** side the URI keys are abbreviations rather than the field
names: `image` → `image_name`, `tag` → `image_tag`, `server` → `server_version`,
`python` → `python_version`, and for `llamacpp-cli-stock-tools` the pair
`repo` + `version` becomes `repository_url` + `repository_version` nested under
`source`. Round-trip a `runtimes list --format uri` row rather than hand-porting
one shape to the other.

The same model, three ways:

```text
mlx://repo=mlx-community/Qwen3.5-0.8B-4bit&rev=v2
```
```json
{"type":"mlx","source":"huggingface","org":"mlx-community","repo_name":"Qwen3.5-0.8B-4bit","revision":"v2"}
```
```toml
{ type = "mlx", source = "huggingface", org = "mlx-community", repo_name = "Qwen3.5-0.8B-4bit", revision = "v2" }
```

The third is the plan form. It is the same serde model, which is why a plan
entry and a `--model` JSON object look alike. See [plan-runner.md](../pipette-plan/plan-runner.md).

### When you must use JSON

- A value containing `&`, or a `url=` with a query string.
- `auth_token` on the model. There is no URI key for it; prefer
  `PIPETTE_HF_TOKEN` in the environment, which is injected for you.
- A local path source: `absolute_file` / `absolute_dir` (and the
  `relative_*` forms, which parse but have no store entry).
- A uv/MLX/OpenVINO runtime whose requirements are **not** a catalog row, or one
  carrying `install_flags`. A URI names a catalog preset; an off-catalog body
  has no URI that would round-trip.
- An in-app runtime, though the desktop CLI will then refuse to run it.

Note an asymmetry: the URI defaults `flavor` for the docker and MLX schemes, but
the JSON form **requires** it.

```bash
--runtime '{"type":"docker_vllm","image_name":"vllm/vllm-openai","image_tag":"v0.20.2","flavor":"nvidia_gpu"}'
--model '{"type":"gguf_text","source":"absolute_file","path":"/models/m.gguf"}'
```

## The digest form

Once an artifact is installed, `list` shows a `DIGEST` you can use instead of
retyping its coordinates:

```bash
pipette models list
#  MODEL                                                    | TYPE      | DIGEST       | FETCHED
# ----------------------------------------------------------+-----------+--------------+---------
#  unsloth/gemma-3-270m-it-GGUF:gemma-3-270m-it-Q4_K_M.gguf | gguf_text | 794e8d3d3305 | …

pipette benchmarks run --benchmark local/prefill_throughput_smoke \
  --model 'model://sha256=794e8d3d' --runtime 'runtime://sha256=5eb48867'
```

Any prefix of 8 hex chars or more works; an ambiguous one is an error rather
than a guess. Each scheme searches only its own store. It resolves against
**installed** artifacts, so a catalog pin never pulled has nothing to point at.

`--format uri` on either `list` renders the importable URI instead of the human
name, so a row pastes straight back into `pull`.

## Per-cell flags

Three families tune a cell. All are keyed on the cell itself, and **which
settings are legal depends on the benchmark, the runtime, and the model
together**. A setting the cell does not accept is an error, never a silent
no-op.

| Family | CLI surface | Carries its axes? |
|--------|-------------|-------------------|
| runtime flags | `--runtime-flags '<json object>'` | **no**: derived from the cell |
| model flags | `--model-flags '<json object>'`, or `--model-enable-thinking` | **yes**: `model_type` + `benchmark_type` |
| benchmark flags | individual flags: `--http-timeout-seconds`, `--readiness-*`, `--doomloop-*` | n/a, no JSON surface |

### `--runtime-flags`

One JSON **object** of settings for this cell. Not an array, and not carrying
the `runtime_type` / `model_type` / `benchmark_type` keys a plan entry has.
The CLI derives those from `--benchmark`, `--runtime` and `--model`:

```bash
--runtime-flags '{"threads":8,"number_gpu_layers":99,"flash_attention":"on"}'
```

```text
# the plan-style array, rejected:
runtime flags must be a JSON object of knobs, e.g. {"threads":4} (the one-element array is no longer accepted)

# axes restated, rejected:
runtime flags must not carry `benchmark_type`: the cell comes from --benchmark, --runtime and --model, which are parsed first

# legal cell, illegal setting:
knob `ctx_size` is not accepted by prefill_throughput × LlamacppCliStockTools × GgufText
```

What each cell accepts:

| Runtime × model | Benchmarks | Settings |
|-----------------|-----------|----------|
| llama.cpp × `gguf_text` (llama-bench) | prefill, decode, max-memory | `threads`, `number_gpu_layers`, `mmap`, `flash_attention`, `raw` |
| llama.cpp × `gguf_text` (llama-server) | end-to-end, eval | the above + `ctx_size`, `no_cache` |
| llama.cpp × `gguf_vision` | vl | `threads`, `number_gpu_layers`, `mmap`, `flash_attention`, `ctx_size`, `no_cache`, `raw` |
| `docker_vllm` × `torch` | end-to-end, eval, max-memory | `tensor_parallel_size`, `dtype`, `max_model_len`, `prefix_caching`, `gpus`, `shm_size`, `ipc`, `envs`, `raw` |
| `uv_vllm` × `torch` | same | the above minus `gpus`, `shm_size`, `ipc` |
| `docker_sglang` × `torch` | same | `tensor_parallel_size`, `prefix_caching`, `gpus`, `shm_size`, `ipc`, `envs`, `raw` |
| `uv_sglang` × `torch` | same | `tensor_parallel_size`, `prefix_caching`, `envs`, `raw` |
| `uv_openvino` × `openvino` | prefill, decode, end-to-end, max-memory | `device`, `max_prompt_len`, `min_response_len`, `generate_hint` |
| `mlx_macos_pipette` × `mlx` | n/a | **none**: MLX has no runtime-flag cell, so any setting is refused. An empty `{}` means "no flags" and is the one accepted value |

`raw` is the escape hatch: an array of argv tokens passed verbatim to the
underlying tool.

```bash
--runtime-flags '{"raw":["--numa","distribute"]}'
```

It refuses two kinds of token, and both sets are per cell. First, anything that
aliases a typed setting on *that* cell: `-t`/`--threads`, `-ngl`, `--mmap`,
`-fa`, plus `-c`/`--ctx-size` and `--no-cache-prompt` on the server cells. Set
the typed field instead. Second, whatever the harness owns there:

| Cell | Reserved |
|------|----------|
| llama-bench (prefill, decode) | `--output`/`-o`, `--model`/`-m`, `--n-prompt`/`-p`, `--n-gen`/`-n`, `--n-depth`/`-d`, `--repetitions`/`-r` |
| llama-bench (max-memory) | the above, plus `--ctx-size`/`-c` |
| llama-server (end-to-end, eval, vl) | `--model`/`-m`, `--mmproj`, `--host`, `--port`, `--no-warmup` |
| vLLM | `--model`, `--host`, `--port` |
| SGLang | `--model-path`, `--host`, `--port` |

So `--no-warmup` passes through on a prefill cell but is reserved on an eval
cell, and `-p` is the reverse. Matching covers `--flag=value` and the glued
short forms `-t8` and `-ngl99`.

Two notes on secrets: `envs` takes `K=V` or a bare `K` to inherit from the
environment, and env **values are stripped** from the submitted record while
`raw` is submitted verbatim. Put a token in the bare-`K` form, never in `raw`.

### `--model-flags`

Eval benchmarks only. Unlike `--runtime-flags`, it **keeps** the plan's axes. The single setting is `enable_thinking`:

```bash
--model-flags '{"model_type":"mlx","benchmark_type":"eval","enable_thinking":false}'
--model-enable-thinking false     # same thing, axes derived from the cell
```

The two are mutually exclusive. Omitting the flag entirely is not "off": the
kwarg is simply not sent and the engine's own default applies, which today is
thinking-*on* for recent `llama-server` and `mlx_lm`.

### Benchmark flags

There is no JSON surface. These are individual flags, and each is accepted only
on the cells that have somewhere to put it:

| Flag | Valid on |
|------|----------|
| `--http-timeout-seconds` | eval, end-to-end-latency, vl |
| `--readiness-max-wait-secs`, `--readiness-skip-thermal` | prefill, decode, end-to-end-latency, vl |
| `--doomloop-*` (36 flags, 6 detectors) | eval |

Elsewhere they are rejected, naming the cell:

```text
http_timeout_seconds is not a valid benchmark flag for GgufText running PrefillThroughput on LlamacppCliStockTools
```

The doom-loop detectors and their defaults are documented in
[doomloop-detection.md](../pipette-doomloop/doomloop-detection.md). Note that a
`--doomloop-*-window` of `0` does not disable a detector; it fails validation.
Use `--doomloop-<detector>-enabled false`.

## Recipes

Gemma, GGUF, on llama.cpp. This is the smallest thing that works, and it needs
no server and no registration:

```bash
pipette init
pipette benchmarks init-local
pipette benchmarks run \
  --benchmark local/prefill_throughput_smoke \
  --model 'gguf-text://repo=unsloth/gemma-3-270m-it-GGUF&path=gemma-3-270m-it-Q4_K_M.gguf' \
  --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64'
```

The same Gemma repo has other quants: swap `path` for `-Q4_0.gguf`,
`-Q5_K_M.gguf`, or `-Q8_0.gguf` to compare them. That one-axis sweep is what a
plan automates; see [aa-style plans](../pipette-plan/plan-runner.md).

Gemma vision, weights and projector in one ref:

```bash
pipette benchmarks run \
  --benchmark local/vl_throughput_smoke \
  --model 'gguf-vision://repo=ggml-org/gemma-3-4b-it-GGUF&model=gemma-3-4b-it-Q4_K_M.gguf&mmproj=mmproj-model-f16.gguf' \
  --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64'
```

Qwen on llama.cpp, with the cell tuned:

```bash
pipette benchmarks run \
  --benchmark local/decode_throughput_512_100 \
  --model 'gguf-text://repo=unsloth/Qwen3.5-0.8B-GGUF&path=Qwen3.5-0.8B-Q4_0.gguf' \
  --runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64' \
  --runtime-flags '{"threads":8,"number_gpu_layers":99,"flash_attention":"on"}'
```

Qwen on MLX, evaluating with thinking off:

```bash
pipette benchmarks run \
  --benchmark local/eval_smoke \
  --model 'mlx://repo=mlx-community/Qwen3.5-0.8B-4bit' \
  --runtime 'mlx-macos-pipette://version=0.31.3' \
  --model-enable-thinking false
```

Qwen on vLLM under docker:

```bash
export PIPETTE_HF_TOKEN=hf_…   # only if the repo is gated
pipette benchmarks run \
  --benchmark local/end_to_end_latency_smoke \
  --model 'torch://repo=Qwen/Qwen2.5-0.5B-Instruct' \
  --runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.20.2&flavor=nvidia_gpu' \
  --runtime-flags '{"dtype":"bfloat16","max_model_len":4096,"gpus":"all"}'
```

OpenVINO, where the device is a per-cell choice:

```bash
pipette benchmarks run \
  --benchmark local/prefill_throughput_512 \
  --model 'openvino://repo=LiquidAI/LFM2.5-350M-ov&prefix=int4-sym-cw' \
  --runtime 'uv-openvino://version=2026.2.1' \
  --runtime-flags '{"device":"cpu"}'
```
