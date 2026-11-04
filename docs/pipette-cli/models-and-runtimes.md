# Naming models, runtimes, and per-cell flags

**Scope:** the notation `pipette` takes for `--model`, `--runtime`,
`--runtime-flags`, `--model-flags`, and `--benchmark-flags`. Operator workflow is
[usage.md](usage.md); backend specifics are [llama.cpp](llamacpp.md) ·
[MLX](mlx.md) · [Core AI](coreai.md) · [torch-oai](torch-oai.md) ·
[OpenVINO](openvino.md) · [OVMS](../ovms.md) · [ONNX GenAI](onnx-genai.md) ·
[TensorRT Edge-LLM](trt-edgellm.md).

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

Rules:

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

| Scheme | Shape |
|--------|-------|
| `gguf-text` | one GGUF file |
| `gguf-vision` | GGUF weights + mmproj projector |
| `gguf-vision-speculative` | the above plus a drafter, merged or its own file |
| `mlx` | Hugging Face (HF) repo snapshot (quantized safetensors) |
| `core-ai` | Apple Core AI `.aimodel` bundle directory |
| `mlx-speculative` | two HF repo snapshots: a vision-language (VL) target and its drafter |
| `torch` | HF repo snapshot (safetensors) |
| `torch-speculative` | two HF repo snapshots: a VL target and its DSpark drafter (the speculative drafter that `LiquidAI/LFM2.5-VL-3B-DSpark` publishes) |
| `openvino` | compiled OpenVINO IR (Intermediate Representation) bundle |
| `onnx-genai` | GenAI config, ONNX graph, external weights and tokenizer |

Which runtimes load each model type is [below](#model--runtime-compatibility).

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

`gguf-vision-speculative` is `gguf-vision`'s grammar plus the drafter, resolved
against the same source as the weights:

| Key | Required | Meaning |
|-----|----------|---------|
| `repo`, `model`, `mmproj`, `rev`, `model_sha256`, `mmproj_sha256` | as `gguf-vision` | |
| `draft` | no | the drafter's own gguf. Omit it when the drafter is merged into `model`, as DSpark's release ships it |
| `draft_sha256` | no | digest for that file |

The drafter is part of the *model's* identity, not a benchmark parameter: two
rows that differ by their drafter must not collide on one model. The draft
length (`k`) is the parameter, because it changes the measurement on a fixed
(target, drafter) pair.

`mlx-speculative` and `torch-speculative` name **two** repositories under one
model identity: the target, and the drafter that proposes for it. The same
keys on both.

| Key | Required | Meaning |
|-----|----------|---------|
| `repo` | yes | the target bundle, `org/repo_name` |
| `prefix`, `rev` | no | as below, for the target |
| `draft_repo` | yes | the drafter's own bundle |
| `draft_prefix`, `draft_rev` | no | the same two, for the drafter |

Two repositories rather than a file beside the weights, which is where MLX
differs from the gguf pair above: an MLX drafter is published as a bundle of its
own with its own `config.json`, and `mlx_vlm`'s `load_drafter` reads that file
to decide which drafter it is holding. There is no merged form, so `draft_repo`
is required: a speculative model with no drafter is just `mlx`. Both repos are
fetched into one store entry, under `target/` and `draft/`. With
`speculative_enabled = false` the target is launched alone and the drafter is
never loaded.

`torch-speculative` is the same pair for SGLang's DSpark path: the target is
what `--model-path` loads and the drafter goes to
`--speculative-draft-model-path`, a checkpoint of its own whose `config.json`
carries the block size. It runs on the corpus benchmark alone; the throughput
and eval cells take a plain `torch` model.

Neither side has to be an MLX **export**. `mlx_vlm` loads a stock bf16/fp16
safetensors repo directly, so `LiquidAI/LFM2.5-VL-3B` and its
`…-MLX-bf16` conversion generate identical text at the same rate; the drafters
are published in stock form only, and load as they are. An MLX export is
required only when the repo is **quantized**, because a 4/8-bit MLX bundle
carries a `quantization` block and a packed weight layout that only `mlx` reads.
So name the MLX repo for a quantized cell, and either for a bf16 one.

`mlx`, `core-ai`, `torch`, `openvino` and `onnx-genai` share one HF-only grammar:

| Key | Required | Meaning |
|-----|----------|---------|
| `repo` | yes | `org/repo_name` |
| `prefix` | no | subdirectory, when one repo bundles several variants |
| `rev` | no | revision pin |

There is no `sha256` on these five: they name a directory, not a file.
Use `prefix` to pick one variant out of a repo that bundles several:

- an OpenVINO precision: `prefix=int4-sym-cw`
- an MLX quant: `prefix=4bit`
- an ONNX precision and provider:
  `prefix=cpu_and_mobile/cpu-int4-rtn-block-32`

`prefix` works only when the variant directory holds everything the loader
needs. Some ONNX repos publish a top-level `manifest.json` plus a shared
`shared_assets/sha256-*` store, and put part of each variant there. A `prefix`
that names one variant directory in such a repo fetches an incomplete model.
Use the JSON form's `"source": "absolute_dir"` to point at a directory you
assembled yourself.

```bash
--model 'gguf-text://repo=unsloth/gemma-3-270m-it-GGUF&path=gemma-3-270m-it-Q4_K_M.gguf'
--model 'gguf-vision://repo=ggml-org/gemma-3-4b-it-GGUF&model=gemma-3-4b-it-Q4_K_M.gguf&mmproj=mmproj-model-f16.gguf'
--model 'gguf-vision-speculative://repo=example-org/LFM2.5-VL-3B-DSpark-GGUF&model=merged-f16.gguf&mmproj=mmproj-f16.gguf'
--model 'mlx://repo=mlx-community/Qwen3.5-0.8B-4bit'
--model 'mlx-speculative://repo=LiquidAI/LFM2.5-VL-3B-MLX-bf16&draft_repo=LiquidAI/LFM2.5-VL-3B-DSpark'
--model 'torch://repo=Qwen/Qwen2.5-0.5B-Instruct'
--model 'torch-speculative://repo=LiquidAI/LFM2.5-VL-3B&draft_repo=LiquidAI/LFM2.5-VL-3B-DSpark'
--model 'openvino://repo=OpenVINO/LFM2.5-350M-int8-ov'
--model 'onnx-genai://repo=microsoft/Phi-3-mini-4k-instruct-onnx&prefix=cpu_and_mobile/cpu-int4-rtn-block-32'
--model 'gguf-text://url=https://example.com/model-Q4_K_M.gguf'
```

## Runtime schemes

| Scheme | Keys | Notes |
|--------|------|-------|
| `llamacpp-cli-stock-tools` | `version` (+ optional `repo`, `asset`) **xor** `url`; `flavor` | `repo` defaults to `github.com/ggml-org/llama.cpp`; `asset` names a release file the flavor does not |
| `mlx-macos-pipette` | `version`; `flavor` | `flavor` defaults to `macos-arm64` |
| `core-ai-macos-pipette` | optional `version` | macOS 27+; `version` defaults to the bundled `coreai-models` pin and must match it when supplied. The Swift sidecar is built from source. |
| `docker-vllm`, `docker-sglang` | `image`, `tag`; `flavor` | `flavor` defaults to `nvidia_gpu`; pulled into the docker daemon |
| `uv-vllm`, `uv-sglang` | `server`, `build`, `python` | catalog-backed venv |
| `uv-openvino` | `version` | one venv serves cpu/gpu/npu, since the device is per cell |
| `uv-onnx-genai` | `version`, `ep` | one venv per execution provider, since each provider is a separate wheel |
| `ovms` | `version` (+ optional `repo`) **xor** `url`; `flavor` | an OpenVINO Model Server release archive per OS; `repo` defaults to `github.com/openvinotoolkit/model_server`. The device is per cell, as for `uv-openvino` |
| `trt-edgellm` | `url`; `flavor` | a TensorRT Edge-LLM server archive for a Jetson ([TensorRT Edge-LLM](trt-edgellm.md)); upstream publishes none, so there is no catalog and no `version` |

```bash
--runtime 'llamacpp-cli-stock-tools://version=b9305&flavor=macos-arm64'
--runtime 'llamacpp-cli-stock-tools://repo=github.com/acme/llama.cpp&version=b1&flavor=linux-x64-cpu'
--runtime 'llamacpp-cli-stock-tools://version=b11399&asset=<a file in the release>&flavor=windows-x64-cuda-13'
--runtime 'llamacpp-cli-stock-tools://url=https://example.com/llama-b1.tar.gz&flavor=macos-arm64'
--runtime 'mlx-macos-pipette://version=0.31.3'
--runtime 'core-ai-macos-pipette://version=27a66f90e7f3fd9b83a6acb7bcb0a4a5ff71fd60'
--runtime 'docker-vllm://image=vllm/vllm-openai&tag=v0.20.2&flavor=nvidia_gpu'
--runtime 'uv-vllm://server=0.32.0&build=cu129&python=3.12'
--runtime 'uv-openvino://version=2026.4.1'
--runtime 'uv-onnx-genai://version=0.17.1&ep=cpu'
--runtime 'ovms://version=v2026.4.1&flavor=windows-x64'
--runtime 'trt-edgellm://url=https://example.com/trt-edgellm-server-0.11.0-jetson-orin-cuda-13-4e3f9d23f052.tar.gz&flavor=jetson-orin-cuda-13'
```

`flavor` selects the upstream asset, so a wrong flavor installs a binary that
does not match the host. `pipette runtimes flavors` lists the whole
vocabulary with the asset each name resolves to; it needs no workspace and no
network.

The five in-app runtimes (`apple_foundation`, `llamacpp_apk_pipette`,
`llamacpp_ios_pipette`, `mlx_ios_pipette`, `onnx_genai_ios_pipette`) have
**no URI form** and cannot be pulled or run by the desktop CLI. They appear in plan files because the
scheduler dispatches those cells to a phone or a Mac running the pipette app.
Naming one gets you:

```text
runtime `llamacpp-ios-pipette` is not representable as a URI; pass a JSON `--runtime` object instead
```

…and the JSON does parse, but then fails with `is not a desktop CLI runtime`.

## Model ↔ runtime compatibility

Checked before anything is downloaded. The accepted pairs are the arms of
`is_compatible` in pipette-mgmt
[`crates/pipette-plan-types/src/cell/mod.rs`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/cell/mod.rs).
A mismatch fails with ``runtime `…` cannot load a `…` model``.
`core_ai` pairs with `core_ai_macos_pipette` only.

## The JSON form

The URI infers the source from which keys are present; JSON states it outright.
For a **model**, `type` and `source` are both required (except
`apple_foundation_text`, which has neither a source nor a URI). A `core_ai` model is a
directory, with the same source fields as an `mlx` model. **Runtimes** vary:

- `llamacpp_cli_stock_tools`, `ovms`, `trt_edgellm`, `mlx_macos_pipette` and
  the `uv_*` runtimes carry a `source`.
- `docker_vllm` and `docker_sglang` have no `source` field, because an image
  name and tag are the whole coordinate.
- `core_ai_macos_pipette` carries `packages`, the pinned Swift stack, and no
  `source`.

The one field-name difference: the URI's single `repo=org/name` becomes **two**
JSON fields, `org` and `repo_name`. `rev` becomes `revision`.

| URI key (model) | JSON field |
|---------|------------|
| `repo=<org>/<name>` | `"org": "<org>"`, `"repo_name": "<name>"` |
| `rev` | `"revision"` |
| *(none)* | `"auth_token"` |
| everything else | same name |

On the **runtime** side, the URI keys are short forms of the JSON field names:

| URI key (runtime) | JSON field |
|---------|------------|
| `image`, `tag` (docker) | `image_name`, `image_tag` |
| `server`, `python` (`uv-vllm`, `uv-sglang`) | `server_version`, `python_version` |
| `version` (`uv-openvino`, `uv-onnx-genai`) | `server_version`, plus a `python_version` that the URI does not take |
| `ep` (`uv-onnx-genai`) | `execution_provider` |
| `repo` + `version` (`llamacpp-cli-stock-tools`, `ovms`) | `repository_url` + `repository_version`, flat beside `"source": "github_release"`. `version` is accepted as an alias |
| `asset`, `flavor`, `build` | same name |

A URI may omit `repo` for upstream; the JSON form must name it.
Round-trip a `runtimes list --format uri` row rather than hand-porting one shape
to the other.

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
entry and a `--model` JSON object look alike. See [plan-runner.md](../pipette-ctl/plan-runner.md).

### When you must use JSON

- A value containing `&`, or a `url=` with a query string.
- `auth_token` on the model. There is no URI key for it; prefer
  `PIPETTE_HF_TOKEN` (see [Gated models](usage.md#gated-models)).
- A local path source: `absolute_file` / `absolute_dir` (and the
  `relative_*` forms, which parse but have no store entry).
- A uv/MLX/OpenVINO/ONNX GenAI runtime whose requirements are **not** a catalog row, or one
  carrying `install_flags`. A URI names a catalog preset; an off-catalog body
  has no URI that would round-trip.
- A `vault_*` source, on a model or a runtime. The grammar has no key for the
  principal, and a URI that dropped it would name a public fetch of a private
  artifact. See [Vault sources](#vault-sources).
- An in-app runtime, though the desktop CLI will then refuse to run it.

Note an asymmetry: the URI defaults `flavor` for the docker and MLX schemes, but
the JSON form **requires** it.

```bash
--runtime '{"type":"docker_vllm","image_name":"vllm/vllm-openai","image_tag":"v0.20.2","flavor":"nvidia_gpu"}'
--model '{"type":"gguf_text","source":"absolute_file","path":"/models/m.gguf"}'
```

## Vault sources

A **vault** source names a private artifact the management server serves only
to the devices of the principal that owns it: a customer's own checkpoint, an
internal llama.cpp build, a wheelhouse nobody publishes. There is no token in
the spec and nothing to add. The fetch is signed with the workspace's device
identity, exactly as a job claim is. The route and who is entitled to it are
pipette-mgmt
[`docs/api/vault.md`](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/api/vault.md).

Each fetching arm has a vault twin. All four are JSON-only (see
[above](#when-you-must-use-json)):

| Public arm | Vault arm | Where | Bytes |
|------------|-----------|-------|-------|
| `url` | `vault_url` | `gguf_text`, `gguf_vision`, `gguf_vision_speculative` | one gguf per path |
| `huggingface` | `vault_archive` | `mlx`, `core_ai`, `torch`, `openvino`, `onnx_genai`, and either half of the speculative pairs | one tar, tar.gz or zip whose root is the model directory |
| `remote_archive` | `vault_remote_archive` | `llamacpp_cli_stock_tools`, `ovms`, `trt_edgellm` | the release archive |
| `pip_requirements_text` | `vault_wheelhouse` | every `uv_*` runtime and `mlx_macos_pipette` | an archive of wheels, installed with `--find-links` ahead of the plan's `install_flags` |

Every vault arm carries `url`, exactly `/vault/resources/{resource_id}/content`
**relative to the management server** (an absolute URL is refused, so a plan
cannot point a "vault" fetch at another host), and `principal_id`. The fields
are defined in pipette-mgmt
[`crates/pipette-plan-types/src/location.rs`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/location.rs)
and the source enums beside each type. The model arms' optional `sha256` (or
per-file `*_sha256`) must agree with the record's pin. The optional
`custom_label` identifies nothing on the client: a labeled source is fetched
and stored exactly as the unlabeled one, and reported back unchanged in the
submitted descriptor.

```json
{"type":"gguf_text","source":"vault_url","url":"/vault/resources/vault_0195c3e2-a4f3-7b8c-9d1e-2f3a4b5c6d7e/content","principal_id":"org_01936d7e-0f4b-7c1e-9a2b-3c4d5e6f7a8b"}
```

```toml
[runtime]
type = "uv_vllm"
server_version = "0.22.0"
build = "cu129"
python_version = "3.12"

[runtime.source]
type = "vault_wheelhouse"
contents = "vllm==0.22.0\n"
url = "/vault/resources/vault_0195c3e2-a4f3-7b8c-9d1e-2f3a4b5c6d7e/content"
principal_id = "org_01936d7e-0f4b-7c1e-9a2b-3c4d5e6f7a8b"
install_flags = ["--no-index"]
```

How a fetch runs, and what it refuses:

1. The record (`GET /vault/resources/{id}`) is read first, for the pinned
   `sha256` and the size. A plan `sha256` that disagrees with the pin is
   refused before a byte moves.
2. The bytes stream to the store's staging directory, hashed inline, and are
   compared to the record's pin. A download that arrives complete and hashes
   to something else means the object was replaced behind the record; the
   fetch fails and names the repair (an operator re-registers it).
3. An archive's kind is read off its first bytes, since the path names no file.

The path is joined to the server the workspace is **registered with**. An
unregistered workspace refuses every vault source by name (`pipette auth
register` first). A `403` (the device's
owner does not own it, or the device belongs to nobody), a `404` (no such
resource, or its object was removed), a pin disagreement and drifted bytes are
permanent: the worker submits `retriable: false`, since every device with the
same owner would fail identically. So are verified bytes that are not an
archive, or one that will not unpack. A transport failure, a `5xx`, or an
unpack the disk refused (the quota pre-flight knows only the packed size, so a
tree that does not fit is first met there) is retried as usual.

The store keys a vault artifact on its resource id, the way a `url` source
keys on its URL. A `custom_label` is no part of the key: a labeled and an
unlabeled source for one resource share one entry, whose manifest keeps the
label, or none, of the fetch that created it. A record is immutable, but its
object is not: bytes **replaced** behind a record already in the store are
hidden by the cached entry until it is removed (`models delete` /
`runtimes remove`), and are only noticed as drift by a device fetching fresh.

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
A vault artifact resolves without its `custom_label`: the entry holds the label
of whichever fetch created it, so a run named by digest reports none.

The digest is the SHA-256 of the descriptor's canonical JSON, the same
`runtime_descriptor_sha256` / `model_descriptor_sha256` the management server
records on every result, so the prefix also picks out that artifact's warehouse
rows. The exception is a vault copy of a Hugging Face model, which the server
records under the Hugging Face descriptor. A model's digest is taken with the
auth token stripped, so a token never changes it. `runtimes pull` / `remove`,
`models pull` / `delete` and `benchmarks run` all take the digest form.

Treat a digest as a typing shorthand, not an id to write down. It hashes the
descriptor's exact shape, so every digest moves when `Runtime` or `Model` gains
a field. Plans and job files carry the full object.

## Per-cell flags

Three families tune a cell. All are keyed on the cell itself, and **which
settings are legal depends on the benchmark, the runtime, and the model
together**. A setting the cell does not accept is an error, never a silent
no-op.

| Family | CLI surface | Carries its axes? |
|--------|-------------|-------------------|
| runtime flags | `--runtime-flags '<json object>'` | **no**: derived from the cell |
| model flags | `--model-flags '<json object>'`, or `--model-enable-thinking` | **yes**: `model_type` + `benchmark_type` |
| benchmark flags | `--benchmark-flags '<json object>'`, or the single-setting flags below | **no**: derived from the cell |

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

Which settings each cell accepts is the `RuntimeFlags` enum, one variant per
cell, in pipette-mgmt
[`crates/pipette-plan-types/src/cell/runtime_flags.rs`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/cell/runtime_flags.rs).
A cell with no variant (ONNX GenAI, Core AI, and `mlx_macos_pipette` outside
vl-spec-decode) accepts only the empty `{}`.

`speculative_enabled` is vl-spec-decode's switch between its two runs. Unset,
it follows the model: on for a spec-decoding model, off for a regular one. The
run's resolved value is what the row records. The eval-only settings
`parallel` and `repetition_detection` are in
[Eval methodology](../methodology/evals.md#runtime-details).

`raw` is an array of argv tokens that the client passes verbatim to the
underlying tool, for any setting that the typed fields do not cover.

```bash
--runtime-flags '{"raw":["--numa","distribute"]}'
```

`raw` refuses two kinds of token, per cell: a spelling of a setting that cell
types (`-t`, `-ngl`, ...; set the typed field instead), and a flag the harness
owns there (`--model`, `--port`, the slot cap derived from
`parameter_batch_size`, ...). On vl-spec-decode it also refuses speculation
flags on a run with speculation off, since that row is the denominator of every
speedup joined against it. Matching covers `--flag=value` and glued short forms
such as `-t8`. The lists are in pipette-mgmt
[`crates/pipette-plan-types/src/cell/reserved_flags.rs`](https://github.com/Liquid4All/pipette-mgmt/blob/main/crates/pipette-plan-types/src/cell/reserved_flags.rs).

About secrets: `envs` takes `K=V` or a bare `K` to inherit from the
environment, and env **values are stripped** from the submitted record (except
`GGML_*`, whose values are part of the measurement) while `raw` is submitted
verbatim. Put a token in the bare-`K` form, never in `raw`.

### `--model-flags`

Eval, vlm_eval and asr_eval benchmarks only. Unlike `--runtime-flags`, it
**keeps** the plan's axes. It takes `enable_thinking` on every eval model and on
the vlm_eval and asr_eval `gguf_vision` and `torch` cells (an asr_eval cell runs
only with `false`), and `reasoning_effort` (`low`, `medium` or `high`) on the
eval `gguf_text` and `torch` cells alone:

```bash
--model-flags '{"model_type":"mlx","benchmark_type":"eval","enable_thinking":false}'
--model-enable-thinking false     # same thing, axes derived from the cell
--model-flags '{"model_type":"torch","benchmark_type":"eval","enable_thinking":true,"reasoning_effort":"low"}'
--model-flags '{"model_type":"gguf_vision","benchmark_type":"vlm_eval","enable_thinking":true}'
```

`--model-flags` and `--model-enable-thinking` are mutually exclusive. On an
eval, omitting a knob is not "off": it is simply not sent and the engine's own
default applies. That default is thinking-*on* for `llama-server` and
`mlx_lm` at the catalog versions, and `medium` effort for gpt-oss. A vlm_eval never leaves the mode to
the engine: `enable_thinking` must be its recipe's, and is refused unset on a
model whose template can switch thinking
([VLM eval](../methodology/vlm-eval.md#reasoning-models)). Where each knob goes
on each runtime is in [Eval methodology](../methodology/evals.md#model-flags).

### Benchmark flags

`--benchmark-flags` takes them as one JSON object, the shape a claim's
`benchmark_flags` carries without its axes, which are refused as
`--runtime-flags` refuses them:

```bash
--benchmark-flags '{"readiness":{"max_wait_secs":900,"skip_thermal":true}}'
--readiness-max-wait-secs 900 --readiness-skip-thermal   # same thing
```

The single-setting flags (`--http-timeout-seconds`, `--bench-timeout-seconds`,
`--readiness-max-wait-secs`, `--readiness-skip-thermal`, and
`--readiness-skin-below-c` on the Android CLI) are its shorthand, and the two
forms are mutually exclusive. `pipette benchmarks run --help` says
which cells take each. A setting is accepted only on the cells that have
somewhere to put it; elsewhere it is rejected, naming the cell:

```text
http_timeout_seconds is not a valid benchmark flag for GgufText running PrefillThroughput on LlamacppCliStockTools
```

## Recipes

The smallest run (Gemma GGUF on llama.cpp, with no server and no registration)
is [Your first benchmark](usage.md#your-first-benchmark) in usage.md. The same
Gemma repo has other quants: swap `path` for `-Q4_0.gguf`, `-Q5_K_M.gguf`, or
`-Q8_0.gguf` to compare them. A plan automates that one-axis sweep; see
[plan-runner.md](../pipette-ctl/plan-runner.md).

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
  --model 'openvino://repo=OpenVINO/LFM2.5-350M-int8-ov' \
  --runtime 'uv-openvino://version=2026.4.1' \
  --runtime-flags '{"device":"cpu"}'
```
