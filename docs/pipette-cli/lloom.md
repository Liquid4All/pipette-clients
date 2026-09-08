# lloom-serve runtime

Backend library: `pipette-lloom`, driven by the unified `pipette` CLI.

Runs **Apple Silicon** benchmarks against [`lloom-serve`](https://github.com/ocean1/lloom),
the LFM2 engine on Metal, as a short-lived local sidecar: a native binary, no
venv, no Python. Shared operator flow: [usage.md](usage.md). Notation:
[models-and-runtimes.md](models-and-runtimes.md).

## Differences from MLX

| | lloom-serve | MLX |
|---|-----|-----------|
| Runtime artifact | Prebuilt archive with one binary, `lloom-serve` | Python venv + locked `mlx-lm` |
| Install | `runtimes pull` or auto-fetch on first `benchmarks run` (shared store), the llama.cpp archive path | `runtimes pull` or auto-fetch (uv) |
| Models | HF safetensors snapshot (`torch://…`) in shared `models/` | MLX repo snapshot in shared `models/` |
| Sidecar contract | the same endpoints, plus `/tokenize` with `add_special_tokens` and `/v1/completions` | `pipette_mlx_server.py` |
| Host | Apple Silicon only | Apple Silicon only |

## Runtime

```bash
pipette runtimes pull --runtime 'lloom-serve-macos://url=example.com/lloom-serve-macos-arm64.tar.gz&flavor=macos-arm64'
pipette runtimes list
```

`url` is a host/path coordinate (downloaded over `https://`, no scheme in the
value; a query string needs the JSON form). The archive may lay the binary out
however it likes: the runner finds `lloom-serve` anywhere under the unpacked
root. `flavor` defaults to `macos-arm64`, the one build there is.

The JSON form, for plans and for a query-string URL:

```json
{ "type": "lloom_serve_macos", "source": "remote_archive",
  "url": "example.com/lloom-serve-macos-arm64.tar.gz", "flavor": "macos-arm64" }
```

## Model

```bash
--model 'torch://repo=LiquidAI/LFM2.5-2.6B'
# optional: &rev=…
```

`pipette` materializes the HF snapshot into the shared `models/` store and
starts the server with that **local directory** (`lloom-serve --model-dir`).
The directory needs `config.json`, `tokenizer.json` and the safetensors shards;
the checkpoint's `chat_template.jinja` is what `eval` renders with.

## Benchmarks

Supported: `prefill_throughput`, `decode_throughput`, `end_to_end_latency`,
`max_memory_usage`, `eval`. Not supported: `vl_throughput`.

```bash
pipette benchmarks run \
  --benchmark local/decode_throughput_512_100 \
  --model 'torch://repo=LiquidAI/LFM2.5-2.6B' \
  --runtime 'lloom-serve-macos://url=example.com/lloom-serve-macos-arm64.tar.gz'
```

What each cell brackets, on the server's side (`lloom-engine/README.md` in
the lloom repository states it in full):

- `prefill_throughput`: one forward over `P` seed tokens on a fresh cache,
  submitted and drained; `prompt_tps` from the server's own clock.
- `decode_throughput`: the prompt is prefilled and drained outside the
  bracket; `D` forwards and samples on the ring readback are timed to the last
  token landing — `llama-bench --n-depth P --n-gen D`'s shape.
- `end_to_end_latency`: a text prompt built to exactly `P` tokens through
  `/tokenize` (special tokens counted as the request path adds them); the
  runner's HTTP bracket is the figure, EOS suppressed, exact counts validated.
- `max_memory_usage`: the runner polls `phys_footprint` on the server process
  through load, a `P`-token prefill and one decode step, with the Metal shim
  attached for the diagnostic device-side peak.
- `eval`: chat messages rendered with the checkpoint's own template, streamed
  as JSONL; the doom-loop check aborts a sample through `/eval/abort`;
  `finish_reason` and `completion_tokens` from the server classify the stop.

## Runtime flags

None: the server takes no per-cell flags. `runtime_flags` on a lloom-serve
cell are ignored with a warning.
