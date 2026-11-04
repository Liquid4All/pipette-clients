# ONNX GenAI measurement

This page describes how pipette measures `uv_onnx_genai` cells. Operator setup
is in [pipette-cli/onnx-genai.md](pipette-cli/onnx-genai.md), and a plan that
runs every cell is
[examples/plans/onnx-genai-cpu.toml](../examples/plans/onnx-genai-cpu.toml).

A catalog runtime is one row of the [bundled catalog](../crates/pipette-artifacts-catalog/bundled-catalog/onnx_genai/README.md):
an `onnxruntime-genai` version and an execution provider, `cpu` or `cuda`,
installed with its pinned set into a uv venv of its own. A structural JSON
runtime can name another provider ([Model validation](#model-validation)).
Each execution provider is a separate wheel, so the provider is fixed when the
runtime installs. Each cell runs one server process that holds the model. Each
request builds its own generator and key-value (KV) cache.

## Timing and token accounting

Each cell runs one in-house OpenAI-compatible server
(`pipette_onnx_genai_server.py`), loaded before
it listens. The timing cells are measured like every other server-driven
engine, on the client's clock: prefill is the time to the first streamed token,
decode the first-to-last-token interval scaled from `decode_tokens - 1` to
`decode_tokens`, and end-to-end one streamed request, from the send to its
finish chunk. One untimed request at the cell's own shape precedes the five
repetitions. The load stays outside every request. A request waits up to 3600
seconds for its headers, then for each chunk, unless the plan pins
`http_timeout_seconds`; the bound permits a long CPU prefill. The server
generates one sequence at a time, so the timing cells refuse a batched entry.

Every cell sends text sized through the server's `/tokenize`. The server encodes
a string prompt with `Tokenizer.encode`. It keeps whatever special tokens the
tokenizer adds, such as a beginning-of-sequence (BOS) token, and applies no
chat template. `/tokenize` uses the same function, so the sized prompt is the
prompt that `/v1/completions` counts. MLX and OpenVINO count a string prompt the
same way.

With `ignore_eos`, both search bounds equal `len(ids) + max_tokens`: GenAI's
`min_length` and `max_length` count the prompt too, and a minimum equal to the
maximum forces the requested output count despite the end-of-sequence (EOS)
token. The server generates token by token. It decodes each token through
`og.TokenizerStream` and streams one chunk per token as soon as the token
exists, with or without text. The last chunk carries `finish_reason = "length"`. `usage.prompt_tokens` is `len(ids)` and
`usage.completion_tokens` counts the server's own loop. The Rust harness rejects
unexpected counts, a finish reason other than `"length"`, and nonpositive or
nonfinite timing samples. The server speaks HTTP/1.1 and keeps a connection open
between requests, so a client that reuses its connection pays no new connect.

## Host memory

The client watches the server process from its launch with the per-OS sampler
every other engine uses: `VmHWM` on Linux, with the swap term beside it,
`phys_footprint` on macOS, and PSAPI's `PeakWorkingSetSize` on Windows. Every
successful cell records that peak as `observation_max_host_bytes`. The memory
cell sends one request at its prompt size with one output token, then uses the
same peak as its primary metric, and refuses a sampler that read nothing.

The memory cell first runs the same request on a warm-up server of its own and
drops that reading. The measured server starts after the warm-up's stops. The
first import after a runtime install writes the venv's Python bytecode, and that
compile runs inside the server. For Phi-3-mini int4 at `max_memory_usage_512` on
an M3 Max, a server without that bytecode read 6 to 8 MB above one with it:
2,563 to 2,566 MB against 2,556 to 2,557 MB. With the warm-up, both read
2,554 to 2,555 MB. On a Linux Xeon the gap sat inside
the 4.5 MB spread between runs, and evicting the model from the page cache moved
nothing. The CUDA provider has not been run for it.

The peak includes model load, every request the server has answered (including
prompt sizing) and the measured workload. Server rows whose `client_version` is
dated before 2026-10-02 carry the server's own `ru_maxrss` instead of the
sampler's figure. On macOS, `ru_maxrss` also counts resident file-backed pages,
so it reads above `phys_footprint`
([Per-repetition driver results](#per-repetition-driver-results)).

The CPU provider allocates host memory only, so its `max_gpu_bytes` stays
empty. A GPU provider's memory comes from the OS as `max_gpu_bytes`, and the
memory cell fails if the GPU shows none. How each OS reads it is in
[peak-memory-linux.md](methodology/peak-memory-linux.md#ovms-openvino-and-onnx-genai)
and [peak-memory-windows.md](methodology/peak-memory-windows.md#gpu-and-npu-counter):

- CUDA on Linux: NVML (the NVIDIA Management Library), the bytes that the
  driver attributes to the server's process.
- CUDA and DirectML on Windows: the per-adapter billing of WDDM (the Windows
  Display Driver Model).
- A custom provider on Windows picks its own device. pipette records whatever
  GPU or NPU memory it holds, and requires none.
- A custom provider on macOS: macOS has unified memory, so the host figure is
  the whole cost.

pipette refuses a provider whose device memory nothing reads before the server
starts: CUDA and DirectML on macOS, because the host figure cannot show their
GPU, and a custom provider on Linux. pipette watches DirectML on Linux as a
GPU, so its cell fails when the server cannot load it.

On Windows a uv venv's `python.exe` is a launcher: it starts the base
interpreter as a child process and itself holds a few MiB. So the client does
what the launcher does: it runs `python.exe` from the `home` directory that the
venv's `pyvenv.cfg` names, with `__PYVENV_LAUNCHER__` set to the venv's
`python.exe` so the interpreter still uses the venv. The sampler then watches
the server's own process, as on Linux and macOS, where `bin/python` is a
symlink to the interpreter. A Windows row from one release of
2026-10-02 has a memory figure of about 5 MiB, because its sampler watched the
launcher. Do not use such a row. The same holds for `uv_openvino` rows.

## Model validation

Only `genai_config.json` is a fixed required filename. The decoder graph name comes from
`model.decoder.filename`, and must refer to a file. The validator reads only these
keys and ignores the rest of the config, so a config that adds keys still validates.

The builder adds `.data` to the graph name for the external weights file. An existing
sibling must be a nonempty file. If it is missing and the graph is at most 64 MiB, the
validator searches the graph bytes for the sibling's literal filename. A reference
without a file indicates incomplete external weights. A small self-contained graph with
no such reference passes; size alone never requires a sidecar.

This declaration scan requires few resources. It is not a protobuf parser or an
integrity check. It can miss an incomplete graph that ends before its location string.
It can also miss nonstandard external-data names and declarations in graphs above the
scan limit. A byte string that matches outside the external-data metadata can also
trigger the scan.

Loader errors remain responsible for those formats and malformed protobufs. Only a
Hugging Face LFS file in a directory model is checked against a digest, so this check
catches a useful partial-fetch case: a complete small graph whose multi-gigabyte
sibling never arrived.

The client maps the runtime's provider to GenAI's own name: DirectML is `dml`, since
GenAI refuses `directml`; in a runtime, `directml` names the
`onnxruntime-genai-directml` wheel. The server hands that name to `append_provider`, and
the validator reads the export's lowercased provider keys back through the same map. An
empty `provider_options` means CPU. A named runtime provider must occur in the declared
set, so CPU/CUDA mismatches fail in both directions and multi-provider lists use
membership. Custom runtime providers skip this check because pipette does not own their
spelling. The CPU export against the CUDA runtime, for example, stops before the model
loads:

```text
ONNX GenAI provider mismatch: the model declares [cpu], but the runtime uses `cuda`.
```

The client validates the export's `genai_config.json` on disk; the server clears the
provider list only in the config it loads, so the export declaration stays intact for
this check.

Windows ML is refused on every OS before any cell starts a server: GenAI's
`append_provider` takes no `winml` name. For DirectML, a runtime sets
`execution_provider` to `directml` and `server_version` to `0.13.1`, and pins
`onnxruntime-genai-directml==0.13.1`, the newest DirectML wheel that installs from PyPI,
in `source.contents`. A custom `dml`, in any case, is refused too and pointed at
`directml`: it would skip the provider check and record under another spelling.

## The CUDA capability rule

The planner of the management server (the `pipette-planner` crate in
pipette-mgmt) gives a CUDA cell `accel:gpu` and a CPU cell nothing (its
capability rules; matching is in
[docs/plan-ingestion.md §5](https://github.com/Liquid4All/pipette-mgmt/blob/main/docs/plan-ingestion.md#5-capability-matching)).
`accel:gpu` also matches a Linux host whose GPU is AMD and a Windows host without
an NVIDIA card; there the CLI's GPU preflight stops the cell before install
([The CUDA library set](#the-cuda-library-set)).

No client reports `os:linux` or `os:windows`: a Linux client's `os:` names its
distribution (`os:ubuntu`) and a Windows client's its edition
(`os:microsoftwindows11pro`), so a plan that targets one platform requires that
name, or a tag.

DirectML and WinML have no capability rule, so a client that cannot serve a
cell on a structural JSON DirectML runtime can claim it, and the cell stops at
install. Give such a cell a `requires` as above.

## The CUDA library set

The CUDA rows pin `onnxruntime-gpu` 1.30.0 or later (1.31.1 on the 0.18.0
row), which builds against CUDA 13. On a host with driver 580.105.08 and a CUDA
12.8 system toolkit, a row that pinned only the wrapper and the engine failed at
model load:

```text
loading the model on cuda failed: Cuda interface not available: Failed to load
library: libcublasLt.so.13: cannot open shared object file: No such file or directory
```

`onnxruntime-gpu` keeps the CUDA libraries behind its `[cuda]` and `[cudnn]`
extras. Each CUDA row therefore resolves as
`onnxruntime-gpu[cuda,cudnn]==<pinned version>` and pins the seven `nvidia-*`
packages that the extras add, about 1.2 GB of wheels.
The venv then supplies the libraries, so two CUDA hosts measure the same stack
whatever system toolkit each one holds.

The pins alone do not load. Those libraries install under site-packages, which
the dynamic linker does not search, so the server calls
`onnxruntime.preload_dlls()` before it loads a CUDA model. The same load then
succeeds.

CUDA 13 libraries need NVIDIA driver 580 or newer. Before it installs a CUDA row,
a cell run (`pipette benchmarks run` or a worker claim) checks `nvidia-smi -L` and the driver version, and refuses a host
with no `nvidia-smi` or an older driver (`assert_cuda_13_driver` in
`crates/pipette-cli/src/preflight.rs`). `pipette runtimes pull` runs no GPU
preflight, so a pull of the CUDA row succeeds on a host that cannot run it.

## Per-repetition driver results

These rows came from a per-repetition driver: each repetition was a fresh
driver process that timed `append_tokens` and the generator loop itself. Rows
whose `client_version` is dated before 2026-09-25 were measured that way. Do
not compare them with rows taken through the server, whose numbers include the
HTTP exchange and the server-side tokenization. Each used Phi-3-mini int4
([CPU export](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx/tree/main/cpu_and_mobile/cpu-int4-rtn-block-32),
or its CUDA variant on the CUDA host), runtime 0.16.0, five repetitions per
timing cell and one memory pass.

Their memory figures are the process's own `ru_maxrss`, not the per-OS sampler.
On Linux that is the same figure, but on macOS it also counts resident
file-backed pages, the library code and the parts of the model ONNX Runtime
reads in place from its mapped file, which `phys_footprint` leaves out. For
Phi-3-mini int4 on an M3 Ultra that is 2.75 GB against 2.26 GB in one server
process.

### Linux CPU results

Linux x86_64 (Intel Core i9-13900HK), CPU row, default readiness gate.

| Benchmark | Metric | Host peak observation (bytes) |
| --- | --- | --- |
| `prefill_throughput_512` | 5,741.919 ms | 3,150,290,944 |
| `decode_throughput_512_100` | 6,720.616 ms | 3,227,291,648 |
| `end_to_end_latency_512_256` | 23,219.571 ms | 3,352,260,608 |
| `max_memory_usage_512` | 3,087,749,120 bytes | 3,087,749,120 |

These correspond to about 89.17 prefill tokens/s and 14.88 decode tokens/s.

### macOS CPU results

Mac Studio (32-core Apple M3 Ultra, 512 GiB RAM, macOS 26.6.2), CPU row,
default readiness gate.

| Benchmark | Metric | Host peak observation (bytes) |
| --- | --- | --- |
| `prefill_throughput_512` | 2,493.932 ms | 3,138,158,592 |
| `decode_throughput_512_100` | 4,584.530 ms | 3,217,080,320 |
| `end_to_end_latency_512_256` | 14,379.651 ms | 3,343,777,792 |
| `max_memory_usage_512` | 3,082,436,608 bytes | 3,082,436,608 |

These correspond to about 205.30 prefill tokens/s and 21.81 decode tokens/s.

As a sanity check of the order of magnitude, on an Apple M4 Max (not the
M3 Ultra above), `llama-bench` b9305 with
Microsoft's `Phi-3-mini-4k-instruct-q4.gguf` (Q4_K_M) on an Apple M4 Max, CPU
only, reported 176.49 prefill tokens/s against about 83 for an early ONNX CPU
probe on the same machine. Different quantizers, threading defaults and
background desktop activity prevent an exact comparison.

```bash
llama-bench -m Phi-3-mini-4k-instruct-q4.gguf -p 512 -n 0 -ngl 0 -t 8 -r 3 -o json
```

### Linux CUDA results

Ubuntu 22.04 with one NVIDIA H100 80GB HBM3 (Intel Xeon Platinum 8480+, 221 GiB
RAM, driver 580.105.08, kernel 6.8.0-1046-nvidia), CUDA row, the
`cuda/cuda-int4-rtn-block-32` Phi-3 variant. This host exposes no
`/sys/class/thermal/thermal_zone*`, so each timing cell ran with
`--readiness-skip-thermal`; the load criterion still applied. These numbers are
therefore not comparable with the thermally gated CPU rows above.

| Benchmark | Metric | Host peak observation (bytes) |
| --- | --- | --- |
| `prefill_throughput_512` | 12.707 ms | 1,066,594,304 |
| `decode_throughput_512_100` | 297.506 ms | 1,115,820,032 |
| `end_to_end_latency_512_256` | 771.953 ms | 1,105,174,528 |
| `max_memory_usage_512` | 1,089,802,240 bytes | 1,089,802,240 |

These correspond to about 40,291 prefill tokens/s and 336.13 decode tokens/s.
Prefill is about 452 times the i9-13900HK CPU rate and about 196 times the M3 Ultra
rate. That gap is the evidence that the CUDA execution provider carried the work
rather than a silent CPU fallback. The host peak is about one third of the CPU
rows, because the weights sit in GPU memory; the card's own memory is not in
these rows.
