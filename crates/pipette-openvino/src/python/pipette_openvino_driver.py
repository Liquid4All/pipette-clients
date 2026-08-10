"""One-shot OpenVINO benchmark driver.

Reads a JSON request line on stdin and does one of two jobs:

* a measured mode (`prefill` / `decode`) -- invoked once per rep, compiles
  exactly one `LLMPipeline`, runs the workload, writes one result object;
* `tokenize` -- invoked once per cell, compiles *no* pipeline, and answers a
  token count per line until stdin closes.

Why one-shot rather than a long-lived server like the MLX driver: compiling
several pipelines in a single process took an Intel NPU down with
`ZE_RESULT_ERROR_DEVICE_LOST`, and throughput came back degraded for several
runs afterwards. One process, one compiled pipeline, is the constraint the
hardware imposes -- see docs/openvino-ir.md. Tokenize mode is compatible with
that constraint precisely because it compiles nothing.

stdout carries result objects and nothing else, so anything OpenVINO or its
telemetry prints has to go to stderr; `emit` is the only writer to stdout.
"""

import json
import sys
import time

ov_genai = None

RESULT_PREFIX = "PIPETTE_RESULT "


def emit(obj):
    """Write one result line, flushed. Prefixed so a stray library write to
    stdout cannot be mistaken for a result, and flushed because tokenize mode's
    caller is waiting on the line before it sends the next candidate."""
    sys.stdout.write(RESULT_PREFIX + json.dumps(obj) + "\n")
    sys.stdout.flush()


def fail(message, kind="error"):
    emit({"ok": False, "kind": kind, "error": message})
    sys.exit(1)


def peak_host_bytes():
    """Peak resident set of *this* process, which is the process holding the
    compiled pipeline and the weights -- so its own high-water mark is the
    measurement, with no cross-process polling to race.

    Returns None rather than guessing when the platform counter is unavailable;
    the caller reports a missing number instead of a wrong one.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # Declaring the signatures is load-bearing on 64-bit: ctypes defaults a
        # return to C int, so the GetCurrentProcess pseudo-handle (-1) would be
        # truncated to 32 bits and GetProcessMemoryInfo would reject it.
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return int(counters.PeakWorkingSetSize) if ok else None

    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KiB on Linux, bytes on macOS.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def seed_tokens(tokenizer, seed):
    """The shared pipette prompt corpus, tokenized without special tokens.

    Every backend builds its synthetic prompts from this same passage, so a
    decode number from OpenVINO is comparable with one from llama.cpp or MLX.

    Arrives in the request rather than the environment: it is ~24 KB against a
    32,767-character Windows ceiling for one variable.
    """
    if not seed:
        fail("the request carried no prompt_seed", kind="prompt")
    ids = tokenizer.encode(seed, add_special_tokens=False).input_ids.data[0]
    if len(ids) == 0:
        fail("the prompt seed tokenized to nothing", kind="prompt")
    return [int(i) for i in ids]


def make_tokens(pool, n):
    """`n` tokens from the seed pool, repeating it when the pool is short.

    Mirrors `make_tokens` in the MLX driver so the two send the same prompt for
    the same token count.
    """
    out = []
    while len(out) < n:
        out.extend(pool[: n - len(out)])
    return out


def tensor_inputs(ids):
    """Token ids as a `TokenizedInputs` batch of one.

    Pre-tokenized on purpose. Handing `generate` a *string* makes it apply the
    model's chat template -- 9 extra tokens for LFM2 -- but prefill and decode
    are raw-continuation measurements in every other pipette backend: MLX feeds
    token ids straight to `stream_generate`, llama.cpp posts to `/completion`
    rather than `/chat/completions`, and only the eval path templates anything.
    Templating here would have measured a different workload under the same
    benchmark id.
    """
    import numpy as np
    import openvino as ov

    batch = np.array([ids], dtype=np.int64)
    return ov_genai.TokenizedInputs(ov.Tensor(batch), ov.Tensor(np.ones_like(batch)))


def tokenized_inputs(pool, n):
    """`n` seed tokens as a `TokenizedInputs` batch of one.

    The prefill/decode/max-memory shape: repeating the seed pool makes the
    count exact by construction, with no convergence search. Matches
    `make_tokens` in the MLX server, which prepares the same three cells the
    same way.
    """
    return tensor_inputs(make_tokens(pool, n))


def run_tokenize(model_dir):
    """Answer token counts for candidate prompts until stdin closes.

    Compiles no `LLMPipeline` -- only the tokenizer, which runs on CPU -- so
    this never touches the NPU and never spends a second pipeline compile in
    one process, the constraint the one-shot driver exists to respect.

    `add_special_tokens=True` because the measured pass encodes the winning
    prompt the same way; a count taken under different settings would describe
    a different prompt. Same accounting the MLX server's `/tokenize` uses.
    """
    try:
        tokenizer = ov_genai.Tokenizer(model_dir)
    except Exception as exc:  # noqa: BLE001
        fail(f"loading the tokenizer from {model_dir} failed: {exc}", kind="tokenizer")
    # The caller blocks on this: it is what tells a bad model directory apart
    # from a prompt that failed to tokenize.
    emit({"ok": True, "ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            text = json.loads(line)["text"]
        except Exception as exc:  # noqa: BLE001
            fail(f"malformed tokenize request: {exc}", kind="tokenize")
        try:
            ids = tokenizer.encode(text, add_special_tokens=True).input_ids.data[0]
        except Exception as exc:  # noqa: BLE001
            fail(f"tokenizing failed: {exc}", kind="tokenize")
        emit({"ok": True, "tokens": int(len(ids))})


def main():
    # One line, not the whole stream: tokenize mode keeps reading after the
    # request. `serde_json` emits compact single-line JSON, so the request
    # always fits one line whatever the prompt seed contains.
    request = json.loads(sys.stdin.readline())
    model_dir = request["model_dir"]
    mode = request["mode"]

    global ov_genai
    try:
        import openvino_genai
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        fail(f"importing openvino_genai failed: {exc}", kind="import")
    ov_genai = openvino_genai

    if mode == "tokenize":
        run_tokenize(model_dir)
        return

    device = request["device"]
    prefill_tokens = int(request.get("prefill_tokens", 0))
    decode_tokens = int(request.get("decode_tokens", 0))
    warmup = request.get("warmup")
    properties = dict(request.get("properties", {}))
    prompt_text = request.get("prompt")

    compile_start = time.perf_counter()
    try:
        pipe = ov_genai.LLMPipeline(model_dir, device, **properties)
    except Exception as exc:  # noqa: BLE001
        fail(f"compiling the pipeline on {device} failed: {exc}", kind="compile")
    compile_s = time.perf_counter() - compile_start

    if mode == "compile":
        # Only the blob mattered. The zeros keep one result shape for the
        # caller's parser; nothing reads them.
        emit(
            {
                "ok": True,
                "device": device,
                "compile_s": compile_s,
                "wall_ms": 0.0,
                "input_tokens": 0,
                "generated_tokens": 0,
                "ttft_ms": 0.0,
            }
        )
        return

    tokenizer = pipe.get_tokenizer()
    # Only when something needs it: encoding the ~24 KB seed is per-rep work,
    # and a text cell with no warm-up needs no pool at all.
    pool = (
        seed_tokens(tokenizer, request.get("prompt_seed"))
        if prompt_text is None or warmup
        else None
    )
    # A text prompt is encoded inside the timed region below, so tokenization
    # counts toward the latency the way it does for a real caller. Cells that
    # send no text measure raw continuation from seed ids, prepared up front.
    prompt = None if prompt_text is not None else tokenized_inputs(pool, prefill_tokens)

    config = ov_genai.GenerationConfig()
    config.do_sample = False
    # `mode` decides how much of the pipeline is timed. Prefill still has to
    # generate one token -- there is no prefill-only entry point -- so the
    # reported prefill time comes from the metrics, not the wall clock.
    config.max_new_tokens = 1 if mode == "prefill" else max(decode_tokens, 1)
    if mode == "prefill":
        config.ignore_eos = True
    else:
        # Without this a short generation stops early and the rep reports fewer
        # tokens than the cell asked for, which the harness then rejects.
        config.ignore_eos = True

    if warmup:
        # The cell's own shape, sent explicitly rather than reused from
        # prefill_tokens/decode_tokens so the driver never has to guess: kernel
        # selection is shape-keyed, so a lighter rehearsal would leave that cost
        # inside the measured region below.
        warmup_config = ov_genai.GenerationConfig()
        warmup_config.do_sample = False
        warmup_config.ignore_eos = True
        warmup_config.max_new_tokens = max(int(warmup["decode_tokens"]), 1)
        warmup_prompt = tokenized_inputs(pool, max(int(warmup["prefill_tokens"]), 1))
        try:
            pipe.generate(warmup_prompt, warmup_config)
        except Exception as exc:  # noqa: BLE001
            fail(f"warmup generate failed: {exc}", kind="generate")

    start = time.perf_counter()
    if prompt_text is not None:
        # Pre-encoded rather than handed to `generate` as a string: a string
        # makes GenAI apply the chat template, which would add tokens the cell
        # did not ask for. Encoding here keeps the template out and the
        # tokenization in.
        try:
            encoded = tokenizer.encode(prompt_text, add_special_tokens=True)
            ids = encoded.input_ids.data[0]
        except Exception as exc:  # noqa: BLE001
            fail(f"encoding the prompt failed: {exc}", kind="tokenize")
        prompt = tensor_inputs([int(i) for i in ids])
    try:
        result = pipe.generate(prompt, config)
    except Exception as exc:  # noqa: BLE001
        fail(f"generate failed: {exc}", kind="generate")
    wall_s = time.perf_counter() - start

    metrics = result.perf_metrics
    payload = {
        "ok": True,
        "device": device,
        "compile_s": compile_s,
        "wall_ms": wall_s * 1000.0,
        "input_tokens": int(metrics.get_num_input_tokens()),
        "generated_tokens": int(metrics.get_num_generated_tokens()),
        "ttft_ms": float(metrics.get_ttft().mean),
        "peak_host_bytes": peak_host_bytes(),
    }
    # TPOT is undefined for a single generated token, and OpenVINO reports a
    # placeholder rather than erroring. Omit it instead of forwarding a number
    # that means nothing.
    if payload["generated_tokens"] > 1:
        payload["tpot_ms"] = float(metrics.get_tpot().mean)
        payload["throughput_tps"] = float(metrics.get_throughput().mean)
    emit(payload)


if __name__ == "__main__":
    main()
