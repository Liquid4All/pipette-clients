"""Purpose-built HTTP server for pipette-mlx benchmarks."""

import argparse
import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import TCPServer

import mlx_lm
from mlx_lm.sample_utils import make_sampler

logging.basicConfig(
    level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s"
)
logger = logging.getLogger("pipette-mlx-server")

PROMPT_SEED_TEXT_ENV = "PIPETTE_MLX_PROMPT_SEED_TEXT"
GREEDY_SAMPLER = make_sampler(temp=0.0)
EVAL_CHUNK_FLUSH_BYTES = 256


class BadRequest(ValueError):
    pass


class LocalHTTPServer(HTTPServer):
    """A loopback-only server that keeps MLX work on its load thread.

    MLX Metal streams are thread-local.  The model is loaded immediately
    before ``serve_forever`` on this thread, so request handlers must execute
    there too.  A ``ThreadingHTTPServer`` would dispatch each request to a
    worker thread where the model's GPU stream is unavailable.
    """

    def server_bind(self):
        TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


def load_model(repo_id):
    logger.info("loading model %s", repo_id)
    model, tokenizer = mlx_lm.load(
        repo_id, tokenizer_config={"trust_remote_code": True}
    )
    return model, tokenizer


def prompt_seed_text():
    seed = os.environ.get(PROMPT_SEED_TEXT_ENV)
    if seed is None:
        raise RuntimeError(f"{PROMPT_SEED_TEXT_ENV} must be set")
    if not seed:
        raise RuntimeError(f"{PROMPT_SEED_TEXT_ENV} must not be empty")
    return seed


def tokenize_seed(tokenizer, seed_text):
    tokens = tokenizer.encode(seed_text, add_special_tokens=False)
    tokens = [int(token) for token in tokens]
    if not tokens:
        raise RuntimeError("prompt seed tokenized to an empty token list")
    logger.info("tokenized prompt seed to %d tokens", len(tokens))
    return tokens


def make_tokens(seed_tokens, n):
    out = []
    while len(out) < n:
        remaining = n - len(out)
        out.extend(seed_tokens[:remaining])
    return out


def _suppress_eos(tokenizer):
    tokenizer._eos_token_ids = set()


def parse_json_body(handler):
    try:
        length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise BadRequest("Content-Length must be an integer") from exc
    if length < 0:
        raise BadRequest("Content-Length must be non-negative")

    raw = handler.rfile.read(length)
    try:
        request = json.loads(raw.decode("utf-8") if raw else "{}")
    except UnicodeDecodeError as exc:
        raise BadRequest("request body must be UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise BadRequest(str(exc)) from exc

    if not isinstance(request, dict):
        raise BadRequest("request body must be a JSON object")
    return request


def require_string(request, field):
    value = request.get(field)
    if not isinstance(value, str):
        raise BadRequest(f"field '{field}' must be a string")
    return value


def optional_positive_int(request, field, default):
    value = request.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BadRequest(f"field '{field}' must be a positive integer")
    return value


def require_list(request, field):
    value = request.get(field)
    if not isinstance(value, list):
        raise BadRequest(f"field '{field}' must be a list")
    return value


def optional_nullable_bool(request, field):
    value = request.get(field)
    if value is not None and not isinstance(value, bool):
        raise BadRequest(f"field '{field}' must be null or boolean")
    return value


def optional_temperature(request, field):
    # Sampling temperature: absent or 0.0 means greedy (the prior
    # hardcoded behavior); a positive value samples. Accept int or float,
    # reject bool (a JSON `true` is not a temperature). Bounded to
    # [0.0, 2.0] to match the client's `Temperature` type — the client is
    # the source of truth, but mirroring the bound keeps the contracts
    # from silently diverging.
    value = request.get(field, 0.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 <= value <= 2.0
    ):
        raise BadRequest(f"field '{field}' must be a number in [0.0, 2.0]")
    return float(value)


def stream_generate_last_response(model, tokenizer, tokens, max_tokens):
    resp = None
    # `resp` intentionally holds the final yielded value after the loop.
    for resp in mlx_lm.stream_generate(  # noqa: B007
        model,
        tokenizer,
        tokens,
        max_tokens=max_tokens,
        sampler=GREEDY_SAMPLER,
    ):
        pass
    if resp is None:
        raise RuntimeError("stream_generate yielded no tokens")
    return resp


def run_prefill_throughput(model, tokenizer, seed_tokens, prompt_tokens):
    """Run one prefill measurement. The client drives repetition with a
    readiness wait between requests, so this endpoint deliberately runs
    a single trial per call. Warmup is the client's responsibility — it
    issues one extra request before the measurement loop to prime
    kernel caches."""
    _suppress_eos(tokenizer)
    tokens = make_tokens(seed_tokens, prompt_tokens)

    logger.info("prefill_throughput measurement prompt_tokens=%d", prompt_tokens)
    resp = stream_generate_last_response(model, tokenizer, tokens, max_tokens=1)
    return {
        "prompt_tps": float(resp.prompt_tps),
        "prompt_tokens": prompt_tokens,
    }


def run_decode_throughput(model, tokenizer, seed_tokens, prompt_tokens, decode_tokens):
    """Run one decode measurement. The client drives repetition with a
    readiness wait between requests, so this endpoint deliberately runs
    a single trial per call. Warmup is the client's responsibility (see
    `run_prefill_throughput`)."""
    _suppress_eos(tokenizer)
    tokens = make_tokens(seed_tokens, prompt_tokens)

    logger.info(
        "decode_throughput measurement prompt_tokens=%d decode_tokens=%d",
        prompt_tokens,
        decode_tokens,
    )
    resp = stream_generate_last_response(
        model, tokenizer, tokens, max_tokens=decode_tokens
    )
    return {
        "generation_tps": float(resp.generation_tps),
        "decode_tokens": decode_tokens,
    }


def run_max_memory_usage(model, tokenizer, seed_tokens, prompt_tokens, decode_tokens):
    _suppress_eos(tokenizer)
    tokens = make_tokens(seed_tokens, prompt_tokens)

    logger.info(
        "max_memory_usage prompt_tokens=%d decode_tokens=%d",
        prompt_tokens,
        decode_tokens,
    )
    resp = stream_generate_last_response(
        model,
        tokenizer,
        tokens,
        max_tokens=decode_tokens,
    )
    return {
        "prompt_tokens": int(resp.prompt_tokens),
        "completion_tokens": completion_token_count(resp),
    }


def completion_token_count(resp):
    completion_tokens = getattr(resp, "completion_tokens", None)
    if completion_tokens is None:
        completion_tokens = resp.generation_tokens
    return int(completion_tokens)


def run_end_to_end_latency(model, tokenizer, prompt, decode_tokens):
    """Run one end-to-end latency measurement. The client drives
    repetition with a readiness wait between requests."""
    _suppress_eos(tokenizer)

    logger.info("end_to_end_latency measurement decode_tokens=%d", decode_tokens)
    started = time.monotonic()
    resp = stream_generate_last_response(
        model,
        tokenizer,
        prompt,
        max_tokens=decode_tokens,
    )
    total_ms = (time.monotonic() - started) * 1000.0
    return {
        "total_ms": total_ms,
        "prompt_tokens": int(resp.prompt_tokens),
        "completion_tokens": completion_token_count(resp),
    }


def run_eval(
    model,
    tokenizer,
    samples,
    max_tokens,
    done_ids,
    emit_event,
    should_abort,
    clear_abort,
    enable_thinking=None,
    temperature=0.0,
):
    done = set(done_ids)
    total = len(samples)
    # Greedy (temp=0.0) reuses the shared sampler; a positive temperature
    # builds a sampling sampler. No seed is set, so each call — including
    # IFBench's repeated `#k` ids — is an independent draw.
    sampler = GREEDY_SAMPLER if temperature == 0.0 else make_sampler(temp=temperature)
    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        if sample_id in done:
            logger.info(
                "eval %d/%d id=%s (skipped, already checkpointed)",
                i + 1,
                total,
                sample_id,
            )
            continue
        messages = sample["messages"]

        # Only forward `enable_thinking` when the caller set it explicitly.
        # Leaving it absent preserves mlx_lm's tokenizer-derived default.
        tmpl_kwargs = {}
        if enable_thinking is not None:
            tmpl_kwargs["enable_thinking"] = bool(enable_thinking)
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, **tmpl_kwargs
        )
        if isinstance(prompt, list):
            prompt = tokenizer.decode(prompt)

        logger.info("eval %d/%d id=%s", i + 1, total, sample_id)
        emit_event(
            {
                "kind": "eval_sample_start",
                "sample_id": sample_id,
                "prompt": prompt,
            }
        )

        text = ""
        last_emitted_len = 0
        for resp in mlx_lm.stream_generate(
            model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
        ):
            # mlx_lm yields resp.text as the per-token delta, not cumulative
            # text, so accumulate here before emitting JSONL chunks.
            text += resp.text
            if len(text) - last_emitted_len >= EVAL_CHUNK_FLUSH_BYTES:
                emit_event(
                    {
                        "kind": "eval_sample_chunk",
                        "sample_id": sample_id,
                        "delta": text[last_emitted_len:],
                    }
                )
                last_emitted_len = len(text)
            if should_abort(sample_id):
                break

        stopped_early = should_abort(sample_id)
        if stopped_early:
            clear_abort(sample_id)
        logger.info(
            "eval %d/%d id=%s %s completion=%s",
            i + 1,
            total,
            sample_id,
            "stopped_early" if stopped_early else "done",
            text,
        )
        emit_event(
            {
                "kind": "eval_sample_done",
                "sample_id": sample_id,
                "completion": text,
                "stopped_early": stopped_early,
            }
        )

    emit_event({"kind": "eval_done"})


def make_handler(model, tokenizer, seed_tokens):
    model_lock = threading.Lock()
    abort_lock = threading.Lock()
    abort_ids = set()

    def request_abort(sample_id):
        with abort_lock:
            abort_ids.add(sample_id)

    def should_abort(sample_id):
        with abort_lock:
            return sample_id in abort_ids

    def clear_abort(sample_id):
        with abort_lock:
            abort_ids.discard(sample_id)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "pipette_mlx_server/0.1"

        def _write_json(self, status, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _start_jsonl_stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()

        def _write_jsonl_event(self, event):
            payload = (json.dumps(event) + "\n").encode("utf-8")
            self.wfile.write(f"{len(payload):x}\r\n".encode("ascii"))
            self.wfile.write(payload)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

        def _finish_jsonl_stream(self):
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def do_GET(self):
            if self.path == "/health":
                self._write_json(200, {})
                return
            self._write_json(404, {"error": f"unknown endpoint: {self.path}"})

        def do_POST(self):
            if self.path == "/tokenize":
                self._handle_tokenize()
                return
            if self.path == "/prefill_throughput":
                self._handle_prefill_throughput()
                return
            if self.path == "/decode_throughput":
                self._handle_decode_throughput()
                return
            if self.path == "/max_memory_usage":
                self._handle_max_memory_usage()
                return
            if self.path == "/end_to_end_latency":
                self._handle_end_to_end_latency()
                return
            if self.path == "/eval":
                self._handle_eval()
                return
            if self.path == "/eval/abort":
                self._handle_eval_abort()
                return
            if self.path == "/shutdown":
                self._handle_shutdown()
                return

            self._write_json(404, {"error": f"unknown endpoint: {self.path}"})

        def _handle_tokenize(self):
            try:
                request = parse_json_body(self)
                prompt = require_string(request, "prompt")
            except BadRequest as exc:
                logger.info("bad /tokenize request: %s", exc)
                self._write_json(400, {"error": str(exc)})
                return

            try:
                with model_lock:
                    tokens = tokenizer.encode(prompt, add_special_tokens=True)
                tokens = [int(token) for token in tokens]
            except Exception as exc:
                logger.exception("failed to tokenize prompt")
                self._write_json(500, {"error": str(exc)})
                return

            self._write_json(200, {"tokens": tokens, "count": len(tokens)})

        def _handle_prefill_throughput(self):
            try:
                request = parse_json_body(self)
                prompt_tokens = optional_positive_int(request, "prompt_tokens", None)
            except BadRequest as exc:
                logger.info("bad /prefill_throughput request: %s", exc)
                self._write_json(400, {"error": str(exc)})
                return

            try:
                with model_lock:
                    result = run_prefill_throughput(
                        model,
                        tokenizer,
                        seed_tokens,
                        prompt_tokens,
                    )
            except Exception as exc:
                logger.exception("failed to run prefill_throughput")
                self._write_json(500, {"error": str(exc)})
                return

            self._write_json(200, result)

        def _handle_decode_throughput(self):
            try:
                request = parse_json_body(self)
                prompt_tokens = optional_positive_int(request, "prompt_tokens", None)
                decode_tokens = optional_positive_int(request, "decode_tokens", None)
            except BadRequest as exc:
                logger.info("bad /decode_throughput request: %s", exc)
                self._write_json(400, {"error": str(exc)})
                return

            try:
                with model_lock:
                    result = run_decode_throughput(
                        model,
                        tokenizer,
                        seed_tokens,
                        prompt_tokens,
                        decode_tokens,
                    )
            except Exception as exc:
                logger.exception("failed to run decode_throughput")
                self._write_json(500, {"error": str(exc)})
                return

            self._write_json(200, result)

        def _handle_max_memory_usage(self):
            try:
                request = parse_json_body(self)
                prompt_tokens = optional_positive_int(request, "prompt_tokens", None)
                decode_tokens = optional_positive_int(request, "decode_tokens", None)
                if decode_tokens != 1:
                    raise BadRequest("field 'decode_tokens' must be 1")
            except BadRequest as exc:
                logger.info("bad /max_memory_usage request: %s", exc)
                self._write_json(400, {"error": str(exc)})
                return

            try:
                with model_lock:
                    result = run_max_memory_usage(
                        model,
                        tokenizer,
                        seed_tokens,
                        prompt_tokens,
                        decode_tokens,
                    )
            except Exception as exc:
                logger.exception("failed to run max_memory_usage")
                self._write_json(500, {"error": str(exc)})
                return

            self._write_json(200, result)

        def _handle_end_to_end_latency(self):
            try:
                request = parse_json_body(self)
                prompt = require_string(request, "prompt")
                decode_tokens = optional_positive_int(request, "decode_tokens", None)
            except BadRequest as exc:
                logger.info("bad /end_to_end_latency request: %s", exc)
                self._write_json(400, {"error": str(exc)})
                return

            try:
                with model_lock:
                    result = run_end_to_end_latency(
                        model,
                        tokenizer,
                        prompt,
                        decode_tokens,
                    )
            except Exception as exc:
                logger.exception("failed to run end_to_end_latency")
                self._write_json(500, {"error": str(exc)})
                return

            self._write_json(200, result)

        def _handle_eval(self):
            stream_started = False
            try:
                request = parse_json_body(self)
                samples = require_list(request, "samples")
                max_tokens = optional_positive_int(request, "max_tokens", None)
                done_ids = require_list(request, "completions_done_ids")
                enable_thinking = optional_nullable_bool(request, "enable_thinking")
                temperature = optional_temperature(request, "temperature")
            except BadRequest as exc:
                logger.info("bad /eval request: %s", exc)
                self._write_json(400, {"error": str(exc)})
                return

            try:
                self._start_jsonl_stream()
                stream_started = True
                with model_lock:
                    run_eval(
                        model,
                        tokenizer,
                        samples,
                        max_tokens,
                        done_ids,
                        self._write_jsonl_event,
                        should_abort,
                        clear_abort,
                        enable_thinking=enable_thinking,
                        temperature=temperature,
                    )
                self._finish_jsonl_stream()
                self.close_connection = True
            except Exception as exc:
                logger.exception("failed to stream eval")
                if stream_started:
                    try:
                        self._write_jsonl_event(
                            {"kind": "eval_error", "error": str(exc)}
                        )
                        self._finish_jsonl_stream()
                    except Exception:
                        logger.exception("failed to finish errored eval stream")
                self.close_connection = True

        def _handle_eval_abort(self):
            try:
                request = parse_json_body(self)
                sample_id = require_string(request, "sample_id")
            except BadRequest as exc:
                logger.info("bad /eval/abort request: %s", exc)
                self._write_json(400, {"error": str(exc)})
                return

            request_abort(sample_id)
            self._write_json(200, {"sample_id": sample_id})

        def _handle_shutdown(self):
            self._write_json(200, {})
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, fmt, *args):
            logger.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="pipette_mlx_server")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    seed_text = prompt_seed_text()
    model, tokenizer = load_model(args.model)
    seed_tokens = tokenize_seed(tokenizer, seed_text)
    server = LocalHTTPServer(
        ("127.0.0.1", args.port),
        make_handler(model, tokenizer, seed_tokens),
    )
    host, port = server.server_address
    print(json.dumps({"kind": "ready", "host": host, "port": port}), flush=True)
    logger.info("serving on %s:%s", host, port)
    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
