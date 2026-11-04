"""OpenVINO GenAI behind a minimal OpenAI-compatible server.

Compiles one `LLMPipeline` before listening, so a slow NPU compile is startup
time and the process never compiles twice (a second compile in one process
took an NPU down; see docs/openvino-ir.md). Generation streams through a
token-level streamer, detokenized here, because GenAI's text streamer holds
tokens back to print whole words.

Endpoints: `GET /v1/models` (the readiness probe), `POST /tokenize` and
`POST /v1/completions` (streamed or not). Standard library only for HTTP, so
the uv environment needs nothing beyond the runtime wheel. HTTP/1.1, so a
client's connection carries request after request. One thread serves every
connection a client keeps, whichever sends a request next: a client's pool can
open a second connection while it still holds the first, then send on either.
A thread per connection raised the memory cell's peak by 8 MB on Linux.

Streaming sends one chunk per generated token as soon as the token exists,
text or not: a token that completes no character yet carries empty text, and
the last carries `length`. A generation that stops on EOS ends with a textless
`stop` chunk instead.
"""

import argparse
import contextlib
import itertools
import json
import os
import selectors
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

MODEL_ID = "default"
DEFAULT_MAX_TOKENS = 16
PARENT_POLL_S = 1.0
# Past pipette-http's 90 s pool idle timeout, so the client lets go first.
IDLE_S = 300.0
SYNCHRONIZE = 0x00100000
INFINITE = 0xFFFFFFFF
ERROR_INVALID_PARAMETER = 87


class RequestError(Exception):
    """A request this server refuses, answered with HTTP 400."""


def log(message):
    print(f"pipette_openvino_server: {message}", file=sys.stderr, flush=True)


def wait_for_exit(pid):
    """Block until process `pid` exits; False when it cannot be watched."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not handle:
            # An invalid pid is one that has already exited.
            return ctypes.get_last_error() == ERROR_INVALID_PARAMETER
        kernel32.WaitForSingleObject(handle, INFINITE)
        return True
    if os.getppid() == pid:
        # Re-parenting is immediate even while the dead parent is an unreaped
        # zombie, which `kill(pid, 0)` would still report alive.
        while os.getppid() == pid:
            time.sleep(PARENT_POLL_S)
        return True
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        time.sleep(PARENT_POLL_S)


def exit_with_parent(pid):
    """Exit when the harness that started this server does: one killed outright
    cannot stop its server, which would hold the model's memory into the next
    run."""

    def watch():
        if not wait_for_exit(pid):
            log(f"cannot watch the harness (pid {pid}); running unwatched")
            return
        # The harness was stderr's reader, so this note may have nowhere to go.
        with contextlib.suppress(OSError):
            log(f"the harness (pid {pid}) exited; stopping")
        os._exit(1)

    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def usage(prompt_tokens, completion_tokens):
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


class Completion:
    """A validated `/v1/completions` body."""

    def __init__(self, body):
        self.prompt = body.get("prompt")
        if not isinstance(self.prompt, str):
            raise RequestError("prompt must be a string")
        self.max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
        if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise RequestError("max_tokens must be a positive integer")
        if body.get("temperature", 0) not in (0, 0.0):
            raise RequestError("only greedy decoding is served: temperature must be 0")
        self.ignore_eos = bool(body.get("ignore_eos", False))
        self.stream = bool(body.get("stream", False))
        options = body.get("stream_options") or {}
        self.include_usage = bool(options.get("include_usage", False))


class Handler(BaseHTTPRequestHandler):
    # Keeps a connection open between requests, so a client's pooled
    # connection carries the next one without a new connect.
    protocol_version = "HTTP/1.1"
    # Per-token chunks are small writes; Nagle would hold one back until the
    # previous one is acknowledged, which moves the client's token stamps.
    disable_nagle_algorithm = True
    engine = None
    ids = itertools.count(1)

    def __init__(self, request, client_address, server):
        # Set up only: the server answers each request as it arrives.
        self.request = request
        self.client_address = client_address
        self.server = server
        self.setup()

    def do_GET(self):
        if self.path == "/v1/models":
            model = {"id": MODEL_ID, "object": "model", "owned_by": "pipette"}
            self.send_json(200, {"object": "list", "data": [model]})
        else:
            self.send_error_json(404, f"no route for GET {self.path}")

    def do_POST(self):
        routes = {"/tokenize": self.tokenize, "/v1/completions": self.complete}
        try:
            # Read before any answer, a 404 included: closing on an unread body
            # resets the connection on Windows, and the reset loses the answer.
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            route = routes.get(self.path)
            if route is None:
                self.send_error_json(404, f"no route for POST {self.path}")
                return
            route(self.parse_json(raw))
        except RequestError as exc:
            self.send_error_json(400, str(exc))
        except Exception as exc:
            log(f"{self.path} failed: {exc!r}")
            self.send_error_json(500, str(exc))

    @staticmethod
    def parse_json(raw):
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise RequestError(f"the body is not JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise RequestError("the body must be a JSON object")
        return body

    def tokenize(self, body):
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise RequestError("prompt must be a string")
        # One accounting only, the one a string completion gets.
        if body.get("add_special_tokens", True) is not True:
            raise RequestError(
                "add_special_tokens must be true: /tokenize counts a prompt the "
                "way /v1/completions feeds it"
            )
        tokens = [int(i) for i in self.engine.encode(prompt)]
        self.send_json(200, {"tokens": tokens, "count": len(tokens)})

    def complete(self, body):
        request = Completion(body)
        ids = [int(i) for i in self.engine.encode(request.prompt)]
        if not ids:
            raise RequestError("the prompt holds no tokens")
        header = {
            "id": f"cmpl-{next(self.ids)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": MODEL_ID,
        }
        if request.stream:
            self.stream(request, ids, header)
            return
        pieces = []
        generated = self.generate(
            request, ids, lambda text, finish: pieces.append(text)
        )
        finish_reason = "length" if generated == request.max_tokens else "stop"
        choice = {"index": 0, "text": "".join(pieces), "finish_reason": finish_reason}
        self.send_json(
            200, {**header, "choices": [choice], "usage": usage(len(ids), generated)}
        )

    def generate(self, request, ids, on_chunk):
        """Run the engine, handing `on_chunk(text, finish_reason)` every token.
        Returns the count generated."""
        generated = 0

        def on_token(text):
            nonlocal generated
            generated += 1
            if generated > request.max_tokens:
                raise RuntimeError(
                    f"the engine generated past max_tokens {request.max_tokens}"
                )
            on_chunk(text, "length" if generated == request.max_tokens else None)

        self.engine.generate(ids, request.max_tokens, request.ignore_eos, on_token)
        return generated

    def stream(self, request, ids, header):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # Framed, so the body has an end and the connection outlives it.
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def event(chunk):
            self.wfile.write(framed(b"data: " + json.dumps(chunk).encode() + b"\n\n"))
            self.wfile.flush()

        def on_chunk(text, finish_reason):
            choice = {"index": 0, "text": text, "finish_reason": finish_reason}
            event({**header, "choices": [choice]})

        try:
            generated = self.generate(request, ids, on_chunk)
            if generated < request.max_tokens:
                on_chunk("", "stop")
            if request.include_usage:
                event({**header, "choices": [], "usage": usage(len(ids), generated)})
        except Exception as exc:
            log(f"/v1/completions failed mid-stream: {exc!r}")
            try:
                event({"error": {"message": str(exc), "type": "server_error"}})
            except OSError:
                self.close_connection = True
                return
        try:
            # The last chunk rides with `[DONE]`: a client that stops reading
            # there finds the body complete and can reuse the connection.
            self.wfile.write(framed(b"data: [DONE]\n\n") + b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            self.close_connection = True

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status, message):
        kind = "invalid_request_error" if status < 500 else "server_error"
        self.send_json(status, {"error": {"message": message, "type": kind}})


# A text that ends in U+FFFD stops mid-character: its token has no text yet.
PARTIAL = "\ufffd"


class Detokenizer:
    """Incremental detokenization: each token's new text, decoding only the
    window since the last text was emitted."""

    def __init__(self, decode):
        self.decode = decode
        self.ids = []
        self.prefix = 0
        self.read = 0

    def push(self, token):
        self.ids.append(int(token))
        before = self.decode(self.ids[self.prefix : self.read])
        text = self.decode(self.ids[self.prefix :])
        if len(text) <= len(before) or text.endswith(PARTIAL):
            return ""
        self.prefix, self.read = self.read, len(self.ids)
        return text[len(before) :]


class Engine:
    def __init__(self, model_dir, device, properties):
        import numpy
        import openvino
        import openvino_genai

        self.np, self.ov, self.ov_genai = numpy, openvino, openvino_genai
        self.pipe = openvino_genai.LLMPipeline(model_dir, device, **properties)
        self.tokenizer = self.pipe.get_tokenizer()

    def encode(self, text):
        encoded = self.tokenizer.encode(text, add_special_tokens=True)
        return [int(i) for i in encoded.input_ids.data[0]]

    def generate(self, ids, max_tokens, ignore_eos, on_token):
        # Token ids, never a string: a string makes GenAI apply the chat
        # template, adding tokens the request did not send.
        batch = self.np.array([ids], dtype=self.np.int64)
        inputs = self.ov_genai.TokenizedInputs(
            self.ov.Tensor(batch), self.ov.Tensor(self.np.ones_like(batch))
        )
        config = self.ov_genai.GenerationConfig()
        config.do_sample = False
        config.max_new_tokens = max_tokens
        config.ignore_eos = ignore_eos
        self.pipe.generate(inputs, config, self.streamer(on_token))

    def streamer(self, on_token):
        ov_genai = self.ov_genai
        detokenizer = Detokenizer(self.tokenizer.decode)

        class Streamer(ov_genai.StreamerBase):
            def write(self, token):
                for t in token if isinstance(token, list) else [token]:
                    on_token(detokenizer.push(t))
                return ov_genai.StreamingStatus.RUNNING

            def end(self):
                pass

        return Streamer()


def framed(data):
    """`data` as one chunk of a chunked body."""
    return b"%X\r\n" % len(data) + data + b"\r\n"


class Server(HTTPServer):
    """One thread, so a half-sent request holds every other connection;
    clients here write each request whole and never pipeline, so nothing
    waits unselected in a connection's `rfile`."""

    def __init__(self, address, handler_class):
        # Before the bind: a failed bind calls `server_close`.
        self.selector = selectors.DefaultSelector()
        super().__init__(address, handler_class)
        self.selector.register(self.socket, selectors.EVENT_READ)
        self.stopping = threading.Event()
        self.stopped = threading.Event()

    def serve_forever(self, poll_interval=0.5):
        try:
            while not self.stopping.is_set():
                ready = self.selector.select(poll_interval)
                for key, _ in ready:
                    if key.data is None:
                        self._handle_request_noblock()
                    else:
                        self.serve(key.data)
                # Only when nothing waits, so a request sent during a long
                # generation is answered before its connection's idle time counts.
                if not ready:
                    self.release_idle()
        finally:
            self.stopped.set()

    def shutdown(self):
        self.stopping.set()
        self.stopped.wait()

    def process_request(self, request, client_address):
        handler = self.RequestHandlerClass(request, client_address, self)
        handler.idle_since = time.monotonic()
        self.selector.register(request, selectors.EVENT_READ, handler)

    def serve(self, handler):
        try:
            handler.handle_one_request()
        except ConnectionError:
            # A client that closes a kept-alive connection with bytes unread
            # resets it; that is no server error.
            handler.close_connection = True
        except Exception:
            self.handle_error(handler.connection, handler.client_address)
            handler.close_connection = True
        if handler.close_connection:
            self.release(handler)
        else:
            handler.idle_since = time.monotonic()

    def release_idle(self):
        cutoff = time.monotonic() - IDLE_S
        for key in list(self.selector.get_map().values()):
            if key.data is not None and key.data.idle_since < cutoff:
                self.release(key.data)

    def release(self, handler):
        self.selector.unregister(handler.connection)
        handler.finish()
        self.shutdown_request(handler.connection)

    def server_close(self):
        for key in list((self.selector.get_map() or {}).values()):
            if key.data is not None:
                self.release(key.data)
        self.selector.close()
        super().server_close()


def make_server(engine, port):
    """A server for `engine` on 127.0.0.1:`port`; port 0 picks a free one."""
    handler = type("EngineHandler", (Handler,), {"engine": engine})
    return Server(("127.0.0.1", port), handler)


def main(argv=None):
    """Load the engine, then listen: nothing answers the readiness probe until
    the model can serve."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--parent-pid", type=int, help="exit when this process does")
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--properties", type=json.loads, default={}, help="LLMPipeline properties"
    )
    args = parser.parse_args(argv)
    # Before the load, which can take minutes on a cold cache or an NPU compile.
    if args.parent_pid is not None:
        exit_with_parent(args.parent_pid)

    started = time.perf_counter()
    engine = Engine(args.model_dir, args.device, dict(args.properties))
    log(f"loaded {args.model_dir} in {time.perf_counter() - started:.1f}s")
    server = make_server(engine, args.port)
    log(f"listening on http://127.0.0.1:{server.server_address[1]}")
    server.serve_forever()


if __name__ == "__main__":
    main()
