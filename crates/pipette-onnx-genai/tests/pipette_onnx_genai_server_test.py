"""The ONNX GenAI server without a wheel, model download or accelerator: the
engine against a fake `onnxruntime_genai`, and one exchange over HTTP."""

import http.client
import importlib.util
import json
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SERVER = Path(__file__).resolve().parents[1] / "src/python/pipette_onnx_genai_server.py"
spec = importlib.util.spec_from_file_location("pipette_onnx_genai_server", SERVER)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)

BOS = 1


class FakeGenai:
    """One token per character plus a BOS; the model generates `z`, then `é`
    split over two tokens, repeating, and stops at EOS after `eos_after`
    unless the minimum length masks it."""

    def __init__(self):
        self.providers = []
        self.options = []
        self.prompts = []
        self.eos_after = None
        owner = self

        class Config:
            def __init__(self, path):
                owner.path = path

            def clear_providers(self):
                owner.providers.clear()

            def append_provider(self, provider):
                owner.providers.append(provider)

        class Tokenizer:
            def __init__(self, model):
                pass

            def encode(self, text):
                return [BOS] + [ord(c) for c in text]

            def create_stream(self):
                return Stream()

        class Stream:
            def __init__(self):
                self.pending = []

            def decode(self, token):
                self.pending.append(token)
                if self.pending == [0xC3]:
                    return ""
                text = bytes(self.pending).decode()
                self.pending = []
                return text

        class Params:
            def __init__(self, model):
                pass

            def set_search_options(self, **options):
                owner.options.append(options)
                self.options = options

        class Generator:
            def __init__(self, model, params):
                self.options = params.options
                self.out = []

            def append_tokens(self, ids):
                owner.prompts.append(list(ids))
                self.length = len(ids)

            def is_done(self):
                total = self.length + len(self.out)
                if total >= self.options["max_length"]:
                    return True
                eos = owner.eos_after is not None and len(self.out) >= owner.eos_after
                return eos and total >= self.options.get("min_length", 0)

            def generate_next_token(self):
                self.out.append([ord("z"), 0xC3, 0xA9][len(self.out) % 3])

            def get_next_tokens(self):
                return [self.out[-1]]

        self.module = types.ModuleType("onnxruntime_genai")
        self.module.Config = Config
        self.module.Model = lambda config: object()
        self.module.Tokenizer = Tokenizer
        self.module.GeneratorParams = Params
        self.module.Generator = Generator


class EngineContract(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.genai = FakeGenai()
        self.preloads = []
        runtime = types.ModuleType("onnxruntime")
        runtime.preload_dlls = lambda: self.preloads.append("preload")
        modules = patch.dict(
            sys.modules,
            {"onnxruntime_genai": self.genai.module, "onnxruntime": runtime},
        )
        modules.start()
        self.addCleanup(modules.stop)

    def engine(self, provider="cpu"):
        return server.Engine(self.directory.name, provider)

    def generate(self, engine, ids, max_tokens, ignore_eos=True):
        texts = []
        engine.generate(ids, max_tokens, ignore_eos, texts.append)
        return texts

    def test_cpu_is_the_cleared_provider_list(self):
        self.engine()
        self.assertEqual(self.genai.providers, [])
        self.assertEqual(self.preloads, [])

    def test_cuda_is_appended_and_preloads_its_libraries(self):
        self.engine("cuda")
        self.assertEqual(self.genai.providers, ["cuda"])
        self.assertEqual(self.preloads, ["preload"])

    def test_encode_keeps_the_tokenizers_bos(self):
        self.assertEqual(self.engine().encode("ab"), [BOS, 97, 98])

    def test_ignore_eos_masks_eos_through_equal_bounds(self):
        self.genai.eos_after = 1
        texts = self.generate(self.engine(), [5, 6], 4)
        self.assertEqual(len(texts), 4)
        self.assertEqual(
            self.genai.options, [{"do_sample": False, "min_length": 6, "max_length": 6}]
        )

    def test_eos_ends_early_without_ignore_eos(self):
        self.genai.eos_after = 1
        texts = self.generate(self.engine(), [5, 6], 4, ignore_eos=False)
        self.assertEqual(len(texts), 1)
        self.assertEqual(self.genai.options, [{"do_sample": False, "max_length": 6}])

    def test_each_token_is_reported_with_its_own_text(self):
        # `é` is two tokens: the first completes no character yet.
        self.assertEqual(self.generate(self.engine(), [5], 3), ["z", "", "é"])


def serve(test, engine):
    """`engine` behind a server on a free port, shut down and closed at cleanup."""
    quiet = patch.object(server.Handler, "log_message", lambda *args: None)
    quiet.start()
    test.addCleanup(quiet.stop)
    httpd = server.make_server(engine, 0)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    test.addCleanup(thread.join)
    test.addCleanup(httpd.server_close)
    test.addCleanup(httpd.shutdown)
    return httpd


class OverHttp(unittest.TestCase):
    """The engine behind the real framework: `/tokenize` counts what a
    string completion feeds the model."""

    def setUp(self):
        EngineContract.setUp(self)
        engine = server.Engine(self.directory.name, "cpu")
        self.port = serve(self, engine).server_address[1]

    def post(self, path, body):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(conn.close)
        conn.request("POST", path, json.dumps(body))
        return conn.getresponse().read()

    def test_a_streamed_completion_matches_tokenize(self):
        tokens = json.loads(self.post("/tokenize", {"prompt": "hello"}))
        raw = self.post(
            "/v1/completions",
            {
                "prompt": "hello",
                "max_tokens": 3,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        events = [e.removeprefix(b"data: ") for e in raw.split(b"\n\n") if e]
        self.assertEqual(events[-1], b"[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertEqual(
            [c["choices"][0] for c in chunks[:-1]],
            [
                {"index": 0, "text": "z", "finish_reason": None},
                {"index": 0, "text": "", "finish_reason": None},
                {"index": 0, "text": "é", "finish_reason": "length"},
            ],
        )
        self.assertEqual(
            chunks[-1]["usage"],
            {
                "prompt_tokens": tokens["count"],
                "completion_tokens": 3,
                "total_tokens": 9,
            },
        )
        self.assertEqual(self.genai.prompts, [tokens["tokens"]])


class FakeEngine:
    """One token per character; each generated token's text comes from
    `texts`, cycled, and EOS lands after `eos_after` tokens unless ignored."""

    def __init__(self):
        self.texts = ["a"]
        self.eos_after = None
        self.fail_after = None
        self.gate = None
        self.prompts = []

    def encode(self, text):
        return [ord(c) for c in text]

    def generate(self, ids, max_tokens, ignore_eos, on_token):
        self.prompts.append(ids)
        for i in range(max_tokens):
            if not ignore_eos and self.eos_after is not None and i == self.eos_after:
                on_token("")
                return
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("device lost")
            on_token(self.texts[i % len(self.texts)])
            if self.gate is not None and i == 0:
                self.gate.wait(timeout=10)


class ServerContract(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.httpd = serve(self, self.engine)
        self.port = self.httpd.server_address[1]

    def connect(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        self.addCleanup(conn.close)
        return conn

    def request(self, method, path, body=None, conn=None):
        conn = conn or self.connect()
        conn.request(method, path, body=None if body is None else json.dumps(body))
        return conn.getresponse()

    def json(self, method, path, body=None, conn=None):
        response = self.request(method, path, body, conn)
        return response.status, json.loads(response.read())

    def sweeps(self):
        """An event set after each idle sweep from here on."""
        swept = threading.Event()
        release_idle = self.httpd.release_idle

        def sweep():
            release_idle()
            swept.set()

        sweeping = patch.object(self.httpd, "release_idle", sweep)
        sweeping.start()
        self.addCleanup(sweeping.stop)
        return swept

    def stream(self, **body):
        response = self.request(
            "POST",
            "/v1/completions",
            {"prompt": "hello", "max_tokens": 4, "stream": True, **body},
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
        events = [
            line.removeprefix(b"data: ").decode()
            for line in response.read().split(b"\n\n")
            if line
        ]
        self.assertEqual(events[-1], "[DONE]")
        return [json.loads(e) for e in events[:-1]]

    def test_models_is_the_readiness_probe(self):
        status, body = self.json("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(body["data"][0]["id"], server.MODEL_ID)

    def test_tokenize_matches_the_completions_prompt_count(self):
        status, tokens = self.json(
            "POST", "/tokenize", {"prompt": "héllo", "add_special_tokens": True}
        )
        self.assertEqual(status, 200)
        self.assertEqual(tokens["count"], 5)
        self.assertEqual(tokens["tokens"], [ord(c) for c in "héllo"])
        _, completion = self.json(
            "POST", "/v1/completions", {"prompt": "héllo", "max_tokens": 1}
        )
        self.assertEqual(completion["usage"]["prompt_tokens"], tokens["count"])

    def test_tokenize_refuses_another_accounting(self):
        status, body = self.json(
            "POST", "/tokenize", {"prompt": "x", "add_special_tokens": False}
        )
        self.assertEqual(status, 400)
        self.assertIn("add_special_tokens", body["error"]["message"])

    def test_non_streamed_completion(self):
        self.engine.texts = ["a", "b"]
        status, body = self.json(
            "POST",
            "/v1/completions",
            {"prompt": "hi", "max_tokens": 3, "temperature": 0, "ignore_eos": True},
        )
        self.assertEqual(status, 200)
        self.assertEqual(
            body["choices"], [{"index": 0, "text": "aba", "finish_reason": "length"}]
        )
        self.assertEqual(
            body["usage"],
            {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        )

    def test_max_tokens_has_a_default(self):
        _, body = self.json("POST", "/v1/completions", {"prompt": "hi"})
        self.assertEqual(body["usage"]["completion_tokens"], server.DEFAULT_MAX_TOKENS)

    def test_a_stream_is_one_chunk_per_token_then_usage(self):
        chunks = self.stream(ignore_eos=True, stream_options={"include_usage": True})
        texts = [c["choices"][0]["text"] for c in chunks[:-1]]
        finishes = [c["choices"][0]["finish_reason"] for c in chunks[:-1]]
        self.assertEqual(texts, ["a"] * 4)
        self.assertEqual(finishes, [None, None, None, "length"])
        self.assertEqual(chunks[-1]["choices"], [])
        self.assertEqual(
            chunks[-1]["usage"],
            {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        )

    def test_usage_only_when_asked(self):
        chunks = self.stream(ignore_eos=True)
        self.assertTrue(all("usage" not in c for c in chunks))

    def test_a_textless_token_is_still_a_chunk(self):
        # The client stamps every chunk, so a token that completes no
        # character yet still goes out when it exists.
        self.engine.texts = ["", "é", "", ""]
        chunks = self.stream(ignore_eos=True, stream_options={"include_usage": True})
        choices = [c["choices"][0] for c in chunks[:-1]]
        self.assertEqual(
            choices,
            [
                {"index": 0, "text": "", "finish_reason": None},
                {"index": 0, "text": "é", "finish_reason": None},
                {"index": 0, "text": "", "finish_reason": None},
                {"index": 0, "text": "", "finish_reason": "length"},
            ],
        )
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 4)

    def test_ignore_eos_generates_exactly_max_tokens(self):
        self.engine.eos_after = 1
        chunks = self.stream(ignore_eos=True, stream_options={"include_usage": True})
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 4)
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "length")

    def test_an_early_stop_says_stop(self):
        self.engine.eos_after = 1
        chunks = self.stream(stream_options={"include_usage": True})
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 2)
        _, body = self.json(
            "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 4}
        )
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")

    def test_a_chunk_is_on_the_wire_before_the_next_token(self):
        # The engine holds its second token until the test has read the first
        # chunk: a buffering server would deadlock here and time out.
        self.engine.gate = threading.Event()
        response = self.request(
            "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 2, "stream": True}
        )
        first = response.readline()
        self.assertIn(b'"text": "a"', first)
        self.engine.gate.set()
        self.assertTrue(response.read().endswith(b"data: [DONE]\n\n"))

    def test_a_connection_carries_request_after_request(self):
        # A stream's body ends, so the next request can follow on its socket.
        conn = self.connect()
        socks = []
        for stream in [True, False, True]:
            body = {"prompt": "hi", "max_tokens": 2, "stream": stream}
            conn.request("POST", "/v1/completions", json.dumps(body))
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            response.read()
            socks.append(conn.sock)
        self.assertIsNotNone(socks[0])
        self.assertEqual(socks, [socks[0]] * 3)

    def test_the_last_chunk_rides_with_done(self):
        writes = []

        class Recorder(self.httpd.RequestHandlerClass):
            def setup(self):
                super().setup()
                write = self.wfile.write

                def record(data):
                    writes.append(bytes(data))
                    return write(data)

                self.wfile.write = record

        self.httpd.RequestHandlerClass = Recorder
        self.stream()
        self.assertEqual(writes[-1], b"E\r\ndata: [DONE]\n\n\r\n0\r\n\r\n")

    def test_a_kept_connection_is_answered_after_another_connects(self):
        # Waiting on one would stall the other; closing one would lose its
        # next request.
        kept = self.connect()
        self.assertEqual(self.json("GET", "/v1/models", conn=kept)[0], 200)
        other = self.connect()
        self.assertEqual(self.json("GET", "/v1/models", conn=other)[0], 200)
        self.assertEqual(self.json("GET", "/v1/models", conn=kept)[0], 200)

    def test_a_kept_connection_is_answered_while_a_spare_is_silent(self):
        kept = self.connect()
        self.assertEqual(self.json("GET", "/v1/models", conn=kept)[0], 200)
        spare = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        self.addCleanup(spare.close)
        self.assertEqual(self.json("GET", "/v1/models", conn=kept)[0], 200)

    def test_a_silent_connection_stalls_no_other(self):
        # A pool's spare connection may never send a request.
        silent = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        self.addCleanup(silent.close)
        self.assertEqual(self.json("GET", "/v1/models")[0], 200)

    def test_a_connection_the_client_ends_is_let_go(self):
        # Kept at EOF, it would stay readable and spin the loop.
        conn = self.connect()
        self.assertEqual(self.json("GET", "/v1/models", conn=conn)[0], 200)
        conn.sock.shutdown(socket.SHUT_WR)
        self.assertEqual(conn.sock.recv(1), b"")

    def test_an_idle_connection_is_let_go(self):
        conn = self.connect()
        self.assertEqual(self.json("GET", "/v1/models", conn=conn)[0], 200)
        with patch.object(server, "IDLE_S", 0):
            self.assertEqual(conn.sock.recv(1), b"")

    def test_a_silent_connection_is_let_go_when_idle(self):
        spare = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        self.addCleanup(spare.close)
        with patch.object(server, "IDLE_S", 0):
            self.assertEqual(spare.recv(1), b"")

    def test_a_new_connection_outlives_a_sweep(self):
        # Idle time counts from the accept, so a sweep spares a fresh connection.
        swept = self.sweeps()
        kept = self.connect()
        kept.connect()
        deadline = time.monotonic() + 5
        while len(self.httpd.selector.get_map()) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        swept.clear()
        self.assertTrue(swept.wait(timeout=10))
        self.assertEqual(self.json("GET", "/v1/models", conn=kept)[0], 200)

    def test_a_request_restarts_the_idle_clock(self):
        # The clock passes the limit mid-request, so only the restart after the
        # answer keeps `kept` through the next sweep.
        clock = [0.0]
        fake = types.SimpleNamespace(monotonic=lambda: clock[0], time=time.time)
        swept = self.sweeps()
        self.engine.gate = threading.Event()
        with patch.object(server, "time", fake):
            kept = self.connect()
            body = {"prompt": "hi", "max_tokens": 2, "stream": True}
            response = self.request("POST", "/v1/completions", body, conn=kept)
            response.readline()
            clock[0] = server.IDLE_S + 1
            self.engine.gate.set()
            response.read()
            swept.clear()
            self.assertTrue(swept.wait(timeout=10))
            self.assertEqual(self.json("GET", "/v1/models", conn=kept)[0], 200)

    def test_a_request_sent_during_a_generation_outlives_the_idle_limit(self):
        # The engine holds the server while `kept` sends, so `kept` is past a
        # zero limit with its request waiting.
        self.engine.gate = threading.Event()
        held = self.request(
            "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 2, "stream": True}
        )
        held.readline()
        with patch.object(server, "IDLE_S", 0):
            kept = self.connect()
            kept.request("GET", "/v1/models")
            self.engine.gate.set()
            response = kept.getresponse()
            response.read()
            self.assertEqual(response.status, 200)

    def test_an_unexpected_error_drops_only_its_connection(self):
        errors = []
        self.httpd.handle_error = lambda request, address: errors.append(address)
        kept = self.connect()
        self.assertEqual(self.json("GET", "/v1/models", conn=kept)[0], 200)
        with (
            patch.object(server.Handler, "do_GET", side_effect=RuntimeError),
            self.assertRaises(ConnectionError),
        ):
            self.json("GET", "/v1/models")
        self.assertEqual(self.json("GET", "/v1/models", conn=kept)[0], 200)
        self.assertEqual(len(errors), 1)

    def test_a_reset_connection_is_no_server_error(self):
        errors = []
        self.httpd.handle_error = lambda request, address: errors.append(address)
        conn = self.connect()
        conn.request("GET", "/v1/models")
        conn.getresponse()
        # A zero linger closes with a reset, as a close with bytes unread does.
        linger = struct.pack("HH" if sys.platform == "win32" else "ii", 1, 0)
        conn.sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
        conn.close()
        self.assertEqual(self.json("GET", "/v1/models")[0], 200)
        # The reset can reach the server after that request does.
        deadline = time.monotonic() + 5
        while len(self.httpd.selector.get_map()) > 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(errors, [])
        # Only the listener and the second connection stay selected: kept, a
        # reset socket stays readable and on Windows every read of it fails
        # again, so the loop would spin on it.
        self.assertEqual(len(self.httpd.selector.get_map()), 2)

    def test_closing_the_server_closes_a_held_connection(self):
        kept = self.connect()
        self.assertEqual(self.json("GET", "/v1/models", conn=kept)[0], 200)
        self.httpd.shutdown()
        held = [k.data for k in self.httpd.selector.get_map().values() if k.data]
        self.httpd.server_close()
        self.assertEqual([h.connection.fileno() for h in held], [-1])

    def test_a_failed_bind_raises_the_bind_error(self):
        with (
            patch.object(server.Server, "server_bind", side_effect=OSError("taken")),
            self.assertRaisesRegex(OSError, "taken"),
        ):
            server.make_server(FakeEngine(), 0)

    def test_a_failure_after_the_headers_is_an_error_chunk(self):
        self.engine.fail_after = 2
        chunks = self.stream(ignore_eos=True, stream_options={"include_usage": True})
        self.assertEqual(
            chunks[-1], {"error": {"message": "device lost", "type": "server_error"}}
        )

    def test_refused_requests(self):
        for body in [
            {"prompt": "hi", "temperature": 0.7},
            {"prompt": ["a", "b"]},
            {"prompt": [7, 8, 9]},
            {"prompt": "hi", "max_tokens": 0},
            {"prompt": ""},
        ]:
            with self.subTest(body=body):
                status, error = self.json("POST", "/v1/completions", body)
                self.assertEqual(status, 400)
                self.assertEqual(error["error"]["type"], "invalid_request_error")

    def test_an_engine_failure_before_the_headers_is_a_500(self):
        with patch.object(self.engine, "encode", side_effect=RuntimeError("no vocab")):
            status, error = self.json("POST", "/v1/completions", {"prompt": "hi"})
        self.assertEqual(status, 500)
        self.assertIn("no vocab", error["error"]["message"])

    def test_unknown_routes(self):
        self.assertEqual(self.request("GET", "/health").status, 404)
        self.assertEqual(self.request("POST", "/v1/chat/completions", {}).status, 404)
        self.assertEqual(self.request("POST", "/tokenize", None).status, 400)

    def test_a_404_waits_for_the_body(self):
        # A 404 sent before the body is read is reset away on Windows.
        conn = self.connect()
        conn.putrequest("POST", "/v1/chat/completions")
        conn.putheader("Content-Length", "2")
        conn.endheaders()
        conn.sock.settimeout(0.2)
        with self.assertRaises(TimeoutError):
            conn.sock.recv(1, socket.MSG_PEEK)
        conn.sock.settimeout(10)
        conn.send(b"{}")
        self.assertEqual(conn.getresponse().status, 404)


class Main(unittest.TestCase):
    def test_the_engine_loads_before_the_server_listens(self):
        order = []

        class Stop(Exception):
            pass

        def make_server(engine, port):
            order.append(("listen", port))
            return types.SimpleNamespace(
                server_address=("127.0.0.1", port),
                serve_forever=lambda: (_ for _ in ()).throw(Stop()),
            )

        def engine(*args):
            order.append(("load", *args))
            return FakeEngine()

        with (
            patch.object(server, "Engine", engine),
            patch.object(
                server, "exit_with_parent", lambda pid: order.append(("watch", pid))
            ),
            patch.object(server, "make_server", make_server),
            patch.object(server, "log"),
            self.assertRaises(Stop),
        ):
            server.main(
                [
                    "--port",
                    "8123",
                    "--parent-pid",
                    "99",
                    "--model-dir",
                    "/m",
                    "--execution-provider",
                    "cuda",
                ]
            )
        self.assertEqual(
            order, [("watch", 99), ("load", "/m", "cuda"), ("listen", 8123)]
        )


class ParentWatch(unittest.TestCase):
    def test_reparenting_ends_the_wait(self):
        with (
            patch.object(sys, "platform", "linux"),
            patch.object(server.os, "getppid", side_effect=[42, 42, 1]),
            patch.object(server.time, "sleep"),
        ):
            self.assertTrue(server.wait_for_exit(42))

    def test_a_process_that_is_not_the_parent_is_polled(self):
        with (
            patch.object(sys, "platform", "linux"),
            patch.object(server.os, "getppid", return_value=7),
            patch.object(server.os, "kill", side_effect=[None, ProcessLookupError()]),
            patch.object(server.time, "sleep"),
        ):
            self.assertTrue(server.wait_for_exit(42))

    def test_a_real_process_is_seen_to_exit(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(child.kill)
        done = threading.Event()
        watcher = threading.Thread(
            target=lambda: server.wait_for_exit(child.pid) and done.set(), daemon=True
        )
        with patch.object(server, "PARENT_POLL_S", 0.01):
            watcher.start()
            child.kill()
            child.wait()
            self.assertTrue(done.wait(timeout=10))

    def test_a_closed_stderr_does_not_keep_the_server_alive(self):
        # The dead harness was reading stderr, so the exit note fails to write.
        calls = []
        inline = types.SimpleNamespace(
            Thread=lambda target, **kwargs: types.SimpleNamespace(start=target)
        )
        with (
            patch.object(server, "wait_for_exit", return_value=True),
            patch.object(server.os, "_exit", calls.append),
            patch.object(server, "threading", inline),
            patch.object(server, "log", side_effect=BrokenPipeError()),
        ):
            server.exit_with_parent(42)
        self.assertEqual(calls, [1])

    def test_only_a_gone_parent_stops_the_server(self):
        for gone, exits in [(True, [1]), (False, [])]:
            with self.subTest(gone=gone):
                calls = []
                inline = types.SimpleNamespace(
                    Thread=lambda target, **kwargs: types.SimpleNamespace(start=target)
                )
                with (
                    patch.object(server, "wait_for_exit", return_value=gone),
                    patch.object(server.os, "_exit", calls.append),
                    patch.object(server, "threading", inline),
                    patch.object(server, "log"),
                ):
                    server.exit_with_parent(42)
                self.assertEqual(calls, exits)


if __name__ == "__main__":
    unittest.main()
