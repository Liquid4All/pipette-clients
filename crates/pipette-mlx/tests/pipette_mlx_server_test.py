import http.client
import importlib.util
import io
import json
import pathlib
import sys
import threading
import types
import unittest
from types import SimpleNamespace


def load_server_module():
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_sample_utils = types.ModuleType("mlx_lm.sample_utils")
    fake_sample_utils.make_sampler = lambda temp: ("sampler", temp)
    fake_mlx_lm.sample_utils = fake_sample_utils
    fake_mlx_lm.stream_generate = None

    old_mlx_lm = sys.modules.get("mlx_lm")
    old_sample_utils = sys.modules.get("mlx_lm.sample_utils")
    sys.modules["mlx_lm"] = fake_mlx_lm
    sys.modules["mlx_lm.sample_utils"] = fake_sample_utils
    try:
        server_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "src"
            / "python"
            / "pipette_mlx_server.py"
        )
        spec = importlib.util.spec_from_file_location("pipette_mlx_server", server_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if old_mlx_lm is None:
            sys.modules.pop("mlx_lm", None)
        else:
            sys.modules["mlx_lm"] = old_mlx_lm
        if old_sample_utils is None:
            sys.modules.pop("mlx_lm.sample_utils", None)
        else:
            sys.modules["mlx_lm.sample_utils"] = old_sample_utils


class PipetteMlxServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = load_server_module()

    def test_make_tokens_repeats_seed_exactly(self):
        self.assertEqual(
            self.server.make_tokens([1, 2, 3], 7),
            [1, 2, 3, 1, 2, 3, 1],
        )

    def test_optional_positive_int_validates_request_fields(self):
        self.assertEqual(self.server.optional_positive_int({}, "num_trials", 5), 5)
        self.assertEqual(
            self.server.optional_positive_int({"num_trials": 2}, "num_trials", 5), 2
        )

        invalid_cases = [
            ({}, "prompt_tokens", None),
            ({"prompt_tokens": 0}, "prompt_tokens", None),
            ({"prompt_tokens": False}, "prompt_tokens", None),
            ({"prompt_tokens": "8"}, "prompt_tokens", None),
        ]
        for request, field, default in invalid_cases:
            with self.subTest(request=request):
                with self.assertRaises(self.server.BadRequest):
                    self.server.optional_positive_int(request, field, default)

    def test_prefill_throughput_runs_one_measurement_per_call(self):
        calls = self._patch_stream_generate()
        result = self.server.run_prefill_throughput(
            object(),
            SimpleNamespace(_eos_token_ids={1}),
            seed_tokens=[10, 11],
            prompt_tokens=5,
        )

        self.assertEqual(calls, [([10, 11, 10, 11, 10], 1)])
        self.assertEqual(result, {"prompt_tps": 101.0, "prompt_tokens": 5})

    def test_decode_throughput_runs_one_measurement_per_call(self):
        calls = self._patch_stream_generate()
        result = self.server.run_decode_throughput(
            object(),
            SimpleNamespace(_eos_token_ids={1}),
            seed_tokens=[10, 11],
            prompt_tokens=5,
            decode_tokens=4,
        )

        self.assertEqual(calls, [([10, 11, 10, 11, 10], 4)])
        self.assertEqual(result, {"generation_tps": 201.0, "decode_tokens": 4})

    def test_max_memory_usage_uses_single_full_prompt_call(self):
        calls = self._patch_stream_generate()
        tokenizer = SimpleNamespace(_eos_token_ids={1})
        result = self.server.run_max_memory_usage(
            object(),
            tokenizer,
            seed_tokens=[10, 11],
            prompt_tokens=5,
            decode_tokens=1,
        )

        self.assertEqual(calls, [([10, 11, 10, 11, 10], 1)])
        self.assertEqual(tokenizer._eos_token_ids, set())
        self.assertEqual(result, {"prompt_tokens": 5, "completion_tokens": 1})

    def test_end_to_end_latency_times_full_string_prompt(self):
        calls = self._patch_stream_generate()
        tokenizer = SimpleNamespace(_eos_token_ids={1})
        result = self.server.run_end_to_end_latency(
            object(),
            tokenizer,
            prompt="12345",
            decode_tokens=4,
        )

        self.assertEqual(calls, [(["1", "2", "3", "4", "5"], 4)])
        self.assertEqual(tokenizer._eos_token_ids, set())
        self.assertEqual(result["prompt_tokens"], 5)
        self.assertEqual(result["completion_tokens"], 4)
        self.assertGreater(result["total_ms"], 0.0)

    def test_eval_streams_events_without_suppressing_eos(self):
        self._patch_eval_stream(["he", "llo"])
        tokenizer = FakeTokenizer()
        events = []
        self.server.run_eval(
            object(),
            tokenizer,
            samples=[{"id": "s1", "messages": [{"role": "user", "content": "hi"}]}],
            max_tokens=8,
            done_ids=[],
            emit_event=events.append,
            should_abort=lambda _sample_id: False,
            clear_abort=lambda _sample_id: None,
            enable_thinking=None,
        )

        self.assertEqual(tokenizer._eos_token_ids, {1})
        self.assertEqual(
            [event["kind"] for event in events],
            [
                "eval_sample_start",
                "eval_sample_done",
                "eval_done",
            ],
        )
        self.assertEqual(events[1]["completion"], "hello")
        self.assertFalse(events[1]["stopped_early"])

    def test_eval_temperature_selects_sampler(self):
        # temperature=0.0 reuses GREEDY_SAMPLER; a positive temperature
        # builds a sampling sampler via make_sampler(temp=...). The fake
        # make_sampler returns ("sampler", temp) so we can assert which
        # one reached stream_generate.
        for temperature, expected in [
            (0.0, self.server.GREEDY_SAMPLER),
            (0.6, ("sampler", 0.6)),
        ]:
            with self.subTest(temperature=temperature):
                seen = []

                # `_seen=seen` binds the current iteration's list at
                # definition time (ruff B023: don't capture the loop var).
                def fake_stream_generate(
                    _model, _tokenizer, prompt, max_tokens, sampler, _seen=seen
                ):
                    _seen.append(sampler)
                    yield SimpleNamespace(text="ok")

                self.server.mlx_lm.stream_generate = fake_stream_generate
                self.addCleanup(
                    setattr,
                    self.server.mlx_lm,
                    "stream_generate",
                    None,
                )
                self.server.run_eval(
                    object(),
                    FakeTokenizer(),
                    samples=[
                        {"id": "s1", "messages": [{"role": "user", "content": "hi"}]}
                    ],
                    max_tokens=8,
                    done_ids=[],
                    emit_event=lambda _event: None,
                    should_abort=lambda _sample_id: False,
                    clear_abort=lambda _sample_id: None,
                    temperature=temperature,
                )
                self.assertEqual(seen, [expected])

    def test_optional_temperature_validates_request_fields(self):
        self.assertEqual(self.server.optional_temperature({}, "temperature"), 0.0)
        self.assertEqual(
            self.server.optional_temperature({"temperature": 0.6}, "temperature"), 0.6
        )
        self.assertEqual(
            self.server.optional_temperature({"temperature": 1}, "temperature"), 1.0
        )
        self.assertEqual(
            self.server.optional_temperature({"temperature": 2.0}, "temperature"), 2.0
        )
        for bad in [-0.1, 2.1, True, "0.6", None]:
            with self.assertRaises(self.server.BadRequest):
                self.server.optional_temperature({"temperature": bad}, "temperature")

    def test_eval_honors_abort_between_tokens(self):
        self._patch_eval_stream(["partial", "ignored"])
        tokenizer = FakeTokenizer()
        events = []
        cleared = []
        self.server.run_eval(
            object(),
            tokenizer,
            samples=[{"id": "s1", "messages": [{"role": "user", "content": "hi"}]}],
            max_tokens=8,
            done_ids=[],
            emit_event=events.append,
            should_abort=lambda sample_id: sample_id == "s1",
            clear_abort=cleared.append,
            enable_thinking=True,
        )

        self.assertEqual(events[1]["kind"], "eval_sample_done")
        self.assertEqual(events[1]["completion"], "partial")
        self.assertTrue(events[1]["stopped_early"])
        self.assertEqual(cleared, ["s1"])
        self.assertEqual(tokenizer.apply_kwargs, {"enable_thinking": True})

    def test_eval_http_endpoint_streams_chunked_jsonl_and_honors_abort(self):
        release_second_token = threading.Event()
        self._patch_eval_stream(
            ["x" * 300, "y", "z"], before_second=release_second_token
        )
        port = self._start_test_http_server(FakeTokenizer())

        eval_conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(eval_conn.close)
        self._request_json(
            eval_conn,
            "POST",
            "/eval",
            {
                "samples": [
                    {"id": "s1", "messages": [{"role": "user", "content": "hi"}]}
                ],
                "max_tokens": 8,
                "completions_done_ids": [],
                "enable_thinking": None,
            },
        )
        response = eval_conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Transfer-Encoding"), "chunked")

        start = json.loads(response.readline())
        chunk = json.loads(response.readline())
        self.assertEqual(start["kind"], "eval_sample_start")
        self.assertEqual(chunk["kind"], "eval_sample_chunk")
        self.assertEqual(chunk["delta"], "x" * 300)

        # Generation is deliberately served on the model-load thread because
        # MLX Metal streams are thread-local.  Invoke the abort handler
        # directly instead of issuing a concurrent HTTP request.
        abort_body = json.dumps({"sample_id": "s1"}).encode("utf-8")
        abort_handler = type(
            "AbortHandler",
            (),
            {
                "headers": {"Content-Length": str(len(abort_body))},
                "rfile": io.BytesIO(abort_body),
                "_write_json": lambda _self, status, payload: self.assertEqual(
                    (status, payload), (200, {"sample_id": "s1"})
                ),
            },
        )()
        self._test_handler_class._handle_eval_abort(abort_handler)
        release_second_token.set()

        done = json.loads(response.readline())
        final = json.loads(response.readline())
        self.assertEqual(done["kind"], "eval_sample_done")
        self.assertTrue(done["stopped_early"])
        self.assertTrue(done["completion"].startswith("x" * 300))
        self.assertNotIn("z", done["completion"])
        self.assertEqual(final["kind"], "eval_done")
        self.assertEqual(response.readline(), b"")

    def test_eval_http_endpoint_emits_eval_error_event(self):
        self._patch_eval_stream_error(RuntimeError("stream broke"))
        port = self._start_test_http_server(FakeTokenizer())

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(conn.close)
        self._request_json(
            conn,
            "POST",
            "/eval",
            {
                "samples": [
                    {"id": "s1", "messages": [{"role": "user", "content": "hi"}]}
                ],
                "max_tokens": 8,
                "completions_done_ids": [],
                "enable_thinking": None,
            },
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 200)

        start = json.loads(response.readline())
        error = json.loads(response.readline())
        self.assertEqual(start["kind"], "eval_sample_start")
        self.assertEqual(error["kind"], "eval_error")
        self.assertIn("stream broke", error["error"])
        self.assertEqual(response.readline(), b"")

    def _patch_stream_generate(self):
        calls = []

        def fake_stream_generate(_model, _tokenizer, tokens, max_tokens):
            calls.append((list(tokens), max_tokens))
            return SimpleNamespace(
                prompt_tokens=len(tokens),
                generation_tokens=max_tokens,
                prompt_tps=100.0 + len(calls),
                generation_tps=200.0 + len(calls),
            )

        self.addCleanup(
            setattr,
            self.server,
            "stream_generate_last_response",
            self.server.stream_generate_last_response,
        )
        self.server.stream_generate_last_response = fake_stream_generate
        return calls

    def _patch_eval_stream(self, parts, before_second=None):
        def fake_stream_generate(_model, _tokenizer, prompt, max_tokens, sampler):
            for index, part in enumerate(parts):
                if index == 1 and before_second is not None:
                    before_second.wait(timeout=2)
                yield SimpleNamespace(text=part)

        self.addCleanup(
            setattr,
            self.server.mlx_lm,
            "stream_generate",
            self.server.mlx_lm.stream_generate,
        )
        self.server.mlx_lm.stream_generate = fake_stream_generate

    def _patch_eval_stream_error(self, exc):
        def fake_stream_generate(_model, _tokenizer, prompt, max_tokens, sampler):
            raise exc
            yield SimpleNamespace(text="")

        self.addCleanup(
            setattr,
            self.server.mlx_lm,
            "stream_generate",
            self.server.mlx_lm.stream_generate,
        )
        self.server.mlx_lm.stream_generate = fake_stream_generate

    def _start_test_http_server(self, tokenizer):
        self._test_handler_class = self.server.make_handler(
            object(), tokenizer, seed_tokens=[1]
        )
        try:
            httpd = self.server.LocalHTTPServer(
                ("127.0.0.1", 0),
                self._test_handler_class,
            )
        except PermissionError as exc:
            raise unittest.SkipTest("loopback socket bind is not permitted") from exc
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        def cleanup():
            httpd.shutdown()
            thread.join(timeout=2)
            httpd.server_close()

        self.addCleanup(cleanup)
        return httpd.server_address[1]

    def _request_json(self, conn, method, path, payload):
        body = json.dumps(payload).encode("utf-8")
        conn.request(
            method,
            path,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
        )


class FakeTokenizer:
    def __init__(self):
        self._eos_token_ids = {1}
        self.apply_kwargs = None

    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        self.apply_kwargs = kwargs
        return "prompt"

    def decode(self, tokens):
        return "".join(str(token) for token in tokens)


if __name__ == "__main__":
    unittest.main()
