"""Contract tests for the pipette-mlx tokenizer helper.

The pure logic is imported with faked tokenizer backends, so it needs neither
`mlx_lm` nor `transformers` and counts toward coverage; the spawned helper and
real-backend counting run as subprocesses.
"""

import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import ModuleType


def load_helper_module():
    helper_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "python"
        / "pipette_mlx_tokenize.py"
    )
    spec = importlib.util.spec_from_file_location("pipette_mlx_tokenize", helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StubTokenizer:
    """Stands in for either backend's tokenizer: `encode` plus EOS ids."""

    def __init__(self, counts, eos_ids):
        self._counts = counts
        self.eos_token_ids = eos_ids

    def encode(self, text):
        return list(range(self._counts[text]))


def make_auto_tokenizer(tokenizer, trust_remote_code):
    expected = trust_remote_code

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_dir, trust_remote_code):
            assert trust_remote_code is expected
            assert model_dir == "/models/x"
            return tokenizer

    return FakeAutoTokenizer


HELPER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "python",
    "pipette_mlx_tokenize.py",
)

TOKENIZER_JSON = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {"[UNK]": 0, "[EOS]": 1, "hello": 2, "world": 3},
        "unk_token": "[UNK]",
    },
}

TOKENIZER_CONFIG_JSON = {
    "eos_token": "[EOS]",
    "unk_token": "[UNK]",
}


def run_helper(model_dir, texts):
    # As pipette-mlx spawns it: the script's text through `-c`.
    return subprocess.run(
        [
            sys.executable,
            "-c",
            pathlib.Path(HELPER).read_text(),
            "--model-dir",
            model_dir,
        ],
        input="".join(json.dumps({"text": text}) + "\n" for text in texts),
        capture_output=True,
        text=True,
        timeout=300,
    )


def reply_lines(stdout):
    return [json.loads(line) for line in stdout.splitlines() if line.strip()]


def has_tokenizer_backend():
    try:
        import mlx_lm.tokenizer_utils  # noqa: F401
    except ImportError:
        pass
    else:
        return True
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


class HelperUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helper = load_helper_module()

    def test_parse_text_rejects_malformed_lines(self):
        for body in [{}, {"text": 1}, {"text": None}, ["hello"], "hello"]:
            with self.assertRaises(ValueError, msg=f"{body!r}"):
                self.helper.parse_text(body)

    def test_parse_text_accepts_strings_and_empty(self):
        self.assertEqual(self.helper.parse_text({"text": "a"}), "a")
        self.assertEqual(self.helper.parse_text({"text": ""}), "")

    def test_eos_ids_reads_list_int_and_missing_shapes(self):
        no_config = "/nonexistent-model-dir"
        make = lambda eos: StubTokenizer({}, eos)  # noqa: E731
        self.assertEqual(self.helper.eos_ids(make([7, 1]), no_config), [1, 7])
        self.assertEqual(self.helper.eos_ids(make(9), no_config), [9])

        single = StubTokenizer({}, [])
        single.eos_token_ids = None
        single.eos_token_id = 4
        self.assertEqual(self.helper.eos_ids(single, no_config), [4])

        neither = StubTokenizer({}, [])
        neither.eos_token_ids = None
        self.assertEqual(self.helper.eos_ids(neither, no_config), [])

    def test_eos_ids_include_the_config_files_ids(self):
        # The servers stop on these, which can name ids the tokenizer alone
        # does not report; missing one ends decode early.
        cases = [
            ({"config.json": {"eos_token_id": [1, 5, 9]}}, [1, 5, 9]),
            ({"config.json": {"eos_token_id": 5}}, [1, 5]),
            ({"config.json": {"eos_token_id": None}}, [1]),
            ({"config.json": {"text_config": {"eos_token_id": 6}}}, [1, 6]),
            ({"config.json": {"text_config": None}}, [1]),
            ({"generation_config.json": {"eos_token_id": [2, 3]}}, [1, 2, 3]),
        ]
        for files, expected in cases:
            with tempfile.TemporaryDirectory() as model_dir:
                for name, body in files.items():
                    with open(os.path.join(model_dir, name), "w") as fid:
                        json.dump(body, fid)
                self.assertEqual(
                    self.helper.eos_ids(StubTokenizer({}, [1]), model_dir),
                    expected,
                    f"{files!r}",
                )

    def test_serve_reports_eos_ids_then_answers_each_line(self):
        stdin = io.StringIO(
            json.dumps({"text": "hello world"})
            + "\n\n"
            + json.dumps({"text": "hello"})
            + "\n"
        )
        stdout = io.StringIO()
        self.helper.serve(
            StubTokenizer({"hello world": 2, "hello": 1}, [3]),
            "/nonexistent-model-dir",
            stdin,
            stdout,
        )
        self.assertEqual(
            reply_lines(stdout.getvalue()),
            [{"eos_ids": [3]}, {"count": 2}, {"count": 1}],
        )

    def test_serve_rejects_a_malformed_line(self):
        with self.assertRaises(ValueError):
            self.helper.serve(
                StubTokenizer({}, []),
                "/nonexistent-model-dir",
                io.StringIO(json.dumps({"text": 1}) + "\n"),
                io.StringIO(),
            )

    def test_main_loads_from_the_model_dir_as_the_server_does(self):
        # Remote code only when asked: `mlx_lm.server` runs with it,
        # `mlx_vlm.server` without.
        seen = []
        real_load = self.helper.load_tokenizer
        self.helper.load_tokenizer = lambda model_dir, trust: (
            seen.append((model_dir, trust)),
            StubTokenizer({"ab": 2}, [3]),
        )[1]
        old_stdin = sys.stdin
        try:
            for flags in ([], ["--trust-remote-code"]):
                sys.stdin = io.StringIO(json.dumps({"text": "ab"}) + "\n")
                out = io.StringIO()
                with redirect_stdout(out):
                    self.helper.main(["--model-dir", "/models/entry", *flags])
                self.assertEqual(
                    reply_lines(out.getvalue()), [{"eos_ids": [3]}, {"count": 2}]
                )
        finally:
            sys.stdin = old_stdin
            self.helper.load_tokenizer = real_load
        self.assertEqual(seen, [("/models/entry", False), ("/models/entry", True)])

    def test_load_tokenizer_prefers_mlx_lm(self):
        sentinel = StubTokenizer({}, [])
        saved = {
            name: sys.modules.get(name) for name in ("mlx_lm", "mlx_lm.tokenizer_utils")
        }
        fake_utils = ModuleType("mlx_lm.tokenizer_utils")
        seen = []
        fake_utils.load = lambda path, extra: (seen.append(extra), sentinel)[1]
        fake_mlx_lm = ModuleType("mlx_lm")
        fake_mlx_lm.tokenizer_utils = fake_utils
        sys.modules["mlx_lm"] = fake_mlx_lm
        sys.modules["mlx_lm.tokenizer_utils"] = fake_utils
        try:
            for trust_remote_code in (True, False):
                self.assertIs(
                    self.helper.load_tokenizer("/models/x", trust_remote_code),
                    sentinel,
                )
            self.assertEqual(
                seen, [{"trust_remote_code": True}, {"trust_remote_code": False}]
            )
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_load_tokenizer_falls_back_to_transformers(self):
        # `None` is mlx_lm not installed, `"boom"` is its load raising: both
        # must reach transformers.
        for mlx_load in (None, "boom"):
            sentinel = StubTokenizer({}, [])
            saved = {
                name: sys.modules.get(name)
                for name in (
                    "mlx_lm",
                    "mlx_lm.tokenizer_utils",
                    "transformers",
                )
            }
            sys.modules.pop("mlx_lm", None)
            sys.modules.pop("mlx_lm.tokenizer_utils", None)
            if mlx_load is not None:
                fake_utils = ModuleType("mlx_lm.tokenizer_utils")

                def _boom(path, extra):
                    raise RuntimeError("no weights here")

                fake_utils.load = _boom
                fake_mlx_lm = ModuleType("mlx_lm")
                fake_mlx_lm.tokenizer_utils = fake_utils
                sys.modules["mlx_lm"] = fake_mlx_lm
                sys.modules["mlx_lm.tokenizer_utils"] = fake_utils
            fake_transformers = ModuleType("transformers")
            sys.modules["transformers"] = fake_transformers
            try:
                for trust_remote_code in (True, False):
                    fake_transformers.AutoTokenizer = make_auto_tokenizer(
                        sentinel, trust_remote_code
                    )
                    self.assertIs(
                        self.helper.load_tokenizer("/models/x", trust_remote_code),
                        sentinel,
                    )
            finally:
                for name, module in saved.items():
                    if module is None:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = module


class HelperContract(unittest.TestCase):
    @unittest.skipUnless(
        has_tokenizer_backend(), "neither mlx_lm nor transformers importable"
    )
    def test_counts_texts_and_reports_eos_ids(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, "tokenizer.json"), "w") as fid:
                json.dump(TOKENIZER_JSON, fid)
            with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as fid:
                json.dump(TOKENIZER_CONFIG_JSON, fid)
            proc = run_helper(model_dir, ["hello world", "hello"])
        self.assertEqual(proc.returncode, 0, f"helper failed: {proc.stderr!r}")
        self.assertEqual(
            reply_lines(proc.stdout),
            [{"eos_ids": [1]}, {"count": 2}, {"count": 1}],
        )

    @unittest.skipUnless(
        has_tokenizer_backend(), "neither mlx_lm nor transformers importable"
    )
    def test_rejects_a_malformed_line(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with open(os.path.join(model_dir, "tokenizer.json"), "w") as fid:
                json.dump(TOKENIZER_JSON, fid)
            with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as fid:
                json.dump(TOKENIZER_CONFIG_JSON, fid)
            proc = run_helper(model_dir, [1])
        self.assertNotEqual(
            proc.returncode, 0, f"expected failure, got {proc.stdout!r}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
