"""Tests for the image token helper, with faked processors."""

import importlib.util
import io
import json
import pathlib
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace


def load_helper_module():
    helper_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "python"
        / "pipette_mlx_image_tokens.py"
    )
    spec = importlib.util.spec_from_file_location(
        "pipette_mlx_image_tokens", helper_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_helper_module()


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self, vocab):
        self.vocab = vocab

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)


class PlaceholderIdTest(unittest.TestCase):
    def test_the_processors_own_id_wins(self):
        processor = SimpleNamespace(image_token_id=396, tokenizer=None)
        self.assertEqual(helper.placeholder_id(processor, {"image_token_id": 7}), 396)

    def test_the_tokenizers_id(self):
        processor = SimpleNamespace(tokenizer=SimpleNamespace(image_token_id=151655))
        self.assertEqual(helper.placeholder_id(processor, {}), 151655)

    def test_the_processors_token_resolved_through_the_tokenizer(self):
        processor = SimpleNamespace(
            image_token="<image>", tokenizer=FakeTokenizer({"<image>": 396})
        )
        self.assertEqual(helper.placeholder_id(processor, {}), 396)

    def test_an_unknown_token_falls_back_to_the_config(self):
        processor = SimpleNamespace(image_token="<nope>", tokenizer=FakeTokenizer({}))
        self.assertEqual(
            helper.placeholder_id(processor, {"image_token_index": 262144}), 262144
        )

    def test_the_text_configs_id(self):
        processor = SimpleNamespace(tokenizer=None)
        config = {"text_config": {"image_token_id": 248056}}
        self.assertEqual(helper.placeholder_id(processor, config), 248056)

    def test_internvls_context_token_id(self):
        processor = SimpleNamespace(img_context_token_id=151667, tokenizer=None)
        self.assertEqual(helper.placeholder_id(processor, {}), 151667)

    def test_no_id_anywhere_is_refused(self):
        with self.assertRaises(ValueError):
            helper.placeholder_id(SimpleNamespace(tokenizer=None), {})


class ExpandedTest(unittest.TestCase):
    def test_a_count_per_image_above_one_passes(self):
        self.assertEqual(helper.expanded(512, 2), 512)

    def test_one_placeholder_per_prompt_is_refused(self):
        for tokens, images in ((1, 1), (1, 3), (3, 3)):
            with self.assertRaises(ValueError):
                helper.expanded(tokens, images)


class MainTest(unittest.TestCase):
    def test_main_prints_the_count_as_json(self):
        seen = {}

        def fake_count(model_dir, width, height, num_images, trust_remote_code):
            seen.update(
                model_dir=model_dir,
                width=width,
                height=height,
                num_images=num_images,
                trust=trust_remote_code,
            )
            return 768

        original = helper.count
        helper.count = fake_count
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                helper.main(
                    [
                        "--model-dir",
                        "/m",
                        "--width",
                        "512",
                        "--height",
                        "256",
                        "--num-images",
                        "3",
                        "--trust-remote-code",
                    ]
                )
        finally:
            helper.count = original
        self.assertEqual(json.loads(out.getvalue()), {"image_tokens": 768})
        self.assertEqual(
            seen,
            {
                "model_dir": "/m",
                "width": 512,
                "height": 256,
                "num_images": 3,
                "trust": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
