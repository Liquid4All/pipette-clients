"""Tokenizer session for pipette-mlx benchmarks.

Loads the tokenizer once, writes `{"eos_ids": [...]}` as the first stdout
line, then answers each stdin line `{"text": "..."}` with `{"count": N}` until
stdin closes.

Counting uses the same tokenizer the benchmark server runs
(`mlx_lm.tokenizer_utils.load`, falling back to transformers), loaded with the
server's own `--trust-remote-code` setting and counted with default encode
options — the same call shape `mlx_lm.server` uses to tokenize a
`/v1/completions` string prompt, so counts agree with its `usage` block.
"""

import argparse
import json
import sys
from pathlib import Path


def load_tokenizer(model_dir, trust_remote_code):
    try:
        from mlx_lm.tokenizer_utils import load as mlx_load_tokenizer
    except ImportError:
        mlx_load_tokenizer = None
    if mlx_load_tokenizer is not None:
        try:
            return mlx_load_tokenizer(
                Path(model_dir), {"trust_remote_code": trust_remote_code}
            )
        except Exception as exc:  # noqa: BLE001 - fall back below
            print(
                f"mlx tokenizer load failed ({exc}); trying transformers",
                file=sys.stderr,
            )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)


def parse_text(request):
    text = request.get("text") if isinstance(request, dict) else None
    if not isinstance(text, str):
        raise ValueError('request line must be {"text": str}')
    return text


def as_id_list(ids):
    if ids is None:
        return []
    if isinstance(ids, int):
        return [ids]
    return [int(i) for i in ids]


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def config_eos_ids(model_dir):
    """`eos_token_id`s a server can stop on that the tokenizer config may not
    name: config.json's, its `text_config`'s (where VL checkpoints keep it) and
    generation_config.json's."""
    config = read_json(Path(model_dir) / "config.json")
    text_config = config.get("text_config") or {}
    generation = read_json(Path(model_dir) / "generation_config.json")
    return sorted(
        set(as_id_list(config.get("eos_token_id")))
        | set(as_id_list(text_config.get("eos_token_id")))
        | set(as_id_list(generation.get("eos_token_id")))
    )


def eos_ids(tokenizer, model_dir):
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids is None:
        ids = getattr(tokenizer, "eos_token_id", None)
    return sorted(set(as_id_list(ids)) | set(config_eos_ids(model_dir)))


def serve(tokenizer, model_dir, stdin, stdout):
    def reply(body):
        stdout.write(json.dumps(body) + "\n")
        stdout.flush()

    reply({"eos_ids": eos_ids(tokenizer, model_dir)})
    for line in stdin:
        if not line.strip():
            continue
        text = parse_text(json.loads(line))
        reply({"count": len(tokenizer.encode(text))})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    tokenizer = load_tokenizer(args.model_dir, args.trust_remote_code)
    serve(tokenizer, args.model_dir, sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
