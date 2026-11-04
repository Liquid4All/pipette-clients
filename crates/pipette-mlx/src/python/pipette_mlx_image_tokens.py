"""Counts the image placeholder ids the model's processor builds for `--num-images` gray
frames, excluding boundary markers. Loads no weights; writes `{"image_tokens": N}`."""

import argparse
import json
import sys
from pathlib import Path

ID_KEYS = ("image_token_id", "image_token_index", "img_context_token_id")


def placeholder_id(processor, config):
    """The processor's image placeholder id, else the config's."""
    for owner in (processor, getattr(processor, "tokenizer", None)):
        for key in ("image_token_id", "img_context_token_id"):
            value = getattr(owner, key, None)
            if isinstance(value, int):
                return value
    token = getattr(processor, "image_token", None)
    tokenizer = getattr(processor, "tokenizer", processor)
    if isinstance(token, str):
        value = tokenizer.convert_tokens_to_ids(token)
        if isinstance(value, int) and value != getattr(tokenizer, "unk_token_id", None):
            return value
    for section in (config, config.get("text_config") or {}):
        for key in ID_KEYS:
            if isinstance(section.get(key), int):
                return section[key]
    raise ValueError("the processor and config name no image placeholder id")


def count(model_dir, width, height, num_images, trust_remote_code):
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import (
        get_model_and_args,
        load_config,
        load_image_processor,
        load_processor,
        prepare_inputs,
    )
    from PIL import Image

    path = Path(model_dir)
    config = load_config(path, trust_remote_code=trust_remote_code)
    # Mirrors `mlx_vlm.load` minus the weights.
    get_model_and_args(config)
    processor = load_processor(path, trust_remote_code=trust_remote_code)
    image_processor = load_image_processor(path, trust_remote_code=trust_remote_code)
    if image_processor is not None:
        processor.image_processor = image_processor
    prompt = apply_chat_template(processor, config, "Describe", num_images=num_images)
    image = Image.new("RGB", (width, height), (128, 128, 128))
    inputs = prepare_inputs(
        processor,
        images=[image] * num_images,
        prompts=prompt,
        image_token_index=config.get("image_token_index"),
    )
    ids = inputs["input_ids"].tolist()
    ids = ids[0] if ids and isinstance(ids[0], list) else ids
    wanted = placeholder_id(processor, config)
    return expanded(sum(1 for token in ids if token == wanted), num_images)


def expanded(tokens, num_images):
    """Rejects a count with no more than one token per image: the processor left the
    placeholder unexpanded, so the count says nothing about the image."""
    if tokens <= num_images:
        raise ValueError(f"{tokens} placeholder tokens for {num_images} images")
    return tokens


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-images", type=int, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    tokens = count(
        args.model_dir, args.width, args.height, args.num_images, args.trust_remote_code
    )
    sys.stdout.write(json.dumps({"image_tokens": tokens}) + "\n")


if __name__ == "__main__":
    main()
