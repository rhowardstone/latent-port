"""End-to-end dense rendered-text baseline for Qwen3-VL.

Unlike the representation probe, this exercises the entire deployed path: image
processor -> vision tower -> language model -> generated transcription. Uniform
random base32 characters carry exactly five source bits each and are difficult for
the language model to repair from context.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor


BASE32_PATTERN = re.compile(r"[A-Z2-7]")
DEFAULT_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")


def random_base32(characters: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    raw = rng.bytes(math.ceil(characters * 5 / 8) + 2)
    return base64.b32encode(raw).decode("ascii")[:characters]


def render_text_grid(
    text: str,
    *,
    side: int = 512,
    font_size: int = 12,
    margin: int = 8,
    font_path: Path = DEFAULT_FONT,
) -> tuple[Image.Image, dict]:
    font = ImageFont.truetype(str(font_path), font_size)
    probe_box = font.getbbox("M")
    char_width = max(1, probe_box[2] - probe_box[0])
    line_height = max(1, probe_box[3] - probe_box[1] + 3)
    columns = max(1, (side - margin * 2) // char_width)
    rows = max(1, (side - margin * 2) // line_height)
    if len(text) > columns * rows:
        raise ValueError(
            f"{len(text)} characters do not fit; capacity is {columns * rows} "
            f"at font size {font_size}"
        )
    lines = [text[start : start + columns] for start in range(0, len(text), columns)]
    image = Image.new("RGB", (side, side), "white")
    draw = ImageDraw.Draw(image)
    for row, line in enumerate(lines):
        draw.text((margin, margin + row * line_height), line, font=font, fill="black")
    return image, {
        "columns": columns,
        "rows": rows,
        "character_capacity": columns * rows,
        "char_width": char_width,
        "line_height": line_height,
    }


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def parse_ints(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",")})
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def load_model(model_id: str, device: str):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    preferred = "flash_attention_2" if device.startswith("cuda") else "eager"
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True, attn_implementation=preferred
        )
        attention = preferred
    except (ImportError, ValueError, RuntimeError):
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
        )
        attention = "sdpa"
    return model.to(device).eval(), processor, attention


def transcribe(model, processor, image: Image.Image, expected_characters: int, device: str):
    prompt = (
        "Transcribe the random base32 code in this image exactly. "
        "Return only the characters A-Z and 2-7, with no spaces, punctuation, or explanation."
    )
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    input_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max(64, math.ceil(expected_characters * 1.5)),
            do_sample=False,
        )
    raw = processor.decode(generated[0, input_length:], skip_special_tokens=True)
    normalized = "".join(BASE32_PATTERN.findall(raw.upper()))
    grid = inputs["image_grid_thw"]
    merge = int(getattr(model.config.vision_config, "spatial_merge_size", 2))
    visual_tokens = int((grid.prod(dim=-1) // (merge * merge)).sum().item())
    return raw, normalized, visual_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--characters", type=parse_ints, default=parse_ints("64,128,256,512"))
    parser.add_argument("--side", type=int, default=512)
    parser.add_argument("--font-size", type=int, default=12)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/rendered_text.json"))
    args = parser.parse_args()

    model, processor, attention = load_model(args.model, args.device)
    results = []
    for characters in args.characters:
        source = random_base32(characters, args.seed + characters)
        image, layout = render_text_grid(
            source, side=args.side, font_size=args.font_size, font_path=args.font
        )
        started = time.monotonic()
        raw, recovered, visual_tokens = transcribe(model, processor, image, characters, args.device)
        distance = levenshtein(source, recovered)
        result = {
            "characters": characters,
            "source_bits": characters * 5,
            "visual_tokens": visual_tokens,
            "source_bits_per_visual_token": characters * 5 / visual_tokens,
            "exact": recovered == source,
            "edit_distance": distance,
            "character_accuracy": max(0.0, 1.0 - distance / max(len(source), len(recovered), 1)),
            "recovered_characters": len(recovered),
            "seconds": time.monotonic() - started,
            "layout": layout,
            "raw_output": raw,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "attention_implementation": attention,
                    "font": str(args.font),
                    "font_size": args.font_size,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()

