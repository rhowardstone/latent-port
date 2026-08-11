"""Trace held-out visual bits through Qwen3-VL's frozen language model.

The vision features are loaded from qwen_probe's cache. This isolates the causal
decoder and avoids rerunning the expensive vision tower. A shared per-visual-token
ridge probe is fit at the decoder input, after each DeepStack injection, at selected
later layers, and after the final decoder norm.

This remains a representation lower bound: it asks whether token-local information
survives inside the LLM, not whether the unadapted generative head can spell it out.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .patterns import random_payload_images
from .qwen_probe import (
    bsc_equivalent_rate,
    cached_extract_features,
    effective_patch,
    fit_probe,
    parse_bits,
    parse_floats,
)


def parse_layers(value: str) -> list[int]:
    layers = sorted({int(item) for item in value.split(",")})
    if not layers or layers[0] < 0:
        raise argparse.ArgumentTypeError("layers must be non-negative comma-separated integers")
    return layers


def load_decoder(model_id: str, device: str):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    preferred = "flash_attention_2" if device.startswith("cuda") else "eager"
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=preferred,
        )
        attention = preferred
    except (ImportError, ValueError, RuntimeError) as exc:
        if preferred != "flash_attention_2":
            raise
        print(f"FlashAttention unavailable ({exc}); falling back to SDPA", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        attention = "sdpa"
    # Cached final/DeepStack features replace the visual tower, and no logits are
    # needed. Drop both before GPU transfer to reduce peak as well as steady VRAM.
    model.model.visual = None
    model.lm_head = None
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor, attention


def _template_batch(processor, images, device: str):
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Inspect the image."},
                ],
            }
        ]
        for image in images
    ]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    )
    return {
        key: value.to(device)
        for key, value in inputs.items()
        if key in {"input_ids", "attention_mask", "image_grid_thw"}
    }


@torch.inference_mode()
def trace_decoder(
    model,
    processor,
    images,
    feature_views: dict[str, torch.Tensor],
    *,
    batch_size: int,
    layers: list[int],
    device: str,
    label: str,
) -> tuple[dict[str, torch.Tensor], float]:
    decoder = model.model.language_model
    invalid = [layer for layer in layers if layer >= len(decoder.layers)]
    if invalid:
        raise ValueError(f"decoder has {len(decoder.layers)} layers; invalid requests: {invalid}")
    captured: dict[str, list[torch.Tensor]] = {
        "llm_input": [],
        **{f"before_layer_{layer}": [] for layer in layers if layer != 0},
        "llm_final": [],
    }
    started = time.monotonic()
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        batch_features = {
            name: values[start:stop].to(device=device, dtype=next(decoder.parameters()).dtype)
            for name, values in feature_views.items()
        }
        template = _template_batch(processor, images[start:stop], device)
        input_ids = template["input_ids"]
        attention_mask = template["attention_mask"]
        image_grid_thw = template["image_grid_thw"]
        inputs_embeds = model.model.get_input_embeddings()(input_ids)
        final_flat = batch_features["final"].flatten(0, 1)
        image_mask, _ = model.model.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=final_flat
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, final_flat)
        visual_mask = image_mask[..., 0]
        position_ids, _ = model.model.get_rope_index(
            input_ids,
            image_grid_thw,
            None,
            attention_mask=attention_mask,
        )
        batch_capture: dict[str, torch.Tensor] = {}

        def capture(name: str):
            def hook(_module, args):
                values = args[0][visual_mask].reshape(stop - start, -1, args[0].shape[-1])
                batch_capture[name] = values.detach().cpu()

            return hook

        handles = []
        for layer in layers:
            name = "llm_input" if layer == 0 else f"before_layer_{layer}"
            handles.append(decoder.layers[layer].register_forward_pre_hook(capture(name)))
        try:
            outputs = decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                visual_pos_masks=visual_mask,
                deepstack_visual_embeds=[
                    batch_features[f"deepstack_{index}"].flatten(0, 1)
                    for index in range(3)
                ],
            )
        finally:
            for handle in handles:
                handle.remove()
        final_values = outputs.last_hidden_state[visual_mask].reshape(
            stop - start, -1, outputs.last_hidden_state.shape[-1]
        )
        batch_capture["llm_final"] = final_values.detach().cpu()
        missing = set(captured) - set(batch_capture)
        if missing:
            raise RuntimeError(f"decoder hooks did not capture: {sorted(missing)}")
        for name, values in batch_capture.items():
            captured[name].append(values)
        print(
            f"  {label} traced {stop}/{len(images)} images in {time.monotonic() - started:.1f}s",
            flush=True,
        )
    return {name: torch.cat(parts) for name, parts in captured.items()}, time.monotonic() - started


def cached_views(model, processor, images, args, bits: int, patch: int, split: str):
    seed = (1000 if split == "train" else 9000) + bits
    return cached_extract_features(
        model,
        processor,
        images,
        args.batch_size,
        args.device,
        label=f"{bits}b {split}",
        cache_dir=args.cache_dir,
        metadata={
            "model": args.model,
            "grid": args.grid,
            "bits": bits,
            "patch": patch,
            "amplitude": args.amplitude,
            "jpeg_quality": args.jpeg_quality,
            "scale": args.scale,
            "blur": args.blur,
            "noise": args.noise,
            "split": "validation" if split == "val" else split,
            "samples": len(images),
            "seed": seed,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--bits", type=parse_bits, default=parse_bits("4,8"))
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("0,1,2,3,8,16,24"))
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--amplitude", type=float, default=64.0)
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--blur", type=float, default=0.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--ridge-lambdas", type=parse_floats, default=parse_floats("0.01,0.1,1,10,100"))
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/feature_cache"))
    parser.add_argument("--output", type=Path, default=Path("runs/qwen_llm_probe.json"))
    args = parser.parse_args()

    model, processor, attention = load_decoder(args.model, args.device)
    patch = effective_patch(model)
    target_pixels = (args.grid * patch) ** 2
    processor.image_processor.min_pixels = target_pixels
    processor.image_processor.max_pixels = target_pixels
    processor.image_processor.size = {
        "shortest_edge": target_pixels,
        "longest_edge": target_pixels,
    }
    results = []
    for bits in args.bits:
        common = dict(
            grid=args.grid,
            bits=bits,
            patch=patch,
            amplitude=args.amplitude,
            jpeg_quality=args.jpeg_quality,
            scale=args.scale,
            blur=args.blur,
            noise=args.noise,
        )
        train_images, train_y = random_payload_images(
            samples=args.train_samples, seed=1000 + bits, **common
        )
        val_images, val_y = random_payload_images(
            samples=args.val_samples, seed=9000 + bits, **common
        )
        train_views = cached_views(model, processor, train_images, args, bits, patch, "train")
        val_views = cached_views(model, processor, val_images, args, bits, patch, "val")
        train_depths, train_seconds = trace_decoder(
            model,
            processor,
            train_images,
            train_views,
            batch_size=args.batch_size,
            layers=args.layers,
            device=args.device,
            label=f"{bits}b train",
        )
        val_depths, val_seconds = trace_decoder(
            model,
            processor,
            val_images,
            val_views,
            batch_size=args.batch_size,
            layers=args.layers,
            device=args.device,
            label=f"{bits}b val",
        )
        truth_train = torch.from_numpy(train_y)
        truth_val = torch.from_numpy(val_y)
        for depth, train_x in train_depths.items():
            scores, probe_seconds = fit_probe(
                train_x.float(),
                truth_train,
                val_depths[depth].float(),
                truth_val,
                kind="ridge",
                epochs=1,
                device=args.device,
                seed=args.seed + bits,
                ridge_lambdas=args.ridge_lambdas,
            )
            scores["bsc_equivalent_bits_per_token"] = bsc_equivalent_rate(
                bits, scores["bit_accuracy"]
            )
            result = {
                "bits_per_visual_token": bits,
                "depth": depth,
                "hidden_size": train_x.shape[-1],
                "train_rows": train_x.shape[0] * train_x.shape[1],
                "rows_per_feature": train_x.shape[0] * train_x.shape[1] / train_x.shape[-1],
                "decoder_train_seconds": train_seconds,
                "decoder_validation_seconds": val_seconds,
                "probe_seconds": probe_seconds,
                **scores,
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "attention_implementation": attention,
                    "layers": args.layers,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
