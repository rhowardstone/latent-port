"""Measure exact random-bit capacity of a frozen Qwen visual encoder.

This is intentionally a representation probe, not an OCR benchmark. It asks:
given a patch-aligned visual code and frozen Qwen weights, how many independent
held-out bits can a position-invariant receiver recover from token *i alone*?

Because the ViT mixes information between patches, this local probe is a lower
bound on joint image-channel capacity. A decoder attending to every visual token
may recover distributed information that no individual token exposes linearly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForImageTextToText, AutoProcessor

from .patterns import clipping_fraction, random_payload_images


@dataclass
class ProbeResult:
    bits_per_visual_token: int
    feature_view: str
    train_samples: int
    validation_samples: int
    visual_tokens_per_image: int
    hidden_size: int
    train_rows: int
    rows_per_feature: float
    probe_kind: str
    selected_ridge: float | None
    train_bit_accuracy: float
    train_token_exact_accuracy: float
    bit_accuracy: float
    bsc_equivalent_bits_per_token: float
    bit_accuracy_bootstrap_se: float
    bit_accuracy_ci95_low: float
    bit_accuracy_ci95_high: float
    token_exact_accuracy: float
    image_exact_accuracy: float
    image_bit_error_histogram: dict[str, int]
    validation_loss: float
    validation_loss_kind: str
    clipping_fraction: float
    encoder_seconds: float
    probe_seconds: float
    jpeg_quality: int
    scale: float
    blur: float
    noise: float


class Probe(nn.Module):
    def __init__(self, width: int, bits: int, kind: str = "linear") -> None:
        super().__init__()
        if kind == "linear":
            self.net = nn.Linear(width, bits)
        elif kind == "mlp":
            hidden = min(2048, max(256, width // 2))
            self.net = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, bits)
            )
        else:
            raise ValueError(f"unknown probe kind: {kind}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def effective_patch(model) -> int:
    visual = model.config.vision_config
    patch = int(getattr(visual, "patch_size", 14))
    merge = int(getattr(visual, "spatial_merge_size", 2))
    return patch * merge


def _image_feature_views(output) -> dict[str, list[torch.Tensor]]:
    """Split final and Qwen3 DeepStack streams into equal per-image tensors."""
    deepstack = []
    final = output
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[0], (tuple, list)):
        final, deepstack = output
    final_list = [final] if isinstance(final, torch.Tensor) else list(final)
    split_sizes = [item.shape[0] for item in final_list]
    views = {"final": final_list}
    for index, features in enumerate(deepstack):
        views[f"deepstack_{index}"] = list(torch.split(features, split_sizes))
    return views


@torch.inference_mode()
def extract_features(
    model, processor, images, batch_size: int, device: str, label: str = ""
) -> dict[str, torch.Tensor]:
    batches: dict[str, list[torch.Tensor]] = {}
    expected_tokens = None
    started = time.monotonic()
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        preprocess_started = time.monotonic()
        inputs = processor.image_processor(images=batch, return_tensors="pt")
        preprocess_seconds = time.monotonic() - preprocess_started
        pixels = inputs["pixel_values"].to(device)
        grid_thw = inputs["image_grid_thw"].to(device)
        forward_started = time.monotonic()
        output = model.get_image_features(pixel_values=pixels, image_grid_thw=grid_thw)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        forward_seconds = time.monotonic() - forward_started
        feature_views = _image_feature_views(output)
        lengths = {item.shape[0] for item in feature_views["final"]}
        if len(lengths) != 1:
            raise RuntimeError(f"images produced different visual-token counts: {lengths}")
        stacked_views = {
            name: torch.stack([item.detach().cpu() for item in items])
            for name, items in feature_views.items()
        }
        features = stacked_views["final"]
        expected_tokens = expected_tokens or features.shape[1]
        if features.shape[1] != expected_tokens:
            raise RuntimeError("visual-token count changed between batches")
        for name, view_features in stacked_views.items():
            batches.setdefault(name, []).append(view_features)
        print(
            f"  {label} encoded {min(start + batch_size, len(images))}/{len(images)} "
            f"images in {time.monotonic() - started:.1f}s "
            f"(preprocess {preprocess_seconds:.2f}s, vision {forward_seconds:.2f}s)",
            flush=True,
        )
    return {name: torch.cat(parts, dim=0) for name, parts in batches.items()}


CACHE_VERSION = 2


def cached_extract_features(
    model,
    processor,
    images,
    batch_size: int,
    device: str,
    label: str,
    cache_dir: Path,
    metadata: dict,
) -> dict[str, torch.Tensor]:
    document = {"cache_version": CACHE_VERSION, **metadata}
    digest = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()[:20]
    path = cache_dir / f"features_{digest}.pt"
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        print(f"  {label} loaded feature cache {path.name}", flush=True)
        return cached["features"]
    features = extract_features(model, processor, images, batch_size, device, label=label)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": document, "features": features}, path)
    print(f"  {label} saved feature cache {path.name}", flush=True)
    return features


def fit_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    kind: str,
    epochs: int,
    device: str,
    seed: int,
    ridge_lambdas: list[float],
) -> tuple[dict, float]:
    started = time.monotonic()
    torch.manual_seed(seed)
    width = train_x.shape[-1]
    bits = train_y.shape[-1]
    loss_fn = nn.BCEWithLogitsLoss()
    x = train_x.reshape(-1, width).to(device)
    y = train_y.reshape(-1, bits).float().to(device)
    vx = val_x.reshape(-1, width).to(device)
    vy = val_y.reshape(-1, bits).float().to(device)
    selected_ridge = None

    if kind == "ridge":
        # Select lambda on a deterministic 20% calibration split, then refit on
        # every training row. The reported validation split remains untouched.
        order = torch.randperm(len(x), device=device)
        split = max(1, int(len(order) * 0.8))
        fit_indices, calibration_indices = order[:split], order[split:]
        best_score = -1.0
        for ridge, (weight, bias, mean, scale) in zip(
            ridge_lambdas, _fit_ridge_path(x[fit_indices], y[fit_indices], ridge_lambdas)
        ):
            logits = _ridge_predict(x[calibration_indices], weight, bias, mean, scale)
            score = ((logits > 0) == y[calibration_indices].bool()).float().mean().item()
            if score > best_score:
                best_score = score
                selected_ridge = ridge
        assert selected_ridge is not None
        weight, bias, mean, scale = _fit_ridge(x, y, selected_ridge)
        train_logits = _ridge_predict(x, weight, bias, mean, scale)
        val_logits = _ridge_predict(vx, weight, bias, mean, scale)
        validation_loss = torch.mean((val_logits - (vy * 2.0 - 1.0)) ** 2).item()
        validation_loss_kind = "ridge_mse_on_signed_bits"
    else:
        probe = Probe(width, bits, kind=kind).to(device)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=2e-3, weight_decay=1e-4)
        for _ in range(epochs):
            order = torch.randperm(len(x), device=device)
            for indices in order.split(4096):
                logits = probe(x[indices])
                loss = loss_fn(logits, y[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        with torch.inference_mode():
            train_logits = probe(x)
            val_logits = probe(vx)
            validation_loss = loss_fn(val_logits, vy).item()
            validation_loss_kind = "binary_cross_entropy"

    with torch.inference_mode():
        train_truth = y.bool()
        train_predicted = train_logits > 0
        truth = vy.bool()
        predicted = val_logits > 0
        bit_accuracy = (predicted == truth).float().mean().item()
        token_exact = (predicted == truth).all(dim=-1)
        tokens_per_image = val_x.shape[1]
        image_exact = token_exact.reshape(val_x.shape[0], tokens_per_image).all(dim=-1)
        per_image_errors = (predicted != truth).sum(dim=-1).reshape(
            val_x.shape[0], tokens_per_image
        ).sum(dim=-1)
        per_image_accuracy = (predicted == truth).float().reshape(
            val_x.shape[0], tokens_per_image, bits
        ).mean(dim=(1, 2)).cpu().numpy()
        bootstrap_rng = np.random.default_rng(seed + 1701)
        bootstrap_indices = bootstrap_rng.integers(
            0, len(per_image_accuracy), size=(2000, len(per_image_accuracy))
        )
        bootstrap_means = per_image_accuracy[bootstrap_indices].mean(axis=1)
        unique, counts = torch.unique(per_image_errors, return_counts=True)
        histogram = {str(int(key)): int(count) for key, count in zip(unique, counts)}
        result = {
            "selected_ridge": selected_ridge,
            "train_bit_accuracy": (train_predicted == train_truth).float().mean().item(),
            "train_token_exact_accuracy": (train_predicted == train_truth)
            .all(dim=-1)
            .float()
            .mean()
            .item(),
            "bit_accuracy": bit_accuracy,
            "bit_accuracy_bootstrap_se": float(bootstrap_means.std(ddof=1)),
            "bit_accuracy_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
            "bit_accuracy_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
            "token_exact_accuracy": token_exact.float().mean().item(),
            "image_exact_accuracy": image_exact.float().mean().item(),
            "image_bit_error_histogram": histogram,
            "validation_loss": validation_loss,
            "validation_loss_kind": validation_loss_kind,
        }
    return result, time.monotonic() - started


def _fit_ridge(x: torch.Tensor, y01: torch.Tensor, ridge: float):
    """Fit standardized multi-output ridge regression to {-1,+1} bit targets."""
    mean = x.mean(dim=0)
    scale = x.std(dim=0).clamp_min(1e-5)
    z = (x - mean) / scale
    targets = y01 * 2.0 - 1.0
    bias = targets.mean(dim=0)
    centered_targets = targets - bias
    covariance = z.T @ z / len(z)
    covariance.diagonal().add_(ridge)
    rhs = z.T @ centered_targets / len(z)
    weight = torch.linalg.solve(covariance, rhs)
    return weight, bias, mean, scale


def _fit_ridge_path(x: torch.Tensor, y01: torch.Tensor, ridges: list[float]):
    """Fit a ridge path while building/eigendecomposing the covariance once."""
    mean = x.mean(dim=0)
    scale = x.std(dim=0).clamp_min(1e-5)
    z = (x - mean) / scale
    targets = y01 * 2.0 - 1.0
    bias = targets.mean(dim=0)
    centered_targets = targets - bias
    covariance = z.T @ z / len(z)
    rhs = z.T @ centered_targets / len(z)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0.0)
    projected_rhs = eigenvectors.T @ rhs
    results = []
    for ridge in ridges:
        weight = eigenvectors @ (projected_rhs / (eigenvalues[:, None] + ridge))
        results.append((weight, bias, mean, scale))
    return results


def _ridge_predict(x, weight, bias, mean, scale):
    return ((x - mean) / scale) @ weight + bias


def bsc_equivalent_rate(bits_per_token: int, bit_accuracy: float) -> float:
    """Binary-symmetric-channel equivalent rate, not a correlated-channel capacity."""
    error = min(max(1.0 - bit_accuracy, 0.0), 0.5)
    if error == 0.0:
        return float(bits_per_token)
    entropy = -error * math.log2(error) - (1.0 - error) * math.log2(1.0 - error)
    return bits_per_token * (1.0 - entropy)


def load_model(model_id: str, device: str):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(model_id)
    preferred_attention = "flash_attention_2" if device.startswith("cuda") else "eager"
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=preferred_attention,
        )
        attention = preferred_attention
    except (ImportError, ValueError, RuntimeError) as exc:
        if preferred_attention != "flash_attention_2":
            raise
        print(f"FlashAttention unavailable ({exc}); falling back to SDPA", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        attention = "sdpa"
    # get_image_features only needs the vision tower/projector. Dropping the
    # causal LM before GPU transfer avoids both its steady-state and peak VRAM.
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        model.model.language_model = None
    if hasattr(model, "lm_head"):
        model.lm_head = None
    model = model.to(device)
    model.eval()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor, attention


def _select_view(views: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name == "all_concat":
        return torch.cat([views[key] for key in sorted(views)], dim=-1)
    if name not in views:
        raise ValueError(f"feature view {name!r} unavailable; choose from {sorted(views)}")
    return views[name]


def run_one(model, processor, args, bits: int, patch: int) -> list[ProbeResult]:
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
    train_images, train_y = random_payload_images(samples=args.train_samples, seed=1000 + bits, **common)
    val_images, val_y = random_payload_images(samples=args.val_samples, seed=9000 + bits, **common)
    clip = clipping_fraction(
        train_y.reshape(args.train_samples * args.grid, args.grid, bits),
        patch=patch,
        amplitude=args.amplitude,
    )
    started = time.monotonic()
    cache_base = {
        "model": args.model,
        "grid": args.grid,
        "bits": bits,
        "patch": patch,
        "amplitude": args.amplitude,
        "jpeg_quality": args.jpeg_quality,
        "scale": args.scale,
        "blur": args.blur,
        "noise": args.noise,
    }
    train_views = cached_extract_features(
        model,
        processor,
        train_images,
        args.batch_size,
        args.device,
        label=f"{bits}b train",
        cache_dir=args.cache_dir,
        metadata={
            **cache_base,
            "split": "train",
            "samples": args.train_samples,
            "seed": 1000 + bits,
        },
    )
    val_views = cached_extract_features(
        model,
        processor,
        val_images,
        args.batch_size,
        args.device,
        label=f"{bits}b val",
        cache_dir=args.cache_dir,
        metadata={
            **cache_base,
            "split": "validation",
            "samples": args.val_samples,
            "seed": 9000 + bits,
        },
    )
    encoder_seconds = time.monotonic() - started
    expected = args.grid * args.grid
    if train_views["final"].shape[1] != expected:
        raise RuntimeError(
            f"expected {expected} merged visual tokens from a {args.grid}x{args.grid} grid, "
            f"but Qwen produced {train_views['final'].shape[1]}; processor resizing changed the experiment"
        )
    print(f"  available feature views: {', '.join(sorted(train_views))}", flush=True)
    results = []
    for view_name in args.feature_views:
        train_x = _select_view(train_views, view_name).float()
        val_x = _select_view(val_views, view_name).float()
        scores, probe_seconds = fit_probe(
            train_x,
            torch.from_numpy(train_y),
            val_x,
            torch.from_numpy(val_y),
            kind=args.probe,
            epochs=args.epochs,
            device=args.device,
            seed=args.seed + bits,
            ridge_lambdas=args.ridge_lambdas,
        )
        scores["bsc_equivalent_bits_per_token"] = bsc_equivalent_rate(
            bits, scores["bit_accuracy"]
        )
        results.append(
            ProbeResult(
                bits_per_visual_token=bits,
                feature_view=view_name,
                train_samples=args.train_samples,
                validation_samples=args.val_samples,
                visual_tokens_per_image=expected,
                hidden_size=train_x.shape[-1],
                train_rows=train_x.shape[0] * train_x.shape[1],
                rows_per_feature=round(
                    train_x.shape[0] * train_x.shape[1] / train_x.shape[-1], 3
                ),
                probe_kind=args.probe,
                clipping_fraction=clip,
                encoder_seconds=encoder_seconds,
                probe_seconds=probe_seconds,
                jpeg_quality=args.jpeg_quality,
                scale=args.scale,
                blur=args.blur,
                noise=args.noise,
                **scores,
            )
        )
    return results


def parse_bits(value: str) -> list[int]:
    bits = sorted({int(item) for item in value.split(",")})
    if not bits or bits[0] < 1:
        raise argparse.ArgumentTypeError("bits must be positive comma-separated integers")
    return bits


def parse_floats(value: str) -> list[float]:
    values = sorted({float(item) for item in value.split(",")})
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("values must be positive comma-separated floats")
    return values


def parse_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated feature view names")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--bits", type=parse_bits, default=parse_bits("1,2,4,8,16,32"))
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--probe", choices=("ridge", "linear", "mlp"), default="ridge")
    parser.add_argument("--feature-views", type=parse_strings, default=parse_strings("final"))
    parser.add_argument("--ridge-lambdas", type=parse_floats, default=parse_floats("0.01,0.1,1,10,100"))
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--amplitude", type=float, default=64.0)
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--blur", type=float, default=0.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/qwen_probe.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/feature_cache"))
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    model, processor, attention = load_model(args.model, args.device)
    patch = effective_patch(model)
    target_pixels = (args.grid * patch) ** 2
    # Prevent the dynamic-resolution processor from silently changing token count.
    processor.image_processor.min_pixels = target_pixels
    processor.image_processor.max_pixels = target_pixels
    processor.image_processor.size = {
        "shortest_edge": target_pixels,
        "longest_edge": target_pixels,
    }
    print(
        f"model={args.model} effective_patch={patch}px grid={args.grid}x{args.grid} "
        f"image={args.grid * patch}px probe={args.probe} attention={attention}",
        flush=True,
    )
    results = []
    for bits in args.bits:
        bit_results = run_one(model, processor, args, bits, patch)
        results.extend(asdict(result) for result in bit_results)
        for result in bit_results:
            print(json.dumps(asdict(result), sort_keys=True), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "attention_implementation": attention,
                    "seed": args.seed,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
