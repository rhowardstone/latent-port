"""Patch-aligned synthetic visual codes for encoder-capacity probes."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .codec import apply_channel


def make_basis(bits: int, patch: int, seed: int = 12345) -> np.ndarray:
    """Create deterministic approximately orthogonal RGB spread-spectrum bases."""
    if bits < 1:
        raise ValueError("bits must be positive")
    dimensions = patch * patch * 3
    if bits > dimensions:
        raise ValueError(f"cannot fit {bits} bases in {dimensions} RGB dimensions")
    rng = np.random.default_rng(seed)
    # Random Rademacher vectors are nearly orthogonal in this dimensionality and
    # cheaper to construct than a full QR decomposition for large patches.
    basis = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), (bits, dimensions))
    basis -= basis.mean(axis=1, keepdims=True)
    basis /= np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-8)
    return basis.reshape(bits, patch, patch, 3)


def render_payload_grid(
    labels: np.ndarray,
    *,
    patch: int,
    amplitude: float = 64.0,
    basis_seed: int = 12345,
) -> Image.Image:
    """Render ``[rows, cols, bits]`` binary labels into one RGB image.

    Every bit modulates a global micro-pattern inside its effective ViT patch.
    The total modulation energy stays approximately constant as bit count grows.
    """
    if labels.ndim != 3:
        raise ValueError("labels must have shape [rows, cols, bits]")
    raw = _render_float(labels, patch=patch, amplitude=amplitude, basis_seed=basis_seed)
    image = np.clip(np.rint(raw), 0, 255).astype(np.uint8)
    rows, cols, _ = labels.shape
    image = image.transpose(0, 3, 1, 4, 2).reshape(rows * patch, cols * patch, 3)
    return Image.fromarray(image, mode="RGB")


def _render_float(
    labels: np.ndarray, *, patch: int, amplitude: float, basis_seed: int
) -> np.ndarray:
    if labels.ndim != 3:
        raise ValueError("labels must have shape [rows, cols, bits]")
    _, _, bits = labels.shape
    basis = make_basis(bits, patch, seed=basis_seed)
    signs = labels.astype(np.float32) * 2.0 - 1.0
    # Unit-length bases need sqrt(pixel dimensions) scaling to occupy a useful
    # fraction of the 8-bit range. sqrt(bits) keeps aggregate energy controlled.
    scale = amplitude * math.sqrt(patch * patch * 3) / math.sqrt(bits)
    return 127.5 + scale * np.einsum("rcb,bhwk->rchwk", signs, basis)


def clipping_fraction(
    labels: np.ndarray, *, patch: int, amplitude: float = 64.0, basis_seed: int = 12345
) -> float:
    """Fraction of pre-quantization RGB values outside the 8-bit gamut."""
    raw = _render_float(labels, patch=patch, amplitude=amplitude, basis_seed=basis_seed)
    return float(np.mean((raw < 0.0) | (raw > 255.0)))


def random_payload_images(
    *,
    samples: int,
    grid: int,
    bits: int,
    patch: int,
    seed: int,
    amplitude: float = 64.0,
    jpeg_quality: int = 100,
    scale: float = 1.0,
    blur: float = 0.0,
    noise: float = 0.0,
) -> tuple[list[Image.Image], np.ndarray]:
    """Generate held-out random payloads and their patch-aligned code images."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, (samples, grid, grid, bits), dtype=np.uint8)
    images = []
    for index, sample in enumerate(labels):
        image = render_payload_grid(sample, patch=patch, amplitude=amplitude)
        image = apply_channel(
            image,
            jpeg_quality=jpeg_quality,
            scale=scale,
            blur=blur,
            noise=noise,
            seed=seed + index,
        )
        images.append(image)
    return images, labels.reshape(samples, grid * grid, bits)
