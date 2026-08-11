"""Deterministic image codecs used as controls for learned visual channels.

The codecs deliberately operate on bytes, rather than natural-language tokens.  A
held-out random byte is incompressible and cannot be guessed by a language model,
which makes byte recovery an honest channel-capacity measurement.
"""

from __future__ import annotations

import base64
import io
import math
import struct
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


MAGIC = b"VEC1"
HEADER = struct.Struct(">4sBIII")  # magic, flags, original length, stored length, crc32


class DecodeError(ValueError):
    """Raised when an image does not contain a valid, intact packet."""


@dataclass(frozen=True)
class PacketInfo:
    original_bytes: int
    stored_bytes: int
    compressed: bool
    crc32: int


@dataclass(frozen=True)
class EncodedImage:
    image: Image.Image
    packet: PacketInfo
    mode: str
    bits_per_cell: int = 0
    cell_size: int = 1
    repetition: int = 1


def pack_payload(payload: bytes, compress: bool = True) -> tuple[bytes, PacketInfo]:
    compressed = zlib.compress(payload, level=9)
    use_compressed = compress and len(compressed) < len(payload)
    stored = compressed if use_compressed else payload
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    flags = 1 if use_compressed else 0
    packet = HEADER.pack(MAGIC, flags, len(payload), len(stored), crc) + stored
    return packet, PacketInfo(len(payload), len(stored), use_compressed, crc)


def unpack_payload(packet: bytes) -> tuple[bytes, PacketInfo]:
    if len(packet) < HEADER.size:
        raise DecodeError("packet is shorter than its header")
    magic, flags, original_len, stored_len, expected_crc = HEADER.unpack_from(packet)
    if magic != MAGIC:
        raise DecodeError("magic header mismatch")
    end = HEADER.size + stored_len
    if len(packet) < end:
        raise DecodeError(f"packet is truncated: need {end} bytes, found {len(packet)}")
    stored = packet[HEADER.size:end]
    try:
        payload = zlib.decompress(stored) if flags & 1 else stored
    except zlib.error as exc:
        raise DecodeError(f"compressed payload is damaged: {exc}") from exc
    if len(payload) != original_len:
        raise DecodeError(f"length mismatch: expected {original_len}, decoded {len(payload)}")
    actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise DecodeError(
            f"CRC mismatch: expected {expected_crc:08x}, decoded {actual_crc:08x}"
        )
    return payload, PacketInfo(original_len, stored_len, bool(flags & 1), actual_crc)


def _square_side(items: int) -> int:
    return max(1, math.ceil(math.sqrt(items)))


def encode_raw(payload: bytes, compress: bool = True) -> EncodedImage:
    """Put exactly three packet bytes in every RGB pixel.

    This is the lossless-file upper control, not a VLM-readable representation.
    """
    packet, info = pack_payload(payload, compress=compress)
    framed = struct.pack(">I", len(packet)) + packet
    side = _square_side(math.ceil(len(framed) / 3))
    raw = np.zeros(side * side * 3, dtype=np.uint8)
    raw[: len(framed)] = np.frombuffer(framed, dtype=np.uint8)
    image = Image.fromarray(raw.reshape(side, side, 3), mode="RGB")
    return EncodedImage(image=image, packet=info, mode="raw")


def decode_raw(image: Image.Image) -> tuple[bytes, PacketInfo]:
    raw = np.asarray(image.convert("RGB"), dtype=np.uint8).reshape(-1).tobytes()
    if len(raw) < 4:
        raise DecodeError("image is too small")
    packet_len = struct.unpack_from(">I", raw)[0]
    if packet_len > len(raw) - 4:
        raise DecodeError(f"packet length {packet_len} exceeds image capacity")
    return unpack_payload(raw[4 : 4 + packet_len])


def _candidate_colors() -> np.ndarray:
    # Avoid extreme black/white: they suffer clipping and dominate JPEG blocks.
    levels = np.asarray([24, 88, 168, 232], dtype=np.float32)
    return np.asarray(np.meshgrid(levels, levels, levels), dtype=np.float32).reshape(3, -1).T


def palette(bits_per_cell: int) -> np.ndarray:
    """Return a deterministic, widely separated RGB codebook."""
    if bits_per_cell not in (1, 2, 3, 4):
        raise ValueError("bits_per_cell must be between 1 and 4")
    wanted = 1 << bits_per_cell
    candidates = _candidate_colors()
    # Greedy farthest-point sampling provides substantially wider color margins
    # than taking the first N colors from an RGB cube.
    chosen = [int(np.argmax(np.linalg.norm(candidates - 128.0, axis=1)))]
    while len(chosen) < wanted:
        distances = np.min(
            np.linalg.norm(candidates[:, None, :] - candidates[chosen][None, :, :], axis=2),
            axis=1,
        )
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return candidates[chosen].astype(np.uint8)


def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    usable = len(bits) - (len(bits) % 8)
    return np.packbits(bits[:usable], bitorder="big").tobytes()


INTERLEAVE_BLOCK_BITS = 256


def _repeat_interleaved(bits: np.ndarray, repetition: int) -> np.ndarray:
    """Repeat bits in blocks, keeping copies far apart in the image stream."""
    if repetition == 1:
        return bits
    padding = (-len(bits)) % INTERLEAVE_BLOCK_BITS
    padded = np.pad(bits, (0, padding)) if padding else bits
    blocks = padded.reshape(-1, INTERLEAVE_BLOCK_BITS)
    return np.repeat(blocks[:, None, :], repetition, axis=1).reshape(-1)


def _majority_interleaved(bits: np.ndarray, repetition: int) -> np.ndarray:
    if repetition == 1:
        return bits
    encoded_block = INTERLEAVE_BLOCK_BITS * repetition
    usable = len(bits) - (len(bits) % encoded_block)
    if not usable:
        return np.empty(0, dtype=np.uint8)
    copies = bits[:usable].reshape(-1, repetition, INTERLEAVE_BLOCK_BITS)
    return (copies.sum(axis=1) > repetition // 2).astype(np.uint8).reshape(-1)


def encode_palette(
    payload: bytes,
    *,
    bits_per_cell: int = 4,
    cell_size: int = 12,
    repetition: int = 1,
    compress: bool = True,
) -> EncodedImage:
    """Encode bytes as a square grid of separated colors.

    ``repetition`` copies 256-bit blocks before symbol packing. Corresponding
    copies are therefore separated by an entire block rather than placed in the
    same JPEG neighborhood. It remains a simple explicit-cost control, not a
    substitute for a proper burst-error code.
    """
    if cell_size < 1:
        raise ValueError("cell_size must be positive")
    if repetition < 1 or repetition % 2 == 0:
        raise ValueError("repetition must be a positive odd integer")
    codebook = palette(bits_per_cell)
    packet, info = pack_payload(payload, compress=compress)
    bits = _repeat_interleaved(_bytes_to_bits(packet), repetition)
    padding = (-len(bits)) % bits_per_cell
    if padding:
        bits = np.pad(bits, (0, padding))
    weights = (1 << np.arange(bits_per_cell - 1, -1, -1)).astype(np.uint16)
    symbols = bits.reshape(-1, bits_per_cell).dot(weights)
    side = _square_side(len(symbols))
    cells = np.zeros(side * side, dtype=np.uint16)
    cells[: len(symbols)] = symbols
    rgb_cells = codebook[cells].reshape(side, side, 3)
    rgb = np.repeat(np.repeat(rgb_cells, cell_size, axis=0), cell_size, axis=1)
    return EncodedImage(
        image=Image.fromarray(rgb, mode="RGB"),
        packet=info,
        mode="palette",
        bits_per_cell=bits_per_cell,
        cell_size=cell_size,
        repetition=repetition,
    )


def decode_palette(
    image: Image.Image,
    *,
    bits_per_cell: int = 4,
    cell_size: int = 12,
    repetition: int = 1,
) -> tuple[bytes, PacketInfo]:
    if repetition < 1 or repetition % 2 == 0:
        raise ValueError("repetition must be a positive odd integer")
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    height = arr.shape[0] // cell_size
    width = arr.shape[1] // cell_size
    if not height or not width:
        raise DecodeError("image is smaller than one cell")
    arr = arr[: height * cell_size, : width * cell_size]
    cells = arr.reshape(height, cell_size, width, cell_size, 3).mean(axis=(1, 3))
    codebook = palette(bits_per_cell).astype(np.float32)
    symbols = np.argmin(
        np.sum((cells.reshape(-1, 1, 3) - codebook.reshape(1, -1, 3)) ** 2, axis=2),
        axis=1,
    )
    shifts = np.arange(bits_per_cell - 1, -1, -1)
    encoded_bits = ((symbols[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    decoded_bits = _majority_interleaved(encoded_bits, repetition)
    packet_bytes = _bits_to_bytes(decoded_bits)
    if len(packet_bytes) < HEADER.size:
        raise DecodeError("not enough recovered data for a packet header")
    _, _, _, stored_len, _ = HEADER.unpack_from(packet_bytes)
    total = HEADER.size + stored_len
    return unpack_payload(packet_bytes[:total])


def apply_channel(
    image: Image.Image,
    *,
    jpeg_quality: int = 100,
    scale: float = 1.0,
    blur: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
) -> Image.Image:
    """Apply a deterministic digital channel and restore the original dimensions."""
    result = image.convert("RGB")
    original_size = result.size
    if scale != 1.0:
        scaled = (max(1, round(result.width * scale)), max(1, round(result.height * scale)))
        result = result.resize(scaled, Image.Resampling.LANCZOS)
        result = result.resize(original_size, Image.Resampling.LANCZOS)
    if blur > 0:
        result = result.filter(ImageFilter.GaussianBlur(radius=blur))
    if noise > 0:
        rng = np.random.default_rng(seed)
        arr = np.asarray(result, dtype=np.float32)
        arr = np.clip(arr + rng.normal(0.0, noise, arr.shape), 0, 255).astype(np.uint8)
        result = Image.fromarray(arr, mode="RGB")
    if jpeg_quality < 100:
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=jpeg_quality, subsampling=2)
        buffer.seek(0)
        result = Image.open(buffer).convert("RGB")
    return result


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def image_from_data_url(url: str) -> Image.Image:
    try:
        encoded = url.split(",", 1)[1]
        raw = base64.b64decode(encoded, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise DecodeError("invalid image data URL") from exc


def visual_token_estimate(image: Image.Image, effective_patch: int = 32) -> int:
    """Approximate Qwen3-VL merged visual tokens for an image.

    Qwen3-VL uses 32px effective merged patches. Pass 28 for Qwen2.5-VL.
    """
    return math.ceil(image.width / effective_patch) * math.ceil(image.height / effective_patch)


def metrics(encoded: EncodedImage) -> dict[str, float | int | bool | str]:
    tokens = visual_token_estimate(encoded.image)
    pixels = encoded.image.width * encoded.image.height
    return {
        "mode": encoded.mode,
        "width": encoded.image.width,
        "height": encoded.image.height,
        "pixels": pixels,
        "estimated_visual_tokens": tokens,
        "original_bytes": encoded.packet.original_bytes,
        "stored_bytes": encoded.packet.stored_bytes,
        "compressed": encoded.packet.compressed,
        "source_bits_per_visual_token": round(encoded.packet.original_bytes * 8 / tokens, 3),
        "stored_bits_per_visual_token": round(encoded.packet.stored_bytes * 8 / tokens, 3),
        "source_bits_per_pixel": round(encoded.packet.original_bytes * 8 / pixels, 5),
    }
