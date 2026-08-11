import numpy as np
import pytest

from visual_encoder.codec import (
    DecodeError,
    apply_channel,
    decode_palette,
    decode_raw,
    encode_palette,
    encode_raw,
    metrics,
)


RNG = np.random.default_rng(20260808)


@pytest.mark.parametrize("payload", [b"", b"hello world", RNG.bytes(4096)])
def test_raw_round_trip(payload):
    encoded = encode_raw(payload)
    decoded, _ = decode_raw(encoded.image)
    assert decoded == payload


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
@pytest.mark.parametrize("repetition", [1, 3])
def test_palette_round_trip(bits, repetition):
    payload = b"A visual token is not a pixel." * 8
    encoded = encode_palette(
        payload, bits_per_cell=bits, cell_size=8, repetition=repetition
    )
    decoded, _ = decode_palette(
        encoded.image, bits_per_cell=bits, cell_size=8, repetition=repetition
    )
    assert decoded == payload


def test_palette_survives_moderate_jpeg():
    payload = np.random.default_rng(11).bytes(512)
    encoded = encode_palette(payload, bits_per_cell=2, cell_size=16, repetition=3)
    damaged = apply_channel(encoded.image, jpeg_quality=75)
    decoded, _ = decode_palette(
        damaged, bits_per_cell=2, cell_size=16, repetition=3
    )
    assert decoded == payload


def test_raw_detects_damage():
    encoded = encode_raw(np.random.default_rng(12).bytes(1024), compress=False)
    damaged = apply_channel(encoded.image, jpeg_quality=95)
    with pytest.raises(DecodeError):
        decode_raw(damaged)


def test_metrics_are_payload_based():
    encoded = encode_palette(b"hello", bits_per_cell=4, cell_size=4)
    result = metrics(encoded)
    assert result["original_bytes"] == 5
    assert result["estimated_visual_tokens"] >= 1
    assert result["source_bits_per_visual_token"] > 0
