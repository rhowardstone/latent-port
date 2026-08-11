import numpy as np
import torch

from visual_encoder.latent_port import (
    BASE32_ALPHABET,
    LatentSender,
    normalize,
    payload_to_bits,
    random_payload,
)


def test_payload_bits_round_trip_structure():
    payload = random_payload(32, np.random.default_rng(3))
    bits = payload_to_bits(payload, 16)
    assert bits.shape == (16, 10)
    assert set(np.unique(bits)) <= {-1.0, 1.0}
    # First slot holds the first two characters, 5 bits each, MSB first.
    first = BASE32_ALPHABET.index(payload[0])
    expected = [(first >> shift) & 1 for shift in range(4, -1, -1)]
    assert [int(b > 0) for b in bits[0, :5]] == expected


def test_sender_output_scale_matches_embedding_rms():
    sender = LatentSender(slots=8, chars_per_slot=2, d_model=64, embed_rms=0.05)
    bits = torch.from_numpy(
        payload_to_bits(random_payload(16, np.random.default_rng(1)), 8)
    ).unsqueeze(0)
    latents = sender(bits)
    assert latents.shape == (1, 8, 64)
    rms = latents.pow(2).mean().sqrt().item()
    assert 0.02 < rms < 0.15


def test_normalize_strips_non_base32():
    assert normalize("a b1c 2!d\n7") == "ABC2D7"
