import numpy as np
import torch

from visual_encoder.latent_port import LatentSender, payload_to_bits, random_payload
from visual_encoder.wiretap import ManifoldGauge, Wiretap, read_wire, sender_config


def test_sender_config_inferred_from_checkpoint():
    sender = LatentSender(slots=16, chars_per_slot=3, d_model=64, embed_rms=0.05, hidden=32)
    slots, chars_per_slot, hidden, d_model = sender_config(sender.state_dict())
    assert (slots, chars_per_slot, hidden, d_model) == (16, 3, 32, 64)


def test_wiretap_shapes_and_decode():
    tap = Wiretap(d_model=64, chars_per_slot=2, width=32, layers=1)
    latents = torch.randn(3, 8, 64)
    logits = tap(latents)
    assert logits.shape == (3, 8, 2, 32)
    decoded, confidence = read_wire(tap, latents)
    assert len(decoded) == 3 and all(len(d) == 16 for d in decoded)
    assert confidence.shape == (3,)
    assert float(confidence.min()) > 0


def test_manifold_gauge_flags_noise():
    sender = LatentSender(slots=4, chars_per_slot=1, d_model=32, embed_rms=0.05, hidden=16)
    rng = np.random.default_rng(0)
    payloads = [random_payload(4, rng) for _ in range(128)]
    bits = torch.from_numpy(np.stack([payload_to_bits(p, 4) for p in payloads]))
    with torch.no_grad():
        latents = sender(bits).float()
    gauge = ManifoldGauge(latents)
    in_dist = float(gauge.z(latents[:16]).mean())
    jam = float(gauge.z(torch.randn(16, 4, 32) * 10).mean())
    assert jam > in_dist * 3
