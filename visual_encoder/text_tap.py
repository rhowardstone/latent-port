"""Independent wiretap for LP-2 free-text latent traffic.

Reads the text bridge's 16 slot vectors and emits token IDs directly — per slot,
three token positions, scored against the frozen A-model embedding table (tied
softmax). Neither A's transformer nor B is consulted at decode time; B is never
loaded at all. Ships the same trust gauges as the Base32 tap: mean max-softmax
confidence and the on-manifold z-score.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .latent_bridge import ABrain, GatherBridge
from .text_bridge import WINDOW, load_snippets, masked_read
from .text_baseline import levenshtein
from .wiretap import ManifoldGauge


class TextWiretap(nn.Module):
    def __init__(self, d_model: int, positions_per_slot: int, a_width: int, width: int = 512) -> None:
        super().__init__()
        self.positions = positions_per_slot
        self.a_width = a_width
        self.inlet = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, width), nn.GELU())
        block = nn.TransformerEncoderLayer(
            width, 8, width * 4, dropout=0.0, batch_first=True, norm_first=True
        )
        self.mixer = nn.TransformerEncoder(block, 2)
        self.heads = nn.Linear(width, positions_per_slot * a_width)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, latents: torch.Tensor, embed_table: torch.Tensor) -> torch.Tensor:
        mixed = self.mixer(self.inlet(latents))
        batch, slots, _ = latents.shape
        queries = self.heads(mixed).reshape(batch, slots, self.positions, self.a_width)
        return (queries @ embed_table.T) * self.scale


@torch.no_grad()
def read_text_wire(tap, latents, embed_table, tokenizer) -> tuple[list[str], torch.Tensor]:
    logits = tap(latents, embed_table)
    confidence = logits.softmax(dim=-1).max(dim=-1).values.mean(dim=(1, 2))
    picks = logits.argmax(dim=-1)
    texts = [
        tokenizer.decode(sample.reshape(-1).tolist(), skip_special_tokens=True).strip()
        for sample in picks
    ]
    return texts, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--bridge-checkpoint", type=Path, default=Path("runs/bridges/lp2_text_16slots.pt"))
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/text_tap.json"))
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(args.seed)
    brain = ABrain(args.sender, device)
    train_snippets, val_snippets = load_snippets(brain)
    state = torch.load(args.bridge_checkpoint, map_location=device)
    positions = state["embed.0.weight"].shape[1] // brain.width
    d_model = state["out.weight"].shape[0]
    bridge = GatherBridge(
        brain.width, d_model, args.slots, positions, 0.0, offset=0
    )
    bridge.load_state_dict(state)
    bridge = bridge.to(device).eval()
    for parameter in bridge.parameters():
        parameter.requires_grad_(False)
    embed_table = brain.model.get_input_embeddings().weight.detach().float()

    @torch.no_grad()
    def traffic(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        states, mask = masked_read(brain, texts)
        ids = brain.tokenizer(
            texts, return_tensors="pt", padding="max_length", truncation=True,
            max_length=WINDOW, add_special_tokens=True,
        ).input_ids.to(device)
        return bridge(states, mask).float(), ids

    pad_id = brain.tokenizer.pad_token_id
    tap = TextWiretap(d_model, positions, brain.width).to(device)
    optimizer = torch.optim.AdamW(tap.parameters(), lr=args.lr)
    started = time.monotonic()
    for step in range(args.steps):
        rng = np.random.default_rng(args.seed * 611_953 + step)
        texts = [train_snippets[i] for i in rng.integers(0, len(train_snippets), args.batch_size)]
        latents, ids = traffic(texts)
        logits = tap(latents, embed_table)
        # Ignore pad positions: predicting padding is trivial and would inflate
        # both the tap's apparent confidence and the LP-5a legibility signal.
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids.reshape(-1), ignore_index=pad_id
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"tap step {step}/{args.steps} loss={loss.item():.4f} ({time.monotonic() - started:.0f}s)", flush=True)

    rng = np.random.default_rng(args.seed)
    calibration = [train_snippets[i] for i in rng.choice(len(train_snippets), 256, replace=False)]
    gauge = ManifoldGauge(traffic(calibration)[0])

    exact = 0
    char_accuracy = []
    confidences = []
    zs = []
    examples = []
    picks = np.random.default_rng(args.seed + 555).choice(
        len(val_snippets), size=args.eval_samples, replace=False
    )
    for index in picks:
        text = val_snippets[index]
        latents, _ = traffic([text])
        decoded, confidence = read_text_wire(tap, latents, embed_table, brain.tokenizer)
        distance = levenshtein(text, decoded[0])
        exact += decoded[0] == text
        char_accuracy.append(max(0.0, 1.0 - distance / max(len(text), len(decoded[0]), 1)))
        confidences.append(float(confidence[0]))
        zs.append(float(gauge.z(latents)[0]))
        if len(examples) < 3:
            examples.append({"sent": text, "tap_read": decoded[0]})

    noise = torch.randn(64, args.slots, d_model, device=device) * 0.03
    _, noise_conf = read_text_wire(tap, noise, embed_table, brain.tokenizer)
    result = {
        "experiment": "LP-2 text wiretap",
        "tap_parameters": sum(p.numel() for p in tap.parameters()),
        "tap_exact_rate": exact / args.eval_samples,
        "tap_char_accuracy": float(np.mean(char_accuracy)),
        "mean_confidence": float(np.mean(confidences)),
        "mean_z": float(np.mean(zs)),
        "noise_confidence": float(noise_conf.mean()),
        "noise_z": float(gauge.z(noise).mean()),
        "examples": examples,
    }
    checkpoint = args.output.parent / "wiretaps" / "tap_text_16slots.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tap.state_dict(), checkpoint)
    gauge_path = args.output.parent / "wiretaps" / "tap_text_gauge.pt"
    torch.save({"mean": gauge.mean, "std": gauge.std}, gauge_path)
    result["tap_checkpoint"] = str(checkpoint)
    print(json.dumps(result, sort_keys=True), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
