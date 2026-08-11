"""Render a latent-port message as a literal picture — the vectors as pixels.

Level-1 'translate back to a picture': take the k vectors a message becomes and
lay each out as a tile, colormapped. It's the actual thing the models exchange,
made viewable. (Level 2 — inverting the vision tower to a NATURAL image — is a
separate, heavier experiment; see NOTEBOOK.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .latent_bridge import ABrain, GatherBridge
from .latent_port import load_receiver
from .text_bridge import WINDOW, masked_read


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sender", default="Qwen/Qwen3-4B")
    p.add_argument("--receiver", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--bridge", type=Path, default=Path("runs/bridges/lp2_text_16slots.pt"))
    p.add_argument("--messages", nargs="+", default=[
        "The wifi password is sunflower99.",
        "Meeting Tuesday at 3pm, bring the report.",
    ])
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/vector_pictures.png"))
    args = p.parse_args()

    model, tok = load_receiver(args.receiver, args.device)
    brain = ABrain(args.sender, args.device)
    state = torch.load(args.bridge, map_location=args.device)
    pos = state["embed.0.weight"].shape[1] // brain.width
    d_model = model.get_input_embeddings().weight.shape[1]
    slots = state["slot_pos"].shape[0]
    bridge = GatherBridge(brain.width, d_model, slots, pos, 0.0, offset=0)
    bridge.load_state_dict(state); bridge = bridge.to(args.device).eval()

    pics = []
    for m in args.messages:
        with torch.no_grad():
            states, mask = masked_read(brain, [m])
            vecs = bridge(states, mask).squeeze(0).float().cpu().numpy()  # [slots, d]
        # tile each vector into a square-ish patch, arrange slots in a grid
        d = vecs.shape[1]
        side = int(np.floor(np.sqrt(d)))
        tiles = vecs[:, : side * side].reshape(slots, side, side)
        cols = int(np.ceil(np.sqrt(slots)))
        rows = int(np.ceil(slots / cols))
        canvas = np.full((rows * (side + 2), cols * (side + 2)), np.nan)
        for i in range(slots):
            r, c = divmod(i, cols)
            canvas[r * (side + 2):r * (side + 2) + side, c * (side + 2):c * (side + 2) + side] = tiles[i]
        pics.append((m, canvas))

    fig, axes = plt.subplots(1, len(pics), figsize=(6 * len(pics), 6.6), facecolor="#0b0f16")
    if len(pics) == 1:
        axes = [axes]
    for ax, (m, canvas) in zip(axes, pics):
        v = np.nanmax(np.abs(canvas))
        ax.imshow(canvas, cmap="twilight_shifted", vmin=-v, vmax=v, interpolation="nearest")
        ax.set_title(f'"{m}"\n{pics[0][1].shape and slots} vectors → a picture',
                     color="#eaf0f8", fontsize=11)
        ax.axis("off")
    fig.suptitle("A latent-port message, rendered as pixels", color="#c9a5ff", fontsize=15, fontweight="bold")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130, facecolor="#0b0f16")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
