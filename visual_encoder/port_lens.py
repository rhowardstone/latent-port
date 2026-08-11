"""Interpretability lens for trained LP-1 senders: what did the code become?

For each trained sender checkpoint this renders one figure answering, per slot:
- Does the slot still "contain" its assigned characters? (cosine to the payload's
  character embeddings — block-diagonal means superposed text)
- Is the slot still on the text manifold? (cosine to the nearest vocabulary token,
  calibrated against how close real tokens sit to each other)
- Where does it sit geometrically? (PCA of embedding space: 32 char embeddings
  plus the k slot vectors)
- What does the frozen model read from the slot alone? (solo-slot greedy decode)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from .latent_port import (
    BASE32_ALPHABET,
    LatentSender,
    PortBatcher,
    char_embedding_table,
    greedy_decode,
    load_receiver,
    normalize,
    payload_to_bits,
    random_payload,
)

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "lens_blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return a @ b.T


@torch.no_grad()
def analyze(model, tokenizer, args, characters: int, checkpoint: Path) -> dict:
    device = args.device
    slots = args.slots
    embed_rms = model.get_input_embeddings().weight.detach().float().pow(2).mean().sqrt().item()
    d_model = model.get_input_embeddings().weight.shape[1]
    sender = LatentSender(slots, characters // slots, d_model, embed_rms, hidden=args.hidden)
    sender.load_state_dict(torch.load(checkpoint, map_location=device))
    sender = sender.to(device).eval()

    rng = np.random.default_rng(args.payload_seed)
    payload = random_payload(characters, rng)
    chunk = characters // slots
    chunks = [payload[i * chunk : (i + 1) * chunk] for i in range(slots)]
    bits = torch.from_numpy(payload_to_bits(payload, slots)).to(device)
    latents = sender(bits.unsqueeze(0)).squeeze(0).float()

    vocab = model.get_input_embeddings().weight.detach().float()
    char_table = char_embedding_table(model, tokenizer, device)

    # Slot vs the payload's own character embeddings, [slots, characters].
    char_embeds = char_table[
        torch.tensor([BASE32_ALPHABET.index(c) for c in payload], device=device)
    ]
    slot_char_cos = cosine(latents, char_embeds).cpu().numpy()

    # Nearest vocabulary tokens per slot.
    vocab_cos = cosine(latents, vocab)
    top_cos, top_ids = vocab_cos.topk(3, dim=-1)
    nearest = [
        [
            {"token": tokenizer.decode([tid]), "cos": round(float(c), 3)}
            for tid, c in zip(ids.tolist(), cs.tolist())
        ]
        for ids, cs in zip(top_ids, top_cos)
    ]
    # Calibration: how close do REAL tokens sit to their nearest other token?
    sample_ids = torch.randint(0, vocab.shape[0], (256,), generator=torch.Generator().manual_seed(0))
    ref = cosine(vocab[sample_ids.to(device)], vocab)
    ref.scatter_(1, sample_ids.to(device).unsqueeze(1), -1.0)
    token_selfsim = float(ref.max(dim=-1).values.median())

    norm_ratio = (latents.pow(2).mean(-1).sqrt() / embed_rms).cpu().numpy()

    # PCA over char embeddings + slots.
    stack = torch.cat([char_table, latents]).cpu().numpy()
    centered = stack - stack.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vt[:2].T
    chars_2d, slots_2d = projected[:32], projected[32:]

    # Behavioral: full decode and per-slot solo decode.
    batcher = PortBatcher(model, tokenizer, slots, device)
    full_decode = normalize(greedy_decode(model, batcher, latents, max_new=2 * characters + 8))
    solo_batcher = PortBatcher(model, tokenizer, 1, device)
    solo = [
        greedy_decode(model, solo_batcher, latents[i : i + 1], max_new=8).strip()
        for i in range(slots)
    ]

    return {
        "characters": characters,
        "slots": slots,
        "payload": payload,
        "chunks": chunks,
        "full_decode": full_decode,
        "exact": full_decode == payload,
        "slot_char_cos": slot_char_cos,
        "nearest": nearest,
        "top1_cos": top_cos[:, 0].cpu().numpy(),
        "token_selfsim_reference": token_selfsim,
        "norm_ratio": norm_ratio,
        "chars_2d": chars_2d,
        "slots_2d": slots_2d,
        "solo_decodes": solo,
    }


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)


def render(analysis: dict, output: Path) -> None:
    characters, slots = analysis["characters"], analysis["slots"]
    figure = plt.figure(figsize=(16, 11), facecolor=SURFACE)
    figure.suptitle(
        f"Latent port lens — {characters} chars in {slots} slots "
        f"({5 * characters // slots} bits/slot) — decode "
        + ("EXACT" if analysis["exact"] else "inexact"),
        fontsize=14,
        color=INK,
        fontweight="bold",
    )

    ax = figure.add_subplot(2, 2, 1)
    style_axis(ax)
    image = ax.imshow(
        analysis["slot_char_cos"], cmap=BLUE_RAMP, aspect="auto", vmin=0, vmax=1
    )
    ax.set_xticks(range(characters), list(analysis["payload"]), fontsize=7, family="monospace")
    ax.set_yticks(range(slots), [f"s{i}" for i in range(slots)], fontsize=7)
    chunk = characters // slots
    for boundary in range(chunk, characters, chunk):
        ax.axvline(boundary - 0.5, color=SURFACE, linewidth=1.5)
    ax.set_title("Does slot i still contain its characters?  cos(slot, char embedding)", fontsize=10, color=INK)
    ax.set_xlabel("payload characters (slot chunks separated)", fontsize=8, color=INK2)
    figure.colorbar(image, ax=ax, fraction=0.03, pad=0.02)

    ax = figure.add_subplot(2, 2, 2)
    style_axis(ax)
    ax.grid(color=MUTED, alpha=0.2, linewidth=0.5)
    for (x, y), char in zip(analysis["chars_2d"], BASE32_ALPHABET):
        ax.text(x, y, char, color=MUTED, fontsize=9, ha="center", va="center", family="monospace")
    ax.scatter(
        analysis["slots_2d"][:, 0], analysis["slots_2d"][:, 1],
        color=BLUE, s=42, zorder=3, label="latent slots",
    )
    for i, (x, y) in enumerate(analysis["slots_2d"]):
        ax.annotate(
            analysis["chunks"][i], (x, y), textcoords="offset points", xytext=(5, 4),
            fontsize=6.5, color=ORANGE, family="monospace",
        )
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.set_title("Where do slots live? PCA of embedding space (gray = 32 char tokens)", fontsize=10, color=INK)

    ax = figure.add_subplot(2, 2, 3)
    style_axis(ax)
    ax.grid(axis="y", color=MUTED, alpha=0.2, linewidth=0.5)
    ax.bar(range(slots), analysis["top1_cos"], color=BLUE, width=0.62)
    ax.axhline(
        analysis["token_selfsim_reference"], color=ORANGE, linewidth=2,
        label=f"real tokens' nearest-neighbor cos (median {analysis['token_selfsim_reference']:.2f})",
    )
    ax.set_xticks(range(slots), [f"s{i}" for i in range(slots)], fontsize=7)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.set_title("How text-like is each slot?  cos to nearest vocabulary token", fontsize=10, color=INK)

    ax = figure.add_subplot(2, 2, 4)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    lines = [
        f"payload : {analysis['payload']}",
        f"decoded : {analysis['full_decode']}",
        "",
        "slot  chunk   nearest tokens (cos)              model reads slot alone as",
    ]
    for i in range(slots):
        near = ", ".join(f"{n['token']!r} {n['cos']:.2f}" for n in analysis["nearest"][i][:2])
        solo = analysis["solo_decodes"][i][:24].replace("\n", " ")
        lines.append(
            f"s{i:<4} {analysis['chunks'][i]:<7} {near:<33} {solo}"
        )
    ax.text(
        0.01, 0.99, "\n".join(lines), transform=ax.transAxes, fontsize=7.5,
        family="monospace", color=INK, va="top",
    )
    ax.set_title("What the frozen receiver reads", fontsize=10, color=INK, loc="left")

    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--chars", default="16,32,48,64,96")
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--payload-seed", type=int, default=424242)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("runs/senders"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lens"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, tokenizer = load_receiver(args.model, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for characters in [int(c) for c in args.chars.split(",")]:
        checkpoint = args.checkpoint_dir / f"lp1_{characters}chars_{args.slots}slots.pt"
        if not checkpoint.exists():
            print(f"skip {characters} chars: {checkpoint} not found", flush=True)
            continue
        analysis = analyze(model, tokenizer, args, characters, checkpoint)
        png = args.output_dir / f"lens_{characters}chars.png"
        render(analysis, png)
        serializable = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in analysis.items()
        }
        (args.output_dir / f"lens_{characters}chars.json").write_text(
            json.dumps(serializable, indent=2) + "\n"
        )
        print(f"wrote {png}", flush=True)


if __name__ == "__main__":
    main()
