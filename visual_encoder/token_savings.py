"""Honest token-savings measurement for the free-text port (LP-2).

Answers the question directly and without hand-waving: when A speaks to B over the
latent port instead of in text, how many receiver context positions are saved, at
what fidelity, and — the part that keeps it honest — does B actually RECEIVE the
message, or is it hallucinating something plausible from its language prior?

The causal control is the crux (external-review A5): compare B's teacher-forced
NLL of the true message under the CORRECT latents vs a DERANGED latent (another
message's) and a ZERO latent. If correct ≪ deranged, the vectors carry real
example-specific information. Reported as paired information gain (bits/message)
with message-cluster bootstrap CIs, plus position-compression and fidelity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .channel_metrics import cluster_bootstrap_ci, paired_information_gain
from .latent_bridge import ABrain, GatherBridge
from .latent_port import MARKER, PortBatcher, greedy_decode, load_receiver, normalize, text_positions
from .provenance import provenance
from .text_baseline import levenshtein
from .text_bridge import TRAIN_TEMPLATES, WINDOW, load_snippets, masked_read, synthetic_chat

SLOTS = 16


@torch.no_grad()
def message_nll_nats(model, batcher, latents, message, device) -> float:
    """Teacher-forced total NLL (nats) of `message` under B given `latents`."""
    embeds, mask, labels = batcher.training_batch(latents, [message])
    out = model(
        inputs_embeds=embeds,
        attention_mask=mask,
        position_ids=text_positions(1, embeds.shape[1], device),
    )
    logits = out.logits[:, :-1, :].reshape(-1, out.logits.shape[-1]).float()
    targets = labels[:, 1:].reshape(-1)
    nll = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum")
    return float(nll)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--sender", default="Qwen/Qwen3-4B")
    parser.add_argument("--bridge", type=Path, default=Path("runs/bridges/lp2_text_16slots.pt"))
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/token_savings.json"))
    args = parser.parse_args()

    model, tokenizer = load_receiver(args.receiver, args.device)
    brain = ABrain(args.sender, args.device)
    state = torch.load(args.bridge, map_location=args.device)
    positions = state["embed.0.weight"].shape[1] // brain.width
    d_model = model.get_input_embeddings().weight.shape[1]
    bridge = GatherBridge(brain.width, d_model, SLOTS, positions, 0.0, offset=0)
    bridge.load_state_dict(state)
    bridge = bridge.to(args.device).eval()
    batcher = PortBatcher(model, tokenizer, SLOTS, args.device, message=TRAIN_TEMPLATES[0])

    _, val = load_snippets(brain)
    chat = synthetic_chat(np.random.default_rng(args.seed + 1), 200)
    rng = np.random.default_rng(args.seed)
    pool = val + chat
    messages = [pool[i] for i in rng.choice(len(pool), args.samples, replace=False)]

    # Precompute all latents so we can derange (use another message's vectors).
    latents = []
    for m in messages:
        states, mask = masked_read(brain, [m])
        latents.append(bridge(states, mask).float())
    deranged_index = np.roll(np.arange(len(messages)), 1)

    per_msg = []
    for i, m in enumerate(messages):
        z = latents[i]
        nll_correct = message_nll_nats(model, batcher, z, m, args.device)
        nll_deranged = message_nll_nats(model, batcher, latents[deranged_index[i]], m, args.device)
        nll_zero = message_nll_nats(model, batcher, torch.zeros_like(z), m, args.device)
        # free text: compare raw decode (do NOT base32-normalize, which would strip English)
        decoded = greedy_decode(model, batcher, z.squeeze(0), max_new=90).strip()
        b_tokens = len(tokenizer(m, add_special_tokens=False).input_ids)
        fidelity = max(0.0, 1.0 - levenshtein(m, decoded) / max(len(m), len(decoded), 1))
        per_msg.append({
            "b_tokens": b_tokens,
            "position_compression": b_tokens / SLOTS,
            "fidelity": fidelity,
            "nll_correct": nll_correct,
            "nll_deranged": nll_deranged,
            "nll_zero": nll_zero,
        })

    arr = {k: np.array([r[k] for r in per_msg], dtype=np.float64) for k in per_msg[0]}
    gain_vs_deranged = paired_information_gain(arr["nll_correct"], arr["nll_deranged"])
    gain_vs_zero = paired_information_gain(arr["nll_correct"], arr["nll_zero"])
    # per-message bits so we can bootstrap the causal signal
    di_deranged = (arr["nll_deranged"] - arr["nll_correct"]) / np.log(2)

    result = {
        "experiment": "LP-2 token savings (honest)",
        "provenance": provenance(args),
        "samples": args.samples,
        "position_compression": cluster_bootstrap_ci(arr["position_compression"], seed=args.seed),
        "fidelity": cluster_bootstrap_ci(arr["fidelity"], seed=args.seed),
        "mean_b_tokens": float(arr["b_tokens"].mean()),
        "causal_control": {
            "delta_i_bits_correct_vs_deranged": gain_vs_deranged,
            "delta_i_bits_correct_vs_zero": gain_vs_zero,
            "delta_i_vs_deranged_ci": cluster_bootstrap_ci(di_deranged, seed=args.seed),
            "note": "correct<<deranged NLL => vectors carry example-specific info, not prior hallucination",
        },
        "verdict": {
            "break_even_tokens": SLOTS,
            "reading": (
                "16 vectors cost 16 receiver positions. For a message of L text tokens the "
                "context saving is L/16x, LOSSY at the measured fidelity; net win only for L>16. "
                "Real content transfer is confirmed iff delta_i_vs_deranged CI excludes 0."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
