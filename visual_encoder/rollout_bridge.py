"""Make the rollout work: train the bridge on MULTI-picture messages.

The single-picture bridge (lp2_text_16slots.pt) reads one 16-vector picture well
but B fixates on picture 1 of a multi-picture rollout — reading a *sequence* of
pictures is out-of-distribution. This fine-tunes the bridge (warm-started from the
single-picture checkpoint) on messages up to P pictures long: each ~26-token window
is encoded to 16 vectors, all P*16 vectors are spliced into B's context at once, and
frozen B is teacher-forced to transcribe the WHOLE message. If frozen B can learn
to read the concatenation, rollout works; if it plateaus, the honest conclusion is
that a receiver-side adapter is required (logged either way).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .channel_metrics import cluster_bootstrap_ci
from .latent_bridge import ABrain, GatherBridge
from .latent_port import MARKER, PortBatcher, greedy_decode, load_receiver, text_positions
from .provenance import provenance
from .text_baseline import levenshtein
from .text_bridge import TRAIN_TEMPLATES, WINDOW, load_snippets, masked_read, synthetic_chat

SLOTS = 16
CHUNK = 26  # tokens per picture window (matches the demo packetizer)


def long_messages(brain, base, pictures, rng, n):
    """Concatenate short snippets into messages up to `pictures` windows long."""
    out = []
    for _ in range(n):
        k = int(rng.integers(1, pictures + 1))
        msg = " ".join(base[int(rng.integers(0, len(base)))] for _ in range(k))
        ids = brain.tokenizer(msg, add_special_tokens=False).input_ids[: pictures * CHUNK]
        out.append(brain.tokenizer.decode(ids, skip_special_tokens=True).strip())
    return [m for m in out if m]


def encode_rollout(brain, bridge, message):
    """Split a message into windows, encode each to 16 vectors, concat -> [1, k*16, d]."""
    ids = brain.tokenizer(message, add_special_tokens=False).input_ids
    windows = [ids[i : i + CHUNK] for i in range(0, len(ids), CHUNK)] or [ids]
    texts = [brain.tokenizer.decode(w, skip_special_tokens=True).strip() for w in windows]
    states, mask = masked_read(brain, texts)          # [k, WINDOW, d_a]
    vecs = bridge(states, mask)                        # [k, 16, d]
    return vecs.reshape(1, -1, vecs.shape[-1])         # [1, k*16, d]


def evaluate(model, tokenizer, brain, bridge, batcher, snippets, args, samples):
    bridge.eval()
    fids, exact = [], 0
    rng = np.random.default_rng(args.seed + 7)
    msgs = long_messages(brain, snippets, args.pictures, rng, samples)
    for m in msgs:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            vecs = encode_rollout(brain, bridge, m).squeeze(0).float()
        decoded = greedy_decode(model, batcher, vecs, max_new=2 * len(tokenizer(m).input_ids) + 16).strip()
        fids.append(max(0.0, 1.0 - levenshtein(m, decoded) / max(len(m), len(decoded), 1)))
        exact += decoded == m
    bridge.train()
    return {"samples": len(msgs), "exact_rate": exact / max(1, len(msgs)),
            "char_fidelity": cluster_bootstrap_ci(np.array(fids), seed=args.seed)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receiver", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--sender", default="Qwen/Qwen3-4B")
    p.add_argument("--init", type=Path, default=Path("runs/bridges/lp2_text_16slots.pt"))
    p.add_argument("--pictures", type=int, default=4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-samples", type=int, default=48)
    p.add_argument("--seed", type=int, default=20260817)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/rollout_bridge.json"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer = load_receiver(args.receiver, args.device)
    brain = ABrain(args.sender, args.device)
    state = torch.load(args.init, map_location=args.device)
    positions = state["embed.0.weight"].shape[1] // brain.width
    d_model = model.get_input_embeddings().weight.shape[1]
    bridge = GatherBridge(brain.width, d_model, SLOTS, positions, 0.0, offset=0)
    bridge.load_state_dict(state)  # warm start from the working single-picture bridge
    bridge = bridge.to(args.device).train()
    batcher = PortBatcher(model, tokenizer, SLOTS, args.device, message=TRAIN_TEMPLATES[0])

    train, val = load_snippets(brain)
    train = train + synthetic_chat(np.random.default_rng(args.seed + 1), len(train))

    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    started, interim = time.monotonic(), []
    for step in range(args.steps):
        rng = np.random.default_rng(args.seed * 1_000_003 + step)
        msgs = long_messages(brain, train, args.pictures, rng, args.batch_size)
        loss_acc = 0.0
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            for m in msgs:  # variable picture count per message -> accumulate
                vecs = encode_rollout(brain, bridge, m)
                embeds, attn, labels = batcher.training_batch(vecs.squeeze(0).unsqueeze(0), [m])
                loss = model(inputs_embeds=embeds, attention_mask=attn, labels=labels,
                             position_ids=text_positions(1, embeds.shape[1], embeds.device)).loss / len(msgs)
                loss.backward()
                loss_acc += float(loss)
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step={step}/{args.steps} loss={loss_acc:.4f} ({time.monotonic()-started:.0f}s)", flush=True)
        if args.eval_every and step and step % args.eval_every == 0:
            sc = evaluate(model, tokenizer, brain, bridge, batcher, val, args, 16)
            interim.append({"step": step, **{k: sc[k] for k in ("exact_rate",)}, "fid": sc["char_fidelity"]["mean"]})
            print(f"interim: {json.dumps(interim[-1])}", flush=True)

    final = evaluate(model, tokenizer, brain, bridge, batcher, val, args, args.eval_samples)
    ckpt = args.output.parent / "bridges" / f"lp2_rollout_{args.pictures}pic_16slots.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bridge.state_dict(), ckpt)
    result = {"experiment": "LP-2 multi-picture rollout", "provenance": provenance(args),
              "pictures": args.pictures, "interim_evals": interim, "final": final,
              "bridge_checkpoint": str(ckpt), "train_seconds": time.monotonic() - started}
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
