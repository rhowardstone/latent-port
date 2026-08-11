"""Wide picture: one contiguous block of many vectors, instead of a rollout.

Tests the user's insight — "16 slots was a thumbnail, not a limit." Long messages
don't need a SEQUENCE of 16-vector pictures (which makes B fixate); just use ONE
wider picture with more slots (e.g. 64), injected contiguously, same ~2 tokens/slot
density. This matches how a real image works (one big block of patch-tokens) and
sidesteps the sequence-reading problem entirely. Preliminary comparison vs
rollout_bridge.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .channel_metrics import cluster_bootstrap_ci
from .latent_bridge import ABrain, GatherBridge
from .latent_port import PortBatcher, greedy_decode, load_receiver, text_positions
from .provenance import provenance
from .text_baseline import levenshtein
from .text_bridge import TRAIN_TEMPLATES, load_snippets, synthetic_chat


def wide_messages(brain, base, window, rng, n):
    """Concatenate snippets into messages that roughly fill one `window`-token picture."""
    out = []
    for _ in range(n):
        msg, target = "", int(rng.integers(window // 2, window + 1))
        while len(brain.tokenizer(msg, add_special_tokens=False).input_ids) < target:
            msg = (msg + " " + base[int(rng.integers(0, len(base)))]).strip()
        ids = brain.tokenizer(msg, add_special_tokens=False).input_ids[:window]
        out.append(brain.tokenizer.decode(ids, skip_special_tokens=True).strip())
    return [m for m in out if m]


def read_wide(brain, bridge, texts, window):
    states, mask = brain.read(texts, pad_to=window)
    return bridge((states * mask.unsqueeze(-1)).float(), mask)


def warm_targets_wide(model, tokenizer, texts, slots, window, device):
    """Per-slot mean of the window's RECEIVER token embeddings (on-manifold init).

    Cold start crawls through a frozen receiver (LP-1 lesson); this is the warm
    start wide_picture was missing. Mirrors text_bridge.warm_targets but
    window-parameterized."""
    batch = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                      max_length=window, add_special_tokens=False).to(device)
    with torch.no_grad():
        emb = model.get_input_embeddings()(batch["input_ids"]).float()
    return emb.reshape(len(texts), slots, window // slots, -1).mean(dim=2)


def evaluate(model, tokenizer, brain, bridge, batcher, snippets, args, samples):
    bridge.eval()
    fids, exact = [], 0
    rng = np.random.default_rng(args.seed + 7)
    for m in wide_messages(brain, snippets, args.window, rng, samples):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            vecs = read_wide(brain, bridge, [m], args.window).squeeze(0).float()
        decoded = greedy_decode(model, batcher, vecs, max_new=2 * len(tokenizer(m).input_ids) + 16).strip()
        fids.append(max(0.0, 1.0 - levenshtein(m, decoded) / max(len(m), len(decoded), 1)))
        exact += decoded == m
    bridge.train()
    return {"samples": samples, "exact_rate": exact / max(1, samples),
            "char_fidelity": cluster_bootstrap_ci(np.array(fids), seed=args.seed)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receiver", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--sender", default="Qwen/Qwen3-4B")
    p.add_argument("--slots", type=int, default=64)
    p.add_argument("--window", type=int, default=128)   # 2 tokens/slot at 64 slots
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--warmstart-steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-samples", type=int, default=48)
    p.add_argument("--seed", type=int, default=20260819)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/wide_picture.json"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer = load_receiver(args.receiver, args.device)
    brain = ABrain(args.sender, args.device)
    d_model = model.get_input_embeddings().weight.shape[1]
    embed_rms = model.get_input_embeddings().weight.detach().float().pow(2).mean().sqrt().item()
    bridge = GatherBridge(brain.width, d_model, args.slots, args.window // args.slots, embed_rms, offset=0).to(args.device).train()
    batcher = PortBatcher(model, tokenizer, args.slots, args.device, message=TRAIN_TEMPLATES[0])

    train, val = load_snippets(brain, limit_tokens=48)
    train = train + synthetic_chat(np.random.default_rng(args.seed + 1), len(train))
    # calibrate per-dim stats on a wide batch
    bridge.calibrate(*(lambda s, m: (s * m.unsqueeze(-1), m))(*brain.read(
        wide_messages(brain, train, args.window, np.random.default_rng(args.seed), 64), pad_to=args.window)))

    ck = args.output.parent / "ckpt" / f"wide_{args.slots}s_{args.window}w.pt"
    warm_ck = args.output.parent / "ckpt" / f"wide_{args.slots}s_{args.window}w.warm.pt"
    warm_ck.parent.mkdir(parents=True, exist_ok=True)
    # WARM START (essential — cold start crawls through a frozen receiver). Save the
    # warm-started weights so a restart after warm-but-before-CE reloads them.
    if not ck.exists():
        if warm_ck.exists():
            bridge.load_state_dict(torch.load(warm_ck, map_location=args.device))
            print("loaded warm-start weights", flush=True)
        else:
            warm_opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr)
            for step in range(args.warmstart_steps):
                rng = np.random.default_rng(args.seed * 999_983 + step)
                msgs = wide_messages(brain, train, args.window, rng, args.batch_size)
                vecs = read_wide(brain, bridge, msgs, args.window)
                loss = nn.functional.mse_loss(
                    vecs.float(), warm_targets_wide(model, tokenizer, msgs, args.slots, args.window, args.device))
                warm_opt.zero_grad(set_to_none=True); loss.backward(); warm_opt.step()
                if step % 200 == 0 or step == args.warmstart_steps - 1:
                    print(f"warmstart {step}/{args.warmstart_steps} mse={loss.item():.5f}", flush=True)
            torch.save(bridge.state_dict(), warm_ck)

    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    start = 0
    if ck.exists():
        try:
            c = torch.load(ck, map_location=args.device)
            bridge.load_state_dict(c["bridge"]); opt.load_state_dict(c["opt"]); start = int(c["step"]) + 1
            print(f"RESUMED at {start}", flush=True)
        except Exception as exc:
            print(f"resume failed ({exc}); fresh", flush=True)

    started, interim = time.monotonic(), []
    for step in range(start, args.steps):
        rng = np.random.default_rng(args.seed * 1_000_003 + step)
        msgs = wide_messages(brain, train, args.window, rng, args.batch_size)
        opt.zero_grad(set_to_none=True); loss_acc = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            for m in msgs:
                vecs = read_wide(brain, bridge, [m], args.window)
                embeds, attn, labels = batcher.training_batch(vecs, [m])
                loss = model(inputs_embeds=embeds, attention_mask=attn, labels=labels,
                             position_ids=text_positions(1, embeds.shape[1], embeds.device)).loss / len(msgs)
                loss.backward(); loss_acc += float(loss)
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0); opt.step(); sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step={step}/{args.steps} loss={loss_acc:.4f} ({time.monotonic()-started:.0f}s)", flush=True)
        if step % 250 == 0 and step:
            try:
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"step": step, "bridge": bridge.state_dict(), "opt": opt.state_dict()}, str(ck) + ".tmp")
                os.replace(str(ck) + ".tmp", ck)
            except Exception as exc:
                print(f"ckpt failed ({exc})", flush=True)
        if args.eval_every and step and step % args.eval_every == 0:
            sc = evaluate(model, tokenizer, brain, bridge, batcher, val, args, 16)
            interim.append({"step": step, "exact": sc["exact_rate"], "fid": sc["char_fidelity"]["mean"]})
            print(f"interim: {json.dumps(interim[-1])}", flush=True)

    final = evaluate(model, tokenizer, brain, bridge, batcher, val, args, args.eval_samples)
    result = {"experiment": "wide picture (single contiguous block)", "provenance": provenance(args),
              "slots": args.slots, "window": args.window, "tokens_per_slot": args.window / args.slots,
              "interim_evals": interim, "final": final, "train_seconds": time.monotonic() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if ck.exists(): ck.unlink()
    if warm_ck.exists(): warm_ck.unlink()
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
