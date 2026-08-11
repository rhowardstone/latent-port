"""Train the receiver too: does a small LoRA on B break the frozen ceiling?

Direct test of "would training the receiver help?". Uses the LP-1 bit-packing
sender at the load where FROZEN B fails (20 bits/slot: ~0% exact, content ~95%
present but misordered). Adds a LoRA on B's decoder and co-trains it with the
sender. The wiretap gap predicts big headroom (the vectors carry ~2x what frozen
B reads). Reports exact/char/content-vs-order so the jump vs frozen is one glance,
and how much of the 8% order-error the adapter erases.
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
from peft import LoraConfig, get_peft_model

import os

from .channel_metrics import cluster_bootstrap_ci
from .latent_port import (
    LatentSender,
    PortBatcher,
    char_embedding_table,
    content_vs_order,
    greedy_decode,
    load_receiver,
    normalize,
    payload_to_bits,
    random_payload,
    text_positions,
    warmstart_targets,
)
from .provenance import provenance
from .text_baseline import levenshtein

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def evaluate(model, batcher, sender, args, samples):
    sender.eval()
    exact, char, unordered, order_err = 0, [], [], []
    for i in range(samples):
        rng = np.random.default_rng(args.seed * 7919 + 777_000 + i)
        payload = random_payload(args.chars, rng)
        bits = torch.from_numpy(payload_to_bits(payload, args.slots)).to(args.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            latents = sender(bits.unsqueeze(0)).squeeze(0)
        decoded = normalize(greedy_decode(model, batcher, latents, max_new=2 * args.chars + 8))
        exact += decoded == payload
        char.append(max(0.0, 1.0 - levenshtein(payload, decoded) / max(len(payload), len(decoded), 1)))
        s = content_vs_order(payload, decoded, args.slots)
        unordered.append(s["unordered_char_accuracy"]); order_err.append(s["order_error_share"])
    sender.train()
    return {"samples": samples, "exact_rate": exact / samples,
            "char_accuracy": float(np.mean(char)),
            "unordered_char_accuracy": float(np.mean(unordered)),
            "order_error_share": float(np.mean(order_err)),
            "char_ci": cluster_bootstrap_ci(np.array(char), seed=args.seed)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--chars", type=int, default=64)      # 20 bits/slot at 16 slots
    p.add_argument("--slots", type=int, default=16)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmstart-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260818)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/receiver_lora.json"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer = load_receiver(args.model, args.device)
    embed_rms = model.get_input_embeddings().weight.detach().float().pow(2).mean().sqrt().item()
    d_model = model.get_input_embeddings().weight.shape[1]
    batcher = PortBatcher(model, tokenizer, args.slots, args.device)

    # frozen baseline first (no LoRA), so the comparison is in-run and apples-to-apples
    sender = LatentSender(args.slots, args.chars // args.slots, d_model, embed_rms, hidden=args.hidden).to(args.device)
    char_table = char_embedding_table(model, tokenizer, args.device)
    warm = torch.optim.AdamW(sender.parameters(), lr=args.lr)
    for step in range(args.warmstart_steps):
        rng = np.random.default_rng(args.seed * 999_983 + step)
        pays = [random_payload(args.chars, rng) for _ in range(args.batch_size)]
        bits = torch.from_numpy(np.stack([payload_to_bits(x, args.slots) for x in pays])).to(args.device)
        loss = nn.functional.mse_loss(sender(bits).float(), warmstart_targets(pays, args.slots, char_table))
        warm.zero_grad(set_to_none=True); loss.backward(); warm.step()

    def train_phase(trainables, label):
        opt = torch.optim.AdamW(trainables, lr=args.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
        ck = args.output.parent / "ckpt" / f"rlora_{label}_{args.chars}c.pt"
        # checkpoint only TRAINABLE params (sender + any LoRA) — never the frozen 2B
        def trainable_state():
            return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        def load_trainable(state):
            own = dict(model.named_parameters())
            for n, v in state.items():
                if n in own:
                    own[n].data.copy_(v.to(own[n].device))
        def save_ck(step):
            try:
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"step": step, "sender": sender.state_dict(), "opt": opt.state_dict(),
                            "trainable": trainable_state()}, str(ck) + ".tmp")
                os.replace(str(ck) + ".tmp", ck)
            except Exception as exc:
                print(f"ckpt save failed ({exc}); continuing", flush=True)
        start = 0
        if ck.exists():
            try:
                c = torch.load(ck, map_location=args.device)
                sender.load_state_dict(c["sender"]); opt.load_state_dict(c["opt"])
                load_trainable(c["trainable"]); start = int(c["step"]) + 1
                print(f"RESUMED {label} at step {start}", flush=True)
            except Exception as exc:
                print(f"resume {label} failed ({exc}); fresh", flush=True)
        interim = []
        for step in range(start, args.steps):
            rng = np.random.default_rng(args.seed * 1_000_003 + step)
            pays = [random_payload(args.chars, rng) for _ in range(args.batch_size)]
            bits = torch.from_numpy(np.stack([payload_to_bits(x, args.slots) for x in pays])).to(args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                embeds, mask, labels = batcher.training_batch(sender(bits), pays)
                loss = model(inputs_embeds=embeds, attention_mask=mask, labels=labels,
                             position_ids=text_positions(embeds.shape[0], embeds.shape[1], embeds.device)).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainables, 1.0); opt.step(); sched.step()
            if step % 50 == 0 or step == args.steps - 1:
                print(f"  {label} step={step}/{args.steps} loss={loss.item():.4f}", flush=True)
            if step % 250 == 0 and step:
                save_ck(step)
            if args.eval_every and step and step % args.eval_every == 0:
                sc = evaluate(model, batcher, sender, args, 16)
                interim.append({"step": step, "exact": sc["exact_rate"], "order_err": sc["order_error_share"]})
                print(f"  interim: {json.dumps(interim[-1])}", flush=True)
        return interim

    print("== phase 1: FROZEN receiver (sender only) ==", flush=True)
    frozen_interim = train_phase(list(sender.parameters()), "frozen")
    frozen = evaluate(model, batcher, sender, args, args.eval_samples)
    print(json.dumps({"frozen": frozen}, sort_keys=True), flush=True)

    print("== phase 2: + LoRA on receiver (co-train sender + LoRA) ==", flush=True)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, bias="none", target_modules=LORA_TARGETS))
    lora_params = [p for p in model.parameters() if p.requires_grad]
    trainable = int(sum(p.numel() for p in lora_params))
    lora_interim = train_phase(list(sender.parameters()) + lora_params, "lora")
    lora_scores = evaluate(model, batcher, sender, args, args.eval_samples)

    result = {
        "experiment": "receiver LoRA vs frozen", "provenance": provenance(args),
        "chars": args.chars, "loaded_bits_per_slot": 5 * args.chars / args.slots,
        "lora_rank": args.lora_r, "lora_trainable_params": trainable,
        "frozen": frozen, "frozen_interim": frozen_interim,
        "lora": lora_scores, "lora_interim": lora_interim,
        "headline": {
            "frozen_exact": frozen["exact_rate"], "lora_exact": lora_scores["exact_rate"],
            "frozen_order_err": frozen["order_error_share"], "lora_order_err": lora_scores["order_error_share"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
