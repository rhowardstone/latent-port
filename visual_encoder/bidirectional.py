"""LP-3b: bidirectional latent port between two instances of ONE frozen model.

Both peers are the same frozen model M (e.g. Qwen3-4B), sharing weights but
holding separate conversation contexts. Key economy: because both ends are the
same M, a single bridge — read M's hidden states over a text window, inject k
vectors into M's context, M transcribes — serves BOTH directions. Train one
self-port bridge + one tap; wire it A->B and B->A.

This uses a plain causal-LM receiver (standard RoPE), which is simpler than the
Qwen3-VL receiver path in text_bridge.py (no 3-plane position_ids hack needed).
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from .latent_bridge import GatherBridge
from .text_bridge import (
    NOVEL_TEMPLATE,
    TRAIN_TEMPLATES,
    WINDOW,
    load_snippets,
    synthetic_chat,
    warm_targets,
)
from .text_tap import TextWiretap, read_text_wire
from .text_baseline import levenshtein
from .wiretap import ManifoldGauge

MARKER = "@@PAYLOAD@@"


class Peer:
    """One frozen model M used in both roles: reader (sender) and receiver."""

    def __init__(self, model_id: str, device: str) -> None:
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        preferred = "flash_attention_2" if device.startswith("cuda") else "eager"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, low_cpu_mem_usage=True, attn_implementation=preferred
            )
        except (ImportError, ValueError, RuntimeError):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
            )
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.width = self.model.config.hidden_size
        self.embeddings = self.model.get_input_embeddings()

    @torch.no_grad()
    def read(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.tokenizer(
            texts, return_tensors="pt", padding="max_length", truncation=True,
            max_length=WINDOW, add_special_tokens=True,
        ).to(self.device)
        states = self.model(**batch, output_hidden_states=True).hidden_states[-1]
        masked = states * batch["attention_mask"].unsqueeze(-1)
        return masked.float(), batch["attention_mask"]


class PlainBatcher:
    """Splice latents into a marker prompt; standard 1D position_ids."""

    def __init__(self, peer: Peer, message: str) -> None:
        self.peer = peer
        pre, post = message.split(MARKER)
        ids = lambda s: torch.tensor(
            peer.tokenizer(s, add_special_tokens=False).input_ids, device=peer.device
        )
        with torch.no_grad():
            self.pre = peer.embeddings(ids(pre))
            self.post = peer.embeddings(ids(post))
        self.end_id = peer.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = peer.tokenizer.pad_token_id or self.end_id

    def prompt_embeds(self, latents: torch.Tensor) -> torch.Tensor:
        batch = latents.shape[0]
        return torch.cat(
            [
                self.pre.unsqueeze(0).expand(batch, -1, -1),
                latents.to(self.pre.dtype),
                self.post.unsqueeze(0).expand(batch, -1, -1),
            ],
            dim=1,
        )

    def training_batch(self, latents: torch.Tensor, targets_text: list[str]):
        prompt = self.prompt_embeds(latents)
        targets = [
            self.peer.tokenizer(t, add_special_tokens=False).input_ids + [self.end_id]
            for t in targets_text
        ]
        longest = max(len(t) for t in targets)
        batch, prompt_len = prompt.shape[0], prompt.shape[1]
        ids = torch.full((batch, longest), self.pad_id, device=self.peer.device)
        labels = torch.full((batch, prompt_len + longest), -100, device=self.peer.device)
        mask = torch.zeros(batch, prompt_len + longest, dtype=torch.long, device=self.peer.device)
        mask[:, :prompt_len] = 1
        for row, target in enumerate(targets):
            ids[row, : len(target)] = torch.tensor(target, device=self.peer.device)
            labels[row, prompt_len : prompt_len + len(target)] = torch.tensor(target, device=self.peer.device)
            mask[row, prompt_len : prompt_len + len(target)] = 1
        with torch.no_grad():
            target_embeds = self.peer.embeddings(ids)
        embeds = torch.cat([prompt, target_embeds], dim=1)
        position_ids = torch.arange(embeds.shape[1], device=self.peer.device).unsqueeze(0).expand(batch, -1)
        return embeds, mask, labels, position_ids

    @torch.no_grad()
    def generate(self, latents: torch.Tensor, max_new: int) -> str:
        embeds = self.prompt_embeds(latents.unsqueeze(0))
        position = embeds.shape[1]
        pos = torch.arange(position, device=self.peer.device).unsqueeze(0)
        out = self.peer.model(inputs_embeds=embeds, position_ids=pos, use_cache=True)
        past = out.past_key_values
        token = out.logits[:, -1].argmax(dim=-1)
        pieces: list[int] = []
        for _ in range(max_new):
            if token.item() == self.end_id:
                break
            pieces.append(token.item())
            pos = torch.tensor([[position]], device=self.peer.device)
            out = self.peer.model(
                input_ids=token.view(1, 1), position_ids=pos, past_key_values=past, use_cache=True
            )
            position += 1
            past = out.past_key_values
            token = out.logits[:, -1].argmax(dim=-1)
        return self.peer.tokenizer.decode(pieces).strip()


def evaluate(peer, bridge, batcher, snippets, args, samples: int) -> dict:
    bridge.eval()
    exact = 0
    char_accuracy = []
    rng = np.random.default_rng(args.seed + 555)
    for index in rng.choice(len(snippets), size=samples, replace=False):
        text = snippets[index]
        states, mask = peer.read([text])
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
        ):
            latents = bridge(states, mask).squeeze(0)
        decoded = batcher.generate(latents.float(), max_new=90)
        distance = levenshtein(text, decoded)
        exact += decoded == text
        char_accuracy.append(max(0.0, 1.0 - distance / max(len(text), len(decoded), 1)))
    bridge.train()
    return {
        "eval_samples": samples,
        "exact_rate": exact / samples,
        "char_accuracy": float(np.mean(char_accuracy)),
    }


def build_corpus(peer: Peer, seed: int):
    train, val = load_snippets(peer)  # load_snippets uses .tokenizer, Peer has it
    train = train + synthetic_chat(np.random.default_rng(seed + 77), len(train))
    harvest = Path("runs/a_register_corpus.json")
    if harvest.exists():
        train = train + json.loads(harvest.read_text()) * 2
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--warmstart-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--interim-eval-samples", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/bidirectional.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    peer = Peer(args.model, args.device)
    train_snippets, val_snippets = build_corpus(peer, args.seed)
    print(f"peer={args.model} width={peer.width} corpus={len(train_snippets)} train", flush=True)

    embed_rms = peer.embeddings.weight.detach().float().pow(2).mean().sqrt().item()
    bridge = GatherBridge(peer.width, peer.width, args.slots, WINDOW // args.slots, embed_rms, offset=0).to(args.device)
    rng = np.random.default_rng(args.seed)
    calib = [train_snippets[i] for i in rng.choice(len(train_snippets), 64, replace=False)]
    bridge.calibrate(*peer.read(calib))

    batchers = [PlainBatcher(peer, t) for t in TRAIN_TEMPLATES]
    novel_batcher = PlainBatcher(peer, NOVEL_TEMPLATE)

    warm_opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr)
    for step in range(args.warmstart_steps):
        r = np.random.default_rng(args.seed * 999_983 + step)
        texts = [train_snippets[i] for i in r.integers(0, len(train_snippets), args.batch_size)]
        states, mask = peer.read(texts)
        latents = bridge(states, mask)
        loss = nn.functional.mse_loss(latents.float(), warm_targets(peer.model, peer.tokenizer, texts, args.slots, args.device))
        warm_opt.zero_grad(set_to_none=True)
        loss.backward()
        warm_opt.step()
        if step % 200 == 0 or step == args.warmstart_steps - 1:
            print(f"warmstart {step}/{args.warmstart_steps} mse={loss.item():.5f}", flush=True)

    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))
    )
    started = time.monotonic()
    interim = []
    for step in range(args.steps):
        r = np.random.default_rng(args.seed * 1_000_003 + step)
        texts = [train_snippets[i] for i in r.integers(0, len(train_snippets), args.batch_size)]
        batcher = batchers[int(r.integers(0, len(batchers)))]
        states, mask = peer.read(texts)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            latents = bridge(states, mask)
            embeds, attn, labels, position_ids = batcher.training_batch(latents, texts)
            loss = peer.model(inputs_embeds=embeds, attention_mask=attn, labels=labels, position_ids=position_ids).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step={step}/{args.steps} loss={loss.item():.4f} ({time.monotonic() - started:.0f}s)", flush=True)
        if args.eval_every and step and step % args.eval_every == 0:
            sc = evaluate(peer, bridge, batchers[0], val_snippets, args, args.interim_eval_samples)
            interim.append({"step": step, **sc})
            print(f"interim eval: {json.dumps(interim[-1])}", flush=True)

    final = evaluate(peer, bridge, batchers[0], val_snippets, args, args.eval_samples)
    novel = evaluate(peer, bridge, novel_batcher, val_snippets, args, args.eval_samples)
    ckpt = args.output.parent / "bridges" / f"lp3b_selfport_{args.model.split('/')[-1]}_{args.slots}slots.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bridge.state_dict(), ckpt)
    from .provenance import provenance

    result = {
        "experiment": "LP-3b bidirectional self-port",
        "provenance": provenance(args),
        "model": args.model,
        "note": "one bridge, both directions (peers share weights)",
        "trained_template_eval": final,
        "novel_template_eval": novel,
        "interim_evals": interim,
        "bridge_checkpoint": str(ckpt),
        "train_seconds": time.monotonic() - started,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
