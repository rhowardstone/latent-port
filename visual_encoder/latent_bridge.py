"""LP-3: cross-model latent bridge — one mind's states spoken into another.

A frozen Qwen3-0.6B ("A") reads a sentence containing a payload. A trained
resampler bridge cross-attends over A's hidden states and emits k slot vectors,
which are injected into frozen Qwen3-VL-2B ("B") exactly as in LP-1. B is
teacher-forced to transcribe the payload. Neither model is trained — only the
bridge learns, so success means A's internal representations are translatable
into B's input space by a small learned adapter.

Unlike LP-1's bit-packing sender, tap-readability of this traffic is NOT
guaranteed by construction: the code factors through A's representation
geometry. Run wiretap training on the bridge checkpoint to measure it.
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

from .latent_port import (
    PortBatcher,
    char_embedding_table,
    greedy_decode,
    load_receiver,
    normalize,
    random_payload,
    warmstart_targets,
)
from .text_baseline import levenshtein


A_TEMPLATE = "The access code is: {payload}. Memorize it exactly."


# Qwen tokenizes " X" as one token for letters but splits " 3" into two tokens
# for digits, which would make character positions payload-dependent. Rendering
# digits as substitute lowercase letters keeps one char = one token at a fixed
# position. Bijective; B is still supervised on the true Base32 string.
DIGIT_RENDER = str.maketrans("234567", "bcdefg")


def a_text(payload: str) -> str:
    # Space the characters so each is a single token at a stable position in
    # A's stream — the bridge's queries need addressable targets to align to.
    return A_TEMPLATE.format(payload=" ".join(payload.translate(DIGIT_RENDER)))


class ABrain:
    """Frozen sender-side model: text -> hidden states at a chosen layer."""

    def __init__(self, model_id: str, device: str, layer: int = -1) -> None:
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, low_cpu_mem_usage=True)
            .to(device)
            .eval()
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.layer = layer
        self.device = device
        self.width = self.model.config.hidden_size

    @torch.no_grad()
    def read(self, texts: list[str], pad_to: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if pad_to:
            batch = self.tokenizer(
                texts, return_tensors="pt", padding="max_length", truncation=True,
                max_length=pad_to, add_special_tokens=True,
            ).to(self.device)
        else:
            batch = self.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=True
            ).to(self.device)
        states = self.model(
            **batch, output_hidden_states=True
        ).hidden_states[self.layer]
        return states.float(), batch["attention_mask"]


class Bridge(nn.Module):
    """Perceiver-style resampler: variable-length A states -> k slot vectors."""

    def __init__(
        self,
        a_width: int,
        d_model: int,
        slots: int,
        embed_rms: float,
        width: int = 512,
        layers: int = 2,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(slots, width) * 0.02)
        self.inlet = nn.Sequential(nn.LayerNorm(a_width), nn.Linear(a_width, width))
        # Learned key positions: cross-attention cannot align slot queries to
        # A-stream locations without a positional anchor on the keys.
        self.key_pos = nn.Parameter(torch.randn(512, width) * 0.02)
        self.attention = nn.ModuleList(
            nn.MultiheadAttention(width, heads, batch_first=True) for _ in range(layers)
        )
        self.ffn = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width)
            )
            for _ in range(layers)
        )
        self.out = nn.Linear(width, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("embed_rms", torch.tensor(float(embed_rms)))

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        keys = self.inlet(states) + self.key_pos[: states.shape[1]].unsqueeze(0)
        hidden = self.queries.unsqueeze(0).expand(states.shape[0], -1, -1)
        padding = mask == 0
        for attention, ffn in zip(self.attention, self.ffn):
            attended, _ = attention(hidden, keys, keys, key_padding_mask=padding)
            hidden = hidden + attended
            hidden = hidden + ffn(hidden)
        z = self.norm(self.out(hidden))
        return z * (self.embed_rms * self.gain)


class GatherBridge(nn.Module):
    """Gather A's states at each slot's character positions, then encode.

    The perceiver resampler collapses to a constant (attention locks onto the
    shared prefix keys because the per-dim target signal is tiny next to the
    shared mean — see NOTEBOOK). With one-token-per-char rendering the positions
    are deterministic, so alignment can be wired instead of learned.
    """

    def __init__(
        self,
        a_width: int,
        d_model: int,
        slots: int,
        chars_per_slot: int,
        embed_rms: float,
        offset: int,
        hidden: int = 512,
        mixer_layers: int = 2,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.slots = slots
        self.chars_per_slot = chars_per_slot
        self.offset = offset
        # Per-dim standardization instead of LayerNorm: A's states carry huge
        # near-constant outlier dimensions (30x the median scale) that dominate
        # a joint norm and its gradients, collapsing training to a constant.
        self.register_buffer("a_mean", torch.zeros(a_width))
        self.register_buffer("a_std", torch.ones(a_width))
        self.embed = nn.Sequential(
            nn.Linear(chars_per_slot * a_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.slot_pos = nn.Parameter(torch.randn(slots, hidden) * 0.02)
        block = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=0.0, batch_first=True, norm_first=True
        )
        self.mixer = nn.TransformerEncoder(block, mixer_layers)
        self.out = nn.Linear(hidden, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("embed_rms", torch.tensor(float(embed_rms)))

    @torch.no_grad()
    def calibrate(self, states: torch.Tensor, mask: torch.Tensor) -> None:
        flat = states[mask.bool()]
        self.a_mean.copy_(flat.mean(dim=0))
        self.a_std.copy_(flat.std(dim=0).clamp_min(1e-3))

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = states.shape[0]
        span = states[:, self.offset : self.offset + self.slots * self.chars_per_slot]
        span = (span - self.a_mean) / self.a_std
        chunks = span.reshape(batch, self.slots, -1)
        h = self.embed(chunks) + self.slot_pos
        h = self.mixer(h)
        z = self.norm(self.out(h))
        return z * (self.embed_rms * self.gain)


def payload_offset(brain: ABrain) -> int:
    """Token index where the first payload character lands in A's stream."""
    prefix = A_TEMPLATE.split(" {payload}")[0]
    return len(brain.tokenizer(prefix, add_special_tokens=True).input_ids)


def evaluate_bridge(model, batcher, brain, bridge, args, characters: int, samples: int) -> dict:
    bridge.eval()
    exact = 0
    char_accuracy = []
    a_tokens = []
    examples = []
    for index in range(samples):
        rng = np.random.default_rng(args.seed * 7919 + 777_000 + index)
        payload = random_payload(characters, rng)
        states, mask = brain.read([a_text(payload)])
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
        ):
            latents = bridge(states, mask).squeeze(0)
        decoded = normalize(
            greedy_decode(model, batcher, latents.float(), max_new=2 * characters + 8)
        )
        distance = levenshtein(payload, decoded)
        exact += decoded == payload
        char_accuracy.append(max(0.0, 1.0 - distance / max(len(payload), len(decoded), 1)))
        a_tokens.append(int(mask.sum()))
        if index < 3:
            examples.append({"payload": payload, "decoded": decoded})
    bridge.train()
    return {
        "eval_samples": samples,
        "exact_rate": exact / samples,
        "char_accuracy": float(np.mean(char_accuracy)),
        "mean_a_tokens": float(np.mean(a_tokens)),
        "slots": args.slots,
        "position_compression_vs_a": float(np.mean(a_tokens)) / args.slots,
        "loaded_bits_per_slot": 5 * characters / args.slots,
        "net_exact_bits_per_slot": 5 * characters * (exact / samples) / args.slots,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--sender", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--a-layer", type=int, default=-1)
    parser.add_argument("--bridge", choices=("gather", "perceiver"), default="gather")
    parser.add_argument("--chars", type=int, default=32)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--warmstart-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--interim-eval-samples", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/latent_bridge.json"))
    args = parser.parse_args()

    if args.chars % args.slots:
        raise SystemExit("--chars must divide into --slots")
    torch.manual_seed(args.seed)
    model, tokenizer = load_receiver(args.receiver, args.device)
    brain = ABrain(args.sender, args.device, layer=args.a_layer)
    embed_rms = model.get_input_embeddings().weight.detach().float().pow(2).mean().sqrt().item()
    d_model = model.get_input_embeddings().weight.shape[1]
    if args.bridge == "gather":
        bridge = GatherBridge(
            brain.width,
            d_model,
            args.slots,
            args.chars // args.slots,
            embed_rms,
            offset=payload_offset(brain),
        ).to(args.device)
        calibration_rng = np.random.default_rng(args.seed + 424_243)
        calibration = [random_payload(args.chars, calibration_rng) for _ in range(64)]
        bridge.calibrate(*brain.read([a_text(p) for p in calibration]))
    else:
        bridge = Bridge(brain.width, d_model, args.slots, embed_rms).to(args.device)
    batcher = PortBatcher(model, tokenizer, args.slots, args.device)

    if args.warmstart_steps:
        char_table = char_embedding_table(model, tokenizer, args.device)
        warm_optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr)
        for step in range(args.warmstart_steps):
            rng = np.random.default_rng(args.seed * 999_983 + step)
            payloads = [random_payload(args.chars, rng) for _ in range(args.batch_size)]
            states, mask = brain.read([a_text(p) for p in payloads])
            latents = bridge(states, mask)
            loss = nn.functional.mse_loss(
                latents.float(), warmstart_targets(payloads, args.slots, char_table)
            )
            warm_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            warm_optimizer.step()
            if step % 200 == 0 or step == args.warmstart_steps - 1:
                print(f"warmstart {step}/{args.warmstart_steps} mse={loss.item():.5f}", flush=True)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / args.warmup)
        * 0.5
        * (1 + math.cos(math.pi * min(1.0, step / args.steps))),
    )
    started = time.monotonic()
    interim = []
    for step in range(args.steps):
        rng = np.random.default_rng(args.seed * 1_000_003 + step)
        payloads = [random_payload(args.chars, rng) for _ in range(args.batch_size)]
        states, mask = brain.read([a_text(p) for p in payloads])
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
        ):
            latents = bridge(states, mask)
            embeds, attention, labels = batcher.training_batch(latents, payloads)
            from .latent_port import text_positions

            loss = model(
                inputs_embeds=embeds,
                attention_mask=attention,
                labels=labels,
                position_ids=text_positions(embeds.shape[0], embeds.shape[1], embeds.device),
            ).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(
                f"step={step}/{args.steps} loss={loss.item():.4f} "
                f"({time.monotonic() - started:.0f}s)",
                flush=True,
            )
        if args.eval_every and step and step % args.eval_every == 0:
            scores = evaluate_bridge(
                model, batcher, brain, bridge, args, args.chars, args.interim_eval_samples
            )
            interim.append(
                {"step": step, "exact_rate": scores["exact_rate"], "char_accuracy": scores["char_accuracy"]}
            )
            print(f"interim eval: {json.dumps(interim[-1])}", flush=True)

    final = evaluate_bridge(model, batcher, brain, bridge, args, args.chars, args.eval_samples)
    checkpoint = args.output.parent / "bridges" / f"lp3_{args.chars}chars_{args.slots}slots.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bridge.state_dict(), checkpoint)
    result = {
        "experiment": "LP-3a",
        "bridge_kind": args.bridge,
        "sender_model": args.sender,
        "receiver_model": args.receiver,
        "a_layer": args.a_layer,
        "characters": args.chars,
        "steps": args.steps,
        "warmstart_steps": args.warmstart_steps,
        "bridge_parameters": sum(p.numel() for p in bridge.parameters()),
        "train_seconds": time.monotonic() - started,
        "interim_evals": interim,
        "bridge_checkpoint": str(checkpoint),
        **final,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
