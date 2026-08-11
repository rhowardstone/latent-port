"""LP-1: learned sender, frozen receiver, held-out random bits.

Trains a small sender network that maps random Base32 payloads to k soft
embeddings spliced into the prompt of a frozen Qwen3-VL decoder (vision tower
deleted — this is a pure latent port, no pixels). Teacher-forced cross-entropy on
exact transcription trains ONLY the sender; the receiver never changes.

Each latent slot occupies one KV position, exactly like a text token, so
bits-per-slot is directly comparable to bits-per-text-token for the same payload.
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
from transformers import AutoModelForImageTextToText, AutoTokenizer

from .text_baseline import BASE32_PATTERN, levenshtein

BASE32_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
MARKER = "@@PAYLOAD@@"


def text_positions(batch: int, length: int, device, offset: int = 0) -> torch.Tensor:
    """Text-only mRoPE positions: identical arange in all three planes.

    Qwen3-VL's forward derives multimodal positions from input_ids and crashes on
    pure inputs_embeds; explicit position_ids skip that path entirely.
    """
    positions = torch.arange(offset, offset + length, device=device)
    return positions.view(1, 1, -1).expand(3, batch, -1)


def random_payload(characters: int, rng: np.random.Generator) -> str:
    return "".join(BASE32_ALPHABET[i] for i in rng.integers(0, 32, characters))


def payload_to_bits(payload: str, slots: int) -> np.ndarray:
    """Map C chars to [slots, (C/slots)*5] antipodal bit vectors."""
    if len(payload) % slots:
        raise ValueError(f"{len(payload)} chars do not divide into {slots} slots")
    indices = np.array([BASE32_ALPHABET.index(c) for c in payload], dtype=np.uint8)
    bits = np.unpackbits(indices[:, None], axis=1, bitorder="big")[:, 3:]
    return (bits.reshape(slots, -1).astype(np.float32) * 2.0) - 1.0


class LatentSender(nn.Module):
    """Per-slot MLP + transformer mixer, output rescaled to the embedding RMS."""

    def __init__(
        self,
        slots: int,
        chars_per_slot: int,
        d_model: int,
        embed_rms: float,
        hidden: int = 512,
        mixer_layers: int = 2,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(chars_per_slot * 5, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.slot_pos = nn.Parameter(torch.randn(slots, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 4, dropout=0.0, batch_first=True, norm_first=True
        )
        self.mixer = nn.TransformerEncoder(layer, mixer_layers)
        self.out = nn.Linear(hidden, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("embed_rms", torch.tensor(float(embed_rms)))

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        h = self.embed(bits) + self.slot_pos
        h = self.mixer(h)
        z = self.norm(self.out(h))
        return z * (self.embed_rms * self.gain)


def load_receiver(model_id: str, device: str):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    preferred = "flash_attention_2" if device.startswith("cuda") else "eager"
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True, attn_implementation=preferred
        )
    except (ImportError, ValueError, RuntimeError):
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
        )
    model.model.visual = None  # pure latent port: pixels never exist
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def build_prompt(tokenizer, slots: int, message: str | None = None) -> tuple[list[int], list[int]]:
    if message is None:
        message = (
            f"The {slots} embedding slots between the brackets hold an encoded message. "
            f"[{MARKER}] Transcribe the base32 payload stored in those slots exactly: "
            "reply with only the characters A-Z and 2-7, nothing else."
        )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
    )
    pre, post = text.split(MARKER)
    pre_ids = tokenizer(pre, add_special_tokens=False).input_ids
    post_ids = tokenizer(post, add_special_tokens=False).input_ids
    return pre_ids, post_ids


class PortBatcher:
    def __init__(self, model, tokenizer, slots: int, device: str, message: str | None = None) -> None:
        self.embeddings = model.get_input_embeddings()
        self.device = device
        self.slots = slots
        pre_ids, post_ids = build_prompt(tokenizer, slots, message=message)
        ids = lambda values: torch.tensor(values, device=device)
        with torch.no_grad():
            self.pre = self.embeddings(ids(pre_ids))
            self.post = self.embeddings(ids(post_ids))
        self.end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = tokenizer.pad_token_id or self.end_id
        self.tokenizer = tokenizer

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

    def training_batch(self, latents: torch.Tensor, payloads: list[str]):
        prompt = self.prompt_embeds(latents)
        targets = [
            self.tokenizer(text, add_special_tokens=False).input_ids + [self.end_id]
            for text in payloads
        ]
        longest = max(len(t) for t in targets)
        batch, prompt_len = prompt.shape[0], prompt.shape[1]
        ids = torch.full((batch, longest), self.pad_id, device=self.device)
        labels = torch.full((batch, prompt_len + longest), -100, device=self.device)
        mask = torch.zeros(batch, prompt_len + longest, device=self.device, dtype=torch.long)
        mask[:, :prompt_len] = 1
        for row, target in enumerate(targets):
            ids[row, : len(target)] = torch.tensor(target, device=self.device)
            labels[row, prompt_len : prompt_len + len(target)] = torch.tensor(
                target, device=self.device
            )
            mask[row, prompt_len : prompt_len + len(target)] = 1
        with torch.no_grad():
            target_embeds = self.embeddings(ids)
        return torch.cat([prompt, target_embeds], dim=1), mask, labels


@torch.no_grad()
def greedy_decode(model, batcher: PortBatcher, latents: torch.Tensor, max_new: int) -> str:
    embeds = batcher.prompt_embeds(latents.unsqueeze(0))
    device = embeds.device
    position = embeds.shape[1]
    output = model(
        inputs_embeds=embeds,
        position_ids=text_positions(1, position, device),
        use_cache=True,
    )
    past = output.past_key_values
    token = output.logits[:, -1].argmax(dim=-1)
    generated: list[int] = []
    for _ in range(max_new):
        if token.item() == batcher.end_id:
            break
        generated.append(token.item())
        output = model(
            input_ids=token.view(1, 1),
            position_ids=text_positions(1, 1, device, offset=position),
            past_key_values=past,
            use_cache=True,
        )
        position += 1
        past = output.past_key_values
        token = output.logits[:, -1].argmax(dim=-1)
    return batcher.tokenizer.decode(generated)


@torch.no_grad()
def copy_control(model, tokenizer, payload: str, device: str, max_new: int) -> str:
    message = (
        "Transcribe this base32 payload exactly, replying with only the characters "
        f"A-Z and 2-7, nothing else: {payload}"
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    position = ids.shape[1]
    output = model(input_ids=ids, position_ids=text_positions(1, position, device), use_cache=True)
    past = output.past_key_values
    token = output.logits[:, -1].argmax(dim=-1)
    generated: list[int] = []
    for _ in range(max_new):
        if token.item() == end_id:
            break
        generated.append(token.item())
        output = model(
            input_ids=token.view(1, 1),
            position_ids=text_positions(1, 1, device, offset=position),
            past_key_values=past,
            use_cache=True,
        )
        position += 1
        past = output.past_key_values
        token = output.logits[:, -1].argmax(dim=-1)
    return tokenizer.decode(generated)


def normalize(text: str) -> str:
    return "".join(BASE32_PATTERN.findall(text.upper()))


def evaluate(model, batcher, sender, args, characters: int, samples: int) -> dict:
    sender.eval()
    exact = 0
    char_accuracy = []
    text_tokens = []
    unordered = []
    order_err = []
    examples = []
    for index in range(samples):
        rng = np.random.default_rng(args.seed * 7919 + 777_000 + index)
        payload = random_payload(characters, rng)
        bits = torch.from_numpy(payload_to_bits(payload, args.slots)).to(args.device)
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
        ):
            latents = sender(bits.unsqueeze(0)).squeeze(0)
        decoded = normalize(
            greedy_decode(model, batcher, latents, max_new=2 * characters + 8)
        )
        distance = levenshtein(payload, decoded)
        exact += decoded == payload
        char_accuracy.append(max(0.0, 1.0 - distance / max(len(payload), len(decoded), 1)))
        text_tokens.append(
            len(batcher.tokenizer(payload, add_special_tokens=False).input_ids)
        )
        split = content_vs_order(payload, decoded, args.slots)
        unordered.append(split["unordered_char_accuracy"])
        order_err.append(split["order_error_share"])
        if index < 3:
            examples.append({"payload": payload, "decoded": decoded})
    sender.train()
    exact_rate = exact / samples
    return {
        "eval_samples": samples,
        "exact_rate": exact_rate,
        "char_accuracy": float(np.mean(char_accuracy)),
        # Content vs order (external-review binding control): if unordered stays
        # high while order_error rises, transpositions dominate — the binding signal.
        "unordered_char_accuracy": float(np.mean(unordered)),
        "order_error_share": float(np.mean(order_err)),
        "warmstart_order": args.warmstart_order,
        "net_exact_bits_per_slot": 5 * characters * exact_rate / args.slots,
        "loaded_bits_per_slot": 5 * characters / args.slots,
        "text_tokens_for_same_payload": float(np.mean(text_tokens)),
        "text_baseline_bits_per_token": 5 * characters / float(np.mean(text_tokens)),
        "examples": examples,
    }


def char_embedding_table(model, tokenizer, device) -> torch.Tensor:
    """Embedding of each base32 character's single token, [32, d_model]."""
    rows = []
    with torch.no_grad():
        for char in BASE32_ALPHABET:
            ids = tokenizer(char, add_special_tokens=False).input_ids
            if len(ids) != 1:
                raise RuntimeError(f"{char!r} tokenizes to {len(ids)} tokens; warm start assumes 1")
            rows.append(model.get_input_embeddings()(torch.tensor(ids, device=device))[0])
    return torch.stack(rows).float()


def warmstart_targets(
    payloads: list[str], slots: int, char_table: torch.Tensor,
    order: str = "mean", eps: float = 1.0,
) -> torch.Tensor:
    """Per-slot warm-start target, [batch, slots, d].

    order="mean": average the slot's chunk char embeddings. PERMUTATION-INVARIANT
    — z(AB)=z(BA) — so it erases sub-slot order before CE, a confound for any
    binding claim (external review, 2026-08-10).

    order="role": (1/sqrt m) * sum_j roll(e(c_j), shift_j). Distinct fixed
    circular shift per sub-slot position (orthogonal role binding, HRR-style) →
    order-SENSITIVE. But a dimension roll is orthogonal, NOT semantic (Qwen's
    embedding coords aren't rotation-invariant), so pure role may start far off
    B's native geometry.

    order="blend": RMSNorm[(1-eps)*mean + eps*role], eps in (0,1]. The gentle
    control — inject only a little order into an otherwise on-manifold target.
    If tiny eps kills transpositions, the binding fix is nearly free.
    """
    indices = torch.tensor(
        [[BASE32_ALPHABET.index(c) for c in payload] for payload in payloads],
        device=char_table.device,
    )
    chunks = char_table[indices].reshape(len(payloads), slots, -1, char_table.shape[-1])
    mean_t = chunks.mean(dim=2)
    if order == "mean":
        return mean_t
    m, d = chunks.shape[2], chunks.shape[3]
    shifts = [(j * (d // (m + 1))) for j in range(m)]
    role_t = torch.stack([chunks[:, :, j].roll(shifts[j], dims=-1) for j in range(m)], dim=2).sum(dim=2) / (m ** 0.5)
    if order == "role":
        return role_t
    if order == "blend":
        mixed = (1.0 - eps) * mean_t + eps * role_t
        rms = mixed.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
        return mixed / rms * mean_t.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
    raise ValueError(f"unknown warm-start order: {order}")


def damerau_levenshtein(a: str, b: str) -> int:
    """Edit distance counting an adjacent transposition as ONE operation."""
    da = {}
    inf = len(a) + len(b)
    d = [[inf] * (len(b) + 2) for _ in range(len(a) + 2)]
    d[0][0] = inf
    for i in range(len(a) + 1):
        d[i + 1][0], d[i + 1][1] = inf, i
    for j in range(len(b) + 1):
        d[0][j + 1], d[1][j + 1] = inf, j
    for i in range(1, len(a) + 1):
        db = 0
        for j in range(1, len(b) + 1):
            k, l = da.get(b[j - 1], 0), db
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if not cost:
                db = j
            d[i + 1][j + 1] = min(
                d[i][j] + cost, d[i + 1][j] + 1, d[i][j + 1] + 1,
                d[k][l] + (i - k - 1) + 1 + (j - l - 1),
            )
        da[a[i - 1]] = i
    return d[len(a) + 1][len(b) + 1]


def _nw_align(payload: str, decoded: str) -> list[tuple[int | None, int | None]]:
    """Needleman-Wunsch alignment; returns (payload_idx|None, decoded_idx|None) pairs.

    Aligning first means an insertion/deletion shifts one position instead of
    masquerading as a whole-slot content+order failure (external review, matters
    at the 30-bit regime).
    """
    n, mlen = len(payload), len(decoded)
    dp = [[0] * (mlen + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(mlen + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, mlen + 1):
            cost = 0 if payload[i - 1] == decoded[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + cost, dp[i - 1][j] + 1, dp[i][j - 1] + 1)
    i, j, pairs = n, mlen, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if payload[i - 1] == decoded[j - 1] else 1):
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    return pairs[::-1]


def content_vs_order(payload: str, decoded: str, slots: int) -> dict:
    """Separate content errors (wrong char multiset) from pure ordering errors,
    after edit-aligning decoded to payload so indels don't masquerade."""
    from collections import Counter

    if not payload or len(payload) % slots:
        return {"unordered_char_accuracy": 0.0, "order_error_share": 0.0, "transposition_share": 0.0}
    m = len(payload) // slots
    total = len(payload)
    if not decoded:
        return {"unordered_char_accuracy": 0.0, "order_error_share": 0.0, "transposition_share": 0.0}
    # Map each payload position to its aligned decoded char (or None for a deletion).
    aligned = [None] * total
    positional_hits = 0
    for pi, dj in _nw_align(payload, decoded):
        if pi is not None and dj is not None:
            aligned[pi] = decoded[dj]
            if decoded[dj] == payload[pi]:
                positional_hits += 1
    unordered_hits = 0
    for s in range(slots):
        p = payload[s * m : (s + 1) * m]
        q = [c for c in aligned[s * m : (s + 1) * m] if c is not None]
        unordered_hits += sum((Counter(p) & Counter(q)).values())
    dl = damerau_levenshtein(payload, decoded)
    lev = levenshtein(payload, decoded)
    return {
        "unordered_char_accuracy": unordered_hits / total,
        # content present in the right slot but positionally wrong = ordering failure
        "order_error_share": max(0, unordered_hits - positional_hits) / total,
        # transpositions cost 1 in DL but 2 in Levenshtein; the gap flags them
        "transposition_share": max(0, lev - dl) / max(1, lev),
    }


def run_config(model, tokenizer, args, characters: int) -> dict:
    torch.manual_seed(args.seed + characters)
    embed_rms = (
        model.get_input_embeddings().weight.detach().float().pow(2).mean().sqrt().item()
    )
    d_model = model.get_input_embeddings().weight.shape[1]
    sender = LatentSender(
        args.slots, characters // args.slots, d_model, embed_rms, hidden=args.hidden
    ).to(args.device)
    batcher = PortBatcher(model, tokenizer, args.slots, args.device)
    if args.warmstart_steps and args.warmstart_order != "none":
        char_table = char_embedding_table(model, tokenizer, args.device)
        warm_optimizer = torch.optim.AdamW(sender.parameters(), lr=args.lr)
        for step in range(args.warmstart_steps):
            rng = np.random.default_rng(args.seed * 999_983 + step)
            payloads = [random_payload(characters, rng) for _ in range(args.batch_size)]
            bits = torch.from_numpy(
                np.stack([payload_to_bits(p, args.slots) for p in payloads])
            ).to(args.device)
            latents = sender(bits)
            warm_loss = nn.functional.mse_loss(
                latents.float(),
                warmstart_targets(payloads, args.slots, char_table, order=args.warmstart_order, eps=args.warmstart_eps),
            )
            warm_optimizer.zero_grad(set_to_none=True)
            warm_loss.backward()
            warm_optimizer.step()
            if step % 200 == 0 or step == args.warmstart_steps - 1:
                print(
                    f"  chars={characters} warmstart {step}/{args.warmstart_steps} "
                    f"mse={warm_loss.item():.5f}",
                    flush=True,
                )
    optimizer = torch.optim.AdamW(sender.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / args.warmup)
        * 0.5
        * (1 + math.cos(math.pi * min(1.0, step / args.steps))),
    )
    started = time.monotonic()
    losses = []
    interim = []
    for step in range(args.steps):
        rng = np.random.default_rng(args.seed * 1_000_003 + step)
        payloads = [random_payload(characters, rng) for _ in range(args.batch_size)]
        bits = torch.from_numpy(
            np.stack([payload_to_bits(p, args.slots) for p in payloads])
        ).to(args.device)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
        ):
            latents = sender(bits)
            embeds, mask, labels = batcher.training_batch(latents, payloads)
            loss = model(
                inputs_embeds=embeds,
                attention_mask=mask,
                labels=labels,
                position_ids=text_positions(embeds.shape[0], embeds.shape[1], embeds.device),
            ).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sender.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if step % 50 == 0 or step == args.steps - 1:
            losses.append({"step": step, "loss": round(loss.item(), 4)})
            print(
                f"  chars={characters} step={step}/{args.steps} "
                f"loss={loss.item():.4f} ({time.monotonic() - started:.0f}s)",
                flush=True,
            )
        if args.eval_every and step and step % args.eval_every == 0:
            scores = evaluate(model, batcher, sender, args, characters, args.interim_eval_samples)
            interim.append({"step": step, **{k: scores[k] for k in ("exact_rate", "char_accuracy")}})
            print(f"  interim eval: {json.dumps(interim[-1])}", flush=True)
    final = evaluate(model, batcher, sender, args, characters, args.eval_samples)
    checkpoint = args.output.parent / "senders" / f"lp1_{characters}chars_{args.slots}slots.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sender.state_dict(), checkpoint)
    control_rng = np.random.default_rng(args.seed + 31337)
    control_payload = random_payload(characters, control_rng)
    control = normalize(
        copy_control(model, tokenizer, control_payload, args.device, 2 * characters + 8)
    )
    return {
        "characters": characters,
        "slots": args.slots,
        "loaded_bits_per_slot": 5 * characters / args.slots,
        "steps": args.steps,
        "warmstart_steps": args.warmstart_steps,
        "batch_size": args.batch_size,
        "sender_parameters": sum(p.numel() for p in sender.parameters()),
        "train_seconds": time.monotonic() - started,
        "loss_curve": losses,
        "interim_evals": interim,
        "copy_control_exact": control == control_payload,
        "sender_checkpoint": str(checkpoint),
        **final,
    }


def parse_ints(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",")})
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--chars", type=parse_ints, default=parse_ints("16,32,48,64,96"))
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--warmstart-steps", type=int, default=500)
    parser.add_argument("--warmstart-order", choices=("mean", "role", "blend", "none"), default="mean",
                        help="binding control: 'mean' is permutation-invariant (confound); "
                             "'role' preserves sub-slot order; 'blend' mixes them by --warmstart-eps; "
                             "'none' skips warm start")
    parser.add_argument("--warmstart-eps", type=float, default=1.0,
                        help="blend fraction of the order-preserving target (0=mean, 1=role)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=400)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--interim-eval-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/latent_port.json"))
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    for characters in args.chars:
        if characters % args.slots:
            raise SystemExit(f"--chars {characters} is not divisible by --slots {args.slots}")
    model, tokenizer = load_receiver(args.model, args.device)
    results = []
    for characters in args.chars:
        print(f"== LP-1 config: {characters} chars into {args.slots} slots ==", flush=True)
        result = run_config(model, tokenizer, args, characters)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "loss_curve"}, sort_keys=True), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"model": args.model, "experiment": "LP-1", "results": results}, indent=2)
            + "\n"
        )


if __name__ == "__main__":
    main()
