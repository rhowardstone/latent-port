"""LP-2: the free-text latent port — A's words cross as 16 vectors.

Frozen Qwen3-0.6B ("A") reads an arbitrary text snippet; the bridge resamples a
fixed 48-token window of A's hidden states into 16 slot vectors; frozen
Qwen3-VL-2B ("B") is teacher-forced to write the snippet out. Training randomly
rotates several minimal context templates around the slots so B learns to read
the port wherever it appears in a conversation, not inside one magic prompt —
generalization is checked on a template never seen in training.
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

from .latent_bridge import ABrain, GatherBridge
from .latent_port import MARKER, PortBatcher, greedy_decode, load_receiver, text_positions
from .text_baseline import levenshtein

WINDOW = 32  # A-token window; 16 slots x 2 tokens/slot — the geometry LP-3a proved out


def synthetic_chat(rng: np.random.Generator, count: int) -> list[str]:
    """Conversational-register training lines, structurally chat-like."""
    words = "apple river candle tiger velvet marble thunder pocket lantern cinnamon walnut harbor".split()
    names = "Sam Priya Chen Maria Kofi Lena Omar Josie".split()
    objects = "key notebook charger umbrella ticket badge bottle jacket".split()
    places = "kitchen drawer garage blue box top shelf mailbox glovebox desk".split()
    days = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
    templates = [
        "Sure, I'll remember the word {w} and tell B when asked.",
        "The meeting is at {n}pm on {d}.",
        "The password is {w}{n}{n2}, please keep it safe.",
        "{name} said the {o} is in the {p}.",
        "My favorite word today is {w}.",
        "Remind B that the {o} needs to be back by {d}.",
        "The answer to the riddle is {w}.",
        "Tell them {name} arrives on {d} at {n} o'clock.",
        "The secret number is {n}{n2}{n3}.",
        "Please pass along that the {o} is under the {p}.",
        "Okay: the word is {w}. Just {w}. Nothing else.",
        "{name} prefers {w} over {w2}, remember that.",
    ]
    alphanumeric = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    secret_templates = [
        "Please remember this password: {s}.",
        "The password is {s}, keep it private.",
        "My API key is {s}.",
        "The access code {s} unlocks the side door.",
        "Write down {s}, that's the login.",
        "The wifi key is {s}.",
    ]
    assistant_register = [
        "Hello! How can I assist you today?",
        "I'm sorry, but I don't have access to that information.",
        "Sure! I can help with that. What would you like to know?",
        "Of course. I'll remember that and pass it along.",
        "Got it. Is there anything else you need?",
        "I'm glad to hear that! Let me know how I can help.",
        "Understood. I will tell B exactly that when asked.",
        "Thanks for letting me know. Anything else?",
        "I'm sorry, I don't understand. Could you clarify?",
        "Absolutely, consider it done.",
    ]
    lines = []
    for _ in range(count // 8):
        lines.append(assistant_register[int(rng.integers(0, len(assistant_register)))])
    for _ in range(count // 4):
        secret = "".join(
            alphanumeric[int(i)] for i in rng.integers(0, len(alphanumeric), int(rng.integers(6, 13)))
        )
        template = secret_templates[int(rng.integers(0, len(secret_templates)))]
        lines.append(template.format(s=secret))
    for _ in range(count - len(lines)):
        template = templates[int(rng.integers(0, len(templates)))]
        lines.append(template.format(
            w=words[int(rng.integers(0, len(words)))],
            w2=words[int(rng.integers(0, len(words)))],
            name=names[int(rng.integers(0, len(names)))],
            o=objects[int(rng.integers(0, len(objects)))],
            p=places[int(rng.integers(0, len(places)))],
            d=days[int(rng.integers(0, len(days)))],
            n=int(rng.integers(1, 10)), n2=int(rng.integers(0, 10)), n3=int(rng.integers(0, 10)),
        ))
    return lines

TRAIN_TEMPLATES = [
    "A message from your peer model A just arrived on your latent port as the "
    f"embedding vectors between the brackets. [{MARKER}] Write out A's message exactly.",
    f"[{MARKER}] The vectors above are a latent-port message from model A. "
    "Repeat the message verbatim.",
    "You are linked to another model, A, through a latent port; its messages "
    f"arrive as embedding vectors. A's current message: [{MARKER}] "
    "What does A say? Reply with the message text only.",
    f"Latent port input: [{MARKER}] Transcribe it.",
]
NOVEL_TEMPLATE = (
    f"Something just came through on the wire: [{MARKER}] Please read it back to me word for word."
)


def load_snippets(brain: ABrain, limit_tokens: int = 26):
    from datasets import load_dataset

    def clean(split):
        rows = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)["text"]
        keep = []
        for row in rows:
            row = " ".join(row.split())
            if 40 < len(row) < 240 and not row.startswith("="):
                ids = brain.tokenizer(row, add_special_tokens=True).input_ids[:limit_tokens]
                text = brain.tokenizer.decode(ids, skip_special_tokens=True).strip()
                if len(text) > 30:
                    keep.append(text)
        return keep

    return clean("train"), clean("validation")


def warm_targets(b_model, b_tokenizer, texts: list[str], slots: int, device: str) -> torch.Tensor:
    batch = b_tokenizer(
        texts, return_tensors="pt", padding="max_length", truncation=True,
        max_length=WINDOW, add_special_tokens=False,
    ).to(device)
    with torch.no_grad():
        embeds = b_model.get_input_embeddings()(batch["input_ids"]).float()
    return embeds.reshape(len(texts), slots, WINDOW // slots, -1).mean(dim=2)


def masked_read(brain: ABrain, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    states, mask = brain.read(texts, pad_to=WINDOW)
    return states * mask.unsqueeze(-1), mask


def evaluate(model, batcher, brain, bridge, snippets, args, samples: int) -> dict:
    bridge.eval()
    exact = 0
    char_accuracy = []
    b_tokens = []
    examples = []
    rng = np.random.default_rng(args.seed + 555)
    picks = rng.choice(len(snippets), size=samples, replace=False)
    for index in picks:
        text = snippets[index]
        states, mask = masked_read(brain, [text])
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
        ):
            latents = bridge(states, mask).squeeze(0)
        decoded = greedy_decode(model, batcher, latents.float(), max_new=90).strip()
        distance = levenshtein(text, decoded)
        exact += decoded == text
        char_accuracy.append(max(0.0, 1.0 - distance / max(len(text), len(decoded), 1)))
        b_tokens.append(
            len(batcher.tokenizer(text, add_special_tokens=False).input_ids)
        )
        if len(examples) < 3:
            examples.append({"sent": text, "decoded": decoded})
    bridge.train()
    return {
        "eval_samples": samples,
        "exact_rate": exact / samples,
        "char_accuracy": float(np.mean(char_accuracy)),
        "mean_b_tokens": float(np.mean(b_tokens)),
        "slots": args.slots,
        "position_compression_vs_text": float(np.mean(b_tokens)) / args.slots,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--sender", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--legibility-weight", type=float, default=0.0,
                        help="LP-5a: co-train a reference tap; add this weight of tap CE to the bridge loss")
    parser.add_argument("--tag", default="", help="suffix for output/checkpoint names")
    parser.add_argument("--warmstart-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--interim-eval-samples", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("runs/text_bridge.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer = load_receiver(args.receiver, args.device)
    brain = ABrain(args.sender, args.device)
    print("loading corpus…", flush=True)
    train_snippets, val_snippets = load_snippets(brain)
    train_snippets = train_snippets + synthetic_chat(
        np.random.default_rng(args.seed + 77), len(train_snippets)
    )
    harvest = Path("runs/a_register_corpus.json")
    if harvest.exists():
        harvested = json.loads(harvest.read_text())
        train_snippets = train_snippets + harvested * 2
        print(f"mixed in {len(harvested)} harvested A-register lines (x2)", flush=True)
    print(f"corpus: {len(train_snippets)} train / {len(val_snippets)} val", flush=True)

    embed_rms = model.get_input_embeddings().weight.detach().float().pow(2).mean().sqrt().item()
    d_model = model.get_input_embeddings().weight.shape[1]
    bridge = GatherBridge(
        brain.width, d_model, args.slots, WINDOW // args.slots, embed_rms, offset=0
    ).to(args.device)
    rng = np.random.default_rng(args.seed)
    calibration = [train_snippets[i] for i in rng.choice(len(train_snippets), 64, replace=False)]
    bridge.calibrate(*masked_read(brain, calibration))

    batchers = [
        PortBatcher(model, tokenizer, args.slots, args.device, message=template)
        for template in TRAIN_TEMPLATES
    ]
    novel_batcher = PortBatcher(model, tokenizer, args.slots, args.device, message=NOVEL_TEMPLATE)

    warm_optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr)
    for step in range(args.warmstart_steps):
        rng = np.random.default_rng(args.seed * 999_983 + step)
        texts = [train_snippets[i] for i in rng.integers(0, len(train_snippets), args.batch_size)]
        states, mask = masked_read(brain, texts)
        latents = bridge(states, mask)
        loss = nn.functional.mse_loss(
            latents.float(), warm_targets(model, tokenizer, texts, args.slots, args.device)
        )
        warm_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        warm_optimizer.step()
        if step % 200 == 0 or step == args.warmstart_steps - 1:
            print(f"warmstart {step}/{args.warmstart_steps} mse={loss.item():.5f}", flush=True)

    reference_tap = None
    if args.legibility_weight > 0:
        from .text_tap import TextWiretap

        reference_tap = TextWiretap(d_model, WINDOW // args.slots, brain.width).to(args.device)
        a_embed_table = brain.model.get_input_embeddings().weight.detach().float()
    trainable = list(bridge.parameters()) + (
        list(reference_tap.parameters()) if reference_tap else []
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
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
        texts = [train_snippets[i] for i in rng.integers(0, len(train_snippets), args.batch_size)]
        batcher = batchers[int(rng.integers(0, len(batchers)))]
        states, mask = masked_read(brain, texts)
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
        ):
            latents = bridge(states, mask)
            embeds, attention, labels = batcher.training_batch(latents, texts)
            loss = model(
                inputs_embeds=embeds,
                attention_mask=attention,
                labels=labels,
                position_ids=text_positions(embeds.shape[0], embeds.shape[1], embeds.device),
            ).loss
            if reference_tap is not None:
                window_ids = brain.tokenizer(
                    texts, return_tensors="pt", padding="max_length", truncation=True,
                    max_length=WINDOW, add_special_tokens=True,
                ).input_ids.to(args.device)
                tap_logits = reference_tap(latents.float(), a_embed_table)
                loss = loss + args.legibility_weight * nn.functional.cross_entropy(
                    tap_logits.reshape(-1, tap_logits.shape[-1]), window_ids.reshape(-1),
                    ignore_index=brain.tokenizer.pad_token_id,
                )
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
            scores = evaluate(model, batchers[0], brain, bridge, val_snippets, args, args.interim_eval_samples)
            interim.append({"step": step, "exact_rate": scores["exact_rate"], "char_accuracy": scores["char_accuracy"]})
            print(f"interim eval: {json.dumps(interim[-1])}", flush=True)

    final_trained = evaluate(model, batchers[0], brain, bridge, val_snippets, args, args.eval_samples)
    final_novel = evaluate(model, novel_batcher, brain, bridge, val_snippets, args, args.eval_samples)
    tap_scores = {}
    if reference_tap is not None:
        from .text_tap import read_text_wire

        hits, total = 0.0, 0
        rng = np.random.default_rng(args.seed + 999)
        for index in rng.choice(len(val_snippets), 32, replace=False):
            text = val_snippets[index]
            states, mask = masked_read(brain, [text])
            with torch.no_grad():
                latents = bridge(states, mask).float()
            decoded, _ = read_text_wire(reference_tap, latents, a_embed_table, brain.tokenizer)
            hits += 1 - levenshtein(text, decoded[0]) / max(len(text), len(decoded[0]), 1)
            total += 1
        tap_scores = {"cotrained_tap_char_accuracy": hits / total}
    checkpoint = args.output.parent / "bridges" / f"lp2_text_{args.slots}slots{args.tag}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bridge.state_dict(), checkpoint)
    from .provenance import provenance

    result = {
        "experiment": "LP-2 text bridge",
        "provenance": provenance(args),
        "sender_model": args.sender,
        "legibility_weight": args.legibility_weight,
        **tap_scores,
        "window_a_tokens": WINDOW,
        "steps": args.steps,
        "bridge_parameters": sum(p.numel() for p in bridge.parameters()),
        "train_seconds": time.monotonic() - started,
        "interim_evals": interim,
        "trained_template_eval": final_trained,
        "novel_template_eval": final_novel,
        "bridge_checkpoint": str(checkpoint),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
