"""LP-6: music through the latent port — can a FROZEN text LM re-emit a melody?

The receiver (Qwen3-VL-2B) is a text model, but ABC notation is all over the web,
so it has a prior for it (cf. ChatMusician). We serialize permissively-licensed folk
melodies (IrishMAN, MIT / believed-PD, 216k tunes, already in ABC) as text, inject
them as k vectors, and ask the frozen receiver to write the ABC back.

Two hypotheses (this file tests H1; latent_rollout --music tests H2):
  H1  more information rides through via music than text — BUT music is redundant, so
      note-fidelity can RISE while Shannon bits/slot FALL. The honest adjudicator is
      ΔI vs a deranged tune (token_savings-style), not note-F1. We report BOTH and
      never conflate them.
  H2  latent rollout chains better on music (strong manifold prior resists drift).

Metric discipline mirrors the rest of the lab: parse both sent & decoded ABC to note
events (music21); empty/malformed decode scores zero, never raises; report an
order-aware pitch fidelity AND a content-only histogram cosine so the "generic
plausible melody" failure (high note-content, wrong order — the 20-bit collapse
analog) is visible. A literal-text copy control bounds what the frozen prior can do.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .channel_metrics import cluster_bootstrap_ci
from .latent_bridge import ABrain, GatherBridge
from .latent_port import MARKER, PortBatcher, greedy_decode, load_receiver, text_positions
from .provenance import provenance
from .text_baseline import levenshtein
from .wide_picture import read_wide, warm_targets_wide

warnings.filterwarnings("ignore")  # music21 is chatty on odd ABC

MUSIC_TEMPLATE = (
    f"A melody just arrived on your latent port as the embedding vectors between the "
    f"brackets. [{MARKER}] Write out the ABC notation exactly."
)


# ----------------------------- corpus -----------------------------
def normalize_abc(abc: str) -> str:
    """Clean an ABC tune but PRESERVE its line structure — ABC is line-oriented and
    music21 only parses header fields (X:/L:/M:/K:) when each is on its own line.
    (Collapsing to one line, like wide_picture does for prose, yields zero notes.)
    Drop blank lines and per-line whitespace; keep the newlines."""
    return "\n".join(ln.strip() for ln in abc.strip().splitlines() if ln.strip())


def load_abc_corpus(brain, window, n_train, n_val, seed, cache_dir="runs/music"):
    """Stream IrishMAN, normalize, keep tunes whose token length fits `window`. Cache
    the held-out split to disk (reproducible, no re-download)."""
    cache = Path(cache_dir) / f"irishman_w{window}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["train"], d["val"]
    from datasets import load_dataset

    ds = load_dataset("sander-wood/irishman", split="train", streaming=True)
    keep, need = [], n_train + n_val
    for ex in ds:
        abc = normalize_abc(ex.get("abc notation", ""))
        if not abc:
            continue
        ids = brain.tokenizer(abc, add_special_tokens=False).input_ids
        if 16 <= len(ids) <= window:            # fits one picture, non-trivial
            keep.append(abc)
        if len(keep) >= need * 3:               # oversample, then seed-shuffle
            break
    rng = np.random.default_rng(seed)
    rng.shuffle(keep)
    train, val = keep[:n_train], keep[n_train:n_train + n_val]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"train": train, "val": val}))
    return train, val


def pick(base, window, rng, n, brain):
    """n random tunes, re-truncated to the token window (defensive)."""
    out = []
    for _ in range(n):
        abc = base[int(rng.integers(0, len(base)))]
        ids = brain.tokenizer(abc, add_special_tokens=False).input_ids[:window]
        out.append(brain.tokenizer.decode(ids, skip_special_tokens=True).strip())
    return [m for m in out if m]


# ----------------------------- musical accuracy -----------------------------
def parse_pitches(abc: str):
    """ABC -> (list of MIDI pitches in order, list of (onset,pitch) events). [] on fail."""
    try:
        from music21 import converter

        s = converter.parse(abc, format="abc")
        ev = [(float(n.offset), int(n.pitch.midi)) for n in s.recurse().notes if n.isNote]
        ev.sort()
        return [p for _, p in ev], ev
    except Exception:
        return [], []


def _pstr(pitches):
    return "".join(chr(min(max(p, 0), 127)) for p in pitches)


def music_fidelity(sent: str, dec: str) -> dict:
    """Order-aware pitch fidelity (primary), content-only histogram cosine, note-F1
    (onset+pitch), and best transpose-aligned pitch fidelity (wrong-key ≠ garbled)."""
    ps, es = parse_pitches(sent)
    pd, ed = parse_pitches(dec)
    if not ps:
        return {}  # unparseable *source* — skip (shouldn't happen on IrishMAN)
    # order-aware: 1 - edit_distance over the pitch sequence
    pitch_fid = max(0.0, 1.0 - levenshtein(_pstr(ps), _pstr(pd)) / max(len(ps), len(pd), 1))
    # content-only: 12-bin pitch-class histogram cosine
    hs = np.bincount([p % 12 for p in ps], minlength=12).astype(float)
    hd = np.bincount([p % 12 for p in pd], minlength=12).astype(float)
    hist_cos = float(hs @ hd / (np.linalg.norm(hs) * np.linalg.norm(hd) + 1e-9)) if pd else 0.0
    # note-F1: a decoded note is right if some sent note shares pitch & onset within 0.25 beat
    tp = 0
    used = set()
    for od, pdi in ed:
        for j, (osi, psi) in enumerate(es):
            if j not in used and psi == pdi and abs(osi - od) <= 0.25:
                tp += 1; used.add(j); break
    prec = tp / len(ed) if ed else 0.0
    rec = tp / len(es) if es else 0.0
    note_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # transpose-aligned: wrong key is perceptually near-perfect, keep it SEPARATE
    tpose = max(
        max(0.0, 1.0 - levenshtein(_pstr(ps), _pstr([p + t for p in pd])) / max(len(ps), len(pd), 1))
        for t in range(-5, 7)
    ) if pd else 0.0
    return {"pitch_fid": pitch_fid, "hist_cos": hist_cos, "note_f1": note_f1,
            "transpose_fid": tpose, "n_sent": len(ps), "n_dec": len(pd)}


def _agg(dicts, key):
    xs = [d[key] for d in dicts if key in d]
    return float(np.mean(xs)) if xs else 0.0


def evaluate_music(model, tokenizer, brain, bridge, batcher, tunes, args, samples):
    bridge.eval()
    rng = np.random.default_rng(args.seed + 7)
    msgs = pick(tunes, args.window, rng, samples, brain)
    scored, char_fids, exact = [], [], 0
    for m in msgs:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            vecs = read_wide(brain, bridge, [m], args.window).squeeze(0).float()
        dec = greedy_decode(model, batcher, vecs, max_new=2 * len(tokenizer(m).input_ids) + 16).strip()
        f = music_fidelity(m, dec)
        if f:
            scored.append(f)
        char_fids.append(max(0.0, 1.0 - levenshtein(m, dec) / max(len(m), len(dec), 1)))
        exact += dec == m
    bridge.train()
    pf = np.array([d["pitch_fid"] for d in scored]) if scored else np.array([0.0])
    return {
        "samples": len(msgs), "n_parsed": len(scored),
        "pitch_fid": cluster_bootstrap_ci(pf, seed=args.seed),
        "note_f1": _agg(scored, "note_f1"),
        "hist_cos": _agg(scored, "hist_cos"),          # content
        "transpose_fid": _agg(scored, "transpose_fid"),
        "char_fidelity": float(np.mean(char_fids)),     # secondary sanity only
        "exact_rate": exact / max(1, len(msgs)),
        "mean_notes": _agg(scored, "n_sent"),
    }


@torch.no_grad()
def copy_control(model, tokenizer, tunes, args, samples):
    """Frozen prior's ceiling: give the ABC as LITERAL TEXT in the prompt and ask it
    to echo. If this is low, the receiver can't read ABC at all and we need a LoRA."""
    rng = np.random.default_rng(args.seed + 99)
    scored = []
    for m in pick(tunes, args.window, rng, samples, ABrainless(tokenizer)):
        prompt = f"Echo this ABC melody exactly:\n{m}\nABC:"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        out = model.generate(ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=2 * len(tokenizer(m).input_ids) + 16,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        dec = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        f = music_fidelity(m, dec)
        if f:
            scored.append(f)
    return {"n": len(scored), "pitch_fid": _agg(scored, "pitch_fid"), "note_f1": _agg(scored, "note_f1")}


class ABrainless:
    """Tokenizer-only shim so pick() works in the copy control (no sender needed)."""
    def __init__(self, tok):
        self.tokenizer = tok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receiver", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--sender", default="Qwen/Qwen3-4B")
    p.add_argument("--slots", type=int, default=32)
    p.add_argument("--window", type=int, default=64)     # ~2 tok/slot
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmstart-steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-samples", type=int, default=96)
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=Path("runs/music_port.json"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer = load_receiver(args.receiver, args.device)
    brain = ABrain(args.sender, args.device)
    d_model = model.get_input_embeddings().weight.shape[1]
    embed_rms = model.get_input_embeddings().weight.detach().float().pow(2).mean().sqrt().item()
    bridge = GatherBridge(brain.width, d_model, args.slots, args.window // args.slots, embed_rms, offset=0).to(args.device).train()
    batcher = PortBatcher(model, tokenizer, args.slots, args.device, message=MUSIC_TEMPLATE)

    train, val = load_abc_corpus(brain, args.window, args.n_train, args.n_val, args.seed)
    print(f"IrishMAN: {len(train)} train / {len(val)} val tunes (<= {args.window} tok)", flush=True)
    cc = copy_control(model, tokenizer, val, args, 48)
    print(f"COPY CONTROL (frozen prior reads literal ABC): pitch_fid={cc['pitch_fid']:.3f} note_f1={cc['note_f1']:.3f} (n={cc['n']})", flush=True)

    bridge.calibrate(*(lambda s, m: (s * m.unsqueeze(-1), m))(*brain.read(
        pick(train, args.window, np.random.default_rng(args.seed), 64, brain), pad_to=args.window)))

    ck = args.output.parent / "ckpt" / f"music_{args.slots}s_{args.window}w.pt"
    warm_ck = args.output.parent / "ckpt" / f"music_{args.slots}s_{args.window}w.warm.pt"
    warm_ck.parent.mkdir(parents=True, exist_ok=True)
    if not ck.exists():
        if warm_ck.exists():
            bridge.load_state_dict(torch.load(warm_ck, map_location=args.device)); print("loaded warm-start", flush=True)
        else:
            warm_opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr)
            for step in range(args.warmstart_steps):
                rng = np.random.default_rng(args.seed * 999_983 + step)
                msgs = pick(train, args.window, rng, args.batch_size, brain)
                vecs = read_wide(brain, bridge, msgs, args.window)
                loss = nn.functional.mse_loss(vecs.float(), warm_targets_wide(model, tokenizer, msgs, args.slots, args.window, args.device))
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
        msgs = pick(train, args.window, rng, args.batch_size, brain)
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
            sc = evaluate_music(model, tokenizer, brain, bridge, batcher, val, args, 24)
            interim.append({"step": step, "pitch_fid": sc["pitch_fid"]["mean"], "note_f1": sc["note_f1"], "hist_cos": sc["hist_cos"]})
            print(f"interim: {json.dumps(interim[-1])}", flush=True)

    final = evaluate_music(model, tokenizer, brain, bridge, batcher, val, args, args.eval_samples)
    result = {"experiment": "LP-6 music port (IrishMAN ABC through frozen receiver)",
              "provenance": provenance(args), "slots": args.slots, "window": args.window,
              "tokens_per_slot": args.window / args.slots, "copy_control": cc,
              "interim_evals": interim, "final": final, "train_seconds": time.monotonic() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    bdir = args.output.parent / "bridges"; bdir.mkdir(parents=True, exist_ok=True)
    torch.save(bridge.state_dict(), bdir / f"music_{args.slots}s_{args.window}w.pt")
    if ck.exists(): ck.unlink()
    if warm_ck.exists(): warm_ck.unlink()
    print("=== LP-6 music verdict ===", flush=True)
    print(f"  copy-control pitch_fid {cc['pitch_fid']:.3f} | port pitch_fid {final['pitch_fid']['mean']:.3f} "
          f"note_f1 {final['note_f1']:.3f} hist_cos {final['hist_cos']:.3f} transpose {final['transpose_fid']:.3f}", flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != "provenance"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
