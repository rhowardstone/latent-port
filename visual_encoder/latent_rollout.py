"""LP-4: latent macro-steps — can ONE learned step advance generation by Δ>1 tokens?

A frozen LM generates text autoregressively; at each step its last-layer hidden
state h_t predicts the next token. We train a small "macro-step" net g: h_t -> h_{t+Δ}
(Δ steps ahead), then decode the jumped state through the model's OWN frozen head
and ask: does it predict the token the teacher actually reached Δ steps later?

Honest metrics (the reviewer's warning: if states barely change over Δ steps,
"predicting the future" is trivial):
- noop baseline: decode(h_t) predicting token t+Δ (how far ahead the raw state
  already sees) — the bar the macro-step must beat.
- state change: cosine(h_t, h_{t+Δ}) — is the target even non-trivial?
- macro jump accuracy: decode(g(h_t)) == teacher token t+Δ.
- recursive: apply g repeatedly (h_0 -> h_Δ -> h_2Δ ...); does the horizon survive?

If macro >> noop and holds for Δ=4,8 -> real temporal abstraction. If it only works
at Δ=1 -> a codec, not an accelerator. Either way it's a clean result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class MacroStep(nn.Module):
    """Residual map in hidden space: predict h_{t+Δ} from h_t."""

    def __init__(self, d: int, hidden: int = 2048, layers: int = 3) -> None:
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks += [nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d)]
        self.blocks = nn.ModuleList([nn.Sequential(*blocks[i:i + 4]) for i in range(0, len(blocks), 4)])

    def forward(self, h):
        for b in self.blocks:
            h = h + b(h)
        return h


def load_lm(model_id, device):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, tok


def decode_state(m, h):
    """Frozen readout -> logits for the NEXT token given state h. NOTE: Qwen3's
    hidden_states[-1] is ALREADY post-final-norm (verified: lm_head(h) reconstructs
    the model's own logits at 100%), so applying the norm again would double-norm."""
    return m.lm_head(h)


@torch.no_grad()
def teacher_states(m, tok, prompts, gen_len, device, abc=False):
    """Generate greedily, then one forward with hidden states over the generated region.
    Returns (seqs, mean_entropy_nats, abc_valid_fraction) where seqs is a list of
    (H [L,d] last-layer states, toks [L] the generated token ids). Entropy of the
    teacher's own next-token distribution is the H2 control: music must beat text at
    MATCHED entropy, else it's just winning by being low-entropy/repetitive. abc_valid
    (only meaningful when abc=True) is the self-diagnostic — if the model can't even
    generate parseable ABC, the H2 test is vacuous on this model."""
    out, ents, valid = [], [], []
    for p in prompts:
        ids = tok(p, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
        gen = m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=gen_len,
                         do_sample=False, pad_token_id=tok.pad_token_id)
        full = gen[0]
        fwd = m(full.unsqueeze(0), output_hidden_states=True)
        hs = fwd.hidden_states[-1][0]        # [S, d]
        start = ids.shape[1] - 1  # position whose state predicts the 1st generated token
        H = hs[start:-1]                     # states h_t over generated region
        toks = full[ids.shape[1]:]           # the tokens they predict (t+1)
        lp = torch.log_softmax(fwd.logits[0][start:-1].float(), dim=-1)
        ent = float((-(lp.exp() * lp).sum(-1)).mean())      # mean next-token entropy (nats)
        n = min(H.shape[0], toks.shape[0])
        if n >= 2:
            out.append((H[:n].float(), toks[:n])); ents.append(ent)
            if abc:
                from .music_port import parse_pitches
                ps = parse_pitches(tok.decode(full, skip_special_tokens=True))[0]
                # diversity-aware: alphabet-runs (music21 C-fallback) and degenerate loops
                # collapse to 1-2 distinct pitches — real ABC has many. Reject those.
                valid.append(1 if (len(ps) >= 6 and len(set(ps)) >= 4) else 0)
    return (out,
            float(np.mean(ents)) if ents else 0.0,
            float(np.mean(valid)) if valid else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--deltas", default="1,2,4,8")
    ap.add_argument("--gen-len", type=int, default=48)
    ap.add_argument("--train-prompts", type=int, default=300)
    ap.add_argument("--val-prompts", type=int, default=60)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--music", action="store_true",
                    help="H2: teacher-generate ABC melodies (IrishMAN openings) instead of "
                         "wikitext — does the macro-step chain better on the musical manifold?")
    ap.add_argument("--output", type=Path, default=Path("runs/latent_rollout.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    m, tok = load_lm(args.model, args.device)
    d = m.config.hidden_size

    rng = np.random.default_rng(args.seed)
    if args.music:
        from .music_port import ABrainless, load_abc_corpus
        train, val = load_abc_corpus(ABrainless(tok), 96, 4000, 400, args.seed)
        # prompt = header + a few opening tokens so the model continues IN ABC
        def opening(abc):
            return tok.decode(tok(abc, add_special_tokens=False).input_ids[:18], skip_special_tokens=True)
        tr_prompts = [opening(train[i]) for i in rng.choice(len(train), args.train_prompts, replace=(len(train) < args.train_prompts))]
        va_prompts = [opening(val[i]) for i in rng.choice(len(val), args.val_prompts, replace=(len(val) < args.val_prompts))]
    else:
        from datasets import load_dataset
        rows = [" ".join(r.split()) for r in load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"]
                if 60 < len(r) < 240 and not r.startswith("=")]
        tr_prompts = [rows[i] for i in rng.choice(len(rows), args.train_prompts, replace=False)]
        va_prompts = [rows[i] for i in rng.choice(len(rows), args.val_prompts, replace=False)]
    print(f"collecting teacher states ({args.model}, gen_len {args.gen_len}, music={args.music})...", flush=True)
    train_seqs, tr_ent, tr_valid = teacher_states(m, tok, tr_prompts, args.gen_len, args.device, abc=args.music)
    val_seqs, va_ent, va_valid = teacher_states(m, tok, va_prompts, args.gen_len, args.device, abc=args.music)
    print(f"teacher entropy: train {tr_ent:.2f} nats ({tr_ent/0.6931:.2f} bits), "
          f"ABC-valid frac {tr_valid:.2f} (music only)", flush=True)

    results = []
    for delta in [int(x) for x in args.deltas.split(",")]:
        # (h_t, h_{t+Δ}) pairs
        X, Y = [], []
        for H, _ in train_seqs:
            if H.shape[0] > delta:
                X.append(H[:-delta]); Y.append(H[delta:])
        X = torch.cat(X).to(args.device); Y = torch.cat(Y).to(args.device)

        g = MacroStep(d).to(args.device)
        opt = torch.optim.AdamW(g.parameters(), lr=args.lr)
        t0 = time.time()
        for step in range(args.steps):
            idx = torch.randint(0, X.shape[0], (args.batch,), device=args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                pred = g(X[idx])
                loss = nn.functional.mse_loss(pred.float(), Y[idx].float())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % 500 == 0:
                print(f"  Δ={delta} step {step} mse={loss.item():.4f}", flush=True)

        # ---- honest eval on held-out sequences ----
        g.eval()
        noop_jump, macro_jump, noop_next, cos_change, cos_pred = [], [], [], [], []
        rec_macro = []
        with torch.no_grad():
            for H, toks in val_seqs:
                L = H.shape[0]
                if L <= delta:
                    continue
                h = H[:-delta].to(args.device)          # states h_t
                fut = H[delta:].to(args.device)         # true h_{t+Δ}
                # decode(h_t) predicts token at t+1 == toks[t]; decode(h_{t+Δ}) predicts toks[t+Δ]
                gp = g(h)
                dec_macro = decode_state(m, gp.to(m.dtype)).argmax(-1).cpu()   # predicts token t+Δ+... -> compare to toks[t+Δ]
                dec_noop = decode_state(m, h.to(m.dtype)).argmax(-1).cpu()     # predicts toks[t]
                true_next = toks[:L - delta].cpu()      # toks[t]  (what h_t actually predicts)
                true_jump = toks[delta:L].cpu()         # toks[t+Δ] (what h_{t+Δ} predicts)
                macro_jump.append((dec_macro == true_jump).float().mean().item())
                noop_jump.append((dec_noop == true_jump).float().mean().item())
                noop_next.append((dec_noop == true_next).float().mean().item())
                cos_change.append(torch.cosine_similarity(h, fut, dim=-1).mean().item())
                cos_pred.append(torch.cosine_similarity(gp, fut, dim=-1).mean().item())
                # recursive: from h_0, jump ceil(L/Δ) times, decode each; match teacher at those positions
                cur = H[0:1].to(args.device); mt, nt, cn = 0, 0, 0
                pos = delta
                while pos < L:
                    cur = g(cur)
                    pm = decode_state(m, cur.to(m.dtype)).argmax(-1).cpu().item()
                    mt += int(pm == toks[pos].item()); cn += 1
                    pos += delta
                if cn:
                    rec_macro.append(mt / cn)
        results.append({
            "delta": delta,
            "macro_jump_acc": float(np.mean(macro_jump)),
            "noop_jump_acc": float(np.mean(noop_jump)),
            "noop_next_acc": float(np.mean(noop_next)),   # sanity: ~1.0 (greedy), state predicts its own next token
            "state_cosine_h_t_vs_future": float(np.mean(cos_change)),
            "pred_cosine_vs_future": float(np.mean(cos_pred)),
            "recursive_macro_acc": float(np.mean(rec_macro)) if rec_macro else 0.0,
            "train_seconds": time.time() - t0,
        })
        print(json.dumps(results[-1]), flush=True)

    verdict = {
        "experiment": "LP-4 latent macro-steps" + (" (MUSIC / H2)" if args.music else ""),
        "model": args.model,
        "domain": "music-abc" if args.music else "wikitext",
        "teacher_entropy_nats": {"train": tr_ent, "val": va_ent},
        "teacher_entropy_bits": {"train": tr_ent / 0.69314718, "val": va_ent / 0.69314718},
        "abc_valid_fraction": {"train": tr_valid, "val": va_valid},
        "results": results,
        "reading": "macro_jump_acc >> noop_jump_acc and holding for Δ>1 => one step skips real "
                   "computation (temporal abstraction). If macro only beats noop at Δ=1 => codec, not accelerator. "
                   "noop_next_acc should be ~1.0 (greedy sanity).",
        "h2_reading": ("MUSIC vs TEXT: compare recursive_macro_acc to the wikitext run AT MATCHED "
                       "teacher_entropy_bits. Music 'resists drift' only if it beats text at equal entropy AND "
                       "beats its own noop at Δ>=4. If abc_valid_fraction is low, the base model can't generate "
                       "ABC and the H2 test is vacuous — rerun with a music-capable model." if args.music else
                       "text baseline; run with --music for the H2 comparison."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2) + "\n")
    print("=== LP-4 verdict" + (" (MUSIC/H2)" if args.music else "") + " ===", flush=True)
    if args.music:
        print(f"  teacher entropy {va_ent/0.6931:.2f} bits/tok | ABC-valid {va_valid:.2f} "
              f"({'OK' if va_valid > 0.5 else 'LOW — model barely generates ABC, H2 may be vacuous'})", flush=True)
    for r in results:
        print(f"  Δ={r['delta']}: macro {r['macro_jump_acc']:.3f} vs noop {r['noop_jump_acc']:.3f} "
              f"(state cos {r['state_cosine_h_t_vs_future']:.2f}, recursive {r['recursive_macro_acc']:.3f})", flush=True)


if __name__ == "__main__":
    main()
