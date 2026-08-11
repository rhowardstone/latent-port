# Handoff — A100 node status, lanes, and GPU queue

- **From:** A100 node (Claude Code on the box)
- **Status:** active
- **Built against SHA:** current HEAD (see git log)

## What the A100 node is running / will run (do not duplicate)

1. **LP-5a legibility-tax sweep** — RUNNING. λ ∈ {0, 0.3, 1, 3}, clean (padding
   masked). 1/4 λ-artifacts written (`runs/lp5a_lambda0.0.json`).
2. **Binding control** — QUEUED next (fires when LP-5a frees the GPU). Runs
   `latent_port.py --warmstart-order {mean,role,blend} --warmstart-eps {0.1,0.3,1}`
   at 15/20/30 bits/slot, with `content_vs_order` (indel-aligned) reporting. Tests
   whether the transposition failure is intrinsic or a mean-warm-start artifact.
3. **LP-3b bidirectional self-port** — QUEUED (`bidirectional.py`, one shared 4B
   bridge both directions). Note: currently a self-port reconstruction map; the
   two-context A→B→A demo is the actual bidirectional proof (separate task).
4. Then: Fisher-spectrum on the real receiver; LP-4 macro-step real run.

## Lanes handed to CPU nodes (yours — see COORDINATION.md)

- **CPU node B:** own `visual_encoder/channel_metrics.py` — the shared metrics
  library. Include `I_var` (LP-1, `H(X)=5C`), `ΔI` paired-gain (LP-2, free text),
  message-cluster bootstrap CIs, and a synthetic known-channel oracle
  (`runs/cpu/channel_metrics_reference.json`). Plus the binding-metric property
  tests. Relay patches; A100 node integrates + pushes.
- **CPU node A:** own `notebooks/` + the Colab harness + LP-4 macro-step design.
  Your notebook should CALL `channel_metrics.py`, not re-implement the rate. You
  have GitHub push (user-authorized) — land it under `notebooks/`.

## Integrated from CPU node A this handoff

- The **free-text entropy caveat** (variational rate `H(X)+E log q` is valid only
  for LP-1's known-entropy source; use `ΔI` paired-gain for LP-2) — folded into
  `OPEN-PROBLEMS.md` A2 and `docs/EXPERIMENT-PROTOCOL.md`. Credit: CPU node A.

## Next for whoever picks up

- Once `channel_metrics.py` lands, the A100 node retrofits LP-1/2/3 result
  emission to call it, and re-reports `R_var`/`ΔI` alongside goodput.
