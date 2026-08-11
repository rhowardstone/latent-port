# Multi-agent coordination protocol

This repo is worked by several agents at once (A100 GPU node + CPU nodes + external
reviewers). This file keeps us in separate lanes and defines how work hands off.
**Read this and the open `handoffs/` entries before starting anything.**

Current integration HEAD: see `git rev-parse HEAD`. If you built against an older
SHA, say so in your handoff and the A100 node reconciles.

## Nodes and capabilities (as of 2026-08-11)

| Node | Compute | Push route | Best at |
|---|---|---|---|
| **A100 node** (Claude Code on the box) | A100-40GB + full repo push | direct push | GPU training/eval; integrating others' patches; merge point |
| **CPU node A** | ~3 vCPU/17GB, Drive + GitHub write | direct push (user-authorized) | notebooks, Colab harness, plots, LP-4 design |
| **CPU node B** | ~5 vCPU/6GB, no git remote | relay → A100 integrates | reference libraries, property tests, synthetic oracles |
| Reviewers | — | via `reviews/` | line-level review, math agenda |

**Key topology fact:** the A100 node runs GPU jobs *directly and reliably* — no
12-hour Colab eviction. So for anything that fits the A100, the loop is: CPU node
designs + CPU-validates → commits/relays → **A100 node runs it here** → commits
`runs/` artifacts → CPU node analyzes. Colab is *overflow* capacity for the CPU
nodes' notebooks, not the primary GPU path.

## Lane assignments (avoid collision — both CPU agents proposed variational-rate)

| Lane | Owner | Deliverable |
|---|---|---|
| GPU experiments: binding control, LP-5a, LP-3b, Fisher-spectrum, LP-4 real | **A100 node** | `runs/` artifacts + integration |
| Shared metrics **library** `visual_encoder/channel_metrics.py` (`I_var` for LP-1, `ΔI` paired-gain for LP-2, bootstrap CIs, synthetic known-channel oracle) + `tests/test_channel_metrics.py` | **CPU node B** | tested module + `runs/cpu/*_reference.json` |
| `notebooks/` + Colab job harness + LP-4 macro-step experiment design/spec | **CPU node A** | notebooks that CALL the library |
| `role_geometry.py` lab, wiretap-faithfulness harness, canonical-bus toy (synthetic first) | **CPU node B** | modules + notebooks |
| Fisher-spectrum estimation code (JVP/Lanczos) | A100 node (needs real receiver) | after binding/LP-5a |

**Rules.** (1) One owner per file. (2) Notebooks *orchestrate* tested library
functions — no load-bearing logic hidden in notebook state. (3) The variational
rate lives in ONE library module; the notebook validates it, does not re-implement
it. (4) Every experiment JSON embeds `provenance(args)` (git SHA + full args +
versions) — see `visual_encoder/provenance.py`.

## Handoff mechanism

Claim a lane and pass work via `handoffs/`:
- One markdown per handoff: `handoffs/YYYY-MM-DD-<node>-<topic>.md`.
- Fields: **status** (proposed / in-progress / blocked / done), **lane**, **built
  against SHA**, **deliverable + paths**, **where results land**, **what the next
  node should do**.
- Before starting, scan open handoffs so you don't duplicate a claimed lane.
- The A100 node integrates relayed patches from no-push nodes and updates the
  handoff to `done`.

## GPU job standard

All GPU jobs (here or Colab) follow `docs/EXPERIMENT-PROTOCOL.md`: resumable,
checkpoint-after-each-condition, deterministic run dirs, full manifests. This makes
a Colab eviction lose at most one condition and lets any node interpret a result
without reconstructing notebook state.
