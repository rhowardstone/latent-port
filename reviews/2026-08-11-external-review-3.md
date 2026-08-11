# External review #3 — post-fix pass (2026-08-11)

Reviewer verified head `a59bdaf` (blend warm start, NW alignment, docs/VISION),
confirmed the substantive fixes landed, and raised four more items — two hygiene,
two consequential conceptual. All addressed.

## Hygiene

| # | Finding | Resolution |
|---|---|---|
| P1 | `--warmstart-eps` affects training but isn't saved; `"warmstart_order":"blend"` wouldn't record ε=.1/.3/1 — the 77.7%-class omission. | Added `visual_encoder/provenance.py`; every experiment JSON (LP-1/2/3b) now embeds `provenance` = {git_sha (+dirty), full `cli_args` via vars(args), argv, package versions, timestamp}. Systemic fix, not a per-field patch. |
| P2 | README inconsistent (31 in layout/reproduce, 35 in prose; actual 38); no GitHub CI statuses. | De-hardcoded test counts in README; added `.github/workflows/tests.yml` (CPU suite on every push) + badge. "tests pass" is now machine-verified. |

## Consequential conceptual corrections

**C1 — the "2000 tokens → ~125 latent steps" arithmetic is not earned.** Correct.
LP-2/3 show *representation compression* of an **already-computed** sequence
(32 A-tokens → 16 B positions), NOT `32 sequential reasoning steps → 16 latent
steps`. Coconut-style feedback removes vocab decoding but doesn't leap over
token-generation steps. Fixed in VISION.md and reframed LP-4 around **temporal
abstraction**: a macro-step latent `h_t → z_{t+Δ}`, step-capacity
`C_step = compute-advanced / forward-passes`, and the future-state distillation
experiment `L = D(h^latent_{t+1}, stopgrad(h^teacher_{t+Δ})) + λL_task` swept over
Δ∈{1,2,4,8,16,32}. This is the honest fork: Δ>1 ⇒ reasoning accelerator; Δ=1 only
⇒ codec.

**C2 — "thought translator" overclaims.** The tap reads the *communication state*,
not a faithful account of all internal computation. Softened to
**latent-protocol translator**; added the causal-intervention test (erase/rotate
the tap-labeled direction, check downstream behavior matches the decoded English).

**New crystallization — canonical latent bus / neural ABI.** Adopted into
VISION.md: O(n) adapters `E_i,D_i` into a shared `Z` instead of O(n²) pairwise
bridges; router on `Z`; structured subspaces
`Z = Z_semantic ⊕ Z_control ⊕ Z_exact ⊕ Z_scratch`; gauge symmetry as the
coordinate-system obstacle, semantic anchors as the resolution.

## Two research programs named for the roadmap

1. Latent temporal abstraction (macro-thought steps) — reframes LP-4.
2. Canonical neural ABI/bus (O(n) adapters, structured/auditable shared space).

Status: P1, P2, C1, C2 done and committed; programs recorded in `docs/VISION.md`.
Still queued: run the binding control (behind LP-5a on the GPU); variational
metric retrofit; causal controls; Fisher experiment.
