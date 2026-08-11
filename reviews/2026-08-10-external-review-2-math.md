# External review #2 — deep math/code pass (2026-08-10)

A second, much deeper two-pass review (repository framing, then every line of the
core scientific code + artifacts). It is in several places sharper than the
original work. Summary + resolution; the full mathematical agenda it produced is
folded into `OPEN-PROBLEMS.md` (Addendum A0–A11).

## Correctness / integrity findings

| # | Finding | Verdict | Resolution |
|---|---|---|---|
| M1 | **77.7% free-text result is not artifact-backed** — committed `text_bridge.json` is 71.6/72.4% (wikitext), 77.7% came from an uncommitted chat-register battery that didn't record the sender model. | **Correct, integrity** | 77.7% **withdrawn** in NOTEBOOK pending a self-describing committed battery. Only 71.6/72.4% wikitext stands. |
| M2 | **Binding result has a curriculum confound**: `warmstart_targets(order="mean")` is permutation-invariant, erasing order before CE. Transposition failure partly manufactured. | **Correct — best catch** | Implemented `--warmstart-order {mean,role,none}` + `content_vs_order`/Damerau-Levenshtein metrics + unit tests. Control run queued (top of A100 queue). |
| M3 | **`C_phys ≥ 20 bits/slot` near-trivial** for deterministic injective sender (`I(X;Z)=N`). | **Correct** | OPEN-PROBLEMS A1: replaced with a defined quantized/noisy channel; the *interesting* result is the small tap. |
| M4 | **`net_exact_bits_per_slot` (goodput) vs `b(1−H₂(BER))` share `R(b)`.** Use a variational/GMI rate `N+E[log₂ q(X|Z)]` instead. | **Correct** | OPEN-PROBLEMS A2: adopt `R^var` as primary; `G(b)` demoted to operational goodput. Retrofit queued. |
| M5 | **`wiretap_bridge.json` reports LP-1's receiver score (71.875%)** for the bridge (should be 75%) — `wiretap.py` pulls from `latent_port.json` in bridge mode. | **Correct, bug** | Fixed: receiver scores only attached in sender mode. |
| M6 | **Tap CE trains on padding/EOS** (no mask); LP-5a inherits it; likely depresses the ~52.7% tap and contaminates the λ curve. | **Correct** | Fixed (`ignore_index`) and **LP-5a rerun clean** (now with λ=0 baseline). |
| M7 | **Detector threshold**: docs say conf<0.99, demo uses 0.5. | **Correct** | Documented as per-tap-calibrated; A6 proposes conformal p-value for a real FPR guarantee. |
| M8 | **LP-3 = translation under known alignment** (GatherBridge slices fixed positions; Perceiver collapsed). | **Correct** | Stated in OPEN-PROBLEMS "Cross-model scope" + A8 (CCA/Procrustes → learned alignment). |
| M9 | **Missing deps** (`datasets`, `matplotlib`); stale project metadata. | **Correct** | Fixed (v0.2.0, `latent-port`). |
| M10 | **Tests cover helpers, not headline machinery**; no CI. | **Correct** | README states scope; GPU smoke-suite + CI are open tasks (A11). |

## Mathematical upgrades adopted (see OPEN-PROBLEMS Addendum)

- **A2 variational/GMI receiver rate** — same units for tap and receiver, from CE we already compute.
- **A3 pullback-Fisher geometry** `G = Jᵀ(diag p − ppᵀ)J` as the explanation for the tap/receiver gap, with JVP/Lanczos estimation and a Fisher-separation code loss. (Reviewer's top pick; highest ceiling.)
- **A5 causal controls** — deranged/zero/moment-matched cache + embed-vs-hidden-state ablation (mandatory for the mind-to-mind claim).
- **A6 auditability as bilevel/adversarial game** + conformal detector calibration.
- **A7 information-bottleneck / rate-distortion** with surface vs task distortion + a real structured-fact task battery.
- **A9 LP-4 as dynamical system** (Lyapunov/spectral-radius stability).
- **A10** Problem 5 corrected to identifiability under restricted queries + gauge symmetry (not "white-box ⇒ translator").

## Highest-value directions (reviewer's judgement, adopted)

1. Pullback-Fisher geometry of the frozen latent channel (A3).
2. Auditability/covert capacity as a constrained bilevel game (A6).
3. Order-preserving warm-start binding control (A4) — run **first**, cheapest, may change the story overnight.

## Status of resolutions as of this commit

Done: M1, M3, M4, M5, M6, M7, M8, M9; M2 implemented (run pending GPU); agenda
folded into OPEN-PROBLEMS. Open: run the binding control; variational-metric
retrofit; causal controls; Fisher experiment; self-describing runs + CI;
publication-grade task battery.
