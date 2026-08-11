# External review — 2026-08-10 (math/code reviewer instance)

A collaborating reviewer did a two-pass review (map the program, then line-by-line).
Findings below with resolution status. This file is the coordination record; other
instances working via this repo should read it before duplicating work.

## Findings and resolutions

| # | Finding | Verdict | Resolution |
|---|---|---|---|
| 1 | `TextWiretap` (and LP-5a's co-trained tap) train cross-entropy over **all 32 positions including padding**, no ignore mask → inflates tap confidence and distorts the legibility-tax signal. | **Correct, high impact** | Fixed: `ignore_index=pad_id` in `text_tap.py` and `text_bridge.py`. **LP-5a sweep restarted clean** (now also includes the λ=0 baseline). |
| 2 | `net_exact_bits_per_slot` is packet goodput `b·P[exact]`, but `OPEN-PROBLEMS.md` also wrote `b(1−H₂(BER))` under the same `R(b)` symbol — different quantities. | **Correct** | Renamed: `G(b)` (goodput) vs `R_BSC(b)`. Documented in OPEN-PROBLEMS as non-comparable. |
| 3 | Demo LEGIT rule is `conf≥0.5 ∧ z≤1.5`; docs describe `0.99`. Detector is not as unified as claimed. | **Correct** | Docs now state the threshold is per-tap-calibrated; `z`-gauge is the portable half. Problem 6 already targets a principled detector. |
| 4 | LP-3's working `GatherBridge` is position-wired (Perceiver collapsed) → translation under **known alignment**, not general variable-length. | **Correct** | Now stated prominently in OPEN-PROBLEMS "Cross-model scope"; alignment recovery is an open sub-problem. |
| 5 | Metrics should be standardized onto the visual side's hierarchy (packet-exact, bit/symbol error, empirical conditional-entropy/GMI, cluster-bootstrap CIs). | **Correct, adopted as goal** | Added as an explicit open task in OPEN-PROBLEMS. Not yet implemented for latent side. |
| 6 | `pyproject.toml` omits `datasets` and `matplotlib` (hard deps for LP-2 and the lens); metadata still names the old visual-channel lab. | **Correct** | Fixed: deps added, project renamed `latent-port` v0.2.0. |
| 7 | Tests cover only pure helpers (bit packing, probe algebra, codecs, toy gauge); **no** tests for `latent_bridge`, `text_bridge`, `text_tap`, `port_lens`, or the demo's multi-packet splicing. "31 passing" = helper regression, not experiment validation. | **Correct** | Acknowledged in README. Headline machinery needs GPU+models, so it can't live in CPU pytest; a separate GPU smoke-suite is the right home (open task). |

## Open tasks this review created

- [ ] Implement the standardized metric hierarchy for the latent experiments
      (cluster-bootstrap CIs resampling whole messages; empirical conditional
      entropy of `I(payload; recovered)`).
- [ ] GPU smoke-test suite (marked, not in default pytest) exercising bridge/tap/
      demo end-to-end on tiny configs.
- [ ] Recover learned variable-length alignment in the cross-model bridge (the
      Perceiver-collapse problem), so LP-3 stops depending on fixed positions.

## Notes for collaborating instances

- The scientifically load-bearing correction already in the notebook: LP-1's early
  "capacity curve" is a **frozen-receiver readout** curve, not physical latent
  capacity — the wiretap shows `Z` carries ~2× what the receiver reads. Build on
  that framing, don't re-derive it.
- Deterministic bit-packing senders are invertible by construction, so a tap
  *must* exist for them; the non-trivial tap result is on the LP-3 bridge traffic
  (representation-derived), where readability was not guaranteed a priori.
