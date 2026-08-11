# Open Problems — a mathematical agenda for the latent port

This document restates the project's open questions as math, with the measured
constants attached, so a collaborator can pick a problem and start. Everything
is sized for a single A100-40GB. Notation and measured values are collected at
the end.

The one-paragraph orientation: we inject `k` learned vectors
`Z = (z_1,…,z_k), z_i ∈ ℝ^d` into a frozen decoder `B`'s input stream, one
sequence position each. A **sender** produces `Z` (from bits, from another
model's hidden states, or from text); a **receiver** is `B`, frozen; a
**wiretap** `T` decodes `Z` without querying `B`. Everything below is about the
maps between these objects and the capacities of the channels they define.

---

## Problem 1 — The frozen-readout gap (why 20 ≫ 10 ≫ what B reads)

**Measured.** The same vectors from which an independent tap recovers ~20 bits
are read by the frozen `B` at ~10 bits/position; at the 20-bits/slot load the
tap gets 100% exact and `B` gets 4.7%. So there are (at least) three distinct
capacities of the *same* `Z`:

- `C_phys(Z)` — mutual information `I(payload; Z)` the vectors physically carry;
- `C_tap` — what a small trained `T: ℝ^{k×d} → payload` extracts;
- `C_read` — what the frozen `B` surfaces at its output (generative or probed).

Empirically `C_phys ≥ C_tap ≫ C_read`.

**Question.** Characterize `C_read` as a function of `B`. Concretely: `B` maps
injected `Z` through frozen attention/MLP blocks; the readout is a linear head on
the residual stream at the query position. Is the `C_tap → C_read` collapse
explained by (a) the injected `z_i` lying off the *typical set* of `B`'s input
embeddings (a covariance/whitening mismatch), (b) attention dilution across `k`
positions, or (c) a genuine rank bottleneck in how a single position's
information reaches the answer position?

**Why it's tractable here.** `B` is frozen and differentiable; you can measure
`I(payload; h_ℓ)` at every layer `ℓ` via the existing `llm_probe.py` methodology,
and you can compute the Jacobian `∂ (answer logits) / ∂ z_i` directly. A clean
result would be a bound `C_read ≤ f(σ_min(J), k, ...)` predicting the gap from
frozen-model spectra. **A100 use:** Jacobian/Fisher spectra over a few hundred
prompts; layerwise MI probes (already cached-feature-friendly).

---

## Problem 2 — Optimal slot count `k` and the density frontier

**Measured.** Net exact rate `R(b) = b·(1−H₂(BER(b)))` peaks at `b = 10`
bits/slot (`k=16`, 2 chars/slot); `b=15` still climbing at budget; `b=20`
collapses on exact-match while char accuracy holds 93%; `b=30` collapses
(also a curriculum artifact). See `runs/latent_port.json`.

**Question.** For fixed total payload `N` bits into a frozen `B`, choose `k` and
the per-slot load to maximize *operational* rate. There is a tension:
- large `k` → each vector carries little, easy to read, but costs positions;
- small `k` → dense vectors, but `C_read` per vector saturates.

Model `C_read(k)` and optimize `k`. Is there a variational/rate-distortion
statement: minimize positions `k` subject to `I(payload; \hat{payload}) ≥ (1−ε)N`,
where the channel `Z → \hat{payload}` factors through frozen `B`? The measured
whisper regime (meaning preserved, spelling lost) suggests a **distortion**
formulation `D(payload, \hat{payload})` that weights semantic vs surface errors
differently — what is the right `D`, and what is `R(D)` for this channel?

**A100 use:** sweep `k ∈ {4,8,16,32}` with variable-slot bridges; fit `C_read(k)`.

---

## Problem 3 — The binding/order failure (the cleanest empirical signal)

**Measured.** At every *working* density, content stays per-slot (a lone vector
solo-decodes as its own characters), but errors are dominated by **order** —
within-slot transpositions (`CED→CDE`, `JJVH→JHVJ`) and, at higher load,
cross-slot ordering. See the `runs/lens/` figures and notebook LP-1 B3/B4.

**Question.** This looks exactly like the classic *binding problem*: a set of
vectors preserves "what" but loses "where/when." Formalize the port as
transmitting a sequence `(c_1,…,c_n)` and ask when the frozen `B` can recover the
*permutation* as well as the multiset. Candidate framework: positional binding
via structured phase/rotation (think RoPE-compatible binding, or
Smolensky-style tensor-product / holographic reduced representations — circular
convolution binding `z = Σ_i role_i ⊛ filler_i`). Does adding an explicit,
learned positional "role" basis to the slot vectors convert the order errors to
zero at fixed content capacity? This is a concrete, testable HRR question with a
frozen decoder as the referee.

**A100 use:** small — retrain the ~9M-param bridge with a role-binding
parameterization; measure transposition rate vs the current MLP bridge.

---

## Problem 4 — The legibility tax `R(λ)` (running now: LP-5a)

**Setup.** Train the bridge with total loss
`L = L_B(payload | Z) + λ · L_T(payload | Z)`, where `L_T` is a *co-trained
reference tap*'s cross-entropy. `λ=0` is the free-for-all channel; `λ→∞` forces
traffic to be tap-decodable.

**Question.** Characterize the Pareto frontier
`{ (C_read(λ), C_tap(λ)) : λ ≥ 0 }` — the price, in receiver capacity, of
guaranteed auditability. Is it smooth and cheap (a few %), or is there a phase
transition where forcing legibility craters density? This is the number the
"will the world keep AI-to-AI traffic legible?" debate actually turns on, and it
has never (to our knowledge) been measured. `runs/lp5a_lambda*.json` will hold
the first three points `λ ∈ {0.3, 1, 3}` against the `λ=0` baseline.

**Deeper version.** Give the sender an adversarial incentive:
`L = L_B − μ·L_T` (be readable to `B`, *un*readable to the reference tap) and ask
whether a *fresh* tap family, trained post-hoc, still cracks it. This is a
min–max / steganography game; the equilibrium capacity of a covert channel that
must remain `B`-usable is the object of interest.

---

## Problem 5 — The universal translator: when does a post-hoc tap exist?

**Setup.** Two models co-train a private code with *pure task loss* (no text
supervision, no legibility term): sender `S_θ`, receiver-adapter `A_φ` on `B`,
optimize only downstream task success. A genuinely alien protocol emerges.

**Question.** When can a third party build `T` mapping this code back to language
using only (a) black-box behavior of `B` on the code, or (b) white-box access to
`B`'s weights, but *no* paired (code, meaning) data? Conjecture from our results:
**white-box endpoint access ⇒ a translator exists**, because the code is
optimized to be `B`-decodable and `B`'s own computation exposes the semantics
(probe `B`'s states, à la `llm_probe`). **Black-box + cooperatively obfuscated
(shared secret) ⇒ no translator** (it reduces to breaking encryption). Make this
precise: state the access model under which translation is polynomially learnable
vs information-theoretically impossible. Even a clean statement + toy proof for
linear `B` would be publishable.

**A100 use:** LP-5b — evolve the code (task-only), confirm existing taps fail
(measure z-gauge separation), then attempt translation from behavior alone.

---

## Problem 6 — Manifold gauges: guarantees, not just heuristics

**Measured.** A per-slot diagonal Gaussian on legitimate traffic gives
`z = ‖(z − μ)/σ‖_rms`. In-distribution `z ≈ 1.00`; drifted protocol (same content,
different sender) `z ≈ 1.6`; jamming noise `z ≈ 86–89`. Confidence (mean
max-softmax of the tap) drops 0.99 → 0.44–0.89 off-distribution. The rule
`conf < 0.99 or z > 1.5` separated every tested case — but it also **false-fired
on assistant-refusal register** (z≈59 on legitimate-but-out-of-training-register
traffic).

**Question.** Put a guarantee under the detector. The channel is a distribution
`P_legit` over `ℝ^{k×d}`; a covert or drifted message is a shift to `Q`. Given a
false-positive budget `α`, what detector achieves the best power against the
class of shifts a *capable* adversary can produce while remaining `B`-usable? The
adversary is constrained (`Q` must still decode correctly through `B`), which
bounds how far from `P_legit` it can move — quantify that constraint. This ties
Problems 4–6 together: the spare capacity `C_phys − C_read` is exactly the room a
covert channel has, and the detector's job is to police it.

**A100 use:** small; mostly a statistics/optimization problem on cached traffic.

---

## Problem 7 — Latent rollout (LP-4): the output-token elimination

**Setup.** Today the sender still *generates text*; only the receiver saves
positions. The rollout replaces token generation with vector emission: train a
"vector head" `g` on the sender so it emits `z_1, z_2, …` autoregressively, each
`z_i` decodable (by `B`, and by the tap) into ~`τ` tokens' worth of content.

**Question.** Two coupled objects: (i) the amortized rate — expected content per
emission step — and how it trades against fidelity; (ii) stability of an
autoregressive *continuous* chain (no discretization to snap back to a manifold —
the known drift failure mode of latent CoT, cf. CALM's brittleness). Is there a
contraction/redundancy condition on the emission map that keeps a length-`m`
latent rollout from accumulating error? The tap is the natural
teacher-forcing signal for training `g` (it already reads these vectors), which
makes this immediately buildable here.

**A100 use:** medium — train the emission head against the frozen tap + `B`.

---

## Measured constants (all from `runs/`, seeds fixed)

| symbol | meaning | value |
| --- | --- | --- |
| `d` | receiver hidden width (Qwen3-VL-2B) | 2048 |
| `k` | slots per message (default) | 16 |
| `R*` | peak net exact bits/position (bit-packed) | 7.19 @ 10 bits/slot |
| `C_read` (20b) | frozen-receiver exact recovery at 20 bits/slot | 4.7% |
| `C_tap` (20b) | independent tap exact recovery, same traffic | 100% |
| cross-model exact | 0.6B→2B bridge, 10 bits/slot | 75% |
| free-text fidelity | char accuracy, chat register (4B sender) | 77.7% |
| `z_legit` | on-manifold gauge, in-distribution | ≈ 1.00 |
| `z_drift` | same content, different sender | ≈ 1.6 |
| `z_jam` | Gaussian noise | ≈ 86–89 |
| text baseline | random Base32 bits per text token | ≈ 7.2 |
| bridge params | perceiver/gather bridge | ~9–10 M |
| tap params | independent wiretap | ~2–8 M |

Definitions and a metric-naming correction (thanks to external review, 2026-08-10).
Two distinct quantities were previously both written `R(b)`; they are different and
now kept separate:

- **Packet goodput** `G(b) = b · P[whole packet decoded exactly]`. This is what the
  JSON field `net_exact_bits_per_slot` reports (LP-1/LP-2/LP-3). All-or-nothing per
  message. The "7.19 net exact bits/position" headline is a `G` value.
- **BSC-equivalent rate** `R_BSC(b) = b · (1 − H₂(BER(b)))`, `BER` = per-bit error
  rate. This is the visual-probe side's metric (`bsc_equivalent_rate` in
  `qwen_probe.py`) and is explicitly *only* a binary-symmetric-channel equivalent —
  **not** a certified Shannon capacity, since errors within one message are
  correlated. `G` and `R_BSC` are not comparable point-for-point.

`H₂` = binary entropy. `I(·;·)` = mutual information. A "position/slot" is one KV
cache entry, the same resource a text token consumes.

**Metric standardization (open task, from review).** The latent experiments should
report the same hierarchy the visual side already uses: packet-exact rate, per-bit/
per-symbol error, an empirical conditional-entropy / generalized-MI estimate of
`I(payload; \hat{payload})`, and **cluster-bootstrap CIs** (resampling whole
messages, since intra-message errors are correlated). Until then, treat single
point estimates here as indicative, not certified.

**Detector honesty (from review).** The trust gauge is **not** a single universal
`conf ≥ 0.99` rule. The bit-packed tap calibrates near `conf ≥ 0.99 ∧ z ≤ 1.5`; the
lower-confidence free-text tap uses `conf ≥ 0.5 ∧ z ≤ 1.5` in the demo. The
threshold is **per-tap-calibrated**; the `z`-gauge (on-manifold Mahalanobis-rms) is
the more portable half. Problem 6 is exactly about replacing these hand-set
thresholds with a false-positive-controlled detector.

**Cross-model scope (from review).** LP-3's working bridge is `GatherBridge`, which
reads A's states at **known, fixed token positions** (the learned-alignment Perceiver
collapsed to a constant — see notebook). So LP-3a demonstrates representation
translation **under known alignment**, not yet a general variable-length
mind-to-mind interface. Recovering learned alignment is an open sub-problem (its
failure mode — attention-sink collapse on low-signal targets — is documented).

---

# Addendum — mathematical corrections and upgrades (external review, 2026-08-10)

A reviewer did a two-pass line-level review and supplied the following; these
supersede the looser framing above where they conflict. Credit to that review.

## A0. Name the cost axes; we measure context/KV compression, not bandwidth

Cost is a vector, not a scalar:
`(receiver positions, transport bits, sender FLOPs, receiver FLOPs, latency,
fidelity, auditability)`. A latent slot and a token both cost the receiver one
KV position — so our *context-position* compression comparison is valid — but a
2048-d bf16 vector is **4096 bytes on the wire**, not one integer token id. We
are demonstrating **context/KV compression**, not bandwidth compression. Use
those names. North-star question: *how much task-relevant, independently
auditable information crosses between frozen models per unit of real system
cost?*

## A1. `C_phys` as stated is near-trivial — replace it with a defined channel

For a deterministic sender `Z = S_θ(X)` injective over the message set,
`I(X;Z) = H(X) = N` exactly (in exact arithmetic). So "≥20 bits/slot physically
survive" is almost content-free; the *interesting* result is that a **small**
learnable tap recovers it. To get a real physical-capacity/robustness curve,
define a nontrivial channel: `Z̃ = Q_q(Z + η)`, `Q_q` = 8/4/2-bit quantization,
`η ~ N(0, σ²Σ)`, plus low-rank projection. Then sweep precision/noise/rank and
measure survival. This also starts answering whether latent transport could ever
win on true bandwidth (Problem A0).

## A2. The right receiver metric is a variational / mismatched-decoding rate

This is a **mismatched-decoding** problem: encoder trainable, decoder `B` frozen
and suboptimal for the code. Stop reporting `G(b) = N·P(exact)/k` as a capacity.
Since `H(X)=N` for random payloads, and we already compute teacher-forced CE,
report the **variational lower bound** on decoder-recoverable information:

`I_B^var = N + E[log₂ q_B(X | Z)]`,  `R_B^var = I_B^var / k`,

and identically `R_T^var` for the tap `q_T`. Now the frozen-readout gap is in one
unit, uses all probability mass (not a brittle greedy exact-match endpoint), and
does not crash from 6→0.9 bits because one pair of characters transposed. Adopt
`R^var` as the project's primary rate; keep `G(b)` only as an operational
goodput. (GMI / LM-rate are the standard tools here.)

## A3. Problem 1's framework: pullback-Fisher geometry (not σ_min of J)

`σ_min(J)`, `J = ∂logits/∂Z`, is the wrong summary — `Z` has ~3·10⁴ dims and huge
nullspaces. Use the receiver's **pullback Fisher metric**
`G(z) = J(z)ᵀ (diag(p) − ppᵀ) J(z)` (sum tokenwise for a sequence), so
`KL(p_B(·|z) ‖ p_B(·|z+δ)) ≈ ½ δᵀG(z)δ`. Then the tap/receiver gap has a crisp
statement: **the sender stores information in directions that separate messages
geometrically but have tiny Fisher length under `B`; the tap learns them, `B`
barely reacts.** A100-friendly: never build the 3·10⁴² matrix — use JVP/VJP with
stochastic Lanczos / randomized SVD for the leading spectrum. Measure effective
dimension `r_eff = (tr G)² / tr(G²)` and eigenvalue decay at 5/10/15/20/30
bits/slot, and test whether codeword differences `Δz` increasingly live in
low-`G` directions as density rises. **Then improve the code**: add a
Fisher-separation term `L = L_B − α·E[(z_i−z_j)ᵀ G (z_i−z_j)]` — a code designed
around the receiver's functional geometry, not Euclidean embedding geometry.
(This is the highest-ceiling direction.)

## A4. The binding result has a curriculum confound — run the control first

`warmstart_targets(order="mean")` is `z⁽⁰⁾ = (1/m) Σ_j e(c_j)`, **permutation-
invariant**: `z⁽⁰⁾(AB)=z⁽⁰⁾(BA)`. So the curriculum erases order before CE, and
the transposition-heavy failure is partly manufactured — worse as `m` grows,
exactly where it appears. Controls (now implemented, `--warmstart-order`):
`mean` (confound) vs `role` (order-preserving, `(1/√m)Σ_j R_j e(c_j)`) vs `none`,
crossed with staged `1→2→4` chars/slot, at 15/20/30 bits/slot. Score with the
new `content_vs_order` (unordered/multiset accuracy via Damerau–Levenshtein
separated from ordering). **Only if unordered stays ~100% while permutation
collapses under role-preserving init is the binding bottleneck real.** Run this
before invoking HRR/tensor-product names. *This can change the empirical story
overnight and is the recommended first A100 job.*

## A5. Mandatory causal controls (esp. LP-2/LP-3)

Per the causal-audit result (arXiv:2608.04893): a large effect from injecting a
cache does **not** prove example-specific information crossed — B may hallucinate
from its language prior (and our own "B beats the tap on natural text" is
evidence of exactly that prior). For every LP-2/3 eval run the ablation set
`{Z_correct, Z_deranged, Z=0, Z_moment-matched-random}` and require
`Z_correct ≫ Z_deranged`. Plus the killer: **A input embeddings vs A final hidden
states** at equal bridge capacity — if hidden states win, we're transmitting
something the sender *computed*, not a fancy transcoder. This is what upgrades
"mind-to-mind" from slogan to claim.

## A6. LP-5 as a bilevel/adversarial game; detector via conformal calibration

LP-5a (joint bridge + reference tap) measures a *cooperative* auditable protocol.
The hard question is inherent auditability: `min_S L_B(S) s.t. min_{T∈𝒯} L_T(S) ≤ ε`
(inner = train a fresh auditor to convergence). Adversarial version
`min_S [L_B − μ·min_{T∈𝒯} L_T]` = can `S` hide from the best affordable auditor
while staying `B`-useful (wiretap-channel territory). For the detector, keep any
score (diagonal-z, low-rank Mahalanobis, tap entropy) but convert to a
**conformal p-value** on a held-out legit calibration set → honest
`P_legit(false alarm) ≤ α`. Real test isn't Gaussian jamming (gauges demolish it)
but `δ* = argmax covert-info(z+δ) s.t. B(z+δ)=B(z), detector ≤ τ`. Also:
`C_phys − C_read` is **not** "exactly" covert room — unused info need not be
independently controllable; the right object is `sup I(M_covert; Z | Y_B)` under
a utility + detectability constraint.

## A7. LP-2 wants information-bottleneck / rate-distortion, and a real task battery

Exact transcription is the wrong objective for inter-model comms. With `R = k`
positions and two distortions `D_surface(x,x̂)`, `D_task(y,ŷ)`, map the frontier
`R(D_surface, D_task)` — this makes the "whisper regime" (surface distortion up,
task distortion low) mathematically real. Needs a real battery, not 3/4 and 2/6
anecdotes: random key-value stores, synthetic appointments, graph edges, random
access codes, multi-hop relations (priors can't rescue B), then natural QA.
Note our current LP-2 is only **1.45× context-position compression**; Gist Tokens
reports up to 26× in its setting, so soften/source the "historically 4–10×" line.

## A8. Cross-model alignment: CCA/Procrustes before another Perceiver (LP-3b)

Estimate the A↔B geometry explicitly. With paired positions
`H_A ∈ ℝ^{n×d_A}, H_B ∈ ℝ^{n×d_B}`, try whitened CCA and orthogonal/low-rank
Procrustes as initialization (Procrustes preserves internal geometry; CCA
maximizes cross-model agreement), then a learned monotone/optimal-transport
alignment. The real question isn't "MLP maps A-pos-17→B-slot-6" (proven) but
"discover which portions of A's variable-length state become each B slot" — the
general port.

## A9. LP-4 is a dynamical-systems stability problem

For recursive latent emission `z_{t+1} = g_θ(h_t, z_t)`, accuracy isn't the crux —
stability is. Study `J_t = ∂z_{t+1}/∂z_t`; products `J_T···J_1` growing ⇒ errors
explode, contracting too hard ⇒ information vanishes. Track spectral radius /
Lyapunov exponents; train for the middle regime via spectral normalization,
periodic discrete anchors, or an error-correcting refresh every `m` steps.

## A10. Fix Problem 5's statement (identifiability, not "white-box ⇒ translator")

"White-box access ⇒ translator exists" is trivial (just run `B` as the decoder)
or ill-defined. Latent codes have **gauge symmetries**: an invertible map of the
code plus its inverse in the receiver leaves behavior unchanged, so without
external semantic anchors there may be no uniquely identifiable "meaning." The
real theorem is about **translator complexity and identifiability under
restricted observations/queries**, not a blanket universal-translator conjecture.

## A11. Provenance & reproducibility hygiene (blocking for publication-grade)

Every run must be self-describing: git SHA, full CLI, model+tokenizer revisions,
package versions, dataset revision/hash, GPU, all seeds, and **save every
held-out prediction**, not three examples. (The withdrawn 77.7% is exactly what
this prevents.) Add CI so "tests pass" is GitHub-demonstrated, and report ≥3–5
training seeds around any critical point with cluster-bootstrap CIs (the
visual-probe code already has this discipline; the latent side must match it).

## Reprioritized A100 queue (review's order)

1. **Binding control** (A4) — implemented; run at 15/20/30 bits/slot × {mean,role,none}. Cheap, may rewrite the story.
2. Fixes (A11 + the committed bug-fixes) — mostly done; finish self-describing runs + CI.
3. **Variational/GMI metric** (A2) retrofit across LP-1/2/3.
4. **Causal controls** (A5) — deranged/zero/moment-matched + embed-vs-hidden.
5. **Fisher-spectrum experiment** (A3) — the explanation, then the better code.
6. Outer code at 15–20 bits/slot after characterizing sub/trans/ins/del error mix.
7. LP-5 bilevel (A6), then LP-4 dynamics (A9).

## Good first targets on an A100

- **Fastest to a result:** Problem 4 (running) and Problem 3 (a ~9M-param
  retrain with a role-binding parameterization; a clean win would be
  transposition rate → 0 at fixed capacity).
- **Highest ceiling:** Problem 5 (a crisp translatability theorem) and Problem 1
  (a spectral bound predicting the frozen-readout gap).
- **Most immediately useful to the demo:** Problem 7 (real output-token savings).

Data to request: the cached feature tensors and bridge/tap checkpoints
(gitignored for size) are regenerable by the `python -m` commands in the README;
ask if you want them shipped directly.
