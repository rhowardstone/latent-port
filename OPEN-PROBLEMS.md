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

Definitions. `H₂` = binary entropy. `BER(b)` = bit error rate at load `b`.
`R(b) = b·(1−H₂(BER(b)))` is a BSC-equivalent rate, **not** a certified Shannon
capacity (errors within one image/message are correlated — see notebook caveats).
`I(·;·)` = mutual information. A "position/slot" is one KV cache entry, the same
resource a text token consumes.

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
