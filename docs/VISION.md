# Vision & idea log

Running capture of the conceptual arc behind this repo — the ideas that don't
live in any single experiment. Newest sections at the bottom. `NOTEBOOK.md` is
the experimental record; `OPEN-PROBLEMS.md` is the formal agenda; this is the
"why" and the wild-but-tracked ideas. Ideas here are hypotheses unless a run in
`runs/` backs them.

## The through-line

A vision-language model's "vision tower" is just one adapter that produces
vectors the frozen LLM accepts in place of token embeddings. The LLM doesn't
know they came from pixels. So the tower is replaceable by *anything → learned
adapter → vectors*. Once you see that, a sequence of questions unfolds:

1. **What can a frozen receiver actually use** from arbitrary injected vectors?
   (LP-1: ~10 readable bits/slot through frozen Qwen, though the vectors carry
   ~2× that — the receiver, not the channel, is the bottleneck.)
2. **Can one model speak into another this way?** (LP-3: yes, 0.6B→2B at text
   density, *under known alignment*.)
3. **Can it carry language, not just bits?** (LP-2: yes, a "whisper" — meaning
   survives, spelling smears.)
4. **Can an outside observer read the wire without asking the models?** (Wiretap:
   yes, a 2M-param tap; plus drift/jam gauges. This is the auditability leg.)
5. **What does legibility cost?** (LP-5: the legibility-tax λ-sweep.)
6. **Can the models evolve their own private code, and could we still translate
   it?** (LP-5b + the identifiability question — gauge symmetries mean "meaning"
   may not be uniquely recoverable without semantic anchors.)

The north-star, per external review: *how much task-relevant, independently
auditable information can cross between frozen models per unit of real system
cost* — where cost is a vector (receiver positions, transport bits, sender/
receiver FLOPs, latency, fidelity, auditability). Today we demonstrate **context/
KV-position compression**, not wire-bandwidth compression (a 2048-d bf16 vector
is 4096 bytes, not one token id). Name it correctly.

## The safety frame (why auditability is first-class here)

If capable models communicate in vectors for efficiency, the field loses the
free interpretability that text gives. This repo's bet: make legibility a
*measured property* — every channel ships an independent decoder and a
drift/jam detector, and we measure what auditability costs (LP-5). The honest
limit: *fidelity checks the message; the tap checks for passengers.* Verifying
"B got the right thing" is cheap; proving "nothing else rode along" needs the tap,
capacity starvation, or noise injection, because the vectors carry spare capacity
a covert channel could use.

## Idea: hook a model's out-wire back to its own in-wire (self-port rollout)

**User's insight.** You don't strictly need two models. Give a model a trained
"vector mouth" (LoRA/head that emits a latent instead of a token) and feed that
latent straight back into its own input port. Now it *reasons in vectors* — a
self-loop, no text in the middle. Emit k vectors, each worth ~2 tokens, and a
2000-token thought becomes ~125 latent steps: fewer expensive sequential
decodes. The independent wiretap becomes a **thought-translator** — you can read
the latent reasoning in English without asking the model to narrate it.

**Status / honesty.** This is real and has precedent (Coconut / "chain of
continuous thought" feeds the last hidden state back as the next input; recurrent
-depth transformers, Penelope). The benefit is task-dependent, and the hard part
is *stability*, not plumbing: a continuous self-loop has no discretization to
snap noisy states back onto a manifold, so errors can accumulate or vanish. That
makes it a dynamical-systems problem (OPEN-PROBLEMS A9): track the Jacobian
products' spectral radius / Lyapunov exponents; keep it in a contractive-enough
regime with spectral norm, periodic discrete anchors, or an error-correcting
refresh every m steps. Our `bidirectional.py` self-port map (read M's state →
vector → inject into M) is exactly the reusable piece; LP-4 wires it to itself.
The tap-as-thought-reader is a genuinely novel monitorability angle: alien
internal reasoning that is *translatable by construction* if trained with the
legibility term.

## Idea: a Mixture-of-Experts colony on a shared latent bus

**User's insight.** The payoff isn't two models — it's a *colony* of specialists
(math, code, planning, retrieval, vision) that use each other's strengths by
swapping vectors on a shared bus, with a router that asks "which transformation
does the current latent state need?" rather than "which model answers?". English
is a wasteful inter-process protocol for machines; a learned bus isn't.

**Why our observations should extrapolate.** Everything we measure on the 2-node
case is the primitive the colony is built from: the readable-capacity-per-slot,
the whisper (semantic > surface fidelity — good, since experts need each other's
*meaning* not wording), the tap+gauges (per-edge auditability), and the
legibility tax (the price of a monitorable bus). The standardization sketch (a
canonical bus space, frames with headers + content-types, reliable-exact vs
best-effort service classes, version pinning, the legibility clause) is what a
colony needs to stay compatible and inspectable. The cheapest first colony:
several instances of *one* shared-weight model (so one bridge serves every edge,
like the self-port trick), then heterogeneous specialists via CCA/Procrustes
alignment (A8) before nonlinear bridging.

**What to build here toward it (proposed LP-6).** (a) A 3-node star: two
specialist contexts + a router, all sharing one frozen model, edges = the trained
self-port bridge; measure whether routed latent collaboration beats each node
alone AND beats text relay at equal token budget, with the tap logging every
edge. (b) Then a real specialist pair (e.g. a math-tuned and a code-tuned model)
with a learned-alignment bridge, on a task neither solves alone. The claim to
chase: *a small latent conversation between specialists solves problems neither
can, at less compute than the same exchange in English* — and, because of the
tap, remains auditable.

## What this is NOT — a capability multiplier (honest correction)

A recurring overclaim (mine, and in the source discussions): that a latent colony
is a "hive mind smarter than any member." Push back on this.

- For equal *total* params, a **dense** model is smarter than an **MoE** — the MoE
  activates only a fraction of weights per token. MoE's win is compute/memory
  efficiency, not intelligence. A "colony of specialists on a latent bus" is NOT a
  monolithic MoE (whose experts are FFN slices with lossless shared attention);
  it's separately-trained full models over a **lossy** channel. Don't import the
  MoE efficiency framing to it.
- The colony can **never** beat a single model that already contains the same
  knowledge and fits in memory — a monolith has infinite-fidelity internal
  communication; our channel is the ~72% whisper.
- Two questions were conflated: **(a) composition of specialists** = the capability
  source (A knows what B doesn't) — but *text multi-agent already does this*;
  vectors add nothing to capability here. **(b) vectors vs text** = a *constant-
  factor* efficiency + auditability win, NOT capability.

Honest value proposition of this whole project: **efficiency** (serve specialists
that don't fit as one monolith; fewer positions/tokens/latency — cf. Cache-to-
Cache's ~2× latency), **auditability** (if the field adopts latent comms for
efficiency, we must be able to read them — valuable regardless of capability), and
the **self-loop** (test-time-compute reallocation — a modest, task-dependent
capability angle, per Coconut). Not superintelligence.

Falsifiable test (LP-6): the latent colony must **beat text-relay at equal token
budget** to justify vectors at all. Current honest prediction: it ties text on
capability and wins on efficiency/auditability. State it that way.

## Standing cautions carried from review

- Don't call goodput a capacity; use the variational/GMI rate (A2).
- Don't claim binding without the order-preserving warm-start control (A4).
- Don't claim mind-to-mind without the deranged/zero/embed-vs-hidden controls (A5).
- Every run must be self-describing (git SHA, seeds, model revs, all predictions).
