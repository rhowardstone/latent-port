# Paper strategy for `latent-port`

**Literature/repository review date:** 2026-08-12  
**Scope:** current repository state through `master` at the time this review began, plus a targeted search of current latent-communication, soft-compression, latent-reasoning, interpretability, and latent-channel security papers.

## Executive verdict

The highest-value paper is **not** “LLMs can communicate in vectors.” That claim is already crowded. It is also not “use the VLM visual interface as a universal port,” “build an O(n) canonical latent bus,” “compress text into soft vectors,” “predict future hidden states,” or “a frozen decoder cannot read all information present in its input.” Each of those now has close prior art.

The strongest paper that still appears genuinely differentiated is:

> **Can We Read the Wire? Auditable Latent Communication Under Bounded Adversaries**
>
> Treat model-to-model latent communication as a channel with **three separate properties**: (1) utility/information accessible to the intended receiver, (2) legibility to an independent auditor, and (3) residual covert capacity that can carry information without changing the receiver's intended behavior. Measure the Pareto frontier between them, then deliberately train a sender to hide from bounded auditors while remaining useful to the receiver. Finally, characterize and reduce the surviving covert channel.

A less catchy technical title would be:

> **Utility, Legibility, and Covert Capacity in Learned Latent Communication**

This recommendation uses the strongest pieces already present in this repo—the independent wiretap, the receiver/tap accessibility gap, the LP-5 legibility objective, the manifold detector, and the proposed pullback-Fisher geometry—but changes the scientific object. The object is no longer “a latent port.” It is **the auditability of a learned latent channel**.

That distinction matters because the field is moving quickly toward latent communication, while most work still evaluates **task performance and efficiency**. Recent papers have started showing that latent channels can be causally misleading, can carry attacks invisible to text inspection, and need integrity protection. The missing measurement is much more operational: **if a latent channel is useful, how expensive is it to make independently readable, and how much information can a sender still hide from the monitor?**

I would make the decoder-accessibility/Fisher result a mechanistic section of this paper, not the flagship paper by itself. A February 2026 paper already frames multimodal decoder failure as mismatched decoding and uses GMI plus LoRA interventions, so “decoder mismatch” alone is no longer a clean novelty claim.

---

## 1. What the repository has actually established

The repo is a serious experimental lab, not merely a concept demo. The strongest existing observations are:

1. **A trainable sender can place exact random information into learned context vectors that a frozen language model can recover.** At 16 slots, the current LP-1 artifact peaks operationally around the density of a text baseline at 10 loaded bits/slot. The repo correctly demoted the earlier “capacity” wording: packet goodput is not Shannon capacity.

2. **The same vectors can be dramatically more readable to a small independent decoder than to the intended frozen LLM.** In `runs/wiretap.json`, the ~2.1M-parameter tap reaches 100% packet-exact recovery at 20 loaded bits/slot while the frozen Qwen3-VL-2B receiver reaches 4.6875% packet exact (although its character accuracy remains much higher, ~93.2%). This is a real accessibility gap, not proof of physical channel capacity.

3. **Cross-model transfer exists, but the successful experiment currently assumes known alignment.** `GatherBridge` extracts sender-model hidden states at fixed payload character positions. The Perceiver-style learned alignment collapsed. Therefore LP-3 supports “representation translation under known alignment,” not a general variable-length model-to-model translator.

4. **Free text crosses the port, but the current efficiency number is modest.** LP-2 is ~71.6–72.4% character fidelity at ~1.45x context-position compression on Wikitext. That is scientifically useful as a semantic/distortion regime, but it is not competitive evidence for a generic prompt-compression paper.

5. **The independent free-text tap is nontrivial but substantially worse than the receiver.** `runs/text_tap.json` reports ~52.7% character accuracy. This is useful: auditability is neither automatic nor hopeless.

6. **The cooperative LP-5 legibility sweep is only a pilot.** At lambda=0.3, receiver character fidelity is essentially unchanged from lambda=0, while the co-trained tap is ~40.6% character-accurate. At lambda=1.0, the co-trained tap rises to ~44.7% while receiver fidelity falls to roughly 70–71%. This hints at a tradeoff, but a co-trained tap is not an independent audit test and the sweep is too small to establish a frontier.

7. **Receiver adaptation appears capable of closing a large part of the accessibility gap, but the current comparison is confounded.** `receiver_lora.py` trains the sender for 3,000 steps with the receiver frozen, then adds receiver LoRA and continues training the *same sender* for another 3,000 steps. The reported 82.8% exact rate therefore compares 6,000 sender steps + LoRA against 3,000 sender steps frozen. In addition, `--lora-lr` is parsed but never used: LoRA parameters receive `args.lr`. The result is promising, not yet publication-grade evidence.

8. **LP-4 does not currently establish robust temporal abstraction.** A learned map from `h_t` to `h_{t+Delta}` beats a no-op future-token baseline for Delta=1,2,4,8, but recursive application collapses to roughly 5–13% accuracy. This is evidence that a local future-state predictor can jump toward a later teacher state; it is not evidence that one can safely replace many autoregressive reasoning steps with one recursively composable latent macro-step.

9. **The repo's earlier binding story was correctly challenged by a curriculum confound.** The mean-embedding warm start erases within-slot order. New role/blend/none controls are the right fix; e.g. the eps=0.1 blend run at 15 bits/slot reaches 0.375 packet exact, 0.969 char accuracy, 0.987 unordered accuracy, and ~0.012 order-error share. Binding/order may still be interesting, but it is not the highest-value paper.

The repository's best scientific habit is that it has already revised overclaims when controls exposed them. Keep doing that. The winning paper should be built around a claim that survives hostile controls by construction.

---

## 2. Prior-art map: which attractive claims are already taken?

The literature is moving fast enough that the novelty boundary has shifted in the last several months.

### 2.1 “Models can communicate through embeddings/activations instead of text” — already done

- **CIPHER / Let Models Speak Ciphers (2023)** communicates a weighted output-embedding representation instead of sampled text in multi-agent debate.  
  https://arxiv.org/abs/2310.06272
- **Communicating Activations Between Language Model Agents (2025)** directly combines intermediate activations between agents and reports substantial compute savings.  
  https://arxiv.org/abs/2501.14082
- **Interlat (2025)** explicitly enables agents to communicate entirely in latent space using last hidden states and latent compression.  
  https://arxiv.org/abs/2511.09149
- **LatentMAS (2025)** provides training-free pure latent collaboration through shared latent working memory.  
  https://arxiv.org/abs/2511.20639
- **Cache-to-Cache / C2C (2025)** projects and fuses source KV caches into a target model and reports approximately 2x latency speedup over text communication.  
  https://arxiv.org/abs/2510.03215

**Verdict:** LP-1/LP-3 are useful instruments and replications/extensions, but “latent communication works” cannot carry the paper.

### 2.2 “Use a shared/canonical latent space so heterogeneous models need O(n), not O(n^2), adapters” — already very close

- **Latent Space Communication via K-V Cache Alignment (2026)** learns a shared representation space for multiple models, with adapters translating model states into and out of that space. It also transfers soft prompts across models.  
  https://arxiv.org/abs/2601.06123
- **The Vision Wormhole (2026)** is especially close to this repo's original intuition: it repurposes the VLM visual interface as a model-agnostic latent port, uses a universal codec, supports heterogeneous models, and explicitly adopts a hub-and-spoke O(n) topology instead of pairwise O(n^2) translation.  
  https://arxiv.org/abs/2602.15382

**Verdict:** the proposed “canonical neural ABI” is still worth building as engineering, especially if it gains typed/structured subspaces and safety guarantees, but the basic hub/shared-space architecture is no longer a clean paper novelty.

### 2.3 “The visual interface can be repurposed as a latent communication port” — already done directly

**Vision Wormhole** is effectively direct prior art for this framing. The current repo remains useful because it independently reaches a related architecture and adds a different auditability direction, but the port itself should not be sold as novel.

### 2.4 “Dense latent traffic wastes wire bandwidth and can be compressed” — active prior art

A paper posted **2026-08-10**, **Post-Hoc Sparse Coding of Latent Communication Between Vision-Language Model Agents**, fits a sparse autoencoder to frozen Vision Wormhole traffic and reports a 128x byte reduction relative to the original float32 transport with almost unchanged mean accuracy in its reported setting. The authors explicitly note that matched controls are still needed to isolate sparse coding from precision/rank/position effects.  
https://arxiv.org/abs/2608.10198

**Verdict:** quantization/noise/rank sweeps remain important controls and mitigation experiments here, but “latent vectors can be compressed on the wire” is now an active subfield rather than a fresh headline.

### 2.5 “Soft vectors can compress textual context” — mature prior art

- **Gist Tokens (2023)** trains models to compress prompts into reusable gist tokens and reports up to 26x prompt compression in its setting.  
  https://arxiv.org/abs/2304.08467
- A large soft-prompt / prompt-compression literature predates this repo as well.

**Verdict:** LP-2's 1.45x context-position compression is not a prompt-compression paper. Its value is as a controlled communication channel where semantic and surface fidelity can be separated.

### 2.6 “Reason in continuous/latent states instead of emitting every CoT token” — crowded

- **Coconut / Training LLMs to Reason in a Continuous Latent Space (2024)** feeds the last hidden state back as the next input embedding.  
  https://arxiv.org/abs/2412.06769
- **Recurrent Depth (2025)** scales test-time compute by repeatedly applying latent recurrent computation rather than producing reasoning tokens.  
  https://arxiv.org/abs/2502.05171
- **CODI (2025)** distills chain-of-thought into continuous space and reports 3.1x compression in its setting.  
  https://arxiv.org/abs/2502.21074
- **Training Continuous Chain of Thought Models: A Tale of Two Regimes (2026)** reports severe degradation on longer reasoning traces, which is directly relevant to LP-4's recursive instability.  
  https://arxiv.org/abs/2607.16972

**Verdict:** LP-4 can become a useful stability/mechanism project, but the concept is not novel and the current recursive result is too weak for a flagship paper.

### 2.7 “A frozen language decoder is a mismatched decoder for non-text information” — already formalized

**Modality Collapse as Mismatched Decoding: Information-Theoretic Limits of Multimodal LLMs (2026)** argues that modality-specific information survives in representations but a text-trained decoder fails to use it; it formalizes accessible information with GMI and validates the decoder bottleneck with LoRA interventions.  
https://arxiv.org/abs/2602.23136

This overlaps strongly with `OPEN-PROBLEMS.md` A2 and part of A3.

**Verdict:** do not make “mismatched decoding” itself the novelty. The repo can still contribute a communication-specific geometry result if it shows something stronger, such as a learned sender systematically occupying receiver-insensitive directions and a quantitative relationship between that geometry, independent auditability, and covert capacity.

### 2.8 “Latent communication needs causal controls” — now established as necessary

- **Do Latent Channels Actually Communicate? (2026)** decomposes message presence, identity, example-specific content, and other-agent value through controlled substitutions.  
  https://arxiv.org/abs/2607.26773
- **When Does Latent Communication Pay? (2026-08-05)** uses deranged, zero, and moment-matched cache controls and shows that large apparent cache effects can exist without meaningful example-specific transfer.  
  https://arxiv.org/abs/2608.04893

**Verdict:** these controls are no longer optional reviewer-proofing; they are the emerging standard. `latent-port` should adopt them throughout LP-2/LP-3 and in every task-level experiment.

### 2.9 “Latent channels are opaque/security-sensitive” — known problem, but not yet solved by an auditability frontier

- **Out of Sight, Not Out of Mind (2026)** demonstrates latent-only attacks in latent multi-agent systems and explicitly warns that attacks move beyond visible-text inspection.  
  https://arxiv.org/abs/2605.28214
- **When Latent Agents Lie (2026)** studies KV-cache integrity and shows that text-only verification misses latent tampering; its strongest transport mitigation is cryptographic integrity (HMAC), which solves tampering rather than semantic legibility.  
  https://arxiv.org/abs/2606.28958
- **STATEWITNESS (2026)** trains a separate activation decoder to produce natural-language evidence for deception auditing, establishing that independent natural-language decoding of internal states is a serious safety direction.  
  https://arxiv.org/abs/2606.17478
- **HyLaT (2026)** explicitly identifies latent communication as opaque and keeps a concise text channel alongside a richer latent channel to retain interpretability.  
  https://arxiv.org/abs/2605.25421
- **Prover-Verifier Games improve legibility of LLM outputs (2024)** is not a latent-communication paper, but its “helpful vs sneaky prover against a small verifier” framework is close methodological prior art for the adversarial legibility game proposed below.  
  https://arxiv.org/abs/2407.13692

**Verdict:** the *problem* of opacity is known. What I did **not** find in the current literature is the exact experiment this repo is positioned to run: a quantitative **receiver utility vs independent auditor legibility frontier**, followed by a **bilevel/adversarial sender-vs-fresh-auditor game** that estimates residual covert capacity under fixed receiver utility. That is the novelty to protect and test.

---

## 3. The novelty worth chasing

The paper should define a latent channel by three axes rather than one accuracy number.

Let a sender produce latent traffic `Z = S(X)` and the intended receiver induce a distribution `q_B(Y | Z)`. Let an independent auditor come from a bounded family `T in A` and attempt to recover an audited semantic/message variable `M` from `Z`.

### Axis 1: receiver-accessible information / utility

For controlled random payloads with known entropy, use a variational mismatched-decoding lower bound rather than packet goodput as the primary information metric:

`I_B^var = H(X) + E[log2 q_B(X | Z)]`.

For natural language, do not substitute token count for source entropy. Use paired information gain over causal nulls, e.g. the NLL improvement of `Z_correct` over `Z_deranged`, `Z_zero`, and moment-matched random controls, plus downstream task utility on private-information tasks.

### Axis 2: independent legibility

Define legibility relative to an **auditor class and budget**, not a single hand-picked tap:

`L_A(S) = max_{T in A, budget <= C_A} I_T^var(M; Z)`

or its operational equivalent (semantic recovery, factual recovery, calibrated exactness) after training the auditor on a fixed amount of labeled protocol data.

Crucially, the auditor used for final evaluation must be **fresh**. A channel trained jointly with one tap can overfit to that tap and is not evidence that outside monitors can read it.

### Axis 3: residual covert capacity

The security object is not `C_phys - C_read`. Unused information need not be independently controllable. The object is closer to:

`C_covert(eps, alpha, A) = sup I(M_secret; Z | Y_public)`

subject to:

- intended public receiver utility changing by at most `eps`, and
- a calibrated auditor/detector family failing to identify or recover the secret above a specified level `alpha`.

Operationally, make the sender carry **two messages**:

- a public message/task that B is supposed to use;
- a random secret passenger that B should be invariant to.

Then ask how many passenger bits per latent slot can be transmitted to a colluding secret decoder while (a) B's public behavior remains unchanged and (b) the deployed auditor family cannot decode/detect the passenger.

That experiment turns the repo's spare-capacity observation into a real security result.

### The core paper claim, if the experiments support it

> **Latent communication has a measurable legibility frontier: receiver utility, independent readability, and residual covert capacity can vary independently. Auditability can be trained for, but its cost and robustness must be measured against fresh and adversarial auditors rather than assumed from task performance or a co-trained decoder.**

A strong positive result would show that auditability is cheap and covert capacity can be squeezed close to zero with practical constraints. A strong negative result would show that a receiver-useful protocol can reliably evade affordable monitors. **Either outcome is publishable and decision-relevant.**

---

## 4. Why this is valuable

### Scientific value

Most latent-communication work asks “does it improve accuracy/latency?” This paper asks a different systems question: **what observability is lost when the communication substrate stops being language?** It gives the field metrics and adversarial tests rather than an intuition that hidden vectors are “probably decodable.”

### Safety/security value

Text communication gives developers logs almost for free. Latent traffic does not. If model systems adopt latent messages for efficiency, an independent monitor must either decode them, certify that no unaudited capacity remains, or admit that the channel is opaque. This work would quantify that boundary.

Cryptographic integrity and semantic legibility solve different problems: HMAC can prove that a tensor was not changed in transit; it cannot tell an operator what the tensor says or whether an authorized sender embedded an unwanted passenger. This paper targets the latter.

### Engineering value

If the “legibility tax” is small, the result suggests a deployable design rule: latent protocols should be trained with an independent-auditor constraint and transported through capacity-limited/quantized channels with calibrated monitoring. If the tax is large, hybrid text-latent protocols or stronger explicit service classes become easier to justify.

### Value even if latent communication never beats text on capability

This paper does not depend on a speculative “hive mind” capability gain. It matters if latent communication is adopted for any mundane reason: lower latency, context/KV reuse, model specialization, or internal state sharing. That makes the research thesis more robust than betting on a capability breakthrough.

---

## 5. Required experiments

The current repository contains enough machinery to start, but the paper needs a more adversarial and reproducible protocol.

### E0. Fix validity blockers before spending large GPU budgets

1. **Repair `receiver_lora.py` comparison.** Save one sender checkpoint after the common phase. Fork it into:
   - frozen receiver + another N sender steps;
   - receiver LoRA + the same N sender steps.
   Use identical data streams and separate optimizer parameter groups so `--lora-lr` actually controls LoRA. Run >=5 seeds at the critical density.
2. **Use publication metrics everywhere.** Packet exact, per-symbol error, variational/GMI-style rate for random payloads, paired information gain for natural text, cluster-bootstrap CIs, and full held-out predictions.
3. **Add the causal substitution battery** (`correct`, `deranged`, `zero`, `moment-matched`) to every LP-2/LP-3 task experiment.
4. **For LP-3, compare A input embeddings vs A final hidden states at matched bridge capacity.** Hidden states must win on a task requiring sender computation before “computed state transfer” is earned.
5. Critical results need 3–5 independent training seeds minimum; claims with very small n should be explicitly labeled pilots.

### E1. Build a task battery where the message is causally necessary

Use synthetic/private-information tasks first so language priors cannot fake communication:

- random key-value lookup;
- random access codes / passwords;
- appointments with random names/times/locations;
- graph edges and multi-hop paths on random node labels;
- arithmetic operands or intermediate values revealed only to the sender;
- compositional two-agent tasks where neither agent alone sees all required facts.

Then add natural QA/reasoning as ecological validation.

Every task should have two regimes:

- **message-needed:** B cannot solve without A's private information;
- **message-unneeded:** B can solve from its own prompt, testing whether the latent simply perturbs priors.

### E2. Measure the cooperative legibility frontier with fresh auditors

Train a family of protocols over a grid:

- communication density / slot count;
- legibility weight `lambda`;
- receiver frozen vs small adapter;
- at least 2–3 backbone families or receiver scales.

For each protocol, train **fresh post-hoc auditors** not used in channel training:

- linear probe;
- small MLP;
- the current transformer tap at multiple widths;
- a larger sequence decoder / activation explainer.

Vary auditor data and compute budgets. Report receiver utility against *best fresh-auditor legibility at budget C*. That is the actual Pareto surface.

A co-trained reference tap may remain in the loss, but it cannot be the headline evaluation.

### E3. The decisive experiment: helpful sender vs sneaky sender

Import the logic of prover-verifier games into latent communication.

**Helpful protocol:** maximize receiver task utility while making a small auditor's job easy.

**Sneaky protocol:** maximize receiver task utility while minimizing the performance of an auditor trained to convergence under a fixed budget.

Do not freeze one auditor forever. Use alternating/bilevel training and periodically initialize **fresh auditors** to prevent sender overfitting to a particular decoder.

Measure:

- receiver task success;
- fresh-auditor semantic/message recovery;
- auditor compute/data needed to recover a given fraction of the message;
- protocol drift / detector false-positive rate;
- channel cost.

If sneaky protocols remain easily decoded by fresh small taps, that is evidence that receiver-useful channels have **inherent practical legibility** under the tested class. If they evade fresh taps, the assumption is falsified.

### E4. Make the covert passenger explicit

This should be the paper's memorable experiment.

Train `S(public, secret) -> Z` so that:

1. B recovers/uses `public` exactly as before;
2. a colluding decoder `D_secret(Z)` recovers a random secret payload;
3. the audit system should detect or decode the passenger.

Sweep secret bits per slot while constraining the KL/output change of B on the public task.

Report **covert-goodput vs public-utility delta vs auditor detection/recovery**. A three-axis plot is much more informative than “our tap got 95%.”

This also distinguishes semantic auditability from simple distribution-shift detection: an on-manifold malicious sender may hide a secret while keeping marginal statistics normal.

### E5. Connect covert capacity to receiver geometry

Use the proposed pullback-Fisher analysis, but make it answer the security question.

For B, locally:

`G_B(z) = J_B(z)^T (diag(p_B) - p_B p_B^T) J_B(z)`.

Low-eigenvalue directions are approximately receiver-insensitive: perturbing them changes B's output distribution little. Test whether:

- high-density learned senders place more variance/message separation in low-`G_B` directions;
- covert passenger encoders preferentially use low-`G_B` directions;
- the independent auditor is sensitive to those same directions;
- truly dangerous covert directions are simultaneously low-sensitivity for B's public behavior **and** low-sensitivity for the deployed audit family.

This turns the current tap-vs-receiver gap into mechanism rather than a curiosity.

A useful geometric quantity is the dimension/spectrum of the approximately public-invariant subspace under B after intersecting it with the auditor's blind subspace. That is a local proxy for “room to hide.”

### E6. Test mitigations, not only attacks

Candidate defenses:

- lower-rank transport;
- 8/4/2-bit quantization;
- calibrated additive noise;
- random projection / dimensional bottlenecks;
- capacity-limited structured service classes (`exact`, `semantic`, `control`);
- cooperative legibility loss;
- adversarially trained auditor ensembles;
- Fisher-aware projection that removes directions irrelevant to the public receiver;
- periodic explicit text anchors for high-risk messages.

For each defense report:

`public utility | independent legibility | covert capacity | transport bytes | positions | latency`.

The winning defense is not the one with the prettiest embeddings. It is the one that pushes the covert frontier down without destroying useful communication.

### E7. Calibrate detection properly

Replace hand-set `z > 1.5` / confidence thresholds with held-out calibration and conformal p-values or another explicit false-alarm-controlled procedure. Report power at fixed false-positive rates.

Test adaptive, on-manifold attacks—not only Gaussian jamming. Gaussian noise is an easy sanity check, not a security result.

### E8. Scale and generalize

The most valuable scaling experiment is no longer just “bits/vector vs receiver size.” Measure all three axes across scale:

- Does intended receiver accessibility increase with model size?
- Does a fixed auditor become relatively weaker as the receiver grows?
- Does the residual covert subspace grow or shrink?
- How much auditor capacity is needed per receiver parameter/hidden width?

Also test **protocol transfer**:

- train auditor on sender seed A, test sender seed B;
- train on one checkpoint, test later protocol revisions;
- train on one sender model, test another mapped into the same bus.

If every protocol revision requires a fully retrained tap, auditability is expensive operationally. If a semantically anchored auditor generalizes, that is a major positive result.

### E9. Measure real systems cost honestly

A 2048-dimensional bf16 latent slot is 4096 bytes, even if it occupies one receiver context position. Keep these axes separate:

- receiver positions / KV footprint;
- transport bytes;
- sender FLOPs;
- receiver FLOPs;
- auditor FLOPs;
- wall-clock latency;
- task distortion;
- auditability.

Compare against text and at least one strong latent/KV baseline where feasible. Include quantized/sparse transport because float tensors can erase any network-level efficiency advantage.

---

## 6. What would make the paper genuinely strong?

A top-tier version has **four contributions that lock together**:

1. **Definition/benchmark:** utility, legibility, and covert capacity as separate properties of latent communication.
2. **Empirical frontier:** the first broad measurement of the legibility tax using fresh bounded auditors, multiple channel densities, tasks, and model scales.
3. **Adversarial result:** helpful-vs-sneaky protocols plus the explicit covert-passenger experiment.
4. **Mechanism + mitigation:** receiver/auditor geometry predicts where passengers hide, and a practical bottleneck/regularizer materially reduces covert capacity at small utility cost.

Any one of these by itself is a workshop paper. All four together form a coherent flagship.

The cleanest Figure 1 would be a triangle or 3D frontier:

- x-axis: receiver public utility / accessible rate;
- y-axis: fresh-auditor legibility;
- z-axis (or color): covert passenger bits/slot.

Show baseline latent channels, helpful-legibility training, sneaky training, and defended/quantized channels on the same plot.

The cleanest demo is even simpler:

> B receives the same public message correctly in two cases. In one, the latent also carries a secret password. The monitor either reads/catches the passenger—or provably/empirically fails under a specified audit budget.

That communicates the research question immediately.

---

## 7. Kill criteria and decision points

This direction should be falsifiable early.

### Kill/redirect condition A: fresh auditors always trivially recover everything

If fresh tiny auditors recover essentially all receiver-useful and covert traffic across sneaky training with negligible labeled data, then the safety problem is easier than expected. That is still a useful result: pivot the paper to **inherent practical legibility of learned latent channels** and characterize why.

### Kill/redirect condition B: sneaky channels evade all affordable auditors but only by obvious OOD drift

Then the problem reduces largely to distribution monitoring. Strengthen the detector work and adaptive on-manifold attacks before claiming covert capacity.

### Kill/redirect condition C: covert passengers cannot be added without damaging public utility

Excellent. Quantify the upper bound and identify what property of the receiver/channel starves the covert subspace. A negative covert-capacity result is arguably the safest and most useful outcome.

### Kill/redirect condition D: latent communication loses to text on every real cost axis

The safety result can still publish as a warning/benchmark, but stop investing in a latent-bus product story. The research question becomes “before adopting opaque latent channels, prove they buy enough to justify the audit burden.”

---

## 8. Secondary paper directions, ranked

### #2 — Receiver-accessible geometry of learned latent channels

**Possible title:** *Where Does the Message Go? Functional Geometry of Frozen-Decoder Latent Channels.*

Core experiment: variational accessible rate + pullback-Fisher spectrum across density, sender training, receiver scale, quantization, and receiver LoRA. Demonstrate that learned codeword separation migrates into low-sensitivity directions as density rises, then improve the channel with a geometry-aware loss.

Why second, not first: the mismatched-decoder framing and LoRA recovery now have close prior art in multimodal models. It becomes stronger if tied to **covert capacity** or if the geometry quantitatively predicts channel limits across architectures.

### #3 — Stability limits of latent temporal abstraction

The current LP-4 negative result can motivate a clean dynamical-systems paper: local future-state prediction is easy; recursively composable macro-steps are not. Measure Jacobian products/Lyapunov behavior, train with multi-step rollout losses, spectral constraints, denoising/error-correcting anchors, and compare against Coconut/CODI/recurrent-depth baselines.

Why third: crowded field and current task setup is Wikitext next-token trajectory prediction rather than reasoning.

### #4 — Structured neural ABI / latent bus

Build typed subspaces, O(n) adapters, routing, versioning, exact-vs-semantic service classes, audit hooks, and interoperability tests.

Why fourth as a paper: the basic shared-space and hub-and-spoke architecture now overlaps heavily with KV-cache alignment and Vision Wormhole. It could become novel again if the **auditable/typed ABI** is the main contribution rather than the shared space itself.

---

## 9. Immediate A100 queue I would run

In order:

1. **Fix and rerun receiver-LoRA fairly** (forked common sender checkpoint; proper LoRA LR; 5 seeds). This cleans up one of the repo's most striking observations.
2. **Retrofit causal substitutions + paired information metrics** into LP-2/LP-3. Without this, later task claims are vulnerable to the exact failure mode exposed by the August 2026 causal-audit papers.
3. **Implement a fresh-auditor evaluation harness**: train channel with reference tap, throw that tap away, train 4 fresh auditor classes from scratch under fixed data/compute budgets.
4. **Run a real lambda x density frontier** on random/private payload tasks first. Do not start with natural text.
5. **Implement the two-message passenger task** (`public`, `secret`) with a colluding secret decoder.
6. **Train the sneaky sender** against alternating fresh auditors.
7. **Only then run the Fisher/Jacobian geometry** to explain where successful covert traffic lives.
8. **Mitigation sweep**: rank/precision/noise + audit loss; identify a useful operating point.
9. **Scale across 2–3 receiver sizes** and at least one second model family.
10. **Return to a heterogeneous latent bus** only after the safety primitive is convincing; then make the bus an application of the audit framework rather than the novelty itself.

This ordering maximizes information gained per GPU-hour. Steps 3–6 can kill or validate the paper thesis before expensive multi-model scaling.

---

## 10. Recommended repository build-out

Rather than adding one-off scripts, create an explicit audit layer:

```text
visual_encoder/
  audit/
    metrics.py          # R_B, fresh-auditor R_T, task/info-gain metrics
    auditors.py         # linear / MLP / transformer / larger decoder families
    calibration.py      # conformal/FPR-controlled detector calibration
    adversary.py        # helpful/sneaky alternating training
    covert.py           # public+secret passenger channel
    geometry.py         # JVP/VJP Fisher-spectrum + subspace diagnostics
    defenses.py         # rank/quantization/noise/projection wrappers

experiments/
  audit_frontier.py
  covert_passenger.py
  auditor_generalization.py
  geometry_covert.py
  defense_frontier.py

runs/audit/
  ... self-describing artifacts ...
```

Standardize each result artifact around the same schema:

```text
protocol identity + provenance
public task / secret task
receiver utility + information metric
fresh auditor family + budget + score
covert payload rate
calibrated detection result
positions / bytes / FLOPs / latency
all held-out predictions or hashes/paths
bootstrap / seed statistics
```

That turns `latent-port` from a collection of ingenious experiments into a reusable **latent-channel audit bench**. That is a stronger research artifact and a more defensible contribution.

---

## Bottom line

The project started from an exciting systems intuition—replace text with vectors and let models communicate more directly. In August 2026, the field has already reached that idea from several directions. The opportunity has moved one layer deeper.

**The novel question is no longer whether models can speak latent. It is whether anyone else can reliably tell what they said.**

The repo already contains the unusual seed of that paper: a sender/receiver channel, a separately trained wiretap, a measurable receiver/tap gap, a pilot legibility loss, drift monitoring, and enough control machinery to make the next step rigorous. Turn those pieces into a bounded-adversary communication game and measure the **legibility tax plus covert capacity**. If that works, the result is more important than another 2x latent-agent speedup because it defines what safe deployment of the entire emerging class would require.
