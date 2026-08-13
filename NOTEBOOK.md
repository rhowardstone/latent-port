# Lab Notebook — Visual Channel Lab

Chronological record of experiments, results, and decisions. Newest entries at the
bottom. Numbers cite the JSON artifacts in `runs/` — nothing here is from memory.

Contributors: a Codex (GPT-5.6) instance built the codec/probe phases; a Claude
(Fable 5) instance reviewed those phases and is running the latent-port phase.

---

## 2026-08-08 — Phase 1: Deterministic codecs (backfilled summary)

**Question.** How many exact bytes survive an image that is treated as a picture
(JPEG, resize, blur, noise) rather than as a file?

**Method.** `visual_encoder/codec.py`: CRC-framed packets rendered either as raw
RGB bytes (lossless-file control) or as separated color cells with block-interleaved
repetition coding (robust control). Interactive lab in `visual_encoder/web.py`.

**Findings.** Raw RGB dies under any JPEG, as designed. Palette cells at 2 bits/cell,
16 px cells, 3× repetition survive JPEG 75 exactly. This phase is the honest
"designed sender, designed receiver" baseline.

## 2026-08-08 — Phase 2: Frozen-encoder capacity probe (backfilled summary)

**Question.** How many held-out random bits per merged visual token can a ridge
probe recover from frozen Qwen3-VL-2B representations?

**Method.** `visual_encoder/qwen_probe.py`: patch-aligned spread-spectrum codes
(Rademacher bases, constant total energy, clipping fraction logged), features
cached to `runs/feature_cache/`, closed-form ridge with calibration-split lambda,
image-cluster bootstrap. Rates reported as BSC-equivalent `b·(1−H₂(BER))`.

**Findings** (`runs/qwen_probe_depth_*.json`, clean channel, 32 train / 16 val images):

| Loaded bits/token | Final vision | DeepStack-0 | DeepStack-1 | DeepStack-2 |
| ---: | ---: | ---: | ---: | ---: |
| 4  | 2.16 | 4.00  | 3.97 | 3.93 |
| 8  | 1.79 | 7.84  | 6.72 | 4.57 |
| 16 | 1.30 | 13.88 | 7.97 | 4.06 |
| 32 | 0.90 | 18.55 | 7.60 | 3.28 |

The earliest DeepStack stream preserves the code far better than the final vision
embedding. The information *arrives*; the question is who can read it.

## 2026-08-08 — Phase 3: Depth trace through the frozen decoder (backfilled summary)

**Method.** `visual_encoder/llm_probe.py`: replay cached vision features through the
frozen causal decoder via the real DeepStack injection path, probe the visual token
positions at each depth.

**Findings** (`runs/qwen_llm_probe_*.json`): after the first injection the residual
stream at visual positions holds ~10 bits/token (plateau across 16/32-bit loads);
recoverability then decays monotonically with depth to ~0.1–0.7 bits/token at the
final norm. At 4 bits/token, 13/16 validation images decode exactly at
`before_layer_1` with a shared ridge probe.

**Interpretation caution** (review note): the final-layer ≈ 0 result does NOT bound
generative readout. Generation attends from the answer position into *earlier-layer*
KVs of the visual positions, where the information still lives. The rendered-text
baseline (Phase 4) proves decayed late-layer token-local recoverability coexists
with successful transcription.

## 2026-08-08 — Phase 4: End-to-end rendered-text control (backfilled summary)

**Method.** `visual_encoder/text_baseline.py`: random Base32 rendered as dense
monospace text, full frozen pipeline transcribes, edit-distance scored.

**Findings** (`runs/rendered_text.json`, single sample per point): 95.3% char
accuracy at 1.25 source bits/visual-token, 94.5% at 2.5, 91.8% at 5.0. None exact;
an outer code would be required. The 512-char point has not completed.

**Phase 1–4 conclusion.** The bottleneck is the *receiver*, not the image channel:
~18.5 bits/token enter the LLM, ~10 remain linearly readable after injection, and
the stock generative pathway spells out none of it for alien codes but ~92–95% for
text-shaped codes it was trained to read.

---

## 2026-08-10 — Pivot: from visual channel to latent port

**Decision.** The vision tower was always just one example of a port into the
decoder. The project's real instrument is the ability to (a) inject arbitrary
embeddings into a frozen LLM and (b) measure exactly how many bits survive, where.
We now study the port directly, skipping pixels.

**Relevant prior work** (verified 2026-08-10):
- Cache-to-Cache (arXiv:2510.03215, ICLR'26): learned KV-cache projection between
  LLMs beats text communication by ~3–5% accuracy at ~2× lower latency. Closest
  existing "psychic port"; our differentiator is *measuring channel capacity in bits*,
  which task-accuracy papers do not report.
- Gist tokens / ICAE / xRAG: learned context compression to soft tokens; reliable
  ratios historically ~4–10×, not thousands.
- CALM (arXiv:2510.27688): maximum compression ≠ usable representation; latent
  manifolds need deliberate redundancy. Expect the same here.
- Coconut (arXiv:2412.06769), Penelope (arXiv:2607.25915): latent recurrence helps
  search-shaped tasks; mixed elsewhere. Text tokens act as error-correcting
  repeaters — long analog chains drift. Keep chains short.

**Roadmap.**
1. **LP-1 (this entry): learned sender → frozen receiver, random bits.** Train a
   small sender that maps held-out random Base32 payloads to k soft embeddings
   spliced into the prompt of frozen Qwen3-VL-2B's decoder; teacher-forced CE on
   exact transcription; measure exact/char accuracy vs load. A slot costs one KV
   position — identical accounting to a text token — so "bits per slot vs bits per
   text token" is apples-to-apples.
2. **LP-2: self-port context compression.** Compress a passage into k slots,
   score QA/reconstruction vs k against text baselines (gisting replication with
   our bit-metering).
3. **LP-3: cross-model bridge.** Small sender model's hidden states → projector →
   2B receiver's port.
4. **LP-4: bandwidth-squeeze curve.** 256→64→16→4→1 slots with the random-bit
   control separating channel capacity from receiver competence.

### LP-1 design

**Hypothesis.** With the sender free to place embeddings anywhere in R^2048 (no
pixel constraint, no vision tower), a frozen receiver can be *taught to be spoken
to* at ≥ the text-token rate (~10–15 bits per position for random Base32), because
the sender can at minimum rediscover soft versions of character-token embeddings.
The interesting outcome is how far *above* that it lands, and the shape of the
failure curve.

**Method.** `visual_encoder/latent_port.py`:
- Payload: C random Base32 chars (5 bits each), C/k chars assigned to each of k
  slots as ±1 bit vectors.
- Sender: per-slot MLP → 512-d, 2-layer transformer mixer across slots (lets slots
  coordinate/spread information), project to 2048-d, LayerNorm rescaled to the
  receiver's embedding RMS so injected vectors start on-manifold.
- Receiver: frozen Qwen3-VL-2B (vision tower deleted, lm_head kept). Latents are
  spliced into the chat-templated prompt at a marker; teacher-forced cross-entropy
  on the Base32 string + end token. Only sender parameters train.
- Eval: greedy decode (batch 1, no padding ambiguity), exact-sequence rate, char
  accuracy (Levenshtein), net exact bits/slot = 5C·exact/k; text-token cost of the
  same payload tokenized directly, as the baseline rate.
- Copy control: same prompt with the payload as literal text must transcribe ~100%,
  otherwise failures are about the task, not the port.

**Configs.** k=16 slots; C ∈ {16, 32, 48, 64, 96} → 5–30 bits/slot. Seeded, all
payloads held out between train and eval.

**Risks noted in advance.** (1) Sender may overfit the prompt template rather than
generalize across payloads — mitigated by fresh random payloads every step, so
train and val are both draws from the payload distribution and memorization of
specific strings is impossible. (2) bf16 gradients through 28 frozen layers may be
noisy; watch the loss curve. (3) Greedy decode length drift; cap at 2× expected.

### LP-1 build log (2026-08-10)

- **Integration bug found and fixed.** Qwen3-VL's wrapper derives multimodal RoPE
  positions from `input_ids` and crashes on pure `inputs_embeds`
  (`get_rope_index` → `NoneType has no attribute 'shape'`). Fix: pass explicit
  `position_ids` shaped `[3, batch, seq]` with identical arange planes (text-only
  mRoPE), which skips that path entirely. All three forward sites (training,
  latent decode, copy control) now do this. `visual_encoder/latent_port.py`.
- **Smoke run** (`runs/latent_port_smoke.json`, 40 steps, 16 chars): loss
  6.09 → 4.73, copy control exact, decode produces base32-shaped strings.
  Machinery verified; 40 steps is of course meaningless as a measurement.
- **Calibration surprise worth recording:** random Base32 tokenizes at ~1.42
  chars/token in Qwen3's tokenizer, so the text-token baseline is **~7.1 bits per
  sequence position**, not the 10–15 hypothesized above. The bar for "the port
  beats text" is therefore lower than expected: >7.1 net bits/slot.
- Unit tests for the pure logic in `tests/test_latent_port.py` (bit packing MSB
  order, sender output RMS matches embedding RMS, base32 normalization).
- **Sweep launched:** k=16 slots, C ∈ {16,32,48,64,96} (5–30 bits/slot loaded),
  3,000 steps × batch 16 per config (~0.34 s/step ≈ 17 min/config), fresh random
  payloads every step, 64 held-out eval samples per config, interim evals at
  1k/2k. Log: `runs/latent_port_sweep.log`, results: `runs/latent_port.json`.

### LP-1 result A — cold-start amortized sender (2026-08-10)

`runs/latent_port_coldstart.json` (16 chars → 16 slots, 5 bits/slot, 3,000 steps,
64 held-out evals): **char accuracy 18.6%, exact 0%** — far above the 3.1% chance
floor and clearly transmitting (decodes share substrings with payloads, e.g.
`JX34E46...` vs `VT34T46...`), but convergence is slow: 6.6% → 17.6% char accuracy
between steps 1k and 2k, loss plateauing ~4.4 nats vs the ≈4.9-nat
"format-only, zero-information" ceiling. Copy control exact.

**Interpretation.** Cold-start gradient descent through 28 frozen layers *does*
find the port but crawls. Consistent with the literature gap: per-payload
optimized embeddings work quickly (proto-token results) while amortized senders
need on-manifold initialization (gisting initializes from real tokens; CALM
needed deliberate redundancy). Decision: stopped the cold sweep after config 1
(remaining loads would predictably plateau harder) rather than spend ~80 GPU-min
confirming a foreseeable flat line.

### LP-1 revision B — warm-start curriculum (2026-08-10)

Added `--warmstart-steps`: phase A trains the sender by MSE to the payload
characters' *actual token embeddings* (chunk-mean for >1 char/slot) — no LLM
forward, milliseconds per step — then phase B switches to end-to-end CE. At
5 bits/slot the identity code exists exactly, so warm start should land near a
working port immediately; at 10–30 bits/slot the warm start only supplies the
manifold and CE must learn the density. Relaunched: same loads, 800 warm-start +
3,000 CE steps per config. Log: `runs/latent_port_warm.log`.

### LP-1 result B1 — warm start at 5 bits/slot + first lens (2026-08-10)

Warm start works dramatically: **89% exact, 99.24% char accuracy** at 64 held-out
evals (`runs/latent_port.json`, config 16 chars/16 slots), vs 0%/18.6% cold at the
same budget. Trajectory is reproducible across relaunches (seeded). Net exact rate
4.45 bits/slot at 5 loaded — below the 7.12 bits/token text baseline, as expected
for the identity-code config.

**Lens findings** (`runs/lens/lens_16chars.png`, `visual_encoder/port_lens.py`):
- Nearest-vocab top-1 is the *correct character* for every slot, but at cos
  0.34–0.40 (digits 0.19) — all **below** the median nearest-neighbor cos of real
  tokens (0.56). CE training drifted the slots measurably off the token manifold
  while preserving enough direction to stay legible. "Text with an accent."
- Solo-slot decodes: the frozen model reads almost every slot in isolation as its
  correct character (`P`, `D`, `J`, `V`…, with quirks like `A2345678`, `[7]`).
  At this load the code is per-slot legible, not holographic.
- PCA: slots sit displaced from the character-token cloud as a coherent group;
  the two digit slots drift together, far from the letter slots.

Open question for dense configs: does the drift *increase* with load (toward a
truly alien code) and does per-slot legibility break (holographic distribution)?

### LP-1 result B2 — 10 bits/slot: text parity, off-manifold code (2026-08-10)

**71.9% exact on 64 held-out payloads → net 7.19 bits/slot, statistical parity
with the 7.23 bits/token text baseline** at double the per-position load of the
identity config (`runs/latent_port.json`, 32 chars/16 slots; char accuracy 98.7%).

**Lens findings** (`runs/lens/lens_32chars.png`) — the phase transition happened
between 5 and 10 bits/slot:
- Nearest-vocab cosine collapsed to 0.13–0.25 (real tokens: 0.56). Most slots'
  nearest tokens are unrelated junk (` PSP`, ` yytype`, `зарегистрирова`). The
  slot↔char heatmap washed out: slots no longer point along their characters'
  embedding directions. **Geometrically the code has left the text manifold.**
- Yet solo-slot decodes still read correctly: shown one vector in isolation the
  frozen receiver says its 2-char chunk (`PP`, `VH`, `AR`, `D5`, `C6`, `Z4`,
  `CJ`, `NW`…). **Behaviorally the code is still compositional/per-slot** —
  proto-token-style: off-vocabulary points that the model's machinery maps to
  precise 2-char readouts.
- Partial BPE rediscovery: for ~1/3 of slots the nearest token IS the chunk's own
  2-char BPE token (`DW`, ` XK`, ` CJ`, `NW`), at low cosine.

Interpretation: the sender did not superpose two char embeddings (heatmap would
show two bright columns per slot); it found new off-manifold codepoints the
receiver already decodes as char pairs. Meaning is still local. The holographic
question moves to 15–30 bits/slot.

### LP-1 result B3 — 15 bits/slot: onset of holography (2026-08-10)

**42.2% exact, 97.1% char accuracy → net 6.33 bits/slot** on the all-or-nothing
metric (48 chars/16 slots). The exact-rate curve was still climbing steeply at
the 3,000-step cutoff (12.5%→31%→37.5% over the last three interims), so this is
budget-limited, not capacity-limited. At 97.1% char accuracy the soft
(pre-outer-code) information rate is far higher than the exact-packet rate — the
ECC lesson from the visual phases applies verbatim.

**Lens findings** (`runs/lens/lens_48chars.png`):
- Fully off-manifold now: nearest-vocab cos 0.19–0.26 and the neighbors are
  mostly *untrained/rare placeholder tokens* (unprintable glyphs) — the sender is
  using dead zones of embedding space.
- Slot↔char heatmap completely washed out (no character directions at all).
- **Solo-slot legibility is fracturing** — the holography onset: ~2/3 of slots
  still read alone as their chunk, several with *transpositions* (`CED`→`CDE`,
  `BJZ`→`BZJ`, `WAR`→`W4R`); ~1/4 read as a generic alphabet dump
  (`A-Z234567` = "base32-ish, can't read it alone"). Yet the full 16-slot
  sequence decodes EXACTLY. Chunk content survives per-slot; chunk *order* and
  the illegible slots' content are reconstructed only in ensemble context.

### LP-1 results B4/B5 — the cliff and the wall (2026-08-10)

**20 bits/slot** (64 chars): **4.7% exact / 93.2% char accuracy** — the exact-packet
metric collapses while the channel still carries most characters. Exemplary
failure: 64 chars perfect except one transposition (`MIVYS`→`MIYVS`). Net exact
0.94 bits/slot; soft rate far higher. This is the textbook uncoded-transmission
cliff: from here on, exactness is the outer code's job, not the channel's.

**30 bits/slot** (96 chars): **late partial escape, then collapse.** Interims sat
at the chance floor through step 2,500 (chunk-mean warm start over 6 chars is
nearly information-free, so this config effectively cold-starts), but the final
eval landed at **32.6% char accuracy** — it broke off the format plateau late.
Decodes are garbled-but-structured: recognizable chunk fragments plus
autoregressive repetition collapse (`…455455455`). A hierarchical curriculum
(grow a trained 2-char sender to 4 then 6 chars/slot) is the obvious LP-1b;
this is a training-dynamics limit, not demonstrated channel capacity.

**Lens findings at 20 bits/slot** (`runs/lens/lens_64chars.png`) — **major
revision to the holography story:** solo-slot legibility *returned*. Most slots
read alone as their full 4-char chunk (`PP7D`, `DWAR`, `XKZ4`, `BSCJ`, `TAYL`,
`RB5G`, `FBJZ`, `KLZB`, `EP5A`); the failures are *within-chunk order errors*
(`JJVH`→`JHVJ`, `XIKT`→`XKIT`, `NW4X`→`N4W3`), and the full-sequence decode
fails by exactly one such transposition. Nearest-vocab neighbors are all dead
placeholder tokens at cos 0.15–0.22. Revised interpretation: **content stays
per-slot at every working load; what degrades with density is within-slot
order binding**, not content locality. The 15-bit "alphabet dump" solos were the
transition's noisy edge, not the onset of distributed content. At 30 bits/slot
(`lens_96chars.png`) solo reads finally die (bracketed fragments, alphabet
dumps), matching the end-to-end collapse.

**LP-1 capacity curve at 3,000-step budget (net exact bits per slot):**

| loaded | 5 | 10 | 15 | 20 | 30 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| net exact | 4.45 | **7.19** | 6.33 | 0.94 | ~0 |
| char acc | 99.2% | 98.7% | 97.1% | 93.2% | floor |

Text baseline: ~7.2 bits/token. Peak of the exact-rate curve sits at 10 bits/slot
≈ text parity; the soft-rate peak is higher and rightward (15–20 bits/slot with
an outer code). Both 15 and 20 were still improving at cutoff.

**Published results page** (figures + narrative):
<https://claude.ai/code/artifact/970d6804-40a0-4d39-b0df-bea0d265ecfc>

**Queued next:** LP-1b staged-density curriculum (grow trained 2-char sender to
4/6 chars/slot); LP-1c outer code over the 15–20 bit senders (net goodput);
LP-2 document compression into slots (gisting with bit-metering); LP-3
cross-model bridge (small model's hidden states → this port).

## 2026-08-10 — LP-W: the wiretap (reading the wire without asking the minds)

**Goal.** An independent decoder that translates latent traffic to text with no
cooperation from sender or receiver at decode time — the trust layer for the
three-pane demo (chat with A | the wire, decoded | chat with B), with per-message
efficiency metering (bits delivered per context position vs the ~7.2 bits/token
text baseline).

**Method.** `visual_encoder/wiretap.py`: 2.1M-param decoder (LayerNorm → linear →
2-layer transformer over slots → per-char 32-way heads), trained on frozen-sender
traffic. **Qwen is never loaded during tap training** — independence by
construction. Trust gauges: mean max-softmax confidence (calibration binned) and
a per-slot Gaussian on-manifold z-score. Evaluated on the *identical* held-out
payload stream as LP-1's receiver evals. Taps checkpointed in `runs/wiretaps/`.

**Results** (`runs/wiretap.json`, `runs/wiretap_96.json`):

| loaded bits/slot | 5 | 10 | 15 | 20 | 30 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| tap exact | 100% | 100% | 98.4% | **100%** | 0% |
| tap chars | 100% | 100% | 99.97% | 100% | **95.4%** |
| receiver exact | 89% | 72% | 42% | 4.7% | 0% |
| receiver chars | 99.2% | 98.7% | 97.1% | 93.2% | 32.6% |

**Major reframe of LP-1:** the "capacity curve" was a *receiver-readout* curve,
not a channel-capacity curve. The sender's vectors carry essentially everything
at every load — 320 bits in 16 vectors losslessly at 20 bits/slot, ~95% of 480
bits even from the "collapsed" 30-bit sender — and a 2M-param tap recovers it
where the 2B frozen receiver cannot. This also confirms the binding-error story:
the information (including order) is in the wire; Qwen's generative readout loses
it.

**Honest caveat.** A deterministic bit-packing sender is an invertible map, so
*some* decoder must exist; the findings are (a) how *small* a decoder suffices,
(b) the size of the tap–receiver gap, and (c) that the gauges work. For LP-3
(hidden-state senders) tap-readability stops being guaranteed and becomes the
experiment.

**OOD detection works.** Own-protocol traffic: z ≈ 1.00, confidence ≥ 0.999.
Drifted protocols (other senders' traffic): confidence 0.44–0.89, z 1.6–4.3.
Noise jamming: z 2.3–4.2. Decision rule `confidence < 0.99 or z > 1.5` separates
every tested case — silent mistranslation is a detectable event, at least for
protocol shifts of this magnitude.

## 2026-08-10 — LP-3a: cross-model bridge, and three bugs worth remembering

**Setup.** Frozen Qwen3-0.6B ("A") reads `The access code is: <payload>.
Memorize it exactly.`; a trained bridge maps A's hidden states to 16 slots in
frozen Qwen3-VL-2B ("B"); B transcribes. Only the bridge (8.7M params) trains.
`visual_encoder/latent_bridge.py`.

**Debugging chronicle** (each cost real time; each is general):
1. **Tokenizer position jitter.** Qwen tokenizes ` X` as one token for letters
   but ` 3` as two for digits, so payload positions shifted per sample and the
   perceiver resampler could never align. Fix: bijective digit→letter rendering
   in A's text (`DIGIT_RENDER`), giving one char = one token = one fixed
   position (all inputs exactly 43 tokens). Diagnostic that found it: print the
   token pieces. A linear probe then reads chars from A's states at 100% at
   *every* layer (0, 8, 28) — the information was never the problem.
2. **Perceiver attention collapse.** Even with fixed positions and learned key
   positions, the cross-attention resampler converged to an input-independent
   constant: output variance across samples 0.000000, MSE parked exactly at the
   constant-prediction floor (= target variance 3.2e-4). The per-dim payload
   signal is tiny next to the shared mean, so softmax attention locks onto the
   constant prefix keys (attention-sink local optimum) and never escapes at
   these step counts. Perceiver kept as `--bridge perceiver` for future study;
   default switched to a position-wired `GatherBridge`.
3. **Outlier-dimension domination.** GatherBridge still stalled: A's hidden
   states carry near-constant outlier dims at ~30× the median scale
   (max |mean| 31.6 vs median 1.0), which dominate a joint LayerNorm and its
   gradients. Fix: per-dimension standardization from calibration statistics
   (buffers, not learned). Warm-start MSE then fell 0.00173 → 0.00001 and CE
   dropped 4.6 → 1.1 within 60 steps.

**Results** (`runs/latent_bridge.json`, 64 held-out evals):
**75% exact, 98.68% char accuracy** at 10 bits/slot — statistically identical to
LP-1's same-load direct sender (71.9%/98.75%). A frozen 0.6B's hidden states,
resampled by an 8.7M-param bridge into 16 vectors, drive the frozen 2B receiver
as well as a purpose-built bit-packer does. Positional compression vs A's own
text: 43 tokens → 16 slots (2.7×). Trajectory: 37.5% → 31% → 69% → 81% → 87.5%
interim exact (16-sample noise), 75% final on 64.

**Bridge wiretap** (`runs/wiretap_bridge.json`): the independent 2.1M tap reads
the mind-driven wire at **100% exact / 100% chars** — tap-readability survived
the move from bit-packed to representation-derived traffic (not guaranteed a
priori; now measured). OOD gauge separates protocols that carry the *same
content type at the same load*: own protocol z=1.01 / conf 0.99998; the LP-1
sender's traffic (same payloads, different protocol) z=1.63 / conf 0.987 —
flagged by both criteria of the `conf < 0.99 or z > 1.5` rule; jamming z=2.49.

**LP-3a conclusion.** The psychic port between two different frozen minds works
at text-density, and its traffic is independently interceptable with working
drift detection.

## 2026-08-10 — LP-2: the free-text port, v1→v2

**v1** (48-token window, 3 tokens/slot, wikitext-only): plateaued ~51% chars;
novel-template = trained-template (50%), so context generalization works and
per-slot capacity was the limit. Archived as `runs/text_bridge_v1_48window.json`.

**v2** (32-token window, 2 tokens/slot — LP-3a's proven geometry — corpus 50%
wikitext + 50% synthetic chat, 6k steps): wikitext val 71% chars / 0% exact with
a *whisper* signature — structure and common words survive nearly verbatim, rare
proper nouns and digits smear (`Jasaw Chan K'awiil` → `Josafat Kan'tai`,
`7429` → `7424`). Chat register: 73% chars with near-paraphrase fidelity.
**Comprehension through the port: 3/4** — B correctly answers questions about
what it heard (orange ✓, meeting time ✓, octopus ✓; blue flowerpot → "green
flowerbox" ✗). Novel template ≈ trained (70.4% vs 71.1%).

**Text tap reversal.** On free text the independent tap (55% chars) is now
*worse* than B (71%) — opposite of the Base32 regime — because B's language
prior error-corrects common tokens and the tap has no LM. Fix queued: pipe tap
tokens through a small LM smoother (user's suggestion). Gauges remain excellent:
legit z ≈ 1.0, jamming z ≈ 89.

**Live demo verified** (`psychic_demo.py` v2, port 8766): "remember the word
orange" → A replies "**orange**" → 16 vectors cross → B hears "Orange" → asked,
B answers "Orange". Fixed 16-vector cost per message means density wins only
for messages >16 tokens (1.4–2.5× at 23–40 tokens); short messages cost more
than text — an honest limitation of the fixed-slot design.

## 2026-08-10 — LP-2 v3: 4B sender, complete-register corpus, live demo verified

**Changes.** Sender upgraded 0.6B → Qwen3-4B (0.6B refused relay tasks —
sender-model competence, never the port). Corpus completed: wikitext + synthetic
chat + secret strings (25%) + assistant-register lines + **485 harvested genuine
A replies (×2)** — training the port on its actual sender's voice. Demo: multi-
packet rollout (long replies stream as numbered 16-vector frames, all packets
live in B's context), one-clause port-mechanics addition to A's system line.

**Committed artifact** (`runs/text_bridge.json` v3, 4B sender): wikitext val
**71.6%** chars (trained template), **72.4%** (novel template), first exact
packets. This is the only artifact-backed LP-2 v3 fidelity number.

**PROVENANCE CORRECTION (external review, 2026-08-10).** An earlier version of
this entry cited "chat register 77.7%" against `text_bridge.json` — but that
number came from an *uncommitted* scratchpad battery (`port_tests.py`) on a
*different eval set* (synthetic chat, not wikitext), and the committed artifact
neither contains it nor recorded its sender model. That violated the repo's
"every claim cites its artifact" standard. The 77.7% chat-register figure is
**withdrawn pending a committed, self-describing battery artifact** (re-run
queued: saves sender model, git SHA, seeds, and all held-out predictions, not 3
examples). Treat only the 71.6/72.4% wikitext numbers as backed.
Password/comprehension anecdotes (`A20h9jt6eGa`→~8/11 chars; 2–6 comprehension
items) are indicative only — far too few samples to estimate a rate.

**Live demo verification** (port 8766, all real API transcripts): orange →
verbatim crossing → B answers "orange". Password → B answers `a20h9t2e` (7/11;
two identified fixes: packetizer split the string mid-password at the token-26
boundary; boilerplate packet echoed prior context). Multi-packet: 2-frame
message, both LEGIT, B *summarizes* correctly ("meeting Tuesday 3pm, key under
the blue fruit pocket" — the whisper's poetry).

**Published:** <https://github.com/rhowardstone/latent-port> (private).

## 2026-08-11 — Overnight results: receiver-LoRA (biggest lever), bidirectional, scaling

**Receiver LoRA vs frozen (the headline).** `runs/receiver_lora.json`, 20 bits/slot
(the load where frozen B fails). A 17.4M-param LoRA on B's decoder, co-trained with
the sender: **exact 0% (frozen, reliable binding baseline) → 82.8%; order error
8.1% → 0.1%.** Training the receiver breaks the frozen ceiling and erases the
residual binding limit — confirming the wiretap-gap prediction (the info was always
in the vectors; frozen B just couldn't use it). (This run's *in-run* frozen baseline
read anomalously low — 7.6% char, a restart artifact — so compare against binding's
reliable frozen ~0% exact / 84% char.) Biggest lever, decisively.

**Bidirectional self-port (LP-3b).** `runs/bidirectional.json`: one shared-weight
Qwen3-4B, both directions via one bridge. 57% char, 0% exact, generalizes
(trained ≈ novel template ≈ 0.57). A real two-way vector link, but notably lossier
than the one-way 4B→2B port (75%) — worth a follow-up.

**Scaling sweep — COMPLETE two-axis result** (`runs/scaling/*.json`; wide_picture,
coherent contiguous passages after fixing the random-concatenation bug that stalled
training flat at 22%). The two knobs behave oppositely:

**Image size** (vectors per picture; density fixed at 2 tok/slot):

| slots | message | char fidelity |
| ---: | ---: | ---: |
| 16 | 32 tok | 0.791 |
| 32 | 64 tok | 0.867 |
| 64 | 128 tok | 0.858 |
| 128 | 256 tok | 0.913 |

Scales — even *improves* — as the picture grows 8×. Longer messages aren't a
problem; just use more slots. Each slot = one KV position, so cost is linear in
positions, not fidelity.

**Density** (tokens packed per vector; slots fixed at 32):

| tok/slot | message | char fidelity | trained? |
| ---: | ---: | ---: | --- |
| 1 | 32 tok | 0.980 | climbed 0.78→0.98 |
| 2 | 64 tok | 0.867 | climbed 0.49→0.87 |
| 3 | 96 tok | 0.258 | FLAT 0.24→0.26 |
| 4 | 128 tok | 0.258 | FLAT 0.27→0.26 |

**Sharp cliff between 2 and 3 tok/vector.** 1–2 tok/vector train fine (climb); 3–4
plateau at ~0.26 from step 500 (never learn beyond the warm-start floor — 3 and 4
land at identical 0.258). Consistent with the ~2-tokens/vector frozen ceiling seen
throughout. Honesty caveat: at ≥3 tok/slot the mean warm start is weak (averaging 3+
token embeddings is near information-free — the mechanism that collapsed LP-1's
densest configs), so "fundamental frozen limit" vs "recipe can't reach it" is not
cleanly separated. The receiver-LoRA result (0%→83% at high density) is the lever
that should move this cliff.

**Practical rule:** *widen the picture freely (scales, ~linear in positions); don't
pack past ~2 tokens/vector unless you train the receiver.* The scaling cliff and the
LoRA win are two views of the same ceiling.

Note (density-axis confound): message length grows with density here (slots fixed),
but the image-size axis shows longer messages don't hurt (improve), so the density
drop is attributable to packing, not length.

**Reproducibility packaging (2026-08-11).** Trained artifacts shipped as a GitHub
Release (`trained-artifacts-2026-08-11`, 19 files / 463MB) — see `ARTIFACTS.md` for
what each is, a load-and-use snippet, and regenerate commands. `peft` added to deps;
`wide_picture`/`receiver_lora` now persist their trained bridge/adapter. Every result
JSON embeds `provenance` (git SHA + CLI args + versions); CI is green.


**Large-picture extension (2026-08-12).** Pushed image size past 128: 256 vectors
(512-tok msg) plateaued flat at 0.26 from step 500 (stuck at warm-start floor — a
RECIPE limit: batch dropped to 4 for memory, undertrained for the harder task; NOT
a proven fidelity ceiling), and 512 vectors OOM'd (needed 18.5GB it didn't have —
a memory limit, needs grad checkpointing / smaller receiver). So clean scaling is
CONFIRMED to 128 vectors / 256-tok msg (0.91); 256+ hits engineering walls (train
budget + memory), not a model ceiling. A better-resourced retry (grad accumulation,
more steps) is needed to find the true limit.

## 2026-08-11 — Clarification: slot count is a free choice, not a limit (plan change)

Realization (prompted by the user): "16 vectors per picture" was never a ceiling —
it's a design choice. A real image → vision tower → *a few hundred* patch-tokens
(one per ~28px patch), which is why a rendered-text image holds ~2000 chars. Our
latent port DELETED the vision tower and injects our own vectors; we picked 16 slots
as a small unit to measure per-vector capacity cleanly. Two INDEPENDENT axes, which
the "picture" metaphor had been blurring:

- **slots per picture = free.** 16 was a thumbnail. Use 64/256 for longer messages;
  each slot costs one KV position, exactly like an image's patch-tokens.
- **per-vector density (~2 tokens/vector) = the real frozen-receiver ceiling.** This
  is what the density experiments actually measure, independent of slot count.

**Plan change.** For long messages there are two routes: (a) multi-picture **rollout**
— a *sequence* of 16-vector pictures (what `rollout_bridge.py` trains now); B tends
to fixate on picture 1 because reading a sequence is OOD/hard. (b) one **wide
picture** — more slots (e.g. 64) injected as ONE contiguous block, same 2 tok/slot
density; no sequence, so no fixation. The wide picture is likely the *cleaner* path
and directly matches how a real image works (one big block). The running rollout run
is the test of route (a); if it plateaus, build route (b) — a 64-slot / 128-token
`text_bridge` (needs WINDOW parameterized, deferred so it doesn't destabilize the
running jobs). Also: re-baseline efficiency vs rendered-text at a COMPARABLE budget
(a few hundred tokens), not against our 16-vector thumbnail (A13).

## 2026-08-11 — Binding control VERDICT (external-review A4)

Ran the order-preserving warm-start sweep at 20 bits/slot (`runs/binding/*.json`),
the load where content is ~95% present but misordered. Order-error dose-response:

| warm start | 20b exact | content | order_err |
| --- | ---: | ---: | ---: |
| mean (ε=0, confound) | 0% | 94.5% | 8.1% |
| blend ε=0.1 | 0% | 95.2% | 7.2% |
| blend ε=0.3 | 7.8% | 98.2% | 2.9% |
| role (ε=1) | 7.8% | 98.5% | 2.9% |

**Verdict: partly artifact, partly real — both the finding and the critique were
right.** (1) The reviewer was correct that the permutation-invariant `mean` warm
start *inflated* the transpositions: order-preserving init cut order error 8.1%→2.9%
and recovered exact 0%→7.8%. (2) But a **real residual** survives the best init —
2.9% order error and only 7.8% exact remain, so there is a genuine (smaller)
frozen-receiver binding limit underneath. (3) It **plateaus at ε=0.3**; full `role`
(ε=1) does not beat it — exactly the reviewer's prediction that pure orthogonal
role-binding lands off B's native geometry. The ε-sweep (not a binary mean-vs-role
test) is what separated artifact from residual. The residual 2.9% is the handoff to
`receiver_lora.py` (train B too), queued at the same 20-bit load tonight.

## 2026-08-11 — Token savings, measured honestly (LP-2)

`visual_encoder/token_savings.py`, 96 held-out messages, 4B→2B, `channel_metrics`
with cluster-bootstrap CIs (`runs/token_savings.json`):

- **Content transfer is REAL (causal control):** ΔI(correct vs deranged latent) =
  **150.8 bits/msg**, CI [142.7, 159.0] — excludes 0 decisively; every message
  positive. correct-vs-zero = 133 bits. So B receives example-specific info from
  the vectors, NOT prior hallucination (external-review A5 control passed).
- **Position compression: 1.15×** (CI 1.06–1.24) for ~18-token messages. It is
  L/16 — break-even ~16 tokens, grows with length. Modest and **lossy**.
- **Fidelity: 83.3%** (CI 80.0–86.7) on this wikitext+chat mix (higher than the
  71.6% wikitext-only because chat is shorter/in-distribution).

**Verdict:** the port genuinely transmits (proven), and saves receiver *context*
positions modestly at a fidelity cost — context/KV compression, not a free lunch,
and not the generation-step savings (that's LP-4 self-loop). Bug caught+fixed:
first run reused base32 `normalize()` on free-text decodes → false 6% fidelity;
raw-text comparison gives the real 83%.

## 2026-08-10 — LP-5a launched: the legibility tax

Question: what does auditable traffic cost? `--legibility-weight λ` co-trains a
reference tap with the bridge; total loss = B's CE + λ·tap CE. Sweep λ ∈
{0.3, 1, 3} vs the λ=0 baseline (v3): measure B fidelity (capacity) vs
co-trained tap fidelity (auditability) per λ. Running; results below.
Queued next: LP-5b (evolve unsupervised native code; test post-hoc translation
from behavioral grounding only), LP-4 (latent rollout: A emits vectors without
generating text — the output-token elimination), v4 demo fixes (variable packet
sizes, quote-aware packetizer, tap LM-smoother).

## 2026-08-10 — The three-pane demo is live (v1, Base32 protocol — superseded by v2 above)

`visual_encoder/psychic_demo.py` (port 8766): chat with A (0.6B) | THE WIRE |
chat with B (2B). Every utterance is generated by the real frozen models; every
transmission is the trained bridge resampling A's hidden states into 16 vectors;
the middle pane is the independent tap reading those vectors live, with the
LEGIT/ANOMALOUS gauge (confidence + on-manifold z) and the per-message
efficiency meter (23 text tokens → 16 vectors, 10 bits/vector vs ~7.2 for
text). First end-to-end run: secret → A → wire (tap exact, LEGIT, z 0.98) →
B receives exact → B answers follow-up questions about the code from its own
decode. Honest scope: the trained protocol carries 32-char Base32 payloads;
semantic traffic is LP-2/LP-3b.

## 2026-08-12 — LP-4: latent macro-steps (can one learned step skip Δ>1 tokens?)

**Question.** A frozen LM's last-layer state h_t predicts the next token. Train a
small residual net g: h_t → h_{t+Δ} and decode g(h_t) through the model's OWN frozen
head. Does one learned step jump the model Δ token-generation steps ahead (temporal
abstraction) — or only work at Δ=1 (a codec)? Honest bar: the no-op baseline
decode(h_t)→token_{t+Δ}, i.e. how far the raw state already sees. Sanity:
decode(h_t)→token_{t+1} must be ~1.0 (`runs/latent_rollout.json`, Qwen3-0.6B).

| Δ | macro jump | noop baseline | ratio | recursive (chained) | state cos |
|--:|--:|--:|--:|--:|--:|
| 1 | 0.472 | 0.146 | 3.24× | 0.050 | 0.75 |
| 2 | 0.335 | 0.143 | 2.35× | 0.080 | 0.70 |
| 4 | 0.271 | 0.161 | 1.69× | 0.080 | 0.68 |
| 8 | 0.270 | 0.174 | 1.55× | 0.130 | 0.67 |

Sanity noop_next = 0.986 at every Δ (readout valid). pred_cos (0.87→0.73) > state_cos
(0.75→0.67): g genuinely moves the state toward the future, not just copying h_t.

**Verdict — partial yes, but it does NOT chain.** The single learned jump carries
real look-ahead: it beats the no-op baseline at every Δ (1.55–3.24×), so a lone
vector step does predict a token multiple generation-steps ahead better than the raw
state does. BUT (a) absolute accuracy is modest and decays with Δ (47%→27%), so the
jump is lossy, and (b) crucially, chaining collapses — recursive roll-out (h_0→h_Δ→
h_2Δ…) lands at 5–13%. You cannot stack macro-steps to roll a whole message ahead in
latent space; the horizon dies after ~2 jumps. This is exactly the analog-drift the
Coconut/CALM literature warned of: text tokens act as error-correcting repeaters that
a pure-latent chain lacks. So the honest answer to "just emit vectors and skip
generation" is: one hop is real but weak, and it doesn't compound into a rollout.
A weak accelerator / lossy codec, not strong temporal abstraction.

## 2026-08-12 — LP-6: music through the port (IrishMAN ABC) — built, H1 queued, H2 blocked

**Idea (lab lead).** Transmit music through the latent port. Permissively-trainable
(IrishMAN: 216k folk tunes, MIT/believed-PD, already in ABC notation). Two hypotheses:
H1 more bits ride through via music than text; H2 latent rollout (LP-4) chains better on
music (a strong musical prior resists the analog drift that killed text rollout).

**Built & CPU-validated (`visual_encoder/music_port.py`, `latent_rollout.py --music`).**
- Serialization = **ABC notation** — the one music text-format a web-trained LM has a
  prior for (cf. ChatMusician). Frozen Qwen3-VL-2B re-emits the ABC injected as k vectors.
- Metric parses both sides to note events (music21) and reports, separately: order-aware
  **pitch fidelity**, content-only **pitch-class histogram cosine** (catches the "generic
  plausible melody" failure the way `content_vs_order` catches order-loss at 20 bits/slot),
  note-F1, transpose-aligned (wrong key ≠ garbled). Validated on known cases + `pytest`.
- **ΔI causal control** (vs deranged tune / zero) built in — the *honest* H1 adjudicator,
  because music's redundancy means note-fidelity can RISE while Shannon bits/slot FALL.
  H1 is "true" only if music wins on ΔI/slot, not on note-F1.
- Fix worth recording: ABC is line-oriented; collapsing to one line (the prose habit) makes
  music21 parse **zero** notes. Keep the newlines.

**H1 status:** queued behind the large-picture retries (`nightshift3_music.sh`, waits for
the GPU). A `copy_control` (frozen model echoes literal ABC) will tell us if the receiver
can read ABC at all or needs a receiver-LoRA.

**H2 blocked — honest finding (CPU probe).** Base Qwen3-0.6B **cannot generate real ABC**:
it degenerates to alphabet runs (`e2 f2 g2 h2 i2…`, past 'g' these aren't ABC notes —
music21 silently assumes C) or literal repeats (`|: G2G B2B` ad infinitum). So H2 on a base
model is vacuous — it needs a music-trained model (ChatMusician-7B). The `--music` code is
ready and self-diagnoses via `abc_valid_fraction`; H2 deferred until that model is pulled.
