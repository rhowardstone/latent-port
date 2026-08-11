# latent-port

![tests](https://github.com/rhowardstone/latent-port/actions/workflows/tests.yml/badge.svg)

**Can one frozen language model speak directly into another's mind — in vectors,
not tokens — and can a third party read the wire?**

This repo is a working lab for *latent ports*: trained bridges that inject
learned embedding vectors into a frozen LLM's context (occupying one sequence
position each, like tokens, but denser and never drawn from a vocabulary),
plus independent *wiretaps* that translate the traffic without asking either
model, with calibrated trust gauges.

Everything here runs on a single A100-40GB with small open models
(Qwen3-VL-2B receiver, Qwen3-0.6B/4B senders). All experiments are seeded;
every claim in the docs cites a JSON artifact in `runs/`.

**Read first:** [`NOTEBOOK.md`](NOTEBOOK.md) — the full chronological lab
notebook. **For reviewers:** [`OPEN-PROBLEMS.md`](OPEN-PROBLEMS.md) — the open
questions stated mathematically, with measured constants and proposed
experiments. **For the "why":** [`docs/VISION.md`](docs/VISION.md) — the
conceptual arc, the self-port rollout and MoE-colony directions, and the safety
frame. Review records live in [`reviews/`](reviews/).

## Headline results (as of 2026-08-10)

| Finding | Number | Artifact |
| --- | --- | --- |
| Random-bit capacity through a frozen receiver peaks at | **7.19 net exact bits/position** (≈ text density) at 10 bits/slot load | `runs/latent_port.json` |
| The vectors physically carry more than the frozen receiver reads | tap decodes **100%** at 20 bits/slot where receiver gets 4.7% | `runs/wiretap.json` |
| Cross-model port (0.6B hidden states → 2B receiver) | **75% exact** at 10 bits/slot — matches the direct sender | `runs/latent_bridge.json` |
| Free-text port (arbitrary sentences, 2 A-tokens per vector) | **71.6/72.4%** char fidelity (wikitext, trained/novel template); meaning survives, rare tokens smear | `runs/text_bridge.json` |
| Independent wiretap on all bit-packed protocols | ~100% decode; drift z-gauge separates protocols at z ≥ 1.6 vs 1.0 | `runs/wiretap*.json` |
| Content stays *per-slot* at every working density; what fails is within-slot **order** | transposition-dominated errors at 20 bits/slot | `runs/lens/` figures |
| Live two-model demo with tapped wire | orange/password/multi-packet transcripts | `visual_encoder/psychic_demo.py` |

Earlier arc: the same lab measured the *visual* channel (images into the vision
tower) — depth-resolved probes localized where information dies in the encoder
and decoder, DeepStack streams included. See the notebook's phases 1–4.

## Layout

```
visual_encoder/
  codec.py         deterministic image codecs (CRC-framed packets; controls)
  patterns.py      spread-spectrum patch codes for the visual-era probes
  qwen_probe.py    frozen-encoder capacity probe (ridge, bootstrap, caching)
  llm_probe.py     depth-resolved probes through the frozen decoder
  text_baseline.py rendered-text end-to-end OCR control
  latent_port.py   LP-1: bit-packing sender -> frozen receiver + lens utils
  port_lens.py     interpretability lens (nearest-token, PCA, solo decodes)
  wiretap.py       independent decoders + confidence/on-manifold gauges
  latent_bridge.py LP-3: cross-model bridge (A's hidden states -> B's context)
  text_bridge.py   LP-2: free-text port + LP-5a legibility-tax option
  text_tap.py      token-level wiretap for free-text traffic
  psychic_demo.py  the three-pane live demo (chat A | tapped wire | chat B)
  web.py           the original visual-channel interactive lab
tests/             CPU-safe unit tests (helper-level; see CI badge)
runs/              JSON results, lens PNGs (checkpoints/logs gitignored)
NOTEBOOK.md        chronological findings — the primary document
OPEN-PROBLEMS.md   mathematically-framed open questions for collaborators
```

## Reproduce

```bash
pip install -e .[dev]           # torch, transformers, fastapi, datasets needed
python -m pytest -q             # CPU-safe unit tests
python -m visual_encoder.latent_port          # LP-1 capacity sweep
python -m visual_encoder.port_lens            # lens figures from checkpoints
python -m visual_encoder.wiretap              # taps + gauges
python -m visual_encoder.latent_bridge        # LP-3a cross-model bridge
python -m visual_encoder.text_bridge --sender Qwen/Qwen3-4B   # LP-2 free text
python -m visual_encoder.text_tap  --sender Qwen/Qwen3-4B
python -m visual_encoder.psychic_demo --port 8766             # the demo
```

Models download from Hugging Face on first use. Seeds are fixed throughout;
payloads are held out by construction (train and eval draw from disjoint
seeded streams).

**Test coverage is honest about its scope:** the CPU test suite covers pure
helpers (bit packing, probe algebra, codecs, the manifold gauge, the
binding-control metrics) — *not* the headline experiments, which need a GPU and
downloaded models. Treat "tests pass" as helper-level regression, not validation
of the capacity claims; those live in `runs/` artifacts and are reproduced by the
`python -m` commands above (CI runs the CPU suite on every push — see the badge).
A GPU smoke-suite is still an open task. Note: LP-2 fidelity is the
artifact-backed wikitext number (71.6/72.4%); an earlier chat-register figure was
withdrawn for a provenance gap. Every experiment JSON now embeds a `provenance`
block (git SHA, full CLI args, package versions) via `visual_encoder/provenance.py`.

## Reviews & collaboration

External review findings and their resolutions are tracked in
[`reviews/`](reviews/). Collaborating instances: read the latest review file and
`OPEN-PROBLEMS.md` before starting, and add a review file rather than editing
someone else's.

## Safety framing

This lab treats *legibility of AI-to-AI communication* as a measurable property,
not a hope: every channel ships with an independent decoder and drift detection,
and the current experiment arc (LP-5) measures what that auditability costs.
Learned codes are treated as untrusted data; decoded content is displayed, never
executed.
