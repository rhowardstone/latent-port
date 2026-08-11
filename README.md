# Visual Channel Lab

> **Lab notebook:** chronological experiments, results, and decisions live in
> [`NOTEBOOK.md`](NOTEBOOK.md), including the latent-port phase
> (`visual_encoder/latent_port.py`) that injects learned embeddings into the
> frozen decoder directly, with no pixels involved.

Visual Channel Lab measures how much exact information survives a vision-language
model's image bottleneck. It separates three quantities that are easy to conflate:

1. **File capacity:** bytes recoverable from an untouched lossless image.
2. **Robust image capacity:** bytes recoverable after JPEG, resizing, blur, and noise.
3. **VLM representation capacity:** held-out random bits recoverable from frozen
   visual-token embeddings.
4. **End-to-end VLM rate:** random characters the causal language model can
   actually transcribe from an image.

Natural language is not used as the probe payload. Random held-out bits cannot be
guessed from semantics, spelling, or redundancy, so their recovery is direct evidence
that the image channel preserved them.

## What the controls mean

A 512×512 RGB bitmap physically stores 786,432 bytes. Qwen3-VL merges that image to
roughly 16×16 = 256 visual tokens, making the untouched file-level ceiling 24,576
bits per visual token. That is only an accounting ceiling: resizing, normalization,
patch projection, the ViT, and finite-precision activations can discard nearly all of
those low-order pixel distinctions.

The `raw` codec writes packet bytes directly into RGB values. It should round-trip a
PNG and fail almost immediately under JPEG. The `palette` codec uses separated color
cells and block-interleaved repetition, explicitly trading density for robustness.
Both codecs use a framed payload with original length, optional zlib compression, and
CRC-32 verification.

## Local interface

```bash
python3 -m visual_encoder.web --port 8765
```

Open <http://127.0.0.1:8765>. The interface supports live codec, density, JPEG,
resize, blur, noise, and repetition controls, plus an automatic JPEG sweep.

## Frozen-Qwen capacity probe

```bash
python3 -m visual_encoder.qwen_probe \
  --bits 1,2,4,8,16,32 \
  --feature-views final,deepstack_0,deepstack_1,deepstack_2 \
  --output runs/qwen_probe.json
```

The probe:

- renders independent random bits into spread-spectrum micro-patterns aligned with
  Qwen's effective merged patches;
- holds total per-patch modulation energy approximately constant as bit load grows;
- logs the fraction clipped by 8-bit RGB quantization;
- freezes Qwen3-VL-2B's visual tower and removes the unused language model;
- caches all final and Qwen3 DeepStack feature streams, so alternative probes do
  not rerun the expensive encoder;
- selects a closed-form ridge regularizer on a calibration split;
- reports untouched validation accuracy, image-cluster bootstrap intervals, exact-token accuracy,
  per-image bit-error histograms, and the train/validation gap.

The default 16×16 grid produces 256 probe rows per encoder forward. With 64 training
images this gives 16,384 rows for 2,048-dimensional Qwen embeddings—eight rows per
feature rather than interpolation territory.

This is a **local, position-invariant lower bound**. Qwen's ViT mixes patches globally,
so a receiver attending jointly to every image token may decode distributed
information that a shared per-token probe cannot recover. Conversely, successful
recovery from the frozen visual representation does not yet prove that the causal LLM
can reason from the code; that requires a separate adapter/LoRA experiment.

Qwen3-VL injects three intermediate DeepStack streams into later language-model
layers. Comparing `final` against `deepstack_0..2` localizes where the fixed code
is lost and avoids measuring only one subset of the representation received by
the language model. It still does not show that the unadapted language model can
decode that representation.

To trace the same cached features through the frozen causal decoder:

```bash
python3 -m visual_encoder.llm_probe \
  --bits 4,8 \
  --layers 0,1,2,3,4,6,8,12,16,20,24,27 \
  --output runs/qwen_llm_probe.json
```

`llm_input` is the final vision embedding before any decoder block;
`before_layer_1..3` are immediately after Qwen's three DeepStack additions. Later
points show how much token-local information remains linearly accessible as the
causal decoder transforms the visual positions. The probe removes the vision tower
and language-model head before GPU transfer and reuses cached visual features, so a
depth sweep takes seconds rather than minutes.

The reported `bsc_equivalent_bits_per_token = b·(1-H₂(BER))` is a useful common
scale, not a claim of Shannon capacity: errors from tokens in the same image are
correlated. Image-cluster bootstrap intervals and an empirical outer-code test are
needed before treating the rate as operational.

### Current clean-channel snapshot

The following held-out results use 32 training images (8,192 token rows) and 16
validation images. Values are BSC-equivalent bits per visual token, not certified
operational rates:

| Loaded bits/token | Final vision | DeepStack-0 | After first LLM injection | Final LLM |
| ---: | ---: | ---: | ---: | ---: |
| 4  | 2.155 | 4.000  | 3.987  | 0.702 |
| 8  | 1.795 | 7.841  | 7.219  | 0.380 |
| 16 | 1.297 | 13.885 | 9.939  | 0.152 |
| 32 | 0.899 | 18.546 | 10.453 | 0.087 |

The fixed code is preserved far better in the earliest DeepStack feature than in
the final vision embedding. Qwen injects that clean feature after its first causal
decoder block, but the decoder progressively erases token-local linear
recoverability. At the tested loads, the first post-injection state plateaus around
10 bits/token. The next experiment should therefore adapt the receiver and test
exact end-to-end decoding, rather than continue optimizing a sender against the
final vision embedding.

## End-to-end rendered-text control

```bash
python3 -m visual_encoder.text_baseline \
  --characters 64,128,256,512 \
  --output runs/rendered_text.json
```

This renders uniformly random Base32 rather than English, then asks the complete
Qwen vision-to-language path to transcribe it. Every source character carries five
known bits and cannot be repaired from linguistic context. Character accuracy is an
uncoded measurement; exact recovery requires an outer code and should be reported
at the resulting net rate. This is the practical on-manifold baseline a learned
visual sender must beat.

## Tests

```bash
python3 -m pytest -q
```

The project intentionally treats learned human-illegible codes as untrusted data.
Decoded content is displayed as text and is never executed.
