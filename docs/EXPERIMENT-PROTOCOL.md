# GPU experiment protocol

Standard for every GPU job (A100 node or Colab), so a runtime death loses at most
one condition and any agent can interpret an artifact without notebook state.
Converged from both CPU agents' proposals.

## Every GPU notebook / script must

1. Clone the repo, `git checkout <explicit SHA>`, print dirty status, **abort if
   unexpected** (assert the SHA it was written against).
2. Record the full environment: `provenance(args)` (git SHA + all CLI args +
   package versions) plus CUDA/GPU/model/tokenizer/dataset revisions.
3. **Checkpoint after each experimental condition**, never only after the whole
   sweep.
4. **Resume by detecting completed-condition manifests** — `if result_exists(cfg):
   skip`.
5. Save predictions and loss curves, not only summary statistics.
6. Use deterministic result directory names.

## Result directory layout

```
runs/<node>/<experiment>/<run-id>/
  manifest.json        # run-level: experiment, SHA, args, status, condition list
  environment.json     # provenance() output
  conditions/
    <cond-1>.json       # per-condition metrics + status (the resume unit)
    <cond-2>.json
  predictions.jsonl     # every held-out prediction, not 3 examples
  metrics.json          # aggregated, with bootstrap CIs
  checkpoint.pt         # resumable weights
  stdout.log
```

`<node>` ∈ {`a100`, `colab`, `cpu`}. `<run-id>` is a passed-in timestamp/uuid (do
not call `Date.now()` inside a workflow script; pass it in).

## Metrics discipline (see `channel_metrics.py` once it lands)

- Report packet-exact **goodput** `G(b)` AND the variational rate `R_var`
  separately — they are different quantities (never both as `R(b)`).
- **LP-1** (uniform Base32, `H(X)=5C` known): `I_var = H(X) + E[log₂ q(X|Z)]`.
- **LP-2** (free text, `H(X)` unknown — do NOT use token count):
  `ΔI = (NLL_null − NLL_correct)/ln2`, paired against `Z_deranged`,
  `Z_moment-matched`, `Z=0`. This unifies the variational-rate and causal-control
  lanes.
- Confidence intervals by **message-cluster bootstrap** (resample whole messages;
  intra-message errors are correlated). Record sample count + bootstrap seed.
- Empty/malformed decodes score zero, never raise.

## Handoff back

Emit a compact JSON suitable for committing to git (small metrics + manifest;
large checkpoints to Drive/artifacts). Update the `handoffs/` entry to `done` with
the artifact path so the next node can pick it up.
