# Trained artifacts

Trained model weights for **downstream use without retraining**. They are NOT in
git (they'd bloat history) — they're attached to a GitHub Release.

## Download

```bash
gh release download trained-artifacts-2026-08-11 -D /tmp/lp-artifacts
# then place under runs/ per the paths below, e.g.:
mkdir -p runs/bridges runs/wiretaps runs/senders
mv /tmp/lp-artifacts/lp2_text_16slots.pt runs/bridges/
mv /tmp/lp-artifacts/tap_text_16slots.pt runs/wiretaps/
# ...etc (assets are flat; the table says which subdir each belongs in)
```

Everything here is also **regenerable from source** via the `python -m` commands
in the README (seeds fixed; each result JSON embeds a `provenance` block).

## What each artifact is

| asset | put in | what it is | produced by |
|---|---|---|---|
| `lp2_text_16slots.pt` | `runs/bridges/` | **flagship free-text bridge**: Qwen3-4B states → 16 vectors → frozen Qwen3-VL-2B. ~72–83% char fidelity. | `python -m visual_encoder.text_bridge --sender Qwen/Qwen3-4B` |
| `tap_text_16slots.pt` + `tap_text_gauge.pt` | `runs/wiretaps/` | **wiretap** for the free-text port + its on-manifold gauge (reads traffic without asking either model). | `python -m visual_encoder.text_tap --sender Qwen/Qwen3-4B` |
| `lp3_32chars_16slots.pt` | `runs/bridges/` | LP-3 cross-model bridge (0.6B→2B, bit-packed, 75% exact @10 bits/slot). | `python -m visual_encoder.latent_bridge` |
| `lp3b_selfport_Qwen3-4B_16slots.pt` | `runs/bridges/` | LP-3b bidirectional self-port (one shared-weight 4B, both directions). | `python -m visual_encoder.bidirectional` |
| `lp2_text_16slots_lam{0.0,0.3,1.0}.pt` | `runs/bridges/` | LP-5a legibility-tax bridges (λ = tap-decodability weight). | `python -m visual_encoder.text_bridge --legibility-weight λ` |
| `lp1_{16,32,48,64,96}chars_16slots.pt` | `runs/senders/` | LP-1 bit-packing senders (5–30 bits/slot) — for the lens + wiretap demos. | `python -m visual_encoder.latent_port` |
| `tap_{16,32,48,64,96}chars.pt` | `runs/wiretaps/` | independent wiretaps for the LP-1 senders (decode ~100%). | `python -m visual_encoder.wiretap` |
| `tap_bridge_32chars.pt` | `runs/wiretaps/` | wiretap trained on LP-3 *bridge* traffic (representation-derived, not bit-packed). | `python -m visual_encoder.wiretap --source bridge` |

Not shipped (regenerate if needed): smoke-test checkpoints, superseded bridge
versions (`_v1`, `_v2`), and the scaling-sweep bridges (research datapoints;
`runs/scaling/*.json` has the numbers).

## Load and use (the flagship port, ~15 lines)

```python
import torch
from visual_encoder.latent_bridge import ABrain, GatherBridge
from visual_encoder.latent_port import PortBatcher, greedy_decode, load_receiver
from visual_encoder.text_bridge import TRAIN_TEMPLATES, masked_read

dev = "cuda"
B, tok = load_receiver("Qwen/Qwen3-VL-2B-Instruct", dev)      # frozen receiver
A = ABrain("Qwen/Qwen3-4B", dev)                              # frozen sender
st = torch.load("runs/bridges/lp2_text_16slots.pt", map_location=dev)
bridge = GatherBridge(A.width, B.get_input_embeddings().weight.shape[1],
                      16, st["embed.0.weight"].shape[1] // A.width, 0.0, offset=0)
bridge.load_state_dict(st); bridge = bridge.to(dev).eval()
batcher = PortBatcher(B, tok, 16, dev, message=TRAIN_TEMPLATES[0])

msg = "Meeting Tuesday at 3pm, bring the report."
states, mask = masked_read(A, [msg])                          # A reads the text
with torch.no_grad():
    vectors = bridge(states, mask).squeeze(0).float()         # 16 vectors (the "picture")
    out = greedy_decode(B, batcher, vectors, max_new=64)      # B reads them back
print(out)
```

Wiretap on that traffic: see `visual_encoder/text_tap.py` (`read_text_wire`).
Live 3-pane demo: `python -m visual_encoder.psychic_demo --port 8766`.

## For collaborating instances

Read `docs/COORDINATION.md` and open `handoffs/` before starting; add a handoff
entry rather than editing someone else's. The results these artifacts came from
are in `runs/*.json` (each with a `provenance` block: git SHA, full CLI args,
package versions).
