#!/usr/bin/env bash
# LP-1 binding control (external review A4): is the transposition failure intrinsic
# to the frozen receiver, or manufactured by the permutation-invariant mean warm
# start? Sweep warm-start order at the dense loads where transpositions appear.
# Each run records content_vs_order (indel-aligned) + warmstart_eps in provenance.
set -euo pipefail
cd "$(dirname "$0")/.."

CHARS="${CHARS:-48,64,96}"   # 15/20/30 bits/slot (16 slots)
STEPS="${STEPS:-3000}"

run () {  # order eps tag
  local order="$1" eps="$2" tag="$3"
  local out="runs/binding/${tag}.json"
  if [ -f "$out" ]; then echo "skip $tag (exists)"; return; fi
  echo "== binding: order=$order eps=$eps -> $out =="
  python -m visual_encoder.latent_port --chars "$CHARS" --steps "$STEPS" \
    --warmstart-order "$order" --warmstart-eps "$eps" --output "$out"
}

mkdir -p runs/binding
run mean  1.0 mean          # the confound (permutation-invariant)
run blend 0.1 blend_eps0.1  # gentle order injection (stays near native geometry)
run blend 0.3 blend_eps0.3
run role  1.0 role          # full orthogonal role binding
run none  1.0 none          # no warm start (cold)
echo "BINDING CONTROL COMPLETE"
