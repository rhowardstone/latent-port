#!/usr/bin/env bash
# Overnight autonomous experiment queue. RESUMABLE (skips jobs whose result JSON
# exists; every training checkpoints every 250 steps), auto-commits, prints
# NIGHTSHIFT: milestones. flock guard = one driver; cron watchdog restarts it.
set -uo pipefail
cd "$(dirname "$0")/.."
NOREPLY="$(gh api user --jq .id 2>/dev/null)+rhowardstone@users.noreply.github.com"

commit () {
  git add -A
  git -c user.name="rhowardstone" -c user.email="$NOREPLY" commit -q -m "$1" 2>/dev/null || true
  git push -q 2>/dev/null || true
}
milestone () { echo "NIGHTSHIFT: $*"; }
run () {  # label outfile module args...
  local label="$1" out="$2"; shift 2
  if [ -s "$out" ]; then milestone "SKIP $label (exists $out)"; return; fi
  milestone "START $label"
  if python -m "$@" --output "$out" >> "runs/nightshift_${label}.log" 2>&1; then
    milestone "DONE $label"; commit "nightshift: $label done"
  else
    milestone "FAILED $label (runs/nightshift_${label}.log)"
  fi
}

exec 9>runs/.nightshift.lock
flock -n 9 || { milestone "another driver holds the lock; exiting"; exit 0; }

while pgrep -f "latent_port --chars" >/dev/null 2>&1; do sleep 60; done  # binding
commit "nightshift: binding results"

# 1) rollout fix (resumes/finishes if it was mid-run, or skips if done)
run rollout runs/rollout_bridge.json visual_encoder.rollout_bridge --pictures 4 --steps 5000

# SCALING DOUBLING SWEEP (fixed: coherent passages — verified fidelity climbs).
# image size (slots) x vector content (tokens/slot), density=2 unless noted.
run scale_s16  runs/scaling/s16_w32.json   visual_encoder.wide_picture --slots 16  --window 32  --steps 3000
run scale_s32  runs/scaling/s32_w64.json   visual_encoder.wide_picture --slots 32  --window 64  --steps 3000
run scale_s64  runs/scaling/s64_w128.json  visual_encoder.wide_picture --slots 64  --window 128 --steps 3000
run scale_s128 runs/scaling/s128_w256.json visual_encoder.wide_picture --slots 128 --window 256 --steps 3000
run scale_d1   runs/scaling/s32_w32.json   visual_encoder.wide_picture --slots 32 --window 32  --steps 3000
run scale_d3   runs/scaling/s32_w96.json   visual_encoder.wide_picture --slots 32 --window 96  --steps 3000
run scale_d4   runs/scaling/s32_w128.json  visual_encoder.wide_picture --slots 32 --window 128 --steps 3000
# how large can a picture go? push image size to 256 and 512 vectors (2 tok/slot)
run scale_s256 runs/scaling/s256_w512.json  visual_encoder.wide_picture --slots 256 --window 512  --steps 3000 --batch-size 4
run scale_s512 runs/scaling/s512_w1024.json visual_encoder.wide_picture --slots 512 --window 1024 --steps 3000 --batch-size 3
commit "nightshift: scaling sweep + large-picture extension (256, 512 vectors)"

# LP-4: can one learned latent step skip Δ>1 token-generation steps? (macro-steps)
run lp4 runs/latent_rollout.json visual_encoder.latent_rollout --deltas 1,2,4,8 --gen-len 48 --train-prompts 300 --steps 2500
commit "nightshift: LP-4 latent macro-steps"

# 3) biggest lever — receiver LoRA vs frozen at 20 bits/slot
run receiver_lora runs/receiver_lora.json visual_encoder.receiver_lora --chars 64 --steps 3000
# 4) bidirectional self-port
run bidirectional runs/bidirectional.json visual_encoder.bidirectional --steps 4000

milestone "NIGHTSHIFT COMPLETE"
