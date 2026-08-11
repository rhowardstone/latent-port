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

# NOTE: scaling doubling sweep (wide_picture) DEFERRED — its fresh-training
# underperforms (22% vs the eval-path-verified 77%); needs a training fix before
# it's worth GPU. Running the proven experiments first.

# 3) biggest lever — receiver LoRA vs frozen at 20 bits/slot
run receiver_lora runs/receiver_lora.json visual_encoder.receiver_lora --chars 64 --steps 3000
# 4) bidirectional self-port
run bidirectional runs/bidirectional.json visual_encoder.bidirectional --steps 4000

milestone "NIGHTSHIFT COMPLETE"
