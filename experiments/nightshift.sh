#!/usr/bin/env bash
# Overnight autonomous experiment queue. Runs jobs sequentially on the A100,
# auto-commits every result artifact, and prints NIGHTSHIFT: milestones (a Monitor
# re-invokes the agent at each to analyze + course-correct). Robust: one failing
# job logs and the queue continues.
set -uo pipefail
cd "$(dirname "$0")/.."
NOREPLY="$(gh api user --jq .id 2>/dev/null)+rhowardstone@users.noreply.github.com"

commit () {  # message
  git add -A
  git -c user.name="rhowardstone" -c user.email="$NOREPLY" commit -q -m "$1" 2>/dev/null || true
  git push -q 2>/dev/null || true
}
milestone () { echo "NIGHTSHIFT: $*"; }
run () {  # label module args...
  local label="$1"; shift
  milestone "START $label"
  if python -m "$@" >> "runs/nightshift_${label}.log" 2>&1; then
    milestone "DONE $label"
  else
    milestone "FAILED $label (see runs/nightshift_${label}.log)"
  fi
}

# 1) wait for the binding control to finish (it owns the GPU first)
milestone "waiting for binding control to finish"
while pgrep -f "latent_port --chars" >/dev/null 2>&1; do sleep 60; done
milestone "binding control finished"
commit "nightshift: binding control results"

# 2) THE ROLLOUT FIX — train B to read multi-picture messages
run rollout visual_encoder.rollout_bridge --pictures 4 --steps 5000 --output runs/rollout_bridge.json
commit "nightshift: multi-picture rollout bridge (make rollout work)"

# 3) bidirectional self-port (LP-3b) — one shared 4B, both directions
run bidirectional visual_encoder.bidirectional --steps 4000 --output runs/bidirectional.json
commit "nightshift: LP-3b bidirectional self-port"

# 4) a denser rollout attempt (more pictures) to probe the ceiling
run rollout6 visual_encoder.rollout_bridge --pictures 6 --steps 5000 --output runs/rollout_bridge_6pic.json
commit "nightshift: 6-picture rollout (probe the multi-picture ceiling)"

milestone "NIGHTSHIFT COMPLETE"
