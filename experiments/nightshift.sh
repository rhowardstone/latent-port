#!/usr/bin/env bash
# Overnight autonomous experiment queue. RESUMABLE: skips any job whose result
# JSON already exists, so a restart (session teardown, crash, watchdog) picks up
# where it left off. Auto-commits every result, prints NIGHTSHIFT: milestones.
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

# guard: only one driver at a time (the watchdog may also try to start it)
exec 9>runs/.nightshift.lock
flock -n 9 || { milestone "another driver holds the lock; exiting"; exit 0; }

# 1) binding control must be finished (its results live in runs/binding/*.json)
milestone "waiting for binding control to finish"
while pgrep -f "latent_port --chars" >/dev/null 2>&1; do sleep 60; done
milestone "binding control finished"
commit "nightshift: binding control results"

# 2) THE ROLLOUT FIX — train B to read multi-picture messages
run rollout runs/rollout_bridge.json visual_encoder.rollout_bridge --pictures 4 --steps 5000

# 2b) BIGGEST LEVER — does a small LoRA on B break the frozen ceiling? (20 bits/slot)
run receiver_lora runs/receiver_lora.json visual_encoder.receiver_lora --chars 64 --steps 3000

# 3) bidirectional self-port (LP-3b)
run bidirectional runs/bidirectional.json visual_encoder.bidirectional --steps 4000

# 4) 6-picture rollout ceiling probe
run rollout6 runs/rollout_bridge_6pic.json visual_encoder.rollout_bridge --pictures 6 --steps 5000

milestone "NIGHTSHIFT COMPLETE"
