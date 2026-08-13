#!/usr/bin/env bash
# Autonomous queue v2 — find the TRUE large-picture ceiling with a proper recipe.
# Why: s256 stalled flat at the warm-start floor (0.26) on batch-4 (a weak, noisy
# gradient on a 4x-harder task); s512 OOM'd on the single-message forward. The inner
# loop already accumulates per-message, so a bigger --batch-size costs NO extra memory
# — only time. Fix = strong effective batch + more warmstart/steps; --grad-checkpointing
# fits the 512-slot / 1024-tok forward. Resumable (skip-if-exists, 250-step ckpts),
# auto-commits, prints NIGHTSHIFT2 milestones. flock = one driver.
set -uo pipefail
cd "$(dirname "$0")/.."
NOREPLY="$(gh api user --jq .id 2>/dev/null)+rhowardstone@users.noreply.github.com"
commit () {
  git add -A
  git -c user.name="rhowardstone" -c user.email="$NOREPLY" commit -q -m "$1" 2>/dev/null || true
  git push -q 2>/dev/null || true
}
milestone () { echo "NIGHTSHIFT2: $*"; }
run () {  # label outfile module args...
  local label="$1" out="$2"; shift 2
  if [ -s "$out" ]; then milestone "SKIP $label (exists $out)"; return; fi
  milestone "START $label"
  if python -m "$@" --output "$out" >> "runs/nightshift2_${label}.log" 2>&1; then
    milestone "DONE $label"; commit "nightshift2: $label done"
  else
    milestone "FAILED $label (runs/nightshift2_${label}.log)"
  fi
}

exec 9>runs/.nightshift2.lock
flock -n 9 || { milestone "another driver holds the lock; exiting"; exit 0; }

# 256-vector picture, PROPER recipe: 4x effective batch (free on memory), 2x steps,
# bigger warm start. Does it learn, or is ~0.26 a real ceiling? (interim eval /500 steps)
run scale_s256v2 runs/scaling/s256_w512_v2.json visual_encoder.wide_picture \
  --slots 256 --window 512 --batch-size 16 --warmstart-steps 1000 --steps 4000 --eval-every 500
# 512-vector picture: grad-checkpointed so the 1024-tok forward fits. Push past 256.
run scale_s512v2 runs/scaling/s512_w1024_v2.json visual_encoder.wide_picture \
  --slots 512 --window 1024 --batch-size 8 --warmstart-steps 1000 --steps 4000 --eval-every 500 --grad-checkpointing
commit "nightshift2: large-picture proper-recipe retries (s256v2, s512v2)"

milestone "NIGHTSHIFT2 COMPLETE"
