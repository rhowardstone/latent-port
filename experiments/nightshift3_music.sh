#!/usr/bin/env bash
# LP-6 music port — waits POLITELY for the large-picture queue (nightshift2) to free
# the GPU, then trains the music port on IrishMAN ABC. Separate lock + waits for the
# "NIGHTSHIFT2 COMPLETE" signal, so it never fights nightshift2 for the card and never
# depends on editing a running driver (which silently dropped LP-4 last time).
# Resumable: music_port checkpoints every 250 steps and skips if the output exists.
set -uo pipefail
cd "$(dirname "$0")/.."
NOREPLY="$(gh api user --jq .id 2>/dev/null)+rhowardstone@users.noreply.github.com"
commit () {
  git add -A
  git -c user.name="rhowardstone" -c user.email="$NOREPLY" commit -q -m "$1" 2>/dev/null || true
  git push -q 2>/dev/null || true
}
milestone () { echo "MUSIC: $*"; }
run () {
  local label="$1" out="$2"; shift 2
  if [ -s "$out" ]; then milestone "SKIP $label (exists $out)"; return; fi
  milestone "START $label"
  if python -m "$@" --output "$out" >> "runs/music_${label}.log" 2>&1; then
    milestone "DONE $label"; commit "music: $label done"
  else
    milestone "FAILED $label (runs/music_${label}.log)"
  fi
}

exec 9>runs/.music.lock
flock -n 9 || { milestone "another music driver holds the lock; exiting"; exit 0; }

milestone "waiting for nightshift2 (large-picture) to free the GPU..."
while ! grep -q "NIGHTSHIFT2 COMPLETE" runs/nightshift2_driver.log 2>/dev/null; do sleep 60; done
milestone "GPU free — starting LP-6 music port"

# H1: can the frozen receiver re-emit a melody? matched budget to the text run s32_w64 (0.87).
run music runs/music_port.json visual_encoder.music_port \
  --slots 32 --window 64 --steps 3000 --eval-every 500
commit "music: LP-6 port trained"

milestone "MUSIC COMPLETE"
