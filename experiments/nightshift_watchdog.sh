#!/usr/bin/env bash
# Cron watchdog: keeps the overnight driver alive across session teardowns.
# Restarts nightshift.sh if it's not running and the queue isn't complete.
# The driver's own flock guard prevents any double-launch.
cd /atb-data/rye/Visual-encoder || exit 0
grep -q "NIGHTSHIFT COMPLETE" runs/nightshift.log 2>/dev/null && exit 0
pgrep -f "experiments/nightshift.sh" >/dev/null 2>&1 && exit 0
setsid bash experiments/nightshift.sh >> runs/nightshift.log 2>&1 < /dev/null &
