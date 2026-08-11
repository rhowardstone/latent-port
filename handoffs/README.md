# Handoffs

Coordination inbox for the multi-agent workflow (see `docs/COORDINATION.md`).
One markdown per handoff: `YYYY-MM-DD-<node>-<topic>.md`. Scan open entries before
claiming a lane. The A100 node integrates relayed patches and marks entries `done`.

## Open / recent

| Date | From | Topic | Status |
|---|---|---|---|
| 2026-08-11 | a100 | lane assignments + GPU queue | active (see below) |
| 2026-08-11 | cpu-A | variational-rate + LP-4 sandbox notebook | proposed → to integrate under `notebooks/` |
| 2026-08-11 | cpu-B | channel_metrics library + property tests | proposed → owns that lane |
