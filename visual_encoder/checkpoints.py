"""Mid-training checkpoint + resume, so a server restart loses minutes not hours.

Save model(s), optimizer, scheduler, and step atomically every N steps; on start,
if a checkpoint exists, reload and resume from the next step. Atomic (write-tmp +
rename) so a crash during save can't corrupt the checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def resume(path, device, **objs) -> int:
    """Load state into the given named objects; return the step to resume from (0 if none)."""
    p = Path(path)
    if not p.exists():
        return 0
    try:
        c = torch.load(p, map_location=device)
    except Exception as exc:  # corrupt/partial checkpoint -> start clean
        print(f"checkpoint {p} unreadable ({exc}); starting fresh", flush=True)
        return 0
    for name, obj in objs.items():
        if obj is not None and name in c and c[name] is not None:
            obj.load_state_dict(c[name])
    step = int(c.get("step", -1)) + 1
    print(f"RESUMED from {p} at step {step}", flush=True)
    return step


def checkpoint(path, step: int, **objs) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = {"step": step}
        for name, obj in objs.items():
            blob[name] = obj.state_dict() if obj is not None else None
        tmp = str(p) + ".tmp"
        torch.save(blob, tmp)
        os.replace(tmp, p)  # atomic on POSIX
    except Exception as exc:  # a save failure must never kill a training run
        print(f"checkpoint save failed ({exc}); continuing", flush=True)


def clear(path) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()
