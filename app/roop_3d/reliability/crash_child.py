"""Child process for the hard-kill scenario: renders a job slowly until the parent terminates it.

    python -m roop_3d.reliability.crash_child <root> <frames> [tracks] [delay_seconds]

Prints ``READY <job_id>`` once the job is queued, then keeps running. The parent kills this process with
``TerminateProcess`` / ``SIGKILL`` (no cleanup handlers run) and restarts from the same ``<root>``, which is exactly
what an application crash, a forced end-task or a power loss looks like to the on-disk state.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

from roop_3d.reliability.harness import build_rig
from roop_3d.reliability.processes import bind_children_to_process_lifetime


def main(argv: list[str]) -> int:
    root, frames = Path(argv[0]), int(argv[1])
    tracks = int(argv[2]) if len(argv) > 2 else 1
    delay = float(argv[3]) if len(argv) > 3 else 0.01
    bind_children_to_process_lifetime()
    rig = build_rig(root, frames=frames, tracks=tracks, delay=delay, checkpoint_every=10)
    record = rig.submit("crash-child")
    print(f"READY {record.job_id}", flush=True)
    while True:
        time.sleep(0.2)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
