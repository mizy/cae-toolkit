"""Enforce a wall-clock limit around the mesh CLI's native work."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_status import record_failure
from run_status_lease import LEASE_NAME


def _stop_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    # A descendant may ignore SIGTERM after the group leader exits.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _record_worker_failure(args, pid: int, started: float, error: Exception) -> None:
    """Only modify artifacts while holding the terminated child's own output lease."""
    lease_path = Path(args.output_dir) / LEASE_NAME
    try:
        with lease_path.open("r+") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            owner = json.load(lease)
            if owner.get("pid") != pid:
                return
            status_path = Path(args.output_dir) / "run_status.json"
            status = json.loads(status_path.read_text())
            if status.get("generation_id") != owner.get("generation_id"):
                return
            if status.get("execution_status") == "failed":
                return
            for name in ("source_material_report.json", "source_material_report.html"):
                (Path(args.output_dir) / name).unlink(missing_ok=True)
            status["_started_monotonic"] = started
            record_failure(args, status, error)
    except (OSError, ValueError, json.JSONDecodeError):
        # The child may time out before creating an output directory or lease.
        return


def run_mesh_with_timeout(args, command: list[str], *, timeout_seconds: float) -> int:
    """Run the existing CLI in its own process group so native calls can be stopped."""
    started = time.monotonic()
    label = getattr(args, "surface_strategy", "mesh repair")
    process = subprocess.Popen(command, start_new_session=True)
    try:
        result = process.wait(timeout=timeout_seconds)
        if result not in (0, 2):
            _record_worker_failure(args, process.pid, started, RuntimeError(
                f"{label} worker exited unexpectedly with code {result}"
            ))
        return result
    except subprocess.TimeoutExpired:
        _stop_group(process)
        _record_worker_failure(args, process.pid, started, TimeoutError(
            f"{label} timed out after {timeout_seconds:g} seconds"
        ))
        print(f"{label} timed out after {timeout_seconds:g} seconds; no accepted mesh", file=sys.stderr)
        return 124
    except BaseException:
        _stop_group(process)
        raise
