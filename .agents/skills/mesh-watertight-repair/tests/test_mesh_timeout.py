from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from mesh_timeout import run_mesh_with_timeout  # noqa: E402
from test_run_status import args_for  # noqa: E402


class MeshTimeoutTest(unittest.TestCase):
    def test_projected_shell_timeout_revokes_partial_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = args_for(root)
            args.surface_strategy = "projected-shell"
            script = '''
import fcntl,json,os,pathlib,sys,time
root=pathlib.Path(sys.argv[1])
lease=(root/".mesh_repair_run.lease").open("w+")
fcntl.flock(lease,fcntl.LOCK_EX)
json.dump({"pid":os.getpid(),"generation_id":"test"},lease);lease.flush()
(root/"run_status.json").write_text(json.dumps({"generation_id":"test","stage":"closure_proxy_started","outputs":{}}))
(root/"two_stage_report.json").write_text(json.dumps({"decision":{"status":"accepted"}}))
time.sleep(30)
'''
            result = run_mesh_with_timeout(args, [sys.executable, "-c", script, str(root)], timeout_seconds=.5)
            self.assertEqual(result, 124)
            report = json.loads((root / "two_stage_report.json").read_text())
            self.assertEqual(report["decision"]["status"], "rejected")
            self.assertIsNone(report["outputs"]["accepted_mesh_vtp"])

    def test_timeout_stops_child_and_revokes_partial_accepted_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = args_for(root)
            args.source_timeout_seconds = 0.5
            script = '''
import fcntl, json, os, pathlib, sys, time
root = pathlib.Path(sys.argv[1])
lease = (root / ".mesh_repair_run.lease").open("w+")
fcntl.flock(lease, fcntl.LOCK_EX)
json.dump({"pid": os.getpid(), "generation_id": "test"}, lease)
lease.flush()
(root / "run_status.json").write_text(json.dumps({"generation_id":"test", "stage":"native_union", "outputs":{}}))
(root / "source_material_report.json").write_text(json.dumps({"accepted":True}))
(root / "source_material_report.html").write_text("accepted")
(root / "source_material_candidate.vtp").write_text("diagnostic")
time.sleep(30)
'''
            started = time.monotonic()
            result = run_mesh_with_timeout(args, [sys.executable, "-c", script, str(root)], timeout_seconds=args.source_timeout_seconds)
            self.assertEqual(result, 124)
            self.assertLess(time.monotonic() - started, 5)
            status = json.loads((root / "run_status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["stage"], "native_union")
            self.assertEqual(status["resource_failure"]["kind"], "timeout")
            self.assertIsNone(status["accepted_mesh_vtp"])
            self.assertFalse((root / "source_material_report.json").exists())
            self.assertFalse((root / "source_material_report.html").exists())
            self.assertTrue((root / "source_material_candidate.vtp").exists())
            self.assertEqual(status["outputs"]["rejected_candidate_vtp"], str(root / "source_material_candidate.vtp"))

    def test_timeout_does_not_change_another_runs_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = args_for(root)
            args.source_timeout_seconds = 0.3
            (root / ".mesh_repair_run.lease").write_text(json.dumps({"pid": -1, "generation_id": "other"}))
            (root / "run_status.json").write_text('{"generation_id":"other","status":"running"}')
            (root / "source_material_report.json").write_text('{"accepted":true}')
            result = run_mesh_with_timeout(args, [sys.executable, "-c", "import time; time.sleep(30)"], timeout_seconds=args.source_timeout_seconds)
            self.assertEqual(result, 124)
            self.assertEqual(json.loads((root / "run_status.json").read_text())["status"], "running")
            self.assertTrue(json.loads((root / "source_material_report.json").read_text())["accepted"])

    def test_worker_crash_does_not_leave_running_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = args_for(root)
            args.source_timeout_seconds = 3
            script = '''
import fcntl, json, os, pathlib, signal, sys
root = pathlib.Path(sys.argv[1])
lease = (root / ".mesh_repair_run.lease").open("w+")
fcntl.flock(lease, fcntl.LOCK_EX)
json.dump({"pid": os.getpid(), "generation_id": "test"}, lease)
lease.flush()
(root / "run_status.json").write_text(json.dumps({"generation_id":"test", "stage":"native_union", "outputs":{}}))
os.kill(os.getpid(), signal.SIGKILL)
'''
            result = run_mesh_with_timeout(args, [sys.executable, "-c", script, str(root)], timeout_seconds=args.source_timeout_seconds)
            self.assertNotEqual(result, 0)
            status = json.loads((root / "run_status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["stage"], "native_union")
            self.assertIsNone(status["accepted_mesh_vtp"])

    def test_timeout_kills_descendant_even_if_it_ignores_sigterm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = args_for(root)
            args.source_timeout_seconds = 1.5
            script = '''
import pathlib, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
code = "import os,pathlib,signal,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
subprocess.Popen([sys.executable, "-c", code, str(root / "descendant.pid")])
time.sleep(30)
'''
            result = run_mesh_with_timeout(args, [sys.executable, "-c", script, str(root)], timeout_seconds=args.source_timeout_seconds)
            self.assertEqual(result, 124)
            pid = (root / "descendant.pid").read_text()
            state = subprocess.run(["ps", "-p", pid, "-o", "stat="], capture_output=True, text=True).stdout.strip()
            self.assertTrue(not state or state.startswith("Z"), f"descendant still running: {state}")


if __name__ == "__main__":
    unittest.main()
