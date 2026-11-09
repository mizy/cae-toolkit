import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "two_stage_watertight_remesh.py"


def test_cfd_opening_must_be_reviewed_and_executed_before_acceptance():
    mesh = trimesh.creation.box(extents=[20, 20, 20])
    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    faces = faces[~np.all(np.isclose(points[faces, 2], 10), axis=1)]
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root / "open.vtp"
        pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).ravel()).save(source)
        command = [sys.executable, str(SCRIPT), str(source), "--purpose", "cfd", "--cfd-max-edge-length", "4",
                   "--visibility-grid", "64", "--outside-flood-grid", "32", "--sealed-exterior-grid", "32",
                   "--voxel-pitch", "1", "--repair-timeout-seconds", "20"]
        first = root / "unreviewed"
        subprocess.run([*command, "--output-dir", str(first)], check=False, capture_output=True, timeout=30)
        report = json.loads((first / "two_stage_report.json").read_text())
        packet = json.loads((first / "ai_policy_packet.json").read_text())
        assert packet["item_count_total"] == 1
        assert report["gates"]["opening_policy_resolved"]["required"]
        assert not report["gates"]["opening_policy_resolved"]["passed"]
        assert report["target"]["shell_policy"]["semantic_review_applicability"] == "required_for_acceptance"
        assert report["outputs"]["accepted_mesh_vtp"] is None

        choice = dict(packet["items"][0]["decision_template"])
        choice.update(decision="cap", semantic_label="missing_cover", shape_prior="planar",
                      confidence=1.0, reason="synthetic missing cover for this test")
        decisions = root / "decisions.json"
        decisions.write_text(json.dumps([choice]))
        reviewed = root / "reviewed"
        subprocess.run([*command, "--policy-decisions", str(decisions), "--output-dir", str(reviewed)],
                       check=False, capture_output=True, timeout=30)
        report = json.loads((reviewed / "two_stage_report.json").read_text())
        assert report["gates"]["opening_policy_resolved"]["passed"]
        assert report["policy_decisions"][0]["applied_by_pass_id"]
