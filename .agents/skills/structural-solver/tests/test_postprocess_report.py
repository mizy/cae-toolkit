"""Exercise the solved-field export with native units and tetrahedral sections."""
import base64
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pyvista as pv
import pytest


def test_vtu_postprocess_recovers_original_geometry_and_embeds_sections(tmp_path):
    xyz = np.array([[0., 0., 0.], [.02, 0., 0.], [0., .01, 0.], [0., 0., .01]])
    displacement = np.tile([.1, 0., 0.], (4, 1))
    mesh = pv.UnstructuredGrid([4, 0, 1, 2, 3], [pv.CellType.TETRA], xyz + displacement * .002)
    mesh.point_data["displacement"] = displacement
    mesh.point_data["stress_tensor"] = np.tile([1., 2., 3., 0., 0., 0.], (4, 1))
    mesh.cell_data["von_mises_stress"] = [1.73]
    mesh.cell_data["material_id"] = [7]
    mesh.point_data["wind"] = np.tile([3., 4., 0.], (4, 1))
    mesh.field_data["geometry_displacement_scale"] = [2.]
    source = tmp_path / "solved.vtu"
    mesh.save(source)
    output = tmp_path / "report.html"
    cli = Path(__file__).resolve().parents[1] / "scripts" / "solve.py"
    subprocess.run(
        [sys.executable, str(cli), "postprocess", "--mesh", str(source), "--units", "mm",
         "--mesh-units", "m", "--volume-sections", "--conclusion", "线弹性 <筛查>",
         "--output", str(output)],
        check=True,
    )
    assert "线弹性 &lt;筛查&gt;" in output.read_text()
    payload = json.loads(re.search(r'<script type="application/json" id="result-data">(.*?)</script>',
                                   output.read_text(), re.S)[1])
    points = np.frombuffer(base64.b64decode(payload["points"]), dtype=np.float32).reshape(-1, 3)
    assert points.min(axis=0) == pytest.approx([0, 0, 0])
    assert points.max(axis=0) == pytest.approx([20, 10, 10])
    fields = {item["label"]: item for item in payload["fields"]}
    assert fields["von Mises 等效应力"]["association"] == "cell"
    assert fields["最大主应力"]["max"] == pytest.approx(3.)
    assert fields["material_id (单元)"]["max"] == 7
    assert fields["wind · 幅值 (点)"]["max"] == 5
    assert all(len(payload["sections"][axis]) == 5 for axis in "xyz")
    assert payload["sections"]["x"][2]["position"] == pytest.approx(10.)
