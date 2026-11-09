"""Validate browser inputs against assembled FEM equations and boundary semantics."""
import json

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from export_web_case import export_case


def test_export_multiple_loads_and_reference(tmp_path):
    output = tmp_path / "beam.json"
    metadata = export_case("box:0.1,0.01,0.01", "m", 0.01, 1, "steel",
                           ["x_min", "x_min"], ["x_max: fx=50", "x_max: fz=-100"], output)
    case = json.loads(output.read_text())
    matrix = csr_matrix((case["values"], case["colIdx"], case["rowPtr"]))
    rhs, u = np.array(case["rhs"]), np.array(case["reference"])
    assert np.linalg.norm(matrix @ u - rhs) / np.linalg.norm(rhs) < 1e-9
    assert np.allclose(rhs.reshape(-1, 3).sum(axis=0), [50, 0, -100])
    assert np.asarray(case["coordinates"]).shape == (len(rhs) // 3, 3)
    assert metadata["coordinate_unit"] == metadata["displacement_unit"] == "m"
    assert len(metadata["loads"]) == 2
    assert metadata["load_distribution"] == "equal nodal forces"


@pytest.mark.parametrize("load", ["x_min: fx=100", "y_max: fz=-100"])
def test_reject_load_on_or_intersecting_fixed_face(tmp_path, load):
    with pytest.raises(ValueError, match="overlaps a fixed face"):
        export_case("box:0.1,0.01,0.01", "m", 0.01, 1, "steel",
                    "x_min", load, tmp_path / "bad.json")
