import numpy as np
import pytest
from mesh_tet import build_tet_mesh
from serve_web import prepare_case


def test_mm_and_m_geometry_are_equivalent():
    m = build_tet_mesh('box:0.1,0.01,0.01', 0.005, 2, 'm')
    mm = build_tet_mesh('box:100,10,10', 5, 2, 'mm')
    assert np.allclose(np.ptp(m.nodes, axis=0), np.ptp(mm.nodes, axis=0))
    assert len(m.elements) == len(mm.elements)


def test_automatic_solid_mesh_and_input_boundaries():
    conditions = {'material':'steel','fixes':['x_min'],'loads':['x_max: fx=1000']}
    case = prepare_case({'geometry':'box:100,10,10','units':'mm'}, conditions)
    assert case['metadata']['order'] == 2
    assert case['metadata']['mesh_size'] == 5
    assert case['model']['units'] == 'mm'
    assert case['metadata']['coordinate_unit'] == 'm'
    with pytest.raises(ValueError):
        prepare_case({'geometry':'box:100,0,10','units':'mm'}, conditions)
    with pytest.raises(ValueError):
        prepare_case({'geometry':'shell','units':'mm'}, conditions)


def test_stl_upload_payload_with_gravity(tmp_path):
    import base64
    from validate_loads import make_plate
    path=tmp_path/'plate.stl'
    make_plate(path)
    model={'geometry':'stl','units':'m','stl_base64':base64.b64encode(path.read_bytes()).decode()}
    conditions={'material':'steel','fixes':['x_min'],'gravity':[0,0,-9.80665],'density':7850,'loads':[]}
    case=prepare_case(model,conditions)
    assert case['metadata']['integrated_loads']['gravity']['mass_kg']>0
    assert case['metadata']['integrated_loads']['reference_force_balance']<1e-6
    assert 'stl_base64' not in case['model']
