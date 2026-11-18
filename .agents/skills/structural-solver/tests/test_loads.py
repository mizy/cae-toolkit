import numpy as np
import pytest
from mesh_tet import build_tet_mesh
from loads import gravity_load, cylinder_load


def test_quadratic_body_force_preserves_mass_and_moment():
    mesh = build_tet_mesh('box:0.1,0.01,0.01',0.005,2,'m')
    force, info = gravity_load(mesh.nodes,mesh.elements,2,7850,[0,0,-9.80665])
    force = force.reshape(-1,3)
    expected = 7850*0.1*0.01*0.01
    assert info['mass_kg'] == pytest.approx(expected,rel=1e-10)
    assert force.sum(axis=0) == pytest.approx([0,0,-expected*9.80665],abs=1e-12)
    moment = np.cross(mesh.nodes,force).sum(axis=0)
    assert moment == pytest.approx(np.cross([.05,.005,.005],[0,0,-expected*9.80665]),abs=1e-12)
    with pytest.raises(ValueError):
        gravity_load(mesh.nodes,mesh.elements,2,-1,[0,0,-9.81])


def test_bearing_requires_valid_region_and_radial_load():
    mesh=build_tet_mesh('box:0.1,0.01,0.01',.005,2,'m')
    with pytest.raises(ValueError,match='No interior'):
        cylinder_load(mesh.nodes,mesh.elements,2,dict(type='bearing',center=[0,0,0],axis=[0,0,1],radius=.002,force=[100,0,0]))


def test_stl_bore_force_pressure_and_volume(tmp_path):
    from validate_loads import make_plate
    from stl_volume import mesh_stl
    surface = make_plate(tmp_path/'plate.stl')
    mesh = mesh_stl(tmp_path/'plate.stl','m',.005)
    spec = dict(type='bearing',center=[.05,.02,.004],axis=[0,0,1],radius=.006,tolerance=.0002,force=[1000,0,0])
    force, info = cylinder_load(mesh.nodes,mesh.elements,2,spec)
    assert force.reshape(-1,3).sum(axis=0) == pytest.approx([1000,0,0],abs=.01)
    pressure, _ = cylinder_load(mesh.nodes,mesh.elements,2,{**spec,'type':'pressure','pressure_pa':1e6})
    assert np.linalg.norm(pressure.reshape(-1,3).sum(axis=0)) < 1e-6
    _, gravity = gravity_load(mesh.nodes,mesh.elements,2,1,[0,0,0])
    assert gravity['volume_m3'] == pytest.approx(surface.volume,rel=1e-6)
    with pytest.raises(ValueError,match='perpendicular'):
        cylinder_load(mesh.nodes,mesh.elements,2,{**spec,'force':[0,0,100]})
