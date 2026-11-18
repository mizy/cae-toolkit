"""Single closed STL -> T4/T10 volume mesh, preserving original surface facets."""
from pathlib import Path
import tempfile
import numpy as np
import gmsh
import trimesh
from mesh_tet import TetMesh, read_msh22


def mesh_stl(path, units, mesh_size, order=2):
    scale = {'m':1.0,'mm':0.001}[units]
    surface = trimesh.load_mesh(path, force='mesh', process=True)
    if not surface.is_watertight or not surface.is_winding_consistent or surface.volume <= 0:
        raise ValueError('STL must be closed, consistently oriented and have positive material volume')
    if len(surface.split()) != 1 or np.any(surface.area_faces <= 0):
        raise ValueError("STL import currently requires one closed component without degenerate facets")
    if not np.isfinite(surface.vertices).all():
        raise ValueError('STL coordinates must be finite')
    surface.apply_scale(scale)
    with tempfile.TemporaryDirectory(prefix='structural-stl-') as folder:
        source, output = Path(folder)/'input.stl', Path(folder)/'volume.msh'
        surface.export(source)
        gmsh.initialize()
        try:
            gmsh.option.setNumber('General.Terminal',0)
            gmsh.merge(str(source))
            gmsh.option.setNumber('Mesh.MeshOnlyEmpty',1)
            loop = gmsh.model.geo.addSurfaceLoop([tag for _,tag in gmsh.model.getEntities(2)])
            gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()
            gmsh.option.setNumber('Mesh.MeshSizeMin', mesh_size*scale)
            gmsh.option.setNumber('Mesh.MeshSizeMax', mesh_size*scale)
            gmsh.option.setNumber('Mesh.SecondOrderLinear',1)
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.setOrder(order)
            gmsh.option.setNumber('Mesh.MshFileVersion',2.2)
            gmsh.write(str(output))
            nodes,elements,actual = read_msh22(output)
        finally:
            gmsh.finalize()
    nn = 10 if actual == 2 else 4
    used, inverse = np.unique(elements[:,:nn], return_inverse=True)
    elements[:,:nn] = inverse.reshape(-1,nn)
    mesh = TetMesh(nodes=nodes[used],elements=elements,order=actual,units=units,geometry=str(path))
    return mesh
