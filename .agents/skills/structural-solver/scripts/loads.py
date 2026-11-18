"""Consistent solid body loads and cylindrical boundary tractions (SI units)."""
import numpy as np
from assembly import GAUSS4_RST, GAUSS4_W, TET4_DN, tet10_dN_ref


def gravity_load(nodes, elements, order, density, gravity):
    """Integrate Ni*rho*g over each tetrahedron, including all T10 shape functions."""
    g = np.asarray(gravity, dtype=float)
    if g.shape != (3,) or not np.isfinite(g).all() or not np.isfinite(density) or density <= 0:
        raise ValueError('Gravity needs finite acceleration [m/s2] and positive density [kg/m3]')
    nn = 10 if order == 2 else 4
    conn = elements[:, :nn]
    xyz = nodes[conn]
    weights = np.zeros((len(elements), nn))
    for point, w in zip(GAUSS4_RST, GAUSS4_W):
        bary = np.r_[1 - point.sum(), point]
        if order == 2:
            shape = np.r_[bary * (2*bary-1), [4*bary[a]*bary[b] for a,b in [(0,1),(1,2),(2,0),(0,3),(1,3),(2,3)]]]
            gradient = tet10_dN_ref(*point)
        else:
            shape, gradient = bary, TET4_DN
        det = np.linalg.det(np.einsum('ni,enj->eij', gradient, xyz))
        if np.any(det <= 0):
            raise ValueError('Non-positive element Jacobian in gravity integration')
        weights += det[:, None] * (w/6) * shape
    nodal_volume = np.zeros(len(nodes))
    np.add.at(nodal_volume, conn.ravel(), weights.ravel())
    volume = float(weights.sum())
    return (nodal_volume[:,None] * density*g).ravel(), {
        'density_kg_m3':float(density), 'gravity_m_s2':g.tolist(),
        'volume_m3':volume, 'mass_kg':volume*density, 'total_force_n':(volume*density*g).tolist(),
    }


def boundary_faces(elements, order):
    """Extract exterior tet faces; return their node IDs and opposite corner IDs."""
    local = [([0,1,2,4,5,6],3), ([0,1,3,4,8,7],2),
             ([1,2,3,5,9,8],0), ([0,2,3,6,9,7],1)]
    found = {}
    for tet in elements:
        for face, opposite in local:
            ids = tet[face[:6 if order == 2 else 3]]
            key = tuple(sorted(ids[:3]))
            if key in found:
                found[key] = None
            else:
                found[key] = (ids.copy(), tet[opposite])
    outer = [v for v in found.values() if v is not None]
    return np.asarray([v[0] for v in outer]), np.asarray([v[1] for v in outer])


def cylinder_load(nodes, elements, order, spec):
    """Pressure into material, or cosine bearing load on the compressed half-bore.

    Selection is an explicit cylinder (center/axis/radius/tolerance in meters),
    not inferred CAD identity. Surface geometry must be planar T3/T6 facets.
    """
    center = np.asarray(spec['center'], dtype=float)
    axis = np.asarray(spec['axis'], dtype=float)
    radius = float(spec['radius'])
    tolerance = float(spec.get('tolerance', 0.02*radius))
    if center.shape != (3,) or axis.shape != (3,) or not np.isfinite(np.r_[center,axis,radius,tolerance]).all() or np.linalg.norm(axis) == 0 or radius <= 0 or not 0 < tolerance < radius:
        raise ValueError('Invalid cylinder selection')
    axis /= np.linalg.norm(axis)
    faces, opposite = boundary_faces(elements, order)
    xyz = nodes[faces[:,:3]]
    centroid = xyz.mean(axis=1)
    normal = np.cross(xyz[:,1]-xyz[:,0], xyz[:,2]-xyz[:,0])
    area = np.linalg.norm(normal, axis=1)/2
    if np.any(area <= 0):
        raise ValueError('Degenerate boundary facet')
    normal /= (2*area[:,None])
    reverse = np.einsum('ij,ij->i', normal, nodes[opposite]-centroid) > 0
    normal[reverse] *= -1
    relative = xyz-center
    radial = relative - np.einsum('fki,i->fk', relative, axis)[:,:,None]*axis
    distance = np.linalg.norm(radial, axis=2)
    # In a bore, the outward solid normal points toward the cylinder axis.
    radial_center = radial.mean(axis=1)
    selection = (np.max(np.abs(distance-radius),axis=1)<=tolerance) & (np.abs(normal@axis)<0.8) & (np.einsum('ij,ij->i',normal,radial_center)<0)
    if not selection.any():
        raise ValueError('No interior cylindrical facets match the requested region')
    faces, area, normal = faces[selection], area[selection], normal[selection]
    if order == 2:
        points = nodes[faces]
        midpoint = (points[:,[0,1,2]]+points[:,[1,2,0]])/2
        if np.max(np.linalg.norm(points[:,3:]-midpoint,axis=2)) > 1e-7*max(np.ptp(nodes,axis=0)):
            raise ValueError('Curved T6 traction integration is not yet supported')
    if np.linalg.norm((area[:,None]*normal).sum(axis=0))/area.sum() > 0.01:
        raise ValueError('Cylinder selection is incomplete or too distorted; inspect the selected facets')
    kind = spec['type']
    if kind == 'bearing':
        force = np.asarray(spec['force'],dtype=float)
        if force.shape != (3,) or not np.isfinite(force).all() or np.linalg.norm(force)==0:
            raise ValueError('Bearing needs a nonzero finite force')
        if abs(force@axis)>1e-8*np.linalg.norm(force):
            raise ValueError('Bearing force must be perpendicular to cylinder axis')
        direction = force/np.linalg.norm(force)
        cosine = np.maximum((-normal)@direction,0)
        traction = -normal*cosine[:,None]
        projected = np.sum(area*(traction@direction))
        if projected <= 0:
            raise ValueError('No compressive side found')
        pressure_scale = np.linalg.norm(force)/projected
        traction *= pressure_scale
        actual = (area[:,None]*traction).sum(axis=0)
        if np.linalg.norm(actual-force) > 0.01*np.linalg.norm(force):
            raise ValueError('Bearing resultant differs from requested vector; refine or correct the region')
    elif kind == 'pressure':
        pressure_scale = float(spec['pressure_pa'])
        if not np.isfinite(pressure_scale) or pressure_scale < 0:
            raise ValueError('Pressure must be nonnegative [Pa]')
        traction = -normal*pressure_scale
    else:
        raise ValueError('Cylinder load type must be bearing or pressure')
    force = area[:,None]*traction
    rhs = np.zeros_like(nodes)
    # Integrals of quadratic triangle Ni are [0,0,0,1/3,1/3,1/3].
    weights = np.array([0,0,0,1/3,1/3,1/3]) if order == 2 else np.ones(3)/3
    np.add.at(rhs, faces.ravel(), (force[:,None,:]*weights[None,:,None]).reshape(-1,3))
    moment = np.cross(nodes-center, rhs).sum(axis=0)
    return rhs.ravel(), {'type':kind, 'facets':len(faces), 'area_m2':float(area.sum()),
                         'pressure_scale_pa':float(pressure_scale), 'total_force_n':rhs.sum(axis=0).tolist(),
                         'moment_about_center_nm':moment.tolist(), 'region_nodes':np.unique(faces).tolist()}
