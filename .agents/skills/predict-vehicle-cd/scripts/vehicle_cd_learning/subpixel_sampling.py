"""Sample a real orthographic depth buffer and resolve it onto the model pixel centres."""

import ctypes
import functools
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .render import VIEW_AXES

NATIVE = Path(__file__).resolve().parents[2] / "native/subpixel_depth.cpp"


@functools.lru_cache(maxsize=1)
def _sampler():
    """Compile once per process. All geometry and the first-hit buffer stay local."""
    compiler = shutil.which("clang++") or shutil.which("c++")
    if not compiler:
        raise RuntimeError("The subpixel depth renderer needs a local C++17 compiler")
    with tempfile.TemporaryDirectory(prefix="aero-subpixel-") as temp:
        lib = Path(temp) / "depth.so"
        subprocess.run(
            [
                compiler,
                "-O3",
                "-std=c++17",
                "-ffp-contract=off",
                "-fPIC",
                "-shared",
                str(NATIVE),
                "-o",
                str(lib),
            ],
            check=True,
        )
        sample = ctypes.CDLL(str(lib)).sample_depth
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS")
    i32 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")
    sample.argtypes = [
        f32,
        i32,
        ctypes.c_int64,
        f32,
        *([ctypes.c_int] * 3),
        *([ctypes.c_double] * 4),
    ]
    sample.restype = None
    return sample


def project_vertices(points, view, size):
    """Use the same coarse pixel centres, depth interval and margin as render._rasterize."""
    h, v, d, sign = VIEW_AXES[view]
    scale = (size - 8) / 1.3
    lower = (0.0, 1.3) if d == 2 else (-0.65, 0.65)
    z = lower[1] - points[:, d] if sign > 0 else points[:, d] - lower[0]
    x = (points[:, h] - (0.0 if h == 2 else -0.65)) * scale + 4.0
    y = size - 1 - ((points[:, v] - (0.0 if v == 2 else -0.65)) * scale + 4.0)
    return np.ascontiguousarray(np.column_stack((x, y, z / 1.3)), np.float32)


def acquisition_offset(phase, factor):
    """Place one fine sample on the model ray; returned offset is in coarse pixels."""
    phase = np.asarray(phase, dtype=float)
    if phase.shape != (2,) or not np.isfinite(phase).all() or np.abs(phase).max() > 0.5:
        raise ValueError("Expected a known two-axis phase in [-0.5,+0.5] model pixels")
    return phase - np.rint(factor * phase) / factor


def capture(projected, faces, size, factor, phase, sample_offset, depth=None):
    """@entry Capture actual first hits on a fine lattice with a one-model-pixel halo."""
    if factor not in (1, 4):
        raise ValueError("Supported sampling factors are 1 and 4")
    for position in (phase, sample_offset):
        if (
            np.shape(position) != (2,)
            or not np.isfinite(position).all()
            or np.max(np.abs(position)) > 0.5
        ):
            raise ValueError("Invalid subpixel position")
    width = (size + 2) * factor
    if depth is None:
        depth = np.full((width, width), np.inf, np.float32)
    if depth.shape != (width, width):
        raise ValueError("Depth buffer dimensions disagree with the requested lattice")
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or projected.ndim != 2
        or projected.shape[1] != 3
    ):
        raise ValueError("Expected triangle indices and projected xyz vertices")
    if (
        not len(faces)
        or faces.min() < 0
        or faces.max() >= len(projected)
        or not np.isfinite(projected).all()
    ):
        raise ValueError("Invalid source triangles")
    _sampler()(
        projected,
        faces,
        len(faces),
        depth,
        width,
        factor,
        factor + factor // 2,
        *phase,
        *sample_offset,
    )
    return depth


def _footprint(depth, size, factor, samples):
    centers = np.arange(size, dtype=int) * factor + factor + factor // 2
    weighted = np.zeros((size, size), dtype=np.float64)
    coverage = np.zeros_like(weighted)
    for x, y, weight in samples:
        z = depth[np.ix_(centers + y, centers + x)]
        mask = np.isfinite(z)
        weighted += weight * np.where(mask, z, 0.0)
        coverage += weight * mask
    z = np.divide(weighted, coverage, out=np.zeros_like(weighted), where=coverage > 0)
    return np.stack((z, coverage)).astype(np.float32)


def resolve(depth, size, factor, sample_position=(0.0, 0.0)):
    """@entry Read actual depth/coverage at a given subpixel index; interpolate adjacent hits if needed."""
    if (
        not np.isfinite(sample_position).all()
        or np.shape(sample_position) != (2,)
        or np.max(np.abs(sample_position)) > 1.0
    ):
        raise ValueError("Requested sample falls outside the captured halo")
    scaled = np.asarray(sample_position) * factor
    lower = np.floor(scaled).astype(int)
    dx, dy = scaled - lower
    samples = [
        (lower[0] + x, lower[1] + y, wx * wy)
        for x, wx in enumerate((1 - dx, dx))
        for y, wy in enumerate((1 - dy, dy))
        if wx * wy > 0.0
    ]
    return _footprint(depth, size, factor, samples)


def area_average(depth, size, factor):
    """Symmetric trapezoidal quadrature of coverage and visible depth on the displaced pixel footprint."""
    if factor != 4:
        raise ValueError("Area quadrature uses the four-times fine grid")
    weights = (0.125, 0.25, 0.25, 0.25, 0.125)
    samples = [
        (x - 2, y - 2, wx * wy)
        for x, wx in enumerate(weights)
        for y, wy in enumerate(weights)
    ]
    return _footprint(depth, size, factor, samples)


def registered_views(parts, size, view_phases):
    """@entry Render the model rays directly on a sensor with a known per-view shift.

    Parts are triangles in the *shared, already normalized* physical frame of the
    checkpoint. Positive displacements move the projected geometry right/down.
    The rasterizer places its sample at the corresponding location on the sensor.
    """
    if not parts or set(view_phases) != set(VIEW_AXES):
        raise ValueError(
            "Complete normalized geometry and five declared camera phases required"
        )
    result = {}
    for view, phase in view_phases.items():
        depth = None
        for points, faces in parts:
            depth = capture(
                project_vertices(points, view, size),
                np.ascontiguousarray(faces, np.int32),
                size,
                1,
                phase,
                phase,
                depth,
            )
        result[view] = resolve(depth, size, 1)
    return result
