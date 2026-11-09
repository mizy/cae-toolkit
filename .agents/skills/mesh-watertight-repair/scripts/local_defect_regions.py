"""Map exact face defects to bounded, source-indexed repair regions."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from itertools import pairwise

import numpy as np
from mesh_metrics import UnionFind, edge_topology, face_component_labels


def _face_neighbors(face_id, faces, edge_faces):
    face = faces[face_id]
    for left, right in zip(face, np.roll(face, -1)):
        edge = (min(int(left), int(right)), max(int(left), int(right)))
        yield from (other for other in edge_faces[edge] if other != face_id)


def _boundary(faces, rows, edge_faces):
    selected = set(map(int, rows))
    directed, nonmanifold = [], []
    for edge, incident in edge_faces.items():
        inside = [row for row in incident if row in selected]
        if not inside:
            continue
        if len(incident) > 2 and len(inside) < len(incident):
            nonmanifold.append(list(edge))
        if len(inside) == 1:
            face = faces[inside[0]]
            directed.append(
                next((int(a), int(b)) for a, b in zip(face, np.roll(face, -1)) if {int(a), int(b)} == set(edge))
            )
    outgoing, incoming = defaultdict(list), defaultdict(list)
    for index, (left, right) in enumerate(directed):
        outgoing[left].append(index)
        incoming[right].append(index)
    vertices = set(outgoing) | set(incoming)
    branched = sorted(vertex for vertex in vertices if len(outgoing[vertex]) != 1 or len(incoming[vertex]) != 1)
    if branched:
        return directed, [], {"status": "branched", "vertices": branched, "nonmanifold_source_edges": nonmanifold}
    unused, loops = set(range(len(directed))), []
    while unused:
        first = min(unused)
        start = directed[first][0]
        current, loop, visited = start, [start], set()
        while True:
            edge_id = outgoing[current][0]
            if edge_id not in unused or edge_id in visited:
                return directed, loops, {"status": "non_simple", "nonmanifold_source_edges": nonmanifold}
            visited.add(edge_id)
            unused.remove(edge_id)
            current = directed[edge_id][1]
            loop.append(current)
            if current == start:
                break
        loops.append(loop)
    return (
        directed,
        loops,
        {
            "status": "nonmanifold_source_edge"
            if nonmanifold
            else "closed_loops"
            if loops
            else "closed_surface_region",
            "loop_count": len(loops),
            "nonmanifold_source_edges": nonmanifold,
        },
    )


# @entry: exact face evidence controls a bounded source-face patch proposal.
def extract_defect_regions(
    points,
    faces,
    *,
    seed_faces=None,
    intersection_pairs=None,
    source_face_ids=None,
    halo_rings=2,
    roi_padding=0.0,
    max_region_faces=10000,
    max_total_faces=50000,
):
    """Return source-indexed ROIs; never move, cap, or remove input geometry."""
    points, faces = np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    seeds = np.asarray([] if seed_faces is None else seed_faces, dtype=np.int64).reshape(-1)
    pairs = np.asarray([] if intersection_pairs is None else intersection_pairs, dtype=np.int64).reshape(-1, 2)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.min() < 0
        or faces.max() >= len(points)
    ):
        raise ValueError("expected finite points and valid nonempty triangle faces")
    if not isinstance(halo_rings, int) or halo_rings < 0:
        raise ValueError("halo_rings must be a nonnegative integer")
    if not np.isfinite(roi_padding) or roi_padding < 0 or max_region_faces < 1 or max_total_faces < max_region_faces:
        raise ValueError("invalid ROI padding or face budgets")
    source = (
        np.arange(len(faces), dtype=np.int64)
        if source_face_ids is None
        else np.asarray(source_face_ids, dtype=np.int64)
    )
    if source.shape != (len(faces),) or len(np.unique(source)) != len(source):
        raise ValueError("source_face_ids must uniquely map every face row")
    evidence = np.unique(np.r_[seeds, pairs.ravel()])
    fingerprint = hashlib.sha256(points.tobytes() + faces.tobytes()).hexdigest()
    if not len(evidence):
        return {
            "status": "no_defects",
            "regions": [],
            "source_faces": len(faces),
            "source_geometry_sha256": fingerprint,
        }
    if evidence.min() < 0 or evidence.max() >= len(faces):
        raise ValueError("defect evidence contains an invalid face row")

    _, edge_faces = edge_topology(faces)
    evidence_index = {int(row): index for index, row in enumerate(evidence)}
    clusters = UnionFind(len(evidence))
    for left, right in pairs:
        clusters.union(evidence_index[int(left)], evidence_index[int(right)])
    for incident in edge_faces.values():
        adjacent = [evidence_index[row] for row in incident if row in evidence_index]
        for left, right in pairwise(adjacent):
            clusters.union(left, right)
    seeded = defaultdict(list)
    for row, index in evidence_index.items():
        seeded[clusters.find(index)].append(row)

    labels = face_component_labels(len(faces), edge_faces)
    reports, total = [], 0
    for seed_rows in sorted(seeded.values(), key=min):
        region, frontier = set(seed_rows), deque((row, 0) for row in seed_rows)
        over_budget = len(region) > max_region_faces
        while frontier and not over_budget:
            row, depth = frontier.popleft()
            if depth >= halo_rings:
                continue
            for other in _face_neighbors(row, faces, edge_faces):
                if other not in region:
                    region.add(other)
                    if len(region) > max_region_faces or total + len(region) > max_total_faces:
                        over_budget = True
                        break
                    frontier.append((other, depth + 1))
        if over_budget or total + len(region) > max_total_faces:
            reports.append(
                {
                    "status": "face_budget_exceeded",
                    "seed_face_ids": sorted(int(source[row]) for row in seed_rows),
                    "seed_face_count": len(seed_rows),
                    "max_region_faces": max_region_faces,
                }
            )
            continue
        rows = np.asarray(sorted(region), dtype=np.int64)
        total += len(rows)
        directed, loops, boundary = _boundary(faces, rows, edge_faces)
        region_points = points[faces[rows].ravel()]
        reports.append(
            {
                "id": "local_patch_" + hashlib.sha256((fingerprint + str(rows.tolist())).encode()).hexdigest()[:16],
                "face_rows": rows.astype(int).tolist(),
                "seed_face_rows": sorted(map(int, seed_rows)),
                "status": "ready"
                if boundary["status"] in ("closed_loops", "closed_surface_region")
                else "review_required",
                "source_face_ids": np.sort(source[rows]).astype(int).tolist(),
                "seed_face_ids": sorted(int(source[row]) for row in seed_rows),
                "source_component_ids": np.unique(labels[rows]).astype(int).tolist(),
                "bbox": [
                    (region_points.min(axis=0) - roi_padding).tolist(),
                    (region_points.max(axis=0) + roi_padding).tolist(),
                ],
                "boundary_edges": directed,
                "boundary_loops": loops,
                "boundary": boundary,
                "halo_rings": halo_rings,
                "faces": len(rows),
                "padding": float(roi_padding),
            }
        )
    owners = defaultdict(list)
    for region_index, region in enumerate(reports):
        for face_id in region.get("source_face_ids", []):
            owners[face_id].append(region_index)
    overlaps = [{"source_face_id": row, "regions": indices} for row, indices in owners.items() if len(indices) > 1]
    boxes = [row["bbox"] for row in reports if row.get("bbox")]
    roi_overlaps = []
    roi_check = "computed"
    if len(boxes) > 512:
        roi_check = "budget_exceeded"
    else:
        for left in range(len(boxes)):
            for right in range(left + 1, len(boxes)):
                if np.all(np.asarray(boxes[left][0]) <= boxes[right][1]) and np.all(
                    np.asarray(boxes[right][0]) <= boxes[left][1]
                ):
                    roi_overlaps.append([left, right])
    return {
        "status": "computed",
        "regions": reports,
        "source_faces": len(faces),
        "source_geometry_sha256": fingerprint,
        "region_faces": total,
        "overlapping_region_face_ids": overlaps,
        "roi_overlap_status": roi_check,
        "overlapping_region_rois": roi_overlaps,
    }
