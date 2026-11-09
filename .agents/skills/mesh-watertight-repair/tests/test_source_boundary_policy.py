from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from opening_repair import (  # noqa: E402
    build_preserved_opening_rays,
    cap_loop,
    check_preserved_opening_rays,
)
from repair_policy import build_policy_packet, resolve_policy_decisions  # noqa: E402


def open_box() -> tuple[np.ndarray, np.ndarray, dict]:
    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    points = np.asarray(mesh.vertices, dtype=np.float64)
    top = np.flatnonzero(np.isclose(points[:, 2], 1.0))
    center = points[top].mean(axis=0)
    loop = top[np.argsort(np.arctan2(points[top, 1] - center[1], points[top, 0] - center[0]))]
    faces = np.asarray(mesh.faces, dtype=np.int64)
    faces = faces[~np.all(np.isclose(points[faces, 2], 1.0), axis=1)]
    region = {
        "id": "boundary_loop_0001",
        "simple_closed_loop": True,
        "ordered_vertex_ids": loop.tolist(),
        "normal": [0.0, 0.0, 1.0],
        "planarity_ratio": 0.0,
        "face_ids": list(range(len(faces))),
    }
    return points, faces, region


def functional_preserve(region_id: str) -> list[dict]:
    return [
        {
            "item_id": f"opening_{region_id.rsplit('_', 1)[-1]}",
            "status": "decided",
            "decision": "preserve",
            "semantic_label": "functional_opening",
        }
    ]


class SourceBoundaryPolicyTest(unittest.TestCase):
    def test_cfd_reviews_every_boundary_and_purpose_change_invalidates_decisions(self) -> None:
        _, _, region = open_box()
        region.update(requires_policy=False, classification="near_coincident_part_seam")
        inventory = {"boundary_regions": {"items": [region]}}
        cfd = build_policy_packet(inventory, "watertight-exterior-shell", max_items=10,
                                  run_context={"purpose": "cfd"})
        geometry = build_policy_packet(inventory, "watertight-exterior-shell", max_items=10,
                                       run_context={"purpose": "watertight"})
        self.assertEqual(cfd["item_count_total"], 1)
        self.assertEqual(geometry["item_count_total"], 0)
        printed = build_policy_packet(inventory, "print-material", max_items=10,
                                      run_context={"purpose": "print"})
        decision = dict(cfd["items"][0]["decision_template"])
        decision.update(decision="cap", semantic_label="missing_cover", shape_prior="planar")
        resolved = resolve_policy_decisions(printed, [decision])
        self.assertEqual(resolved[0]["status"], "pending_review")
        self.assertTrue(resolved[0]["stale_policy_decision"])

    def test_open_box_retains_source_clear_rays_and_cap_blocks_them(self) -> None:
        points, faces, region = open_box()
        rays = build_preserved_opening_rays(points, faces, [region], functional_preserve(region["id"]))

        self.assertEqual(rays["status"], "ready")
        self.assertGreater(len(rays["origins"]), 0)
        self.assertTrue(check_preserved_opening_rays(points, faces, rays)["passed"])

        patch = cap_loop(points, faces, region)
        capped_faces = np.vstack((faces, patch))
        report = check_preserved_opening_rays(points, capped_faces, rays)
        self.assertFalse(report["passed"])
        self.assertGreater(report["blocked_ray_count"], 0)
        self.assertIn(region["id"], report["blocked_by_region"])

    def test_open_disk_is_not_reported_as_a_clear_functional_opening(self) -> None:
        points = np.asarray(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        region = {
            "id": "boundary_loop_0001",
            "simple_closed_loop": True,
            "ordered_vertex_ids": [0, 1, 2, 3],
            "normal": [0.0, 0.0, 1.0],
            "planarity_ratio": 0.0,
            "face_ids": [0, 1],
        }

        rays = build_preserved_opening_rays(points, faces, [region], functional_preserve(region["id"]))

        self.assertEqual(rays["status"], "unsupported")
        self.assertEqual(rays["origins"], [])
        self.assertEqual(rays["unsupported_regions"], [region["id"]])
        checked = check_preserved_opening_rays(points, faces, rays)
        self.assertFalse(checked["passed"])
        self.assertEqual(checked["status"], "unsupported")

    def test_nonplanar_loop_is_explicitly_unsupported(self) -> None:
        points, faces, region = open_box()
        points[region["ordered_vertex_ids"][0], 2] += 0.5
        region["planarity_ratio"] = 0.2

        rays = build_preserved_opening_rays(points, faces, [region], functional_preserve(region["id"]))

        self.assertEqual(rays["status"], "unsupported")
        self.assertEqual(rays["regions"][0]["reason"], "functional opening loop is nonplanar")

    def test_region_normal_must_match_the_opening_plane(self) -> None:
        points, faces, region = open_box()
        region["normal"] = [1.0, 0.0, 0.0]

        rays = build_preserved_opening_rays(points, faces, [region], functional_preserve(region["id"]))

        self.assertEqual(rays["status"], "unsupported")
        self.assertEqual(rays["regions"][0]["reason"], "region normal is not aligned with the opening plane")

    def test_missing_region_for_explicit_functional_decision_is_not_skipped(self) -> None:
        decision = functional_preserve("boundary_loop_0007")[0]

        rays = build_preserved_opening_rays(np.zeros((3, 3)), np.asarray([[0, 1, 2]]), [], [decision])

        self.assertEqual(rays["status"], "unsupported")
        self.assertEqual(rays["unsupported_regions"], [decision["item_id"]])
        self.assertFalse(check_preserved_opening_rays(np.zeros((3, 3)), np.asarray([[0, 1, 2]]), rays)["passed"])


if __name__ == "__main__":
    unittest.main()
