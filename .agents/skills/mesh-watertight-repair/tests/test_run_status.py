from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_status import command_fingerprint, initial_run_status, prepare_output_dir, run_parameters  # noqa: E402


def args_for(output_dir: Path, *, overwrite: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        input=Path("/models/input.stl"),
        output_dir=output_dir,
        overwrite=overwrite,
        target_name="watertight-exterior-shell",
        group_source_gltf=None,
        remove_name_regex="internal|hidden",
        visibility_grid=720,
        visibility_min_views=1,
        outside_flood_grid=192,
        sealed_exterior_grid=192,
        sealed_exterior_radius_voxels=1,
        sealed_exterior_band_voxels=1.5,
        depth_tolerance=0.0,
        depth_tolerance_bbox_ratio=2.0e-4,
        depth_tolerance_edge_ratio=0.25,
        dilate_rings=1,
        voxel_pitch=0.0,
        voxel_pitch_bbox_divisor=280.0,
        voxel_pitch_bbox_max_extent=None,
        policy_item_limit=500,
        component_filter_thresholds={},
    )


class RunStatusTest(unittest.TestCase):
    def test_overwrite_cleans_known_artifacts_but_preserves_user_notes(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            output_dir = Path(directory_name)
            known = output_dir / "two_stage_report.json"
            source_report = output_dir / "source_material_report.json"
            note = output_dir / "notes.txt"
            known.write_text("stale", encoding="utf-8")
            source_report.write_text("stale", encoding="utf-8")
            note.write_text("keep", encoding="utf-8")
            cfd = output_dir / "cfd"
            cfd.mkdir()
            (cfd / "cfd_surface_candidate.vtp").write_text("stale")

            prepare_output_dir(args_for(output_dir, overwrite=True))

            self.assertFalse(known.exists())
            self.assertFalse(source_report.exists())
            self.assertFalse(cfd.exists())
            self.assertEqual(note.read_text(encoding="utf-8"), "keep")

    def test_run_parameters_contain_only_geometry_and_diagnostic_inputs(self) -> None:
        values = run_parameters(args_for(Path("/tmp/output")))

        self.assertEqual(values["target_name"], "watertight-exterior-shell")
        self.assertEqual(values["surface_strategy"], "projected-shell")
        self.assertEqual(
            set(values),
            {
                "input",
                "purpose",
                "surface_strategy",
                "source_max_error",
                "source_boundary_mode",
                "source_max_recovery_facets",
                "source_union_backend",
                "source_timeout_seconds",
                "stitch_tolerance",
                "repair_timeout_seconds",
                "certify_reverse_fidelity",
                "cfd_max_edge_length",
                "cfd_max_output_faces",
                "target_name",
                "group_source_gltf",
                "remove_name_regex",
                "visibility_grid",
                "visibility_min_views",
                "outside_flood_grid",
                "sealed_exterior_grid",
                "sealed_exterior_radius_voxels",
                "sealed_exterior_band_voxels",
                "depth_tolerance",
                "requested_depth_tolerance",
                "depth_tolerance_bbox_ratio",
                "depth_tolerance_edge_ratio",
                "dilate_rings",
                "voxel_pitch",
                "requested_voxel_pitch",
                "voxel_pitch_source",
                "voxel_pitch_bbox_divisor",
                "voxel_pitch_bbox_max_extent",
                "scaffold_smoothing_iterations",
                "cross_component_jump_ratio",
                "projection_target",
                "policy_item_limit",
                "component_filter_thresholds",
            },
        )

    def test_projection_controls_change_run_fingerprint(self) -> None:
        args = args_for(Path("/tmp/output"))
        original = command_fingerprint(args)
        args.scaffold_smoothing_iterations = 30
        smooth = command_fingerprint(args)
        args.cross_component_jump_ratio = 0.5
        self.assertNotEqual(original, smooth)
        seam = command_fingerprint(args)
        self.assertNotEqual(smooth, seam)
        args.projection_target = Path("/models/reviewed-exterior.vtp")
        self.assertNotEqual(seam, command_fingerprint(args))

    def test_reverse_fidelity_check_changes_run_fingerprint(self) -> None:
        args = args_for(Path("/tmp/output"))
        before = command_fingerprint(args)
        args.certify_reverse_fidelity = True
        self.assertNotEqual(before, command_fingerprint(args))

    def test_initial_status_never_pretends_an_output_is_accepted(self) -> None:
        status = initial_run_status(args_for(Path("/tmp/output")))

        self.assertEqual(status["status"], "running")
        self.assertIsNone(status["accepted_mesh_vtp"])
        self.assertFalse(status["accepted_mesh_available"])


if __name__ == "__main__":
    unittest.main()
