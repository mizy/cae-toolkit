from __future__ import annotations

from pathlib import Path
from typing import Any

from repair_report import topology_snapshot, trace_row, two_stage_repair_report
from source_projected_contract import (
    build_projection_gates,
    projection_comparisons,
    projection_unhandled_items,
    watertight_shell_policy,
)
from source_preserving_repair import (
    accepted_outputs,
    build_decision,
    build_ignored_outputs,
    input_truth,
)


def build_report(
    args: Any,
    stage_reports: dict[str, dict[str, Any]],
    extraction: dict[str, Any],
    deterministic_passes: list[dict[str, Any]],
    inventory_before: dict[str, Any],
    inventory_after: dict[str, Any],
    policy_packet: dict[str, Any],
    policy_decisions: list[dict[str, Any]],
    remesh: dict[str, Any],
    outputs: dict[str, Any],
    group_filter: dict[str, Any] | None,
    projection_report: dict[str, Any],
    post_write_validation: dict[str, Any],
    projected_visual_drift: dict[str, Any],
) -> dict[str, Any]:
    candidate_path = outputs["source_preserving_candidate_vtp"]
    closure_proxy_path = outputs["closure_proxy_vtp"]
    projected_path = outputs.get("source_projected_watertight_candidate_vtp")
    projected_produced = bool(outputs.get("source_projected_watertight_candidate_produced"))
    projected_metrics = stage_reports["source_projected_watertight_candidate"]
    comparisons = projection_comparisons(
        stage_reports["deterministic_repair_candidate"],
        stage_reports["closure_proxy"],
        projected_metrics,
        projection_report,
        projected_visual_drift,
        post_write_validation,
    )
    shell_policy = watertight_shell_policy(args.target_name)
    if getattr(args, "purpose", "watertight") == "cfd":
        shell_policy["semantic_review_applicability"] = "required_for_acceptance"
        shell_policy["source"] = "cfd_candidate_requires_opening_review"
    gates = build_projection_gates(
        projected_metrics,
        comparisons,
        deterministic_passes,
        projected_path,
        projected_produced,
        closure_proxy_path,
        projection_report,
        shell_policy,
    )
    decision = build_decision(gates, projected_path)
    if getattr(args, "purpose", "watertight") == "cfd" or args.policy_decisions:
        from source_preserving_repair import gate
        items = policy_packet.get("items", [])
        applied = (not policy_packet.get("truncated", True)
                   and policy_packet.get("item_count_total") == len(items) == len(policy_decisions)
                   and all(row.get("status") == "decided" and row.get("applied_by_pass_id")
                           for row in policy_decisions))
        gates["opening_policy_resolved"] = gate(applied, policy_decisions,
            "every detected opening reviewed and executed by deterministic geometry operators",
            "opening decisions are missing, incomplete or not executed")
        decision = build_decision(gates, projected_path)
    outputs = accepted_outputs(
        outputs,
        decision,
        projected_path or candidate_path,
        projected_path or candidate_path,
        source_fallback_path=candidate_path,
    )
    decision = {
        **decision,
        "outcome": outputs["mesh_result"]["status"],
        "source_fallback_path": outputs.get("source_fallback_vtp"),
    }
    ignored_outputs = build_ignored_outputs(
        closure_proxy_path=closure_proxy_path,
        stage1_path=outputs["stage1_exterior_candidate_vtp"],
        candidate_path=candidate_path,
        previews=outputs.get("previews", []),
        group_filter=group_filter,
        output_dir=args.output_dir,
    )
    if decision["status"] != "accepted" and projected_path:
        ignored_outputs.append({
            "path": projected_path,
            "kind": "rejected_source_projected_watertight_candidate",
            "reason": "source-projected shell did not pass every engineering acceptance gate",
            "safe_for_acceptance": False,
        })
    derived_artifacts = [
        outputs["stage1_exterior_candidate_vtp"],
        candidate_path,
        closure_proxy_path,
    ]
    if projected_path:
        derived_artifacts.append(projected_path)
    truth = input_truth(Path(args.input), group_filter, derived_artifacts)
    truth["acceptance_source"] = (
        "retained_prepared_source_surfaces_plus_certified_reconstruction_and_source_projection"
    )
    return {
        "decision": decision,
        "input": {
            "path": str(args.input),
            "kind": "mesh",
            "units": "unknown",
            "coordinate_convention": "unknown",
            "group_metadata_available": group_filter is not None,
        },
        "input_truth": truth,
        "output_contract": {
            "input_kind": "mesh",
            "output_kind": "watertight_mesh",
            "repair_domain": "mesh_domain_source_projected_watertight_shell",
            "cad_output": {
                "supported": False,
                "reason": "this worker does not perform reverse CAD fitting",
            },
            "accepted_final_rule": (
                "decision.final_output_path == outputs.accepted_mesh_vtp == "
                "outputs.source_projected_watertight_candidate_vtp; "
                "closure_proxy_vtp supplies connectivity only"
            ),
            "rejected_result_rule": (
                "outputs.mesh_result identifies the source-preserving fallback as unrepaired"
            ),
        },
        "target": {
            "name": args.target_name,
            "opening_policy": ("review every detected opening for CFD before accepting the shell"
                               if getattr(args, "purpose", "watertight") == "cfd"
                               else "seal all openings for the requested watertight exterior shell"),
            "shell_policy": shell_policy,
            "component_count_role": "diagnostic_only",
        },
        "parameters": parameters_report(args),
        "limitations": limitations(group_filter),
        "group_filter": group_filter,
        "stages": stages_report(args, outputs, stage_reports, extraction, remesh),
        "inventory_before": inventory_before,
        "deterministic_passes": deterministic_passes,
        "unresolved_policy_packet": policy_packet,
        "policy_decisions": policy_decisions,
        "inventory_after": inventory_after,
        "source_projection": projection_report,
        "comparisons": comparisons,
        "gates": gates,
        "outputs": outputs,
        "ignored_outputs": ignored_outputs,
        "unhandled_items": projection_unhandled_items(comparisons),
        "repair_report": two_stage_repair_report(
            stage_reports,
            extraction,
            group_filter,
            remesh,
            deterministic_passes,
            decision=decision,
        ),
    }


def stages_report(
    args: Any,
    outputs: dict[str, Any],
    stage_reports: dict[str, dict[str, Any]],
    extraction: dict[str, Any],
    remesh: dict[str, Any],
) -> dict[str, Any]:
    projected_path = outputs.get("source_projected_watertight_candidate_vtp")
    projected_produced = outputs.get("source_projected_watertight_candidate_produced")
    return {
        "original_input": {"path": str(args.input), "metrics": stage_reports["original_input"]},
        "stage1_exterior_candidate": {
            "path": outputs["stage1_exterior_candidate_vtp"],
            "source": "original_input",
            "provenance_fields": ["source_triangle_index"],
            "metrics": stage_reports["stage1_exterior_candidate"],
            "extraction": extraction,
        },
        "deterministic_repair_candidate": {
            "path": outputs["source_preserving_candidate_vtp"],
            "source": "stage1_exterior_candidate",
            "provenance_fields": ["source_triangle_index"],
            "metrics": stage_reports["deterministic_repair_candidate"],
        },
        "closure_proxy": {
            "path": outputs["closure_proxy_vtp"],
            "implicit_field_path": outputs.get("implicit_field_npz"),
            "source": "deterministic_repair_candidate",
            "metrics": stage_reports["closure_proxy"],
            "remesh": remesh,
            "accepted_final_geometry": False,
            "role": "immutable_watertight_connectivity_template",
        },
        "source_projected_watertight_candidate": {
            "path": projected_path,
            "source": "closure_proxy_topology_plus_deterministic_source_surface_positions",
            "metrics": stage_reports.get("source_projected_watertight_candidate"),
            "status": (
                "accepted"
                if outputs.get("accepted_mesh_vtp") == projected_path
                else "candidate_produced_not_accepted"
                if projected_produced
                else "missing"
            ),
            "provenance_fields": [
                "source_projection_applied",
                "source_projection_fallback",
                "nearest_source_face_id",
                "source_projection_rejection_mask",
            ],
            "accepted_final_geometry": outputs.get("accepted_mesh_vtp") == projected_path,
        },
    }


def parameters_report(args: Any) -> dict[str, Any]:
    return {
        "purpose": getattr(args, "purpose", "watertight"),
        "stitch_tolerance": getattr(args, "stitch_tolerance", 0.),
        "repair_timeout_seconds": getattr(args, "repair_timeout_seconds", None),
        "cfd_max_edge_length": getattr(args, "cfd_max_edge_length", None),
        "cfd_max_output_faces": getattr(args, "cfd_max_output_faces", 2_000_000),
        "visibility_grid": args.visibility_grid,
        "visibility_min_views": getattr(args, "visibility_min_views", 1),
        "outside_flood_grid": getattr(args, "outside_flood_grid", None),
        "depth_tolerance": args.depth_tolerance,
        "requested_depth_tolerance": getattr(args, "requested_depth_tolerance", args.depth_tolerance),
        "depth_tolerance_bbox_ratio": getattr(args, "depth_tolerance_bbox_ratio", None),
        "depth_tolerance_edge_ratio": getattr(args, "depth_tolerance_edge_ratio", None),
        "dilate_rings": args.dilate_rings,
        "voxel_pitch": args.voxel_pitch,
        "requested_voxel_pitch": getattr(args, "requested_voxel_pitch", args.voxel_pitch),
        "voxel_pitch_source": getattr(args, "voxel_pitch_source", "explicit"),
        "voxel_pitch_bbox_divisor": getattr(args, "voxel_pitch_bbox_divisor", None),
        "voxel_pitch_bbox_max_extent": getattr(args, "voxel_pitch_bbox_max_extent", None),
        "sdf_band_voxels": getattr(args, "sdf_band_voxels", None),
        "sdf_smoothing_sigma": getattr(args, "sdf_smoothing_sigma", None),
        "scaffold_smoothing_iterations": getattr(args, "scaffold_smoothing_iterations", 0),
        "cross_component_jump_ratio": getattr(args, "cross_component_jump_ratio", None),
        "projection_target": str(args.projection_target) if getattr(args, "projection_target", None) else None,
        "max_sdf_memory_gb": getattr(args, "max_sdf_memory_gb", None),
        "remove_name_regex": args.remove_name_regex,
        "policy_decisions": str(args.policy_decisions) if args.policy_decisions else None,
        "policy_item_limit": args.policy_item_limit,
        "component_filter_thresholds": getattr(args, "component_filter_thresholds", None),
    }


def limitations(group_filter: dict[str, Any] | None) -> list[str]:
    result = [
        "Watertight material boundaries may retain genuine through-passages. Sub-resolution openings can close during voxel reconstruction and require separate semantic aperture validation.",
        "A sealed voxel shell is partitioned by six-connected far-field flood before marching cubes supplies immutable watertight connectivity and proxy fallback positions.",
        "Only correspondence-safe vertices are placed on source triangles; every rejected match retains its certified proxy position with a reason code.",
        "Certified closed source shells are retained unchanged. Reconstruction currently selects whole remaining surface components, not minimal defect patches inside coupled shells.",
        "The source-projected candidate must survive serialized readback and exact all-face self-intersection certification.",
    ]
    if group_filter:
        result.append(
            "GLTF names and flattened face ranges are diagnostic metadata only and never remove primary mesh triangles."
        )
    else:
        result.extend([
            "No CAD/GLTF group tree was provided for this run.",
            "Stage 1 uses mesh-only multi-view outermost visibility instead of group hide/show.",
        ])
    return result


def record_cfd_refinement(report: dict[str, Any], refinement: dict[str, Any]) -> None:
    """Own final-path selection after the optional CFD stage; keep accepted shell truth separate."""
    shell_path = report["outputs"]["accepted_mesh_vtp"]
    refined_path = refinement["outputs"]["cfd_surface_candidate_vtp"]
    report["cfd_surface_refinement"] = refinement
    report["stages"]["source_projected_watertight_candidate"]["accepted_final_geometry"] = False
    report["stages"]["cfd_refined_surface"] = refinement["stages"]["cfd_refined_surface"]
    report["gates"].update({f"cfd_{name}": value for name, value in refinement["gates"].items()})
    decision = build_decision(report["gates"], refined_path)
    report["outputs"] = accepted_outputs(report["outputs"], decision, refined_path, refined_path, source_fallback_path=shell_path)
    report["outputs"]["watertight_mesh_vtp"] = shell_path
    report["outputs"]["cfd_surface_candidate_vtp"] = refined_path
    report["outputs"]["mesh_result"]["engineering_ready"] = False
    if decision["status"] == "rejected":
        report["outputs"]["mesh_result"].update(
            role="accepted_watertight_fallback",
            warning="CFD surface criteria failed; the fallback is an accepted watertight shell, not a CFD-qualified surface.",
        )
        report["ignored_outputs"].append({"path": refined_path, "kind": "rejected_cfd_surface", "reason": "CFD refinement gates failed", "safe_for_acceptance": False})
    report["decision"] = {**decision, "outcome": report["outputs"]["mesh_result"]["status"], "source_fallback_path": report["outputs"].get("source_fallback_vtp")}
    report["output_contract"]["accepted_final_rule"] = "decision.final_output_path == outputs.accepted_mesh_vtp == outputs.cfd_surface_candidate_vtp; watertight_mesh_vtp retains the accepted pre-refinement shell"
    report["output_contract"]["rejected_result_rule"] = "CFD rejection retains the accepted watertight_mesh_vtp as fallback; this does not qualify it as a CFD surface"
    report["input_truth"]["derived_artifacts"].append(refined_path)
    metrics = refinement["stages"]["cfd_refined_surface"]["metrics"]
    repair = report["repair_report"]
    repair["geometry_to_mesh_trace"].append(trace_row("cfd_refined_surface", metrics, refinement["method"]))
    repair["change_summary"]["source_projected_watertight_shell"]["accepted_final_geometry"] = False
    repair["change_summary"]["cfd_refinement"] = {"parameters": refinement["parameters"], "accepted_final_geometry": decision["status"] == "accepted", "source": shell_path}
    repair["defect_matrix"]["cfd_refined_surface"] = topology_snapshot(metrics)
    repair["requested_capabilities"]["cfd_surface_refinement"] = {"checked": True, "passed": decision["status"] == "accepted", "volume_mesh_validated": False}
    report["limitations"].extend(refinement["limitations"])
