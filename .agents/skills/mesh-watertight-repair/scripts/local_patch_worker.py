"""Run one reviewed local voxel reconstruction or exact intersection cut under a hard deadline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def _write_report(report, output):
    temporary = output / "local_patch_report.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(output / "local_patch_report.json")


def run_file(args):
    import numpy as np
    from html_report import write_html_report
    from local_patch_repair import repair_mesh_patch
    from mesh_io import read_surface, triangle_faces
    from source_projected_validation import roundtrip_validation

    started = time.monotonic()
    source = read_surface(args.input)
    faces = triangle_faces(source)
    fingerprint = hashlib.sha256(
        np.asarray(source.points, dtype=np.float64).tobytes() + np.asarray(faces, dtype=np.int64).tobytes()
    ).hexdigest()
    packet = json.loads(args.regions.read_text())
    if packet.get("source_geometry_sha256") != fingerprint:
        raise ValueError("local region packet does not match the current source geometry")
    regions = [row for row in packet.get("regions", []) if row.get("id") == args.region]
    if len(regions) != 1 or regions[0].get("status") != "ready":
        raise ValueError("one ready source-bound region is required")
    region = regions[0]
    actions = (
        ("cut_intersections", "local_intersection_cut")
        if args.method == "intersection-cut"
        else ("reconstruct", "local_reconstruction")
    )
    intent = region.get("decision", {"action": actions[-1], "source": "explicit_cli_region_selection"})
    if not isinstance(intent, dict) or intent.get("action") not in actions:
        raise ValueError("region intent does not authorize the selected repair method")
    candidate, result = repair_mesh_patch(
        source,
        region["face_rows"],
        method=args.method,
        pitch=args.pitch,
        max_error=args.max_error,
        max_voxels=args.max_voxels,
        max_seam_angle_increase=args.max_seam_angle_increase,
        unreliable_source_face_rows=region.get("defect_face_rows", region.get("seed_face_rows")),
    )
    path = args.output_dir / "local_patch_candidate.vtp"
    candidate.save(path)
    if result["local_repair_accepted"]:
        readback = roundtrip_validation(path, candidate.points, triangle_faces(candidate))
        result["serialized_readback"] = readback
        result["local_repair_accepted"] = bool(readback["passed"])
        result["accepted"] = bool(result["accepted"] and readback["passed"])
        if not readback["passed"]:
            result["failure_reason"] = "serialized local candidate failed readback"
    reason = (
        result.get("failure_reason")
        if not result["local_repair_accepted"]
        else "local replacement passed; source defects remain outside this region"
        if not result["accepted"]
        else "local replacement, full component geometry and readback passed"
    )
    report = {
        **result,
        "input": {"path": str(args.input.resolve())},
        "intent": intent,
        "region": {"id": args.region, "source_geometry_sha256": fingerprint},
        "decision": {"status": "accepted" if result["accepted"] else "rejected", "reason": reason},
        "outputs": {
            "accepted_mesh_vtp": str(path.resolve()) if result["accepted"] else None,
            "accepted_local_candidate_vtp": str(path.resolve()) if result["local_repair_accepted"] else None,
            "mesh_previews": [
                {"label": "Original source / 原始模型", "path": str(args.input.resolve())},
                {
                    "label": "Checked local replacement / 局部修复通过，整体状态见报告"
                    if result["local_repair_accepted"]
                    else "Unchanged source after rejected repair / 修复拒绝，原网格保留"
                    if result.get("rolled_back")
                    else "Rejected serialized candidate / 文件回读未通过的候选",
                    "path": str(path.resolve()),
                },
            ],
        },
        "limitations": [
            "One explicitly selected source region; this does not establish full-car repair or CFD/printing readiness.",
            "Voxel signing supports one closed positive component and one simple cut loop; cutting requires a closed positive source context.",
            "Distance bounds use floating-point queries; exact predicates certify intersections.",
            "Region selection and material intent must be reviewed upstream; the worker does not infer intent.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_report(report, args.output_dir)
    write_html_report(
        report, args.output_dir / "local_patch_report.html", "Source-preserving local repair / 原面保留局部修复"
    )
    return 0 if report["accepted"] else 3 if report["local_repair_accepted"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("voxel", "voxel-band", "intersection-cut"), default="voxel")
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--max-error", type=float, required=True)
    parser.add_argument("--max-voxels", type=int, default=2_000_000)
    parser.add_argument("--max-seam-angle-increase", type=float, default=20.0)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.method in ("voxel", "voxel-band") and args.pitch is None:
        parser.error("--pitch is required for voxel reconstruction")
    if args.method == "intersection-cut" and args.pitch is not None:
        parser.error("--pitch applies only to voxel reconstruction")
    import math

    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("timeout must be finite and positive")
    if args._worker:
        try:
            return run_file(args)
        except (ValueError, RuntimeError, OSError) as exc:
            _write_report(
                {
                    "accepted": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "outputs": {"accepted_mesh_vtp": None},
                },
                args.output_dir,
            )
            return 2
    args.output_dir.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--_worker"]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=args.timeout_seconds)
        if code in (0, 2, 3):
            return code
        reason = f"local patch worker exited unexpectedly with code {code}"
    except subprocess.TimeoutExpired:
        from mesh_timeout import _stop_group

        _stop_group(process)
        code, reason = 124, f"local patch exceeded {args.timeout_seconds:g} seconds"
    except BaseException:
        from mesh_timeout import _stop_group

        _stop_group(process)
        raise
    (args.output_dir / "local_patch_report.html").unlink(missing_ok=True)
    _write_report(
        {"accepted": False, "failure_reason": reason, "outputs": {"accepted_mesh_vtp": None}}, args.output_dir
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
