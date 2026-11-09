"""Deduplicate exact WindsorML mesh aliases while preserving source provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path

SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}


def _stl_payload(path: Path, asset: dict) -> tuple[int, bytes]:
    data = path.read_bytes()
    if (
        len(data) != asset["bytes"]
        or hashlib.sha256(data).hexdigest() != asset["sha256"]
    ):
        raise ValueError(f"Source STL does not match its recorded checksum: {path}")
    if len(data) < 84:
        raise ValueError(f"Invalid binary STL: {path}")
    triangles = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * triangles:
        raise ValueError(f"Invalid binary STL size or triangle count: {path}")
    return triangles, data[80:]


def _geometry_conditions(case: dict) -> tuple:
    geometry = case["geometry"]
    return (
        float(case["reynolds"]),
        float(case["yaw_deg"]),
        float(geometry["unit_scale_to_m"]),
        geometry["front_axis"],
        geometry["up_axis"],
        float(geometry.get("ground_clearance_m", 0.0)),
    )


def deduplicate(source_path: Path, prepared_path: Path, output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError(f"Choose a fresh output path: {output_path}")
    source_bytes = source_path.read_bytes()
    prepared_bytes = prepared_path.read_bytes()
    source = json.loads(source_bytes)
    prepared = json.loads(prepared_bytes)
    source_cases = source["cases"]
    prepared_cases = {case["case_id"]: case for case in prepared["cases"]}
    source_by_id = {case["case_id"]: case for case in source_cases}
    if (
        len(source_by_id) != len(source_cases)
        or source_by_id.keys() != prepared_cases.keys()
    ):
        raise ValueError("Raw and prepared case IDs must match exactly")

    assets = {asset["path"]: asset for asset in source["provenance"]["geometry_assets"]}
    groups = defaultdict(list)
    for case in source_cases:
        rendered = prepared_cases[case["case_id"]]
        if (
            rendered["geometry_id"] != case["geometry_id"]
            or rendered["split"] != case["split"]
        ):
            raise ValueError(
                f"Prepared identity differs from raw source: {case['case_id']}"
            )
        if float(rendered["reynolds"]) != float(case["reynolds"]):
            raise ValueError(
                f"Prepared Reynolds differs from raw source: {case['case_id']}"
            )
        if float(rendered["yaw_deg"]) != float(case["yaw_deg"]):
            raise ValueError(f"Prepared yaw differs from raw source: {case['case_id']}")
        geometry_hash = rendered.get("geometry_hash")
        if not geometry_hash:
            raise ValueError(f"Prepared geometry hash missing: {case['case_id']}")
        groups[geometry_hash].append(case)

    removed = set()
    group_reports = []
    for geometry_hash, members in groups.items():
        if len(members) < 2:
            continue
        conditions = {_geometry_conditions(case) for case in members}
        if len(conditions) != 1:
            raise ValueError(
                f"Duplicate geometry hash has differing flow or geometry conditions: {geometry_hash}"
            )
        condition = next(iter(conditions))

        payloads = []
        triangle_count = None
        for case in members:
            try:
                asset = assets[case["geometry"]["path"]]
            except KeyError as error:
                raise ValueError(
                    f"Missing source asset metadata: {case['case_id']}"
                ) from error
            path = (source_path.parent / asset["path"]).resolve()
            count, payload = _stl_payload(path, asset)
            if triangle_count is None:
                triangle_count = count
            if count != triangle_count:
                raise ValueError(
                    f"Duplicate geometry hash has differing STL facet counts: {geometry_hash}"
                )
            payloads.append(payload)
        if any(payload != payloads[0] for payload in payloads[1:]):
            raise ValueError(
                f"Duplicate geometry hash is only a rounded collision: {geometry_hash}"
            )

        ordered = sorted(
            members,
            key=lambda case: (SPLIT_PRIORITY[case["split"]], case["case_id"]),
        )
        representative = ordered[0]
        excluded = ordered[1:]
        removed.update(case["case_id"] for case in excluded)
        member_records = [
            {
                "case_id": case["case_id"],
                "split": case["split"],
                "source_stl_sha256": assets[case["geometry"]["path"]]["sha256"],
                "cd": case["cd"],
                "cl": case["cl"],
                "design_value": case.get("design_value", {}),
            }
            for case in ordered
        ]
        group_reports.append(
            {
                "geometry_hash": geometry_hash,
                "representative_case_id": representative["case_id"],
                "flow_geometry_conditions": {
                    key: value
                    for key, value in zip(
                        (
                            "reynolds",
                            "yaw_deg",
                            "unit_scale_to_m",
                            "front_axis",
                            "up_axis",
                            "ground_clearance_m",
                        ),
                        condition,
                    )
                },
                "stl_triangle_count": triangle_count,
                "facet_payload_sha256": hashlib.sha256(payloads[0]).hexdigest(),
                "facet_payloads_equal_ignoring_80_byte_header": True,
                "members": member_records,
            }
        )

    cases = [case for case in source_cases if case["case_id"] not in removed]
    raw_counts = {
        split: sum(case["split"] == split for case in source_cases)
        for split in ("train", "val", "test")
    }
    counts = {
        split: sum(case["split"] == split for case in cases)
        for split in ("train", "val", "test")
    }
    excluded_cases = [
        {
            "case_id": member["case_id"],
            "split": member["split"],
            "source_stl_sha256": member["source_stl_sha256"],
            "geometry_hash": group["geometry_hash"],
            "representative_case_id": group["representative_case_id"],
            "reason": "exact duplicate STL facets; representative selected by test>val>train, then case_id",
        }
        for group in group_reports
        for member in group["members"]
        if member["case_id"] in removed
    ]
    source["cases"] = cases
    split_provenance = source.setdefault("split_provenance", {})
    split_provenance["raw_available_counts"] = raw_counts
    split_provenance["counts"] = counts
    split_provenance["method"] = (
        "official full partitions, one case per exact STL geometry hash with heldout split priority"
    )
    split_provenance["excluded_duplicate_cases"] = excluded_cases
    source["provenance"]["geometry_deduplication"] = {
        "raw_source_manifest": {
            "path": source_path.name,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "case_count": len(source_cases),
        },
        "prepared_dataset_sha256": hashlib.sha256(prepared_bytes).hexdigest(),
        "method": "Group by prepared geometry_hash, require identical flow/geometry conditions and byte-identical binary STL facet payloads after the 80-byte header, and keep one unchanged labeled source case per group.",
        "representative_priority": ["test", "val", "train"],
        "tie_break": "lowest case_id",
        "counts": {
            "raw_available": raw_counts,
            "deduplicated": counts,
            "alias_groups": len(group_reports),
            "excluded_cases": len(excluded_cases),
        },
        "groups": group_reports,
        "excluded_cases": excluded_cases,
    }
    if source_path.read_bytes() != source_bytes:
        raise RuntimeError("Raw source manifest changed during deduplication")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
    return {"source": str(output_path), "case_count": len(cases), "counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="immutable raw source manifest")
    parser.add_argument(
        "prepared", type=Path, help="prepared dataset.json with geometry_hash values"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="fresh derived source manifest path"
    )
    args = parser.parse_args()
    print(json.dumps(deduplicate(args.source, args.prepared, args.output), indent=2))


if __name__ == "__main__":
    main()
