"""Unify verified full-geometry caches and explicit physical inputs for scene A/B/C."""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
PARAMETERS = [
    "width_over_length",
    "height_over_length",
    "log_length_m",
    "log_velocity_m_s",
    "log_re_body",
    "mach",
    "alpha_sin",
    "alpha_cos",
    "beta_sin",
    "beta_cos",
    "log_reference_area_over_length2",
    "ground_present",
    "clearance_over_length",
    "ground_speed_over_velocity",
    "wheels_present",
    "wheel_rotation_speed_over_velocity",
    "transition_enabled",
    "gravity_g_length_over_velocity2",
    "altitude_over_10000m",
    "wheel_rotating",
]
MANIFESTS = {
    "vehicle": "outputs/predict-vehicle-cd/builtin-inclusive-20261030/prepared/dataset.json",
    "cocoanet": "outputs/predict-aircraft-coefficients/cocoanet-20261109/prepared/dataset.json",
    "hilift": "outputs/aerodynamic-expansion-20261109/prepared/hilift/dataset.json",
    "drivaerrim": "outputs/aerodynamic-expansion-20261109/prepared/drivaerrim/dataset.json",
    "builtin_orientation": "outputs/predict-vehicle-cd/all-builtin-training/run26-upright/orientation-audit.json",
    "ahmed_source": "outputs/predict-vehicle-cd/public-data/ahmedml/full-source.json",
    "windsor_source": "outputs/predict-vehicle-cd/public-data/windsorml/full-source-deduplicated.json",
}
VIEWS = ("front", "rear", "right", "top", "bottom")


def parameters(length, width, height, conditions):
    """Transform declared physical values; preserve missing/assumed values as None."""
    values = {
        "width_over_length": width / length,
        "height_over_length": height / length,
        "log_length_m": float(np.log(length)),
    }
    for key, value in conditions.items():
        if key in PARAMETERS:
            values[key] = value
    for original, target in (
        ("velocity_m_s", "log_velocity_m_s"),
        ("re_body", "log_re_body"),
    ):
        value = conditions.get(original)
        if value is not None:
            if value <= 0:
                raise ValueError(f"{original} must be positive")
            values[target] = float(np.log(value))
    for original, prefix in (("alpha_deg", "alpha"), ("beta_deg", "beta")):
        value = conditions.get(original)
        if value is not None:
            values[prefix + "_sin"] = float(np.sin(np.deg2rad(value)))
            values[prefix + "_cos"] = float(np.cos(np.deg2rad(value)))
    area = conditions.get("reference_area_m2")
    if area is not None:
        if area <= 0:
            raise ValueError("Reference area must be positive")
        values["log_reference_area_over_length2"] = float(np.log(area / length**2))
    if conditions.get("clearance_m") is not None:
        values["clearance_over_length"] = conditions["clearance_m"] / length
    if conditions.get("altitude_m") is not None:
        values["altitude_over_10000m"] = conditions["altitude_m"] / 10000
    if conditions.get("gravity_m_s2") is not None and conditions.get("velocity_m_s"):
        values["gravity_g_length_over_velocity2"] = (
            conditions["gravity_m_s2"] * length / conditions["velocity_m_s"] ** 2
        )
    result = [values.get(k) for k in PARAMETERS]
    if not np.isfinite([v for v in result if v is not None]).all():
        raise ValueError("Nonfinite physical parameter")
    return result


def resample_views(views, old_size=512, new_size=256, margin=4):
    """Nearest cache resampling through the renderer's actual full-camera transform."""
    ratio = (old_size - 2 * margin) / (new_size - 2 * margin)
    pixels = np.arange(new_size, dtype=np.float64) + 0.5
    cols = np.rint((pixels - margin) * ratio + margin - 0.5).astype(int)
    rows = np.rint(
        old_size - 1 - margin - ratio * (new_size - 1 - pixels - margin) - 0.5
    ).astype(int)
    valid_rows, valid_cols = (
        (rows >= 0) & (rows < old_size),
        (cols >= 0) & (cols < old_size),
    )
    safe_rows, safe_cols = rows.clip(0, old_size - 1), cols.clip(0, old_size - 1)
    result = {}
    for name in VIEWS:
        a = np.asarray(views[name], dtype=np.float32)
        if a.shape != (2, old_size, old_size):
            raise ValueError("Unexpected source view shape")
        b = a[:, safe_rows[:, None], safe_cols[None, :]].copy()
        b[:, ~valid_rows, :] = 0
        b[:, :, ~valid_cols] = 0
        result[name] = b
    return result


def cache_views(path, expected_sha, output, relative, old_size):
    """Verify source cache, convert its fixed camera, and record numerical identity."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise ValueError(f"Source view checksum changed: {path}")
    with np.load(path, allow_pickle=False) as f:
        arrays = {name: f[name] for name in VIEWS}
    if old_size != 256:
        arrays = resample_views(arrays, old_size)
    digest = hashlib.sha256()
    for name in VIEWS:
        a = arrays[name]
        if a.shape != (2, 256, 256) or not np.isfinite(a).all():
            raise ValueError("Invalid numerical view tensor")
        if not np.isin(a[1], [0, 1]).all() or not a[1].any():
            raise ValueError("Invalid silhouette mask")
        if (a[0] < 0).any() or (a[0] > 1).any() or np.any(a[0][a[1] == 0] != 0):
            raise ValueError("Invalid depth/background")
        digest.update(np.asarray(a, dtype=np.float32).tobytes())
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if old_size == 256:
        shutil.copyfile(path, target)
    else:
        np.savez_compressed(target, **arrays)
    return {
        "views": relative,
        "views_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source_views_sha256": expected_sha,
        "numerical_views_sha256": digest.hexdigest(),
    }


def vehicle_records(data, sources):
    """Read legacy car physics without passing source-profile one-hot to the model."""
    orientation = {x["id"]: x["dimensions"] for x in sources["builtin_orientation"]}
    ahmed = {x["case_id"]: x for x in sources["ahmed_source"]["cases"]}
    windsor = {x["case_id"]: x for x in sources["windsor_source"]["cases"]}
    for original in data["cases"]:
        source = original["condition_profile"]
        p = original["parameters"]
        if max(p[:2]) > 1:
            raise ValueError("Legacy car camera is not maximum-extent normalized")
        physical, evidence = (
            {},
            {"source_conditions": data["condition_profiles"][source]["conditions"]},
        )
        targets = list(original["targets"])
        if source == "builtin_assumed":
            length = orientation[original["geometry_id"]]["length_m"]
            physical.update(
                velocity_m_s=evidence["source_conditions"]["speed"]["value_kmh"] / 3.6,
                alpha_deg=0.0,
                beta_deg=float(original["yaw_deg"]),
                ground_present=1.0,
                ground_speed_over_velocity=1.0,
                wheels_present=1.0,
                wheel_rotating=1.0,
                wheel_rotation_speed_over_velocity=1.0,
            )
            evidence["assumptions_not_verified_physics"] = {
                **original.get("assumptions", {}),
                "source_conditions_verified": False,
                "basis": "User-declared nominal low-drag training conditions, 2026-11-12; not verified source CFD conditions",
                "assumed_fields": list(physical),
            }
            evidence["source_label"] = original.get("cd_source")
            # Re/Mach/reference area remain unknown; do not reuse assumed legacy Re.
        else:
            u = {"drivaerml": 38.889, "ahmedml": 1.0, "windsorml": 40.0}[source]
            nu = {
                "drivaerml": 1.507e-5,
                "ahmedml": 3.75e-7,
                "windsorml": 40 * 1.044 / 2.9e6,
            }[source]
            length = original["reynolds"] * nu / u
            physical.update(
                velocity_m_s=u,
                re_body=original["reynolds"],
                alpha_deg=0.0,
                beta_deg=original["yaw_deg"],
                ground_present=1.0,
                ground_speed_over_velocity=0.0,
            )
            if source == "drivaerml":
                area = original["assumptions"]["source_force_reference_area_m2"]
                physical.update(
                    reference_area_m2=area,
                    wheels_present=1.0,
                    wheel_rotating=0.0,
                    wheel_rotation_speed_over_velocity=0.0,
                )
                evidence["reference"] = original["assumptions"]
            elif source == "ahmedml":
                design = ahmed[original["case_id"]]["design_value"]
                area = float(design["body-width"]) * float(design["body-height"]) * 1e-6
                physical.update(
                    reference_area_m2=area, clearance_m=0.05, wheels_present=0.0
                )
                evidence["reference_area_derivation"] = (
                    "body-width_mm*body-height_mm*1e-6; all499 fixed0.112/variable-area coefficient pairs agree within5.92e-10"
                )
            else:
                design = windsor[original["case_id"]]["design_value"]
                raw = sources["windsor_forces"][original["case_id"].rsplit("-", 1)[-1]]
                beta = np.deg2rad(2.5)
                targets = [
                    float(raw["cd"]) * np.cos(beta) - float(raw["cs"]) * np.sin(beta),
                    float(raw["cl"]),
                ]
                physical.update(
                    reference_area_m2=float(design["frontal_area"]),
                    clearance_m=float(design["clearance"]) * 0.001,
                    wheels_present=0.0,
                )
                evidence["source_coefficients"] = {
                    k: float(raw[k]) for k in ("cd", "cs", "cl")
                }
                evidence["wind_axis_conversion"] = (
                    "canonical force=(cd,-cs,cl); wind=(cos2.5,sin2.5,0); Cd_wind=cd*cos2.5-cs*sin2.5; Cl_wind=cl"
                )
                evidence["source_clearance"] = (
                    "STL already includes floor-reaching stilts; no extra cache translation"
                )
            evidence["re_definition"] = (
                "source U*full body length/nu; source Mach not fabricated from temperature/default atmosphere"
            )
        yield (
            original,
            source,
            length,
            p[0] * length,
            p[1] * length,
            physical,
            targets,
            evidence,
        )


def aircraft_records(data, source):
    for original in data["cases"]:
        g = data["geometry"][original["geometry_id"]]
        d = g["dimensions"]
        length, width, height = d["length_m"], d["width_m"], d["height_m"]
        if source == "cocoanet":
            c = original["conditions"]
            physical = {
                "velocity_m_s": c["velocity_magnitude_ms"],
                "alpha_deg": c["alpha_deg"],
                "beta_deg": c["beta_deg"],
                "altitude_m": c["altitude_m"],
                "reference_area_m2": g["reference_area_m2"],
                "wheels_present": 0.0,
            }
            evidence = {
                "source_conditions": c,
                "reference_chord_m": g["reference_chord_m"],
                "unknown_physics": "No source viscosity/Mach/atmosphere/ground BC in pinned prepared data. Altitude is retained, not converted to ISA/Re/Mach.",
                "source_mesh": g["mesh"],
            }
        else:
            c = data["training_conditions"]
            physical = {
                "re_body": c["re_mac"] * length / c["source_reference_chord_m"],
                "mach": c["mach"],
                "alpha_deg": original["conditions"]["alpha_deg"],
                "beta_deg": 0.0,
                "ground_present": 0.0,
                "wheels_present": 0.0,
                "reference_area_m2": c["full_reference_area_m2"],
            }
            evidence = {
                "source_conditions": c,
                "source_re_mac": c["re_mac"],
                "re_conversion": "Re_body=Re_MAC*body_length/source_reference_chord_m",
                "source_mesh": g["mesh"],
                "geometry_transform": g["geometry_transform"],
            }
        yield original, length, width, height, physical, evidence


def main():
    """@entry Freeze the shared physical contract and aligned caches for A/B/C."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    paths = {k: ROOT / p for k, p in MANIFESTS.items()}
    sources = {k: json.loads(p.read_text()) for k, p in paths.items()}
    hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths.values()
    }
    fp = (
        ROOT
        / "outputs/predict-vehicle-cd/source-catalogue/windsorml/force_mom_varref_all.csv"
    )
    sources["windsor_forces"] = {r["run"]: r for r in csv.DictReader(fp.open())}
    hashes[str(fp.relative_to(ROOT))] = hashlib.sha256(fp.read_bytes()).hexdigest()
    output.mkdir(parents=True)
    cases, cached = [], {}

    def add(
        original,
        source,
        scene,
        length,
        width,
        height,
        physical,
        targets,
        evidence,
        old_path,
        old_sha,
        geometry_hash,
        old_size,
        hash_kind,
    ):
        gid = f"{source}__{original['geometry_id']}"
        if gid not in cached:
            cached[gid] = cache_views(
                old_path, old_sha, output, f"views/{gid}.npz", old_size
            )
        cases.append(
            {
                "case_id": f"{source}__{original['case_id']}",
                "geometry_id": gid,
                "family_id": f"{source}__{original['family_id']}",
                "source": source,
                "scene": scene,
                "split": original["split"],
                "parameters": parameters(length, width, height, physical),
                "targets": [None if x is None else float(x) for x in targets],
                "geometry_hash": geometry_hash,
                "source_geometry_hash_kind": hash_kind,
                "conditions": physical,
                "source_evidence": evidence,
                **cached[gid],
            }
        )

    vehicle = sources["vehicle"]
    if vehicle["render_config"] != {
        "ordinary_size": 512,
        "bottom_size": 512,
        "window_l": 1.3,
    }:
        raise ValueError("Legacy source renderer configuration changed")
    for (
        original,
        source,
        length,
        width,
        height,
        physical,
        targets,
        evidence,
    ) in vehicle_records(vehicle, sources):
        add(
            original,
            source,
            1,
            length,
            width,
            height,
            physical,
            targets,
            evidence,
            paths["vehicle"].parent / original["views"],
            original["views_sha256"],
            original["geometry_hash"],
            512,
            "canonical_triangle_hash",
        )
    for source in ("cocoanet", "hilift"):
        data = sources[source]
        for original, length, width, height, physical, evidence in aircraft_records(
            data, source
        ):
            g = data["geometry"][original["geometry_id"]]
            add(
                original,
                source,
                2,
                length,
                width,
                height,
                physical,
                original["targets"],
                evidence,
                paths[source].parent / "views" / f"{original['geometry_id']}.npz",
                g["views_sha256"],
                g["sha256"],
                256,
                "source_stl_sha256",
            )
    data = sources["drivaerrim"]
    for original in data["cases"]:
        g = data["geometry"][original["geometry_id"]]
        length, width, height = g["dimensions"]
        c = data["training_conditions"]
        physical = {
            "velocity_m_s": c["velocity_m_s"],
            "alpha_deg": 0.0,
            "beta_deg": 0.0,
            "reference_area_m2": c["reference_area_m2"],
            "ground_present": 1.0,
            "ground_speed_over_velocity": 1.0,
            "wheels_present": 1.0,
            "wheel_rotating": 1.0,
        }
        evidence = {
            "source_conditions": c,
            "unknown_physics": "Source viscosity/Re/Mach and exact wheel peripheral-speed ratio are not fabricated",
            "complete_geometry": "shared body+front/rear left wheel assemblies mirrored acrossY=0",
        }
        add(
            original,
            "drivaerrim",
            1,
            length,
            width,
            height,
            physical,
            original["targets"],
            evidence,
            paths["drivaerrim"].parent / "views" / f"{original['geometry_id']}.npz",
            g["views_sha256"],
            g["sha256"],
            256,
            "complete_wheel_pair_source_hash",
        )
    groups = defaultdict(list)
    for case in cases:
        groups[case["geometry_hash"]].append(case)
    reassigned = []
    for value, group in groups.items():
        splits = {c["split"] for c in group}
        if len(splits) > 1:
            chosen = max(splits, key={"train": 0, "val": 1, "test": 2}.get)
            for c in group:
                if c["split"] != chosen:
                    reassigned.append(
                        {
                            "case_id": c["case_id"],
                            "from": c["split"],
                            "to": chosen,
                            "geometry_hash": value,
                        }
                    )
                    c["original_split"], c["split"] = c["split"], chosen
    counts, unknowns = {}, {}
    for source in sorted({c["source"] for c in cases}):
        subset = [c for c in cases if c["source"] == source]
        counts[source] = dict(Counter(c["split"] for c in subset))
        unknowns[source] = {
            k: sum(c["parameters"][i] is None for c in subset)
            for i, k in enumerate(PARAMETERS)
        }
    result = {
        "schema_version": 1,
        "parameter_names": PARAMETERS,
        "target_names": ["cd", "cl"],
        "scenes": {
            "0": "unspecified",
            "1": "vehicle_external",
            "2": "aircraft_external",
        },
        "source_manifest_hashes": hashes,
        "source_counts": counts,
        "unknown_counts": unknowns,
        "render_config": {
            "size": 256,
            "window_l": 1.3,
            "scale": "maximum_extent",
            "margin": 4,
        },
        "cache_conversion": "vehicle512→256 nearest affine;cols=round((i+.5-4)*504/248+4-.5); rows=round(512-1-4-(504/248)*(256-1-(i+.5)-4)-.5);out-of-boundszero;not re-rasterized",
        "label_convention": "Common wind-axis drag and upward lift; Windsor source(Cd,Cs,Cl) rotated by+2.5deg wind direction in canonical frame; source Sref retained explicitly",
        "split_policy": "source geometry-disjoint splits preserved; known identical geometry hash assigned to highest heldout priority(test>val>train)",
        "cross_split_reassignments": reassigned,
        "geometry_audit_limit": "Source canonical hashes/STL hashes/assembly hashes have different representations; no complete cross-source near-duplicate certification. All matching published hash values are grouped.",
        "physical_contract_limit": "The20fields are an incomplete BC description; e.g.DrivAer upstreamslip/downstreamnoslip preserved in source_evidence is not encoded by stationary ground speed alone. Scene accuracy differences are predictive, not proof of physical causality. Built-in cars use user-declared nominal operating assumptions recorded in source_evidence; a known input bit means available, not source-verified.",
        "clearance_definition": "body-to-ground gap, not lowest wheel/support contact; DrivAer/Rim remain unknown despite renderer wheel-contact plane0",
        "excluded_pools": [
            "SuperWing: isolated wings",
            "Flow360 UAT: candidate quality/contracts pending",
        ],
        "cases": cases,
    }
    (output / "dataset.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "records": len(cases),
                "cached_geometries": len(cached),
                "counts": counts,
                "reassigned": len(reassigned),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
