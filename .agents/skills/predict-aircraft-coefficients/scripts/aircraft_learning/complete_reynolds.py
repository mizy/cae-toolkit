"""Complete assumed aircraft Re and combine audited, dimensionless source pools."""

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from vehicle_cd_learning.complete_labels import write_table
from vehicle_cd_learning.dimensionless import PARAMETERS, read_prepared

from .scene_experiment import verify_data

CONTRACT = "aerodynamic_dimensionless_v1"


def standard_air(altitude_m):
    """ISA troposphere and Sutherland air, explicitly assumed for this source pool."""
    if not np.isfinite(altitude_m) or not 0 <= altitude_m <= 11000:
        raise ValueError("Atmosphere estimate supports only0–11000 m altitude")
    temperature = 288.15 - 0.0065 * altitude_m
    pressure = 101325 * (temperature / 288.15) ** (9.80665 / (287.05287 * 0.0065))
    density = pressure / (287.05287 * temperature)
    viscosity = (
        1.716e-5
        * (temperature / 273.15) ** 1.5
        * (273.15 + 110.4)
        / (temperature + 110.4)
    )
    return temperature, pressure, density, viscosity


def fill_aircraft_re(case, geometry, index):
    """Preserve existing Re; apply the declared atmosphere only to COCOANet gaps."""
    if case["parameters"][index] is not None:
        return
    if case["source"] != "cocoanet":
        raise ValueError("No approved aircraft Re assumption for this source")
    conditions = case["conditions"]
    length, speed = geometry["dimensions"]["length_m"], conditions["velocity_m_s"]
    if not np.isfinite([length, speed]).all() or min(length, speed) <= 0:
        raise ValueError("Body length and speed must be finite and positive")
    temperature, pressure, density, viscosity = standard_air(conditions["altitude_m"])
    re = density * speed * length / viscosity
    conditions["re_body"] = re
    case["parameters"][index] = float(np.log(re))
    case["source_evidence"]["re_completion"] = {
        "status": "estimated",
        "source_conditions_verified": False,
        "formula": "Re_body = rho_ISA * U * L_body / mu_Sutherland",
        "reference_length": "full aircraft body-X extent in metres, not span or MAC",
        "atmosphere": "Dry ISA troposphere; source altitude treated as geopotential altitude; no temperature offset",
        "audit_inputs": {
            "length_m": length,
            "velocity_m_s": speed,
            "altitude_m": conditions["altitude_m"],
            "temperature_k": temperature,
            "pressure_pa": pressure,
            "density_kg_m3": density,
            "dynamic_viscosity_pa_s": viscosity,
        },
        "basis": "Source U/altitude and audited geometry; source atmosphere/viscosity not verified. User authorized explicit assumptions.",
    }


def complete(source, vehicles, aircraft, output):
    """@entry Merge completed vehicle inputs and estimated aircraft Re into36 inputs."""
    prepared = output / "prepared"
    if prepared.exists():
        raise FileExistsError("Use a fresh prepared directory")
    paths = [
        source / "dataset.json",
        vehicles / "dataset.json",
        aircraft / "dataset.json",
    ]
    frozen = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    data = json.loads(paths[0].read_text())
    verify_data(source, data)
    cars = {c["case_id"]: c for c in read_prepared(vehicles)["cases"]}
    geometries = json.loads(paths[2].read_text())["geometry"]
    names = data["parameter_names"]
    if set(names) != set(PARAMETERS) | {"log_length_m", "log_velocity_m_s"}:
        raise ValueError("Expected audited twenty-field joint source")
    keep, index = [names.index(k) for k in PARAMETERS], names.index("log_re_body")
    completed = []
    for old in data["cases"]:
        if old["scene"] == 1:
            case = copy.deepcopy(cars[old["case_id"]])
            for key in (
                "targets",
                "geometry_id",
                "geometry_hash",
                "split",
                "views",
                "views_sha256",
            ):
                if case[key] != old[key]:
                    raise ValueError(
                        f"Vehicle completion does not match joint source: {key}"
                    )
        else:
            case = old
            if case["source"] == "cocoanet":
                fill_aircraft_re(
                    case, geometries[case["geometry_id"].split("__", 1)[1]], index
                )
            case["parameters"] = [case["parameters"][i] for i in keep]
        if case["parameters"][PARAMETERS.index("log_re_body")] is None:
            raise ValueError("An unsupported Reynolds gap remains")
        completed.append(case)
    data.update(
        cases=completed,
        parameter_names=PARAMETERS,
        input_contract=CONTRACT,
        source_prepared=str(paths[0].resolve()),
        source_dataset_sha256=frozen[str(paths[0].resolve())],
    )
    save(data, source, output, frozen)
    verify_data(prepared, data)
    if any(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha
        for p, sha in frozen.items()
    ):
        raise ValueError("Source artifact changed during completion")
    return data["label_completion"]


def save(data, source, output, frozen):
    prepared = output / "prepared"
    prepared.mkdir(parents=True)
    for case in data["cases"]:
        target = prepared / case["views"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / case["views"], target)
    data["unknown_counts"] = {
        s: {
            k: sum(
                c["parameters"][i] is None for c in data["cases"] if c["source"] == s
            )
            for i, k in enumerate(PARAMETERS)
        }
        for s in data["source_counts"]
    }
    data["label_completion"] = {
        "date": "2026-11-16",
        "source_artifacts": frozen,
        "estimated_re_by_source": dict(
            Counter(
                c["source"]
                for c in data["cases"]
                if "re_completion" in c["source_evidence"]
            )
        ),
        "remaining_missing_re": sum(c["parameters"][2] is None for c in data["cases"]),
        "training_policy": "Re estimates enter inputs. Actual Cd/Cl targets and masks unchanged. Derived U/L/air properties are audit metadata, not independent numeric inputs.",
        "aircraft_assumption_limit": "ISA/Sutherland is an explicit approximation; original atmosphere and altitude definition are not confirmed.",
        "consumer": "Existing scene_experiment freeze/train and SceneCases accept these named numeric fields; select mode A. Vehicle-only and old aircraft prediction checkpoints retain their own contracts.",
    }
    (prepared / "dataset.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    (output / "completion.json").write_text(
        json.dumps(data["label_completion"], indent=2) + "\n"
    )
    write_table(output / "completed-values.csv", data["cases"])
    write_table(
        output / "cocoanet-estimates.csv",
        [c for c in data["cases"] if c["source"] == "cocoanet"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--vehicles", type=Path, required=True)
    parser.add_argument("--aircraft", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            complete(args.source, args.vehicles, args.aircraft, args.output_dir),
            indent=2,
        )
    )
