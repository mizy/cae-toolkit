from __future__ import annotations

import hashlib
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from vehicle_cd_learning import import_drivaerml


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _write_case(root: Path, run_id: int) -> None:
    run_dir = root / f"run_{run_id}"
    run_dir.mkdir()
    (run_dir / f"drivaer_{run_id}.stl").write_text(
        """solid car
facet normal 0 0 1
outer loop
vertex -0.7 -1.0 -0.3
vertex 3.9 1.0 1.2
vertex 0.0 0.0 0.0
endloop
endfacet
endsolid car
""",
        encoding="ascii",
    )
    (run_dir / f"force_mom_{run_id}.csv").write_text(
        "Cd,Cl,Clf,Clr,Cs\n0.31,0.04,-0.02,0.06,0.0\n", encoding="ascii"
    )
    (run_dir / f"geo_ref_{run_id}.csv").write_text(
        "lRef,aRef,forcesCoR,lRefRef,aRefRef,forcesCoRRef\n"
        "2.8,2.22,(1.4 0 -0.3),2.786,2.17,(1.4 0 -0.3)\n",
        encoding="ascii",
    )


def test_case_uses_per_run_labels_and_preserves_shared_family(tmp_path: Path) -> None:
    _write_case(tmp_path, 7)
    (tmp_path / "force_mom_all.csv").write_text(
        "run,cd,cl\n7,9.9,9.9\n", encoding="ascii"
    )

    case = import_drivaerml._case_record(tmp_path, 7, "val")

    assert case["cd"] == pytest.approx(0.31)
    assert case["cl"] == pytest.approx(0.04)
    assert case["split"] == "val"
    assert case["family_id"] == import_drivaerml.FAMILY_ID
    assert case["geometry"]["front_axis"] == "-x"
    assert case["geometry"]["ground_clearance_m"] == 0.0
    assert case["assumptions"]["source_force_reference_area_m2"] == pytest.approx(2.22)
    assert case["reynolds"] == pytest.approx(
        import_drivaerml.SPEED_M_S * 4.6 / import_drivaerml.KINEMATIC_VISCOSITY_M2_S
    )
    _write_case(tmp_path, 8)
    _write_case(tmp_path, 9)
    split_cases = [
        case,
        import_drivaerml._case_record(tmp_path, 8, "train"),
        import_drivaerml._case_record(tmp_path, 9, "test"),
    ]
    assert {item["family_id"] for item in split_cases} == {import_drivaerml.FAMILY_ID}
    assert {item["split"] for item in split_cases} == {"train", "val", "test"}


def test_sample_ids_stay_in_their_official_geometry_partitions() -> None:
    splits = {
        "full_train": [f"run_{i}" for i in range(1, 21)],
        "full_val": [f"run_{i}" for i in range(21, 31)],
        "full_test": [f"run_{i}" for i in range(31, 41)],
    }

    selected = import_drivaerml._sample_ids(
        splits, {"train": 5, "val": 3, "test": 3}, seed=42
    )

    assert {name: len(ids) for name, ids in selected.items()} == {
        "train": 5,
        "val": 3,
        "test": 3,
    }
    assert set(selected["train"]) <= set(range(1, 21))
    assert set(selected["val"]) <= set(range(21, 31))
    assert set(selected["test"]) <= set(range(31, 41))
    assert not set(selected["train"]) & set(selected["val"])
    assert not set(selected["train"]) & set(selected["test"])
    assert not set(selected["val"]) & set(selected["test"])


def test_all_ids_selects_every_case_from_the_400_34_50_source_splits() -> None:
    splits = {
        "full_train": [f"run_{i}" for i in range(1, 401)],
        "full_val": [f"run_{i}" for i in range(401, 435)],
        "full_test": [f"run_{i}" for i in range(435, 485)],
    }

    selected = import_drivaerml._all_ids(splits)

    assert {name: len(ids) for name, ids in selected.items()} == {
        "train": 400,
        "val": 34,
        "test": 50,
    }
    assert selected["train"] == list(range(1, 401))
    assert selected["val"] == list(range(401, 435))
    assert selected["test"] == list(range(435, 485))
    assert len(set().union(*(set(ids) for ids in selected.values()))) == 484


def test_all_ids_rejects_unexpected_official_split_sizes() -> None:
    with pytest.raises(ValueError, match="400/34/50"):
        import_drivaerml._all_ids(
            {
                "full_train": [f"run_{i}" for i in range(1, 5)],
                "full_val": ["run_5"],
                "full_test": ["run_6"],
            }
        )


def test_stl_bounds_use_exact_raw_geometry_bounds(tmp_path: Path) -> None:
    geometry = tmp_path / "precision.stl"
    geometry.write_text(
        """solid car
vertex -0.700000019 -1.000000011 -0.300000013
vertex 3.900000061 1.000000017 1.200000019
vertex 0.0 0.0 0.0
endsolid car
""",
        encoding="ascii",
    )

    lower, upper = import_drivaerml._stl_bounds(geometry)

    assert lower == [-0.700000019, -1.000000011, -0.300000013]
    assert upper == [3.900000061, 1.000000017, 1.200000019]


def test_case_record_runs_in_a_spawned_worker(tmp_path: Path) -> None:
    _write_case(tmp_path, 7)

    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        case = pool.submit(import_drivaerml._case_record, tmp_path, 7, "train").result(
            timeout=20
        )

    assert case["case_id"] == "drivaerml-run-007"
    assert case["split"] == "train"


class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        payload, self.payload = self.payload, b""
        return payload if size < 0 else payload[:size]


@pytest.mark.parametrize("source_kind", ["lfs", "git"])
def test_download_verifies_cache_and_replaces_stale_content(
    tmp_path: Path, source_kind: str
) -> None:
    payload = b"pinned source bytes\n"
    if source_kind == "lfs":
        entry = {
            "size": len(payload),
            "lfs": {"oid": hashlib.sha256(payload).hexdigest()},
        }
    else:
        entry = {"size": len(payload), "oid": _git_blob_sha1(payload)}
    path = "run_7/force_mom_7.csv"
    destination = tmp_path / "run_7" / "force_mom_7.csv"
    destination.parent.mkdir()
    destination.write_bytes(payload)

    with patch.object(import_drivaerml.urllib.request, "urlopen") as urlopen:
        cached = import_drivaerml._download(path, destination, entry)
    urlopen.assert_not_called()
    assert cached["source_path"] == path
    assert cached["size_bytes"] == len(payload)
    assert cached["cache_hit"] is True

    destination.write_bytes(b"stale")
    with patch.object(
        import_drivaerml.urllib.request, "urlopen", return_value=_Response(payload)
    ) as urlopen:
        refreshed = import_drivaerml._download(path, destination, entry)
    assert urlopen.call_count == 1
    assert destination.read_bytes() == payload
    assert refreshed["sha256"] == hashlib.sha256(payload).hexdigest()
    assert refreshed["cache_hit"] is False
    sidecar = json.loads(Path(f"{destination}.source.json").read_text())
    assert sidecar["revision"] == import_drivaerml.REVISION
    assert sidecar["source_hash"] == entry.get("lfs", {}).get("oid", entry.get("oid"))


def test_download_logs_mismatch_and_uses_bounded_xet_fallback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = b"pinned source bytes\n"
    entry = {
        "size": len(payload),
        "lfs": {"oid": hashlib.sha256(payload).hexdigest()},
    }
    path = "run_7/drivaer_7.stl"
    destination = tmp_path / path
    responses = [
        _Response(b"x" * len(payload)) for _ in range(import_drivaerml.MAX_RETRIES)
    ]

    def xet_fallback(_path: str, target: Path, _entry: dict) -> dict:
        target.write_bytes(payload)
        return {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "cache_hit": False,
            "transport": "hf_xet",
        }

    with (
        patch.object(
            import_drivaerml.urllib.request, "urlopen", side_effect=responses
        ) as urlopen,
        patch.object(import_drivaerml.time, "sleep"),
        patch.object(
            import_drivaerml, "_download_with_hf_xet", side_effect=xet_fallback
        ) as xet,
    ):
        result = import_drivaerml._download(path, destination, entry)

    assert urlopen.call_count == import_drivaerml.MAX_RETRIES
    xet.assert_called_once_with(path, destination, entry)
    assert destination.read_bytes() == payload
    assert result["transport"] == "hf_xet"
    output = capsys.readouterr().out
    assert "received=20 expected=20" in output
    assert "source_hash=" in output
    assert "expected_hash=" in output
    assert "Content-Range=None" in output
    assert "Content-Encoding=None" in output
    assert not Path(f"{destination}.part").exists()
