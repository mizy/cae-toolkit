from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from vehicle_cd_similarity_lib.multiview import (
    SCALES,
    VIEWS,
    describe_views,
    predict_descriptors,
    score_descriptors,
    validate_descriptors,
)


def car(mirror=False, size=1024):
    images = {}
    for view_index, view in enumerate(VIEWS):
        image = np.zeros((2, size, size), dtype=np.float32)
        image[1, size // 8 : 7 * size // 8, size // 5 : 4 * size // 5] = 1
        image[0, size // 8 : 7 * size // 8, size // 5 : 4 * size // 5] = (
            0.2 + view_index * 0.05
        )
        if mirror:
            image[1, size // 3 : size // 3 + 8, 3 * size // 4 : 7 * size // 8] = 1
            image[0, size // 3 : size // 3 + 8, 3 * size // 4 : 7 * size // 8] = (
                0.7 + view_index * 0.02
            )
        images[view] = image
    return images


def test_float_depth_and_fractional_occupancy_survive_three_scales():
    images = car()
    images["right"][1, 300:305, 800:805] = 1
    images["right"][0, 300:305, 800:805] = 0.123456
    described = describe_views(images)
    assert tuple(described) == tuple(
        key for size in SCALES for key in (f"mask_{size}", f"depth_{size}")
    )
    assert described["mask_32"].shape == (5, 32, 32)
    assert described["mask_128"].shape == (5, 128, 128)
    assert described["mask_256"].shape == (5, 256, 256)
    assert np.any((described["mask_128"] > 0) & (described["mask_128"] < 1))
    assert np.any((described["depth_256"] > 0.123) & (described["depth_256"] < 0.124))
    assert described["mask_32"].dtype == np.float32
    assert described["depth_256"].dtype == np.float32


def test_sub_8_bit_depth_change_survives_descriptor_generation():
    baseline = car(size=256)
    changed = car(size=256)
    changed["front"][0, 100, 100] += 0.001

    original_depth = describe_views(baseline)["depth_256"]
    changed_depth = describe_views(changed)["depth_256"]

    assert 0.001 < 1 / 255
    assert changed_depth[0, 100, 100] - original_depth[0, 100, 100] == pytest.approx(
        0.001, abs=2e-8
    )


def test_reference_self_match_exclusion_and_mirror_change_are_explicit():
    base = describe_views(car())
    modified = describe_views(car(mirror=True))
    changed = score_descriptors(
        modified, {key: value[None] for key, value in base.items()}
    )
    assert changed["scores"][0] < 1
    assert changed["scores"][0] > 0.9
    alternate = describe_views(car(mirror=True))
    bank = {key: np.stack((value, alternate[key])) for key, value in base.items()}
    result = predict_descriptors(
        base,
        bank,
        [
            {"id": "base", "cd": 99.0, "family_id": "sedan"},
            {"id": "mirror", "cd": 0.28, "family_id": "sedan"},
        ],
        exclude_ids={"base"},
    )
    assert result["reference_count"] == 1
    assert result["top_k"][0]["id"] == "mirror"
    assert result["cd"] == pytest.approx(0.28)
    assert result["top_k"][0]["weight"] == pytest.approx(1)
    assert set(result["top_k"][0]["view_scores"]) == set(VIEWS)
    assert len(result["top_k"][0]["scale_mask_scores"]) == 3
    assert result["cd"] < 1


def test_self_match_scores_one_and_is_selected():
    query = describe_views(car(size=256))
    bank = {key: value[None] for key, value in query.items()}
    records = [{"id": "self", "cd": 0.24}]

    scored = score_descriptors(query, bank)
    result = predict_descriptors(query, bank, records)

    assert scored["scores"][0] == pytest.approx(1)
    assert result["top_k"][0]["id"] == "self"
    assert result["top_k"][0]["score"] == pytest.approx(1)
    assert result["cd"] == pytest.approx(0.24)


def test_descriptor_validation_rejects_empty_reference_and_query_views():
    query = describe_views(car(size=256))
    bank = {key: value[None] for key, value in query.items()}
    assert validate_descriptors(bank) == 1

    empty_reference = {key: value.copy() for key, value in bank.items()}
    empty_query = {key: value.copy() for key, value in query.items()}
    for scale in SCALES:
        empty_reference[f"mask_{scale}"][0, 4] = 0
        empty_reference[f"depth_{scale}"][0, 4] = 0
        empty_query[f"mask_{scale}"][4] = 0
        empty_query[f"depth_{scale}"][4] = 0

    with pytest.raises(ValueError, match="reference view"):
        validate_descriptors(empty_reference)
    with pytest.raises(ValueError, match="query view"):
        score_descriptors(empty_query, bank)


def test_identical_reference_ties_are_stable_under_bank_permutation():
    query = describe_views(car(size=256))
    records = [
        {"id": "zeta", "cd": 0.31},
        {"id": "alpha", "cd": 0.22},
        {"id": "mu", "cd": 0.27},
    ]
    order = [2, 0, 1]

    def predict(order):
        bank = {key: np.stack([query[key] for _ in order]) for key in query}
        return predict_descriptors(query, bank, [records[index] for index in order])

    forward = predict([0, 1, 2])
    permuted = predict(order)

    assert [row["id"] for row in forward["top_k"]] == ["alpha", "mu", "zeta"]
    assert [row["id"] for row in permuted["top_k"]] == ["alpha", "mu", "zeta"]
    assert permuted["cd"] == pytest.approx(forward["cd"])


@pytest.mark.parametrize(
    "case",
    [
        "bad-shape",
        "nan",
        "nonbinary-mask",
        "empty-mask",
        "bad-depth",
        "negative-depth",
        "background-depth",
    ],
)
def test_invalid_input_views_fail_closed(case):
    images = car()
    if case == "bad-shape":
        images["front"] = np.zeros((2, 1000, 1000), dtype=np.float32)
    elif case == "nan":
        images["top"][0, 0, 0] = np.nan
    elif case == "nonbinary-mask":
        images["front"][1, 0, 0] = 0.5
    elif case == "empty-mask":
        images["rear"][:] = 0
    elif case == "bad-depth":
        images["right"][0, 300, 300] = 1.1
    elif case == "negative-depth":
        images["right"][0, 300, 300] = -0.1
    else:
        images["right"][0, 0, 0] = 0.1
    with pytest.raises(ValueError):
        describe_views(images)


@pytest.mark.parametrize("top_k", [0, 1.5, True])
def test_predict_requires_positive_integral_top_k(top_k):
    descriptors = describe_views(car(size=256))
    bank = {key: value[None] for key, value in descriptors.items()}
    with pytest.raises(ValueError, match="top_k"):
        predict_descriptors(
            descriptors, bank, [{"id": "reference", "cd": 0.25}], top_k=top_k
        )


@pytest.mark.parametrize("cd", [0, -0.1, np.nan, np.inf])
def test_predict_requires_finite_positive_cd(cd):
    descriptors = describe_views(car(size=256))
    bank = {key: value[None] for key, value in descriptors.items()}
    with pytest.raises(ValueError, match="Cd labels"):
        predict_descriptors(descriptors, bank, [{"id": "reference", "cd": cd}])
