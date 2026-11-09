"""Float-preserving multiview geometry matching for the reference Cd predictor."""

from __future__ import annotations

from typing import Any

import numpy as np

VIEWS = ("front", "rear", "right", "top", "bottom")
SCALES = (32, 128, 256)


def describe_views(views: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Area-pool complete numerical views into occupancy and masked depth pyramids."""
    if set(views) != set(VIEWS):
        raise ValueError(f"Expected exactly these views: {VIEWS}")
    arrays = []
    size = None
    for name in VIEWS:
        image = np.asarray(views[name], dtype=np.float32)
        if image.ndim != 3 or image.shape[0] != 2 or image.shape[1] != image.shape[2]:
            raise ValueError(f"View {name} must have shape (2,H,H)")
        if size is None:
            size = image.shape[1]
        if image.shape[1] != size or size < SCALES[-1] or size % SCALES[-1]:
            raise ValueError("All views must be equal square images, multiples of 256")
        if not np.isfinite(image).all():
            raise ValueError(f"View {name} contains nonfinite values")
        mask = image[1]
        if not np.isin(mask, (0.0, 1.0)).all() or not mask.any():
            raise ValueError(f"View {name} needs a nonempty binary mask")
        if np.any(image[0] < 0) or np.any(image[0] > 1):
            raise ValueError(f"View {name} depth must be in [0,1]")
        if np.any(image[0][mask == 0] != 0):
            raise ValueError(f"View {name} background depth must be zero")
        arrays.append(image)

    images = np.stack(arrays)
    result = {}
    for target in SCALES:
        factor = size // target
        block_shape = (len(VIEWS), 2, target, factor, target, factor)
        blocks = images.reshape(block_shape)
        mask = blocks[:, 1].mean(axis=(2, 4), dtype=np.float32)
        covered_depth = (blocks[:, 0] * blocks[:, 1]).mean(
            axis=(2, 4), dtype=np.float32
        )
        depth = np.divide(
            covered_depth,
            mask,
            out=np.zeros_like(covered_depth),
            where=mask > 0,
        )
        result[f"mask_{target}"] = mask.astype(np.float32, copy=False)
        result[f"depth_{target}"] = depth.astype(np.float32, copy=False)
    return result


def validate_descriptors(bank: dict[str, np.ndarray]) -> int:
    if set(bank) != {
        key for scale in SCALES for key in (f"mask_{scale}", f"depth_{scale}")
    }:
        raise ValueError(
            "Reference descriptors must contain mask/depth at scales 32, 128, 256"
        )
    count = None
    for scale in SCALES:
        mask = np.asarray(bank[f"mask_{scale}"])
        depth = np.asarray(bank[f"depth_{scale}"])
        if (
            mask.ndim != 4
            or mask.shape[1:] != (len(VIEWS), scale, scale)
            or depth.shape != mask.shape
        ):
            raise ValueError(
                f"Reference descriptors at scale {scale} have invalid shape"
            )
        if count is None:
            count = mask.shape[0]
        if mask.shape[0] != count or count < 1:
            raise ValueError("Reference descriptor counts must match and be nonempty")
        if not np.isfinite(mask).all() or not np.isfinite(depth).all():
            raise ValueError("Reference descriptors contain nonfinite values")
        if (
            np.any(mask < 0)
            or np.any(mask > 1)
            or np.any(depth < 0)
            or np.any(depth > 1)
        ):
            raise ValueError("Reference occupancy and depth must be in [0,1]")
        if not np.all(np.any(mask > 0, axis=(2, 3))):
            raise ValueError("Every reference view needs positive occupancy")
        if np.any(depth[mask == 0] != 0):
            raise ValueError("Reference background depth must be zero")
    return int(count)


def _score_validated_descriptors(
    query: dict[str, np.ndarray],
    reference_bank: dict[str, np.ndarray],
    count: int,
) -> dict[str, np.ndarray]:
    if set(query) != {
        key for scale in SCALES for key in (f"mask_{scale}", f"depth_{scale}")
    }:
        raise ValueError(
            "Query descriptors must contain mask/depth at scales 32, 128, 256"
        )

    mask_scores = np.empty((count, len(VIEWS), len(SCALES)), dtype=np.float32)
    depth_scores = np.empty_like(mask_scores)
    for scale_index, scale in enumerate(SCALES):
        qm = np.asarray(query[f"mask_{scale}"])
        qd = np.asarray(query[f"depth_{scale}"])
        expected = (len(VIEWS), scale, scale)
        if qm.shape != expected or qd.shape != expected:
            raise ValueError(f"Query descriptors at scale {scale} have invalid shape")
        if not np.isfinite(qm).all() or not np.isfinite(qd).all():
            raise ValueError("Query descriptors contain nonfinite values")
        if np.any(qm < 0) or np.any(qm > 1) or np.any(qd < 0) or np.any(qd > 1):
            raise ValueError("Query occupancy and depth must be in [0,1]")
        if not np.all(np.any(qm > 0, axis=(1, 2))):
            raise ValueError("Every query view needs positive occupancy")
        if np.any(qd[qm == 0] != 0):
            raise ValueError("Query background depth must be zero")

        rm = np.asarray(reference_bank[f"mask_{scale}"])
        rd = np.asarray(reference_bank[f"depth_{scale}"])
        for start in range(0, count, 16):
            stop = min(start + 16, count)
            masks = rm[start:stop]
            union = np.maximum(qm[None], masks).sum(axis=(2, 3), dtype=np.float64)
            overlap = np.minimum(qm[None], masks)
            intersection = overlap.sum(axis=(2, 3), dtype=np.float64)
            mask_scores[start:stop, :, scale_index] = np.divide(
                intersection, union, out=np.zeros_like(intersection), where=union > 0
            )

            depth = rd[start:stop]
            error = (np.abs(qd[None] - depth) * overlap).sum(
                axis=(2, 3), dtype=np.float64
            )
            mae = np.divide(
                error, intersection, out=np.ones_like(error), where=intersection > 0
            )
            depth_scores[start:stop, :, scale_index] = np.clip(1.0 - mae / 0.30, 0, 1)

    component = 0.6 * mask_scores + 0.4 * depth_scores
    view_scores = component.mean(axis=2)
    scores = view_scores.mean(axis=1)
    return {
        "scores": scores.astype(np.float64),
        "view_scores": view_scores,
        "mask_scores": mask_scores,
        "depth_scores": depth_scores,
    }


def score_descriptors(
    query: dict[str, np.ndarray], reference_bank: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Compare each reference on equally weighted views and spatial scales."""
    return _score_validated_descriptors(
        query, reference_bank, validate_descriptors(reference_bank)
    )


def predict_descriptors(
    query: dict[str, np.ndarray],
    reference_bank: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    *,
    exclude_ids: set[str] | None = None,
    top_k: int = 3,
    temperature: float = 0.08,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Return the legacy softmax-weighted Cd estimate with cached reference views."""
    count = validate_descriptors(reference_bank)
    if len(records) != count:
        raise ValueError("Need one record per reference")
    if (
        isinstance(top_k, (bool, np.bool_))
        or not isinstance(top_k, (int, np.integer))
        or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer")
    if (
        not np.isfinite((temperature, epsilon)).all()
        or temperature <= 0
        or epsilon <= 0
    ):
        raise ValueError("temperature and epsilon must be finite and positive")
    ids = [str(row.get("id", "")) for row in records]
    if any(not value for value in ids) or len(set(ids)) != count:
        raise ValueError("Reference IDs must be unique and nonempty")
    try:
        cds = [float(row["cd"]) for row in records]
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Reference Cd labels must be finite and positive") from exc
    if any(not np.isfinite(cd) or cd <= 0 for cd in cds):
        raise ValueError("Reference Cd labels must be finite and positive")
    excluded = exclude_ids or set()
    candidates = [i for i, car_id in enumerate(ids) if car_id not in excluded]
    if not candidates:
        raise ValueError("No references remain after exclusion")

    scores = _score_validated_descriptors(query, reference_bank, count)
    ranked = sorted(candidates, key=lambda i: (-scores["scores"][i], ids[i]))[:top_k]
    selected = [scores["scores"][i] for i in ranked]
    logits = [temperature / (max(0.0, 1.0 - score) + epsilon) for score in selected]
    peak = max(logits)
    weights = np.exp(np.asarray(logits, dtype=np.float64) - peak)
    weights /= weights.sum()
    matches = []
    for index, weight in zip(ranked, weights):
        row = records[index]
        matches.append(
            {
                **row,
                "score": float(scores["scores"][index]),
                "weight": float(weight),
                "view_scores": {
                    view: float(scores["view_scores"][index, view_index])
                    for view_index, view in enumerate(VIEWS)
                },
                "scale_mask_scores": [
                    float(v) for v in scores["mask_scores"][index].mean(axis=0)
                ],
                "scale_depth_scores": [
                    float(v) for v in scores["depth_scores"][index].mean(axis=0)
                ],
            }
        )
    cd = float(sum(row["cd"] * row["weight"] for row in matches))
    return {"cd": cd, "top_k": matches, "reference_count": len(candidates)}
