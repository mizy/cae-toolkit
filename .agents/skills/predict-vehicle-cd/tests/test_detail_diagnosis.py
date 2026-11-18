import sys
from pathlib import Path

import numpy as np

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL.parent / "predict-aircraft-coefficients/scripts"))

from vehicle_cd_learning.detail_diagnosis import fit_probe


def test_probe_uses_train_and_validation_labels_without_test_label_leakage(tmp_path):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(24, 6))
    targets = np.stack(
        (0.24 + 0.01 * features[:, 0], -0.1 + 0.03 * features[:, 1]), axis=1
    )
    train, val = np.arange(24) < 12, (np.arange(24) >= 12) & (np.arange(24) < 18)
    original, report = fit_probe(features, targets, train, val, tmp_path / "first.npz")
    changed = targets.copy()
    changed[~(train | val)] = 1000
    repeated, second = fit_probe(features, changed, train, val, tmp_path / "second.npz")
    assert report["selected"] == second["selected"]
    np.testing.assert_array_equal(original, repeated)
    assert abs(original[~(train | val)] - targets[~(train | val)]).mean() < 0.01
