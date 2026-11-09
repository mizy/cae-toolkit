"""Exercise the unattended full-data workflow using explicitly synthetic labels."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SKILL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_learning_fixture import write_fixture
from vehicle_cd_learning import train_full_attention
from vehicle_cd_learning.inspect_attention import inspect
from vehicle_cd_learning.predict import load_model


def test_full_attention_run_retains_holdout_and_reuses_completed_checkpoint(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    source = write_fixture(data_root / "fixture")
    source.rename(source.parent / "full-source.json")
    source = source.parent / "full-source.json"
    monkeypatch.setattr(
        train_full_attention, "SOURCES", (("fixture", "full-source.json"),)
    )
    output = tmp_path / "experiment"
    result = train_full_attention.run(
        data_root, output, epochs=1, batch_size=8, device="cpu"
    )
    checkpoint_path = output / "fixture/model/model.pt"
    _, checkpoint = load_model(checkpoint_path)
    manifest = json.loads(source.read_text())
    expected = {c["case_id"] for c in manifest["cases"] if c["split"] == "train"}
    assert set(checkpoint["train_ids"]) == expected
    assert checkpoint["model_config"]["encoder"] == "pyramid_detail_attention"
    assert checkpoint["model_config"]["fusion"] == "attention"
    assert result["fixture"]["counts"]["test"] == 8
    heldout = next(c for c in manifest["cases"] if c["split"] == "test")
    attention = inspect(
        checkpoint_path,
        output / "fixture/prepared",
        heldout["case_id"],
        tmp_path / "attention",
    )
    assert attention["split"] == "test"
    assert attention["data_kind"] == "synthetic"
    with np.load(tmp_path / "attention/spatial-gates.npz") as maps:
        assert maps["right"].shape == (256, 256)
        assert maps["view_weights"].sum() == pytest.approx(1.0)
        assert all(np.isfinite(maps[name]).all() for name in maps.files)
    with Image.open(tmp_path / "attention/complete-view-gates.png") as image:
        assert image.size == (1280, 576)
    modified = checkpoint_path.stat().st_mtime_ns
    train_full_attention.run(data_root, output, epochs=1, batch_size=8, device="cpu")
    assert checkpoint_path.stat().st_mtime_ns == modified
    manifest["cases"][0]["cd"] += 0.01
    source.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Source changed"):
        train_full_attention.run(
            data_root, output, epochs=1, batch_size=8, device="cpu"
        )
