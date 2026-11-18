"""Scene ablation contracts and old-checkpoint compatibility."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from vehicle_cd_learning.model import VIEWS, CoefficientModel


def model(scene_mode="none"):
    torch.manual_seed(52)
    return CoefficientModel(
        7,
        size="small",
        encoder="pyramid_detail_attention",
        fusion="attention",
        scene_mode=scene_mode,
    )


@pytest.fixture
def inputs():
    torch.set_num_threads(2)
    torch.manual_seed(18)
    views = {name: torch.rand(3, 2, 32, 32) for name in VIEWS}
    for image in views.values():
        image[:, 1] = 1
    return views, torch.randn(3, 7), torch.tensor([0, 1, 2])


def test_no_scene_keeps_legacy_state_and_forward_graph(inputs):
    views, parameters, scene = inputs
    original = model()
    restored = CoefficientModel(
        7, size="small", encoder="pyramid_detail_attention", fusion="attention"
    )
    restored.load_state_dict(original.state_dict(), strict=True)
    assert not any(k.startswith("scene_") for k in original.state_dict())
    # Independent old forward graph: four shared views, bottom, conditions, gates.
    body = original.image_encoder(torch.cat([views[n] for n in VIEWS[:4]]))
    features = [*body.chunk(4), original.bottom_encoder(views["bottom"])]
    condition = original.parameter_encoder(parameters)
    features, _ = original.view_attention(features, condition)
    expected = original.head(torch.cat((*features, condition), dim=1))
    torch.testing.assert_close(original(views, parameters), expected, rtol=0, atol=0)
    torch.testing.assert_close(
        restored(views, parameters, scene), expected, rtol=0, atol=0
    )


def test_film_starts_as_concat_and_unknown_stays_unmodulated(inputs):
    views, parameters, scene = inputs
    concat, film = model("concat"), model("film")
    for key, value in concat.state_dict().items():
        torch.testing.assert_close(value, film.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(
        film(views, parameters, scene), concat(views, parameters, scene), rtol=0, atol=0
    )
    with torch.no_grad():
        film.scene_film[-1].bias.fill_(0.5)
    changed = film(views, parameters, scene)
    baseline = concat(views, parameters, scene)
    torch.testing.assert_close(changed[0], baseline[0], rtol=0, atol=0)
    assert not torch.allclose(changed[1:], baseline[1:])
    torch.testing.assert_close(
        film(views, parameters),
        film(views, parameters, torch.zeros_like(scene)),
        rtol=0,
        atol=0,
    )


def test_scene_and_film_receive_gradients_without_updating_unknown_code(inputs):
    views, parameters, scene = inputs
    network = model("film")
    network(views, parameters, scene).square().sum().backward()
    embedding = network.scene_embedding
    assert embedding.weight[0].eq(0).all()
    assert embedding.weight.grad[0].eq(0).all()
    assert embedding.weight.grad[1:].abs().sum() > 0
    assert network.scene_film[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("scene_mode", ["none", "concat", "film"])
def test_pooled_and_raw_paths_share_scene_conditioning(inputs, scene_mode):
    views, parameters, scene = inputs
    network = model(scene_mode)
    if network.scene_film is not None:
        with torch.no_grad():
            network.scene_film[-1].weight.normal_(std=0.02)
            network.scene_film[-1].bias.fill_(0.03)
    pooled = {
        name: (
            network.bottom_encoder if name == "bottom" else network.image_encoder
        ).pooled_features(views[name])
        for name in VIEWS
    }
    torch.testing.assert_close(
        network(views, parameters, scene),
        network.head(network.encode_pooled(pooled, parameters, scene)),
        rtol=1e-5,
        atol=1e-7,
    )


def test_scene_contract_rejects_invalid_mode_shape_or_type(inputs):
    views, parameters, scene = inputs
    with pytest.raises(ValueError, match="scene_mode"):
        CoefficientModel(7, scene_mode="expert")
    with pytest.raises(ValueError, match="five"):
        CoefficientModel(7, variant="parameters", scene_mode="film")
    network = model("concat")
    for invalid in (scene[:, None], scene.float()):
        with pytest.raises(ValueError, match="scene must"):
            network(views, parameters, invalid)


def test_scene_macro_validation_preserves_legacy_observed_target_loss():
    from vehicle_cd_learning.train import evaluate

    class Predictions(torch.nn.Module):
        target_count = 2

        def forward(self, _views, parameters, scene=None):
            return parameters

    truth = [torch.tensor([0.0, float("nan")]), *[torch.zeros(2) for _ in range(3)]]
    predicted = [torch.ones(2), torch.ones(2), torch.ones(2), torch.full((2,), 3.0)]
    rows = [
        ({}, p, y, torch.tensor(1 if i < 3 else 2))
        for i, (p, y) in enumerate(zip(predicted, truth))
    ]
    network = Predictions()
    micro, errors, counts = evaluate(
        network, DataLoader(rows, batch_size=2), torch.device("cpu")
    )
    macro, _, _ = evaluate(
        network,
        DataLoader(rows, batch_size=2),
        torch.device("cpu"),
        balance_scenes=True,
    )
    assert micro == pytest.approx(23 / 7)
    assert macro == pytest.approx((1 + 9) / 2)
    assert counts.tolist() == [4, 3] and torch.isnan(torch.tensor(errors[0, 1]))
    old = [row[:3] for row in rows]
    legacy, _, _ = evaluate(network, DataLoader(old, batch_size=2), torch.device("cpu"))
    assert legacy == micro
    with pytest.raises(ValueError, match="requires scene IDs"):
        evaluate(
            network,
            DataLoader(old, batch_size=2),
            torch.device("cpu"),
            balance_scenes=True,
        )


def test_paired_scene_training_initialization_rng_and_checkpoint_roundtrips(
    tmp_path, monkeypatch
):
    from test_learning_masked_training import TinyViewCases
    from vehicle_cd_learning import train as training

    torch.set_num_threads(2)
    physical = torch.arange(30, dtype=torch.float32).reshape(6, 5) / 10
    targets = torch.tensor(
        [[-1.0, -0.2], [1.0, 0.3], [0.5, 0.1], [-0.5, -0.4], [0.3, 0.8], [-0.3, -0.6]]
    )
    legacy_rows = list(TinyViewCases(physical.numpy(), targets.numpy()))
    rows = [(*row, torch.tensor(i % 2 + 1)) for i, row in enumerate(legacy_rows)]
    stats = {
        "parameter_mean": [0] * 5,
        "parameter_std": [1] * 5,
        "target_mean": [0, 0],
        "target_std": [1, 1],
    }
    metadata = {
        "render_config": {},
        "conditions": {},
        "parameter_ranges": {},
        "data_kind": "synthetic",
        "provenance": {},
        "train_ids": list(map(str, range(6))),
        "train_geometry_hashes": list(map(str, range(6))),
        "heldout_ids": [],
        "dataset_sha256": "synthetic-scene-fixture",
    }
    records, checkpoints, reports = {}, {}, {}

    class Recorded(CoefficientModel):
        def forward(self, views, parameters, scene=None):
            output = super().forward(views, parameters, scene)
            if self.training:
                signature = hashlib.sha256(
                    views["front"].detach().numpy().tobytes()
                ).hexdigest()
                records[label].append(
                    (
                        parameters.detach().clone(),
                        signature,
                        None if scene is None else scene.clone(),
                        output.detach().clone(),
                    )
                )
            return output

    monkeypatch.setattr(training, "CoefficientModel", Recorded)
    for label, mode in (
        ("legacy", "none"),
        ("A", "none"),
        ("B", "concat"),
        ("C", "film"),
    ):
        records[label] = []
        sampler = WeightedRandomSampler(
            torch.ones(6),
            6,
            replacement=True,
            generator=torch.Generator().manual_seed(45),
        )
        path = training.train_model(
            legacy_rows if label == "legacy" else rows,
            legacy_rows if label == "legacy" else rows,
            parameter_names=[f"p{i}" for i in range(5)],
            normalization=stats,
            metadata={
                **metadata,
                **({"scene_experiment": True} if label != "legacy" else {}),
            },
            output_dir=tmp_path / label,
            epochs=2,
            batch_size=3,
            seed=42,
            device_name="cpu",
            augment=True,
            model_size="small",
            encoder="pyramid_detail_attention",
            fusion="attention",
            preload=True,
            scene_mode=mode,
            scene_dropout=0.5 if mode != "none" else 0,
            train_sampler=sampler,
        )
        ck = torch.load(path, map_location="cpu", weights_only=True)
        loaded = CoefficientModel(**ck["model_config"])
        loaded.load_state_dict(ck["model_state"], strict=True)
        assert torch.isfinite(
            loaded(
                {k: v[None] for k, v in rows[0][0].items()},
                rows[0][1][None],
                rows[0][3][None],
            )
        ).all()
        checkpoints[label] = ck
        reports[label] = json.loads((path.parent / "training_report.json").read_text())
        assert len(reports[label]["history"]) == 2
    assert (
        len({checkpoints[k]["scene_common_initial_sha256"] for k in ("A", "B", "C")})
        == 1
    )
    for label in ("A", "B", "C"):
        torch.testing.assert_close(
            records[label][0][3], records["legacy"][0][3], rtol=0, atol=0
        )
        for reference, current in zip(records["legacy"], records[label]):
            torch.testing.assert_close(reference[0], current[0], rtol=0, atol=0)
            assert reference[1] == current[1]
    for b, c in zip(records["B"], records["C"]):
        assert torch.equal(b[2], c[2])
    assert any((record[2] == 0).any() for record in records["B"])
    assert [r["train_mse"] for r in reports["legacy"]["history"]] == [
        r["train_mse"] for r in reports["A"]["history"]
    ]
