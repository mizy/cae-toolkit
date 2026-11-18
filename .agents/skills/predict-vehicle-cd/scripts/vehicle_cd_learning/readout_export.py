"""Export identity-normalization controls to the existing dimensionless predictor."""

import argparse
import hashlib
from pathlib import Path

import torch

from . import dimensionless as dim
from .model import CoefficientModel
from .readout import BUFFER_NAMES, CONTRACT, load_model


def export(source, output):
    """@entry Remove only provably identity buffers; keep trained neural tensors."""
    if output.exists():
        raise FileExistsError("Refusing to overwrite a checkpoint")
    model, ck = load_model(source)
    for name in BUFFER_NAMES:
        value = getattr(model, name)
        expected = (
            torch.ones_like(value) if name.endswith("std") else torch.zeros_like(value)
        )
        if not torch.equal(value, expected):
            raise ValueError(
                "Scaled features require the readout loader; cannot remove normalization"
            )
    state = {k: v for k, v in ck["model_state"].items() if k not in BUFFER_NAMES}
    compatible = CoefficientModel(**ck["model_config"])
    compatible.load_state_dict(state, strict=True)
    ck.update(
        input_contract=dim.CONTRACT,
        model_state=state,
        readout_export={
            "source_contract": CONTRACT,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "method": "Remove only identity pooled-feature buffers; every trained neural tensor is unchanged",
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(export(args.source, args.output))
