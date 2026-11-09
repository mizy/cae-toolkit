---
name: predict-aircraft-coefficients
description: Train and evaluate aircraft Cd and Cl coefficient predictors from complete five-view depth/mask geometry and declared flight conditions, with geometry-grouped COCOANet data.
---

# Aircraft coefficient learning

Initial scope is low-speed COCOANet CCA-family CFD coefficients (source velocity
at most 80 m/s and zero sideslip). The raw pinned CSV has 3,513 rows; this is not
the data card's advertised 3,570. Geometry-disjoint validation/tests measure new
parameterized shapes within this family, not arbitrary aircraft configurations.

The input is complete STL geometry with explicit source axes and units, alpha,
velocity and altitude. Source geometry is nTop mm, body X (nose toward -X), span Y
and up Z. Five complete views use one uniform maximum-extent scale per geometry,
with the same scale for every view. No part crops are generated. Geometry ratios
and physical body length are numeric inputs. Velocity and altitude remain direct
source inputs; a Reynolds/Mach feature is not fabricated without verified source
atmospheric/viscosity conventions. Pressure fields are not needed for this first
coefficient experiment. Source references for area/chord are retained per mesh.

Source: https://huggingface.co/datasets/ShapeBench/COCOANet , CC BY 4.0, revision
`210068b62a4678a3bcfb9a87926bcec44e6416f9`. Keep source attribution with derivatives.
Download only geometries and small reference data, not the 3.7 TB of field data.

From the project root:

```bash
export PYTHONPATH="$PWD/.agents/skills/predict-aircraft-coefficients/scripts:$PWD/.agents/skills/predict-vehicle-cd/scripts"
python -m aircraft_learning.prepare --csv /path/to/pinned/dataset_samples.csv --output-dir /new/prepared
python -m aircraft_learning.train /new/prepared --mode views --epochs 40 --device mps --output-dir /new/model
```

`prepare` owns source download, complete-view rendering, exact-design grouping,
source coefficient axis checks and train/val/test assignment. Matching source
STL bytes cannot cross splits. Grouping does not certify absence of near-duplicate
geometry. Its prepared output must finish without failures before training.

`AircraftCases` caches one view tensor per geometry, shared across flight cases.
`train` reuses `vehicle_cd_learning.train.train_model` and `CoefficientModel` as
concrete shared training consumers; random initialization prevents vehicle
pretraining from contaminating aircraft holdout. The Cd/Cl head remains shared.
All normalization uses training records only. Epoch selection uses validation
only, and the held-out geometry test is evaluated after training. Optional design
and condition-only tabular baselines use the same splits; their inputs are
explicitly different from the deployed mesh path.

Exported `model.pt` belongs to this aircraft input contract; use the aircraft
loader and predictor rather than the car's profile/range validator. Existing car
checkpoints, input contracts and default behavior are preserved.
