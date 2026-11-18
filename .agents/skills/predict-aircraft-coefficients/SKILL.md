---
name: predict-aircraft-coefficients
description: Train and evaluate aircraft Cd and Cl coefficient predictors from complete five-view depth/mask geometry and declared flight conditions, with geometry-grouped COCOANet data.
---

# Aircraft coefficient learning

For actual vehicle/aircraft training sources, splits and downloaded-only candidates,
read [the training data inventory](../cad-surface-mesher/docs/aerodynamic-training-datasets.md).

Initial scope is low-speed COCOANet CCA-family CFD coefficients (source velocity
at most 80 m/s and zero sideslip). The raw pinned CSV has 3,513 rows; this is not
the data card's advertised 3,570. Geometry-disjoint validation/tests measure new
parameterized shapes within this family, not arbitrary aircraft configurations.

For the separate CRM-HL high-lift dataset, use `aircraft_learning.prepare_hilift`
and the existing trainer with its explicit Re/Mach condition schema. See
[the public-data expansion record](../cad-surface-mesher/docs/aerodynamic-data-expansion-20261109.md)
for source hashes, half-airframe mirroring, geometry-held-out splits and results.
These checkpoints have their own input contract; the COCOANet prediction CLI
continues to accept only its original seven-parameter schema.

For the completed joint vehicle/aircraft scene A/B/C experiment, read
[the scene-conditioning record](../cad-surface-mesher/docs/aerodynamic-scene-conditioning-plan.md).
Its concrete entry points are `aircraft_learning.prepare_scenes`,
`scene_experiment freeze/train`, and `compare_scenes`; the existing trainer and
complete-view model are reused. Experimental 40-input checkpoints have a separate
contract and do not replace the original predictors.

For the separately prepared2026-11-16 dimensionless joint dataset and explicitly
assumed COCOANet Reynolds completion, read the [completion record](../predict-vehicle-cd/references/learning-label-completion.md).
User-authorized ISA/Sutherland estimates remain marked as estimates; old aircraft
checkpoints and source labels are unchanged.

The subsequent [full-pool online-jitter experiment](../predict-vehicle-cd/references/learning-online-jitter-full-pool.md)
uses this completed36-field joint dataset in two single-seed six-epoch runs.
All4,121 train records supervise an auxiliary coefficient readout on the Rim B
shared CNN. Aircraft sampling results remain mixed and no candidate passed all
validation gates; existing aircraft and joint A defaults remain separate.

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
