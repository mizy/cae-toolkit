# Public vehicle aerodynamics datasets

## Recommended source for the current learner

The downloaded pilot uses [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml) at immutable revision `5d448b209bf654503c64ce7261c34fa125f46392`. Its [paper](https://arxiv.org/abs/2408.11969) describes 500 designed Ford OCDA DrivAer notchback variants; the current public split file covers 484 complete cases. It supplies original STL geometry and per-case time-averaged force coefficients. The source is [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

The pilot contains 40 train, 10 validation, and 10 test geometries sampled deterministically from the official `full_train`, `full_val`, and `full_test` partitions. All 60 designs share the same `family_id`; the split is geometry-disjoint within one morphed vehicle family. This supports an in-family interpolation check and does not support claims about unseen vehicle families.

The importer reads the per-case `force_mom_i.csv` variable-reference labels and `geo_ref_i.csv`. `Cd` and `Cl` are normalized by each geometry's frontal reference area; the corresponding area and force reference length are retained in `assumptions`. The source paper defines (C_d=F_x/(qA_{ref})), (C_l=F_z/(qA_{ref})), and (q=\frac12\rho U^2). It reports (U_\infty=38.889\,\mathrm{m/s}), (T=293.15\,\mathrm{K}), kinematic viscosity (1.507\times10^{-5}\,\mathrm{m^2/s}), static wheels, zero yaw, and a static ground split into an upstream slip section and downstream no-slip section. The turbulence method is a hybrid RANS-LES/SA-DDES approach. The paper reports baseline (Re=7.19\times10^6) using a 2.786 m wheelbase; density is not reported, so the importer leaves it unknown.

For this learner, `reynolds` is derived using the measured STL body length along raw X, (U_\infty L_{body}/\nu), so it follows the repository's body-length convention. The source CFD holds velocity, viscosity, and yaw fixed. This derived input Reynolds value is a reference-length conversion, not evidence that the source ran a Reynolds sweep. Source geometry uses metres with X streamwise, Y lateral, Z vertical; a run 1 side projection confirms the nose at raw (-X), roof at (+Z), and wheel contact at minimum Z. The manifest records `front_axis: -x`, `up_axis: +z`, and zero ground clearance for the source wheel-contact plane.

Run the bounded importer from the repository root:

```bash
python .agents/skills/predict-vehicle-cd/scripts/vehicle_cd_learning/import_drivaerml.py \
  --train-count 40 --val-count 10 --test-count 10 --workers 4
```

The ignored output contains the original source STLs, coefficient/reference CSVs, source README and license, the pinned official split manifest, the generated `training_manifest.json`, and a compact run 1 side projection. A typical STL is about 142 MB; this pilot is about 8.6 GB. The full DrivAerML repository is about 31 TB, so the importer caps an invocation at 120 cases.

The current collection also contains 24 paired AhmedML and 24 paired WindsorML
cases under `outputs/predict-vehicle-cd/public-data/`, each using official 16/4/4
split prefixes. Their 48 STL files passed pinned size/hash verification. Collect
the same bounded pools into a fresh output root with:

```bash
export PYTHONPATH="$PWD/.agents/skills/predict-vehicle-cd/scripts"
.venv/bin/python -m vehicle_cd_learning.collect_bluff_bodies \
  --output-root outputs/predict-vehicle-cd/secondary-cfd --workers 4
```

An empty catalogue is initialized from the pinned small source files automatically.
The declarative physics/source profiles live in `assets/public-cfd-profiles.json`
and are consumed by this collector. Windsor STLs include support pins reaching
the ground; their variable body clearance is already represented by the mesh, so
the additional renderer clearance is zero. Windsor's canonical freestream yaw
is +2.5 degrees, converted from the source body rotation of -2.5 degrees. Its
published coefficients retain the source yawed-model axes. See
`learning-cnn-design.md` for the three separate CFD training comparisons.

## Other paired vehicle and car-like sources

| Source | Geometry and labels | Conditions and coefficient references | License and use |
| --- | --- | --- | --- |
| [AhmedML](https://huggingface.co/datasets/neashton/ahmedml) | 500 parametric Ahmed-body surfaces with Cd/Cl. These are simplified bluff-body shapes, not complete road cars. | [Paper](https://arxiv.org/abs/2407.20801): (U_\infty=1\,\mathrm{m/s}), (\nu=3.75\times10^{-7}\,\mathrm{m^2/s}); (Re=7.68\times10^5) on the 0.288 m body height or (2.78\times10^6) on the 1.044 m body length. No stilts are modeled; the body is 0.05 m above the floor. OpenFOAM v2212 with Spalart-Allmaras DDES. Time averaging starts after 10 CTU, but the paper and README conflict on total run duration (40 vs 80 CTU), so the duration needs source reconciliation. Baseline reference area is 0.112 m². The source inventory has 500 constant-reference force rows but only 499 varying-reference rows. | CC BY-SA 4.0. Useful as its own low-speed bluff-body experiment; keep its geometry family and 50 mm floor clearance explicit. |
| [WindsorML](https://huggingface.co/datasets/neashton/windsorml) | 355 designed Windsor-body cases; 350 currently have downloadable STL geometry, while the remaining five lack per-run meshes and are unsuitable as paired geometry samples. Time-averaged Cd, Cl and other force/moment labels are available. The source design clearance varies from 10 to 200 mm. | [Paper](https://arxiv.org/abs/2407.19320): 40 m/s inlet, 2.5° yaw, (Re=2.9\times10^6) based on body length. Static no-slip ground with slip/inviscid top and side walls; Volcano ScaLES WMLES with Vreman subgrid closure; outputs are time-averaged, but the paper gives no single per-case averaging duration. Source has fixed-reference and per-geometry frontal-area coefficient tables; record which table is used. | CC BY-SA 4.0. Separate Windsor-body family and yaw convention from DrivAerML; preserve each case's ground clearance. |
| [DrivAerNet++](https://github.com/Mohamedelrefaie/DrivAerNet) | 8,150 reported designs across fastback, notchback, and estateback classes, with STL and Cd/Cl/axle-lift labels. Coefficient and frontal-area tables are separate from the large field archives. | [Paper](https://arxiv.org/abs/2406.09624): steady OpenFOAM v11, (k\)-\(\omega\) SST, (U_\infty=30\,\mathrm{m/s}), air at 298.15 K, (\rho=1.184\,\mathrm{kg/m^3}), (\nu=1.56\times10^{-5}\,\mathrm{m^2/s}), body-length Reynolds range about (8.37\times10^6\)–(1.01\times10^7). The paper specifies moving ground and rotating wheels. Read the released per-design coefficient/reference-area table before import. | CC BY-NC 4.0. Non-commercial only unless a separate commercial permission is obtained. Do not merge with the CC BY-SA pilot without an explicit license policy. |
| [DrivAerNet](https://github.com/Mohamedelrefaie/DrivAerNet) | Earlier 4,000-design DrivAer release. It shares the broad DrivAer design family with DrivAerNet++; deduplicate geometry and keep version/source provenance. | Its conditions and reference areas must be read from the exact selected release and its files. Do not copy DrivAerNet++ settings to it by assumption. | The source project groups DrivAerNet and DrivAerNet++ under CC BY-NC 4.0; verify the selected release metadata. |
| [SHIFT-SUV](https://huggingface.co/datasets/luminary-shift/SUV) | Thousands of AeroSUV platform variants with geometry and transient drag/lift histories. Separate full-scale/quarter-scale campaigns and rear-end configurations exist. | Source README reports (T=293.15\,\mathrm K), (\rho=1.225\,\mathrm{kg/m^3}), dynamic viscosity (1.81\times10^{-5}\,\mathrm{Pa\,s}), zero yaw/AoA, and constant baseline reference area. Full scale uses (U=31.2928\,\mathrm{m/s}), (A_{ref}=2.464\,\mathrm{m^2}); quarter scale uses (U=50\,\mathrm{m/s}), (A_{ref}=0.154\,\mathrm{m^2}). Read the run folder and coefficient history to select the matching scale. | CC BY-NC 4.0 and gated behind contact-information sharing. No gated files were accessed. Keep scale and configuration pools separate. |
| [SHIFT-Truck](https://huggingface.co/datasets/luminary-shift/Truck) | 982 completed cases from a 1,000-variant pickup-truck design; geometry, parameters, and transient-averaged Cd/Cl are described by the source. Surface-only data is about 0.76 TB; fields with volume about 11.6 TB. | Source README: (U=38.889\,\mathrm{m/s}), (T=293.15\,\mathrm K), zero yaw/AoA, 5-belt rolling ground, rotating wheels, (Re\approx1.4\times10^7) on 5.25 m, and constant baseline reference values (A_{ref}=2.65\,\mathrm{m^2}), (L_{ref}=5.25\,\mathrm m). Transient SA-DDES/Vreman, average over the final 10,000 of 20,000 steps. | CC BY-NC 4.0 and gated behind contact-information sharing. No gated files were accessed. Separate pickup family and reference contract. |
| [Car Drag Prediction](https://decode.mit.edu/projects/dragprediction/) | The paper reports 4,535 successful CFD labels from repaired ShapeNet car meshes, including 4,948 original/width-resized cases before filtering; mirrored forms duplicate existing labels. It supplies Cd only. | [Paper](https://arxiv.org/abs/2306.06110): all meshes scaled to 3.5 m length, stationary body and road, 40 km/h inlet, steady OpenFOAM `SimpleFoam`; 300 iterations with the last 50 averaged. The source paper does not state the source license or a complete condition/reference-area table on its project page. | Treat as exploratory metadata only until data redistribution and ShapeNet-derived mesh rights are confirmed. Mirrored samples must stay in the original geometry group. |

## Benchmarks and separate flow domains

[AutoCFD](https://autocfd.org/) is the automotive CFD workshop programme for shared road-car test cases. Its current site describes the DrivAer Notchback and Windsor Squareback resources and identifies DrivAerML as the ML training dataset used for AutoCFD5. Treat workshop cases as validation references unless a specific released multi-case geometry/coefficient bundle is identified.

[HiLiftAeroML](https://huggingface.co/datasets/nvidia/HiLiftAeroML) contains 1,800 cases across 180 NASA CRM-HL aircraft geometries and ten angles of attack (4°–22°). It is CC BY 4.0, but aircraft force coefficients depend on flight conditions such as angle of attack and compressibility. Keep its manifests, splits, and targets in a separate aerospace pool.

[Ship-D](https://github.com/noahbagz/ShipD) contains 30,000 parameterized hulls with 32 wave-drag coefficients per hull from eight speeds and four draft marks. Its inputs and wave-resistance labels describe hydrodynamic hull design, not road-car external aerodynamics. The project README points to a [Harvard Dataverse record](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/MMGAUS); the license was not verified from the accessible record, so check its terms before use. Keep these geometries and targets in a separate ship pool.

## Import rules

- Keep one CFD/experimental condition and coefficient reference definition within each prepared dataset. Source licenses can differ, and measured, simulated, manufacturer-claimed, and assumed labels must remain distinguishable.
- Preserve every morphed sample's actual base-family identifier. Use the source's design-disjoint split protocol where available; do not relabel variants as unrelated families.
- Record geometry units, front/up axes, tire/ground clearance, source conditions, coefficient reference area, and normalization length per source. Leave unavailable source values null or explicitly unknown.
- Do not convert predictions, literature Cd values without matching geometry, or aircraft/ship coefficients into CFD labels for a car mesh.
