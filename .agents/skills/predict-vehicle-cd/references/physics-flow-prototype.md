# Physics-driven conditional flow prototypes

These initial experiments test one offline training run and fixed-weight online
inference without interior CFD flow labels. They do not use vehicle geometry,
the 26 reference cars, turbulent closure, three-dimensional flow, Cd/Cl or the
five-view encoder. They do not establish vehicle accuracy or its runtime.

Run from the repository root with the optional learning dependencies installed:

```bash
export PYTHONPATH="$PWD/.agents/skills/predict-vehicle-cd/scripts"
.venv/bin/python -m vehicle_flow_learning.channel train \
  --output-dir outputs/predict-vehicle-cd/my-channel-run --steps 3000
.venv/bin/python -m vehicle_flow_learning.channel evaluate \
  outputs/predict-vehicle-cd/my-channel-run/model.pt --split test
.venv/bin/python -m vehicle_flow_learning.channel predict \
  outputs/predict-vehicle-cd/my-channel-run/model.pt \
  --re 35 --gradient 0.0125 --height 1.05 --wall 0.15 --size 128 \
  --output outputs/predict-vehicle-cd/my-channel-run/field.npz
```

The channel operator assumes steady, fully developed nondimensional 2D flow:
`v=0`, `p=-g*x`, `u=u(y)` and `u_yy + Re*g = 0`. Height, Reynolds number,
pressure-gradient magnitude and upper-wall velocity vary independently. The
wall envelope `eta*(1-eta)` also matches the exact pressure-driven parabolic
profile, so the experiment primarily tests the coefficient and parameter
dependence. Pressure and continuity are imposed structurally, not learned.

The manufactured solution experiment learns three fields with full steady 2D
incompressible Navier–Stokes residuals and boundary conditions:

```bash
.venv/bin/python -m vehicle_flow_learning.manufactured_2d train \
  --output-dir outputs/predict-vehicle-cd/my-manufactured-run --steps 3000
.venv/bin/python -m vehicle_flow_learning.manufactured_2d predict \
  outputs/predict-vehicle-cd/my-manufactured-run/model.pt \
  --amplitude 0.175 --re 6.5 --size 128 \
  --output outputs/predict-vehicle-cd/my-manufactured-run/field.npz
```

Its synthetic body force and wall values are derived from the manufactured
solution and therefore contain knowledge of it. The interior analytical field
is used exclusively for evaluation. Unseen parameter values are interpolations
within the training range; they are not unseen geometries. Both commands reject
out-of-range conditions at prediction. Training creates a new output directory
and never overwrites an existing run. Checkpoints here do not contain optimizer
state and cannot resume training exactly.

The first frozen runs and case-level metrics are stored under
`outputs/predict-vehicle-cd/physics-channel-moving-wall-20261104/` and
`outputs/predict-vehicle-cd/physics-operator-20261104/manufactured-2d/`. The
reported small CPU forward timing excludes vehicle preparation, 3D output and
real surface integration. Follow the separate neural-flow development plan for
the multiscale vehicle model, physical boundary conditions and acceptance gates.
