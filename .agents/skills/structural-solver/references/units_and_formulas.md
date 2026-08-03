# Units, Materials, and Analytical Formulas

## Units

| `--units` | length | force | stress | stiffness |
|---|---|---|---|---|
| `m` | m | N | Pa | N/m |
| `mm` | mm | N | MPa | N/m |

Internally the solver always works in SI (m, N, Pa); JSON stores user units.
`display_scale` multiplies displacement for the deformed VTP geometry; the true
displacement is always available in the `displacement` point array.

## Materials (SI)

| name | E (GPa) | nu |
|---|---|---|
| steel | 210 | 0.30 |
| aluminum | 69 | 0.33 |
| titanium | 116 | 0.34 |
| cast_iron | 110 | 0.26 |

## Analytical benchmarks

Tension bar (fixed one end, axial force F):

- delta = F·L/(E·A)
- sigma = F/A

Cantilever (fixed one end, transverse force F at tip):

- delta = F·L³/(3·E·I), I = b·h³/12
- k = 3·E·I/L³

Static stiffness used by the report: k = |F| / delta,
delta = mean(load-face displacement) · load direction.

## Hinge demo case

- Geometry (mm): blade 1 x∈[0,26], knuckle block x∈[26,36] z∈[-4,6],
  blade 2 x∈[36,62]; width y∈[0,30]; blade thickness 2 mm.
- Fix: blade 1 end face `x_min`. Load: blade 2 end face `x_max`, F = 100 N downward (-z).
- Measured (1.5 mm mesh, steel): delta ≈ 1.58 mm, k ≈ 63.3 kN/m,
  max von Mises ≈ 143.5 MPa, strain energy ≈ 0.079 J.
- Convergence: 3.0 mm vs 1.5 mm delta change ≈ 0.9% (gate < 15%).
