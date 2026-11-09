# Visual cleanup, fidelity, and opening-policy validation

Date: 2026-10-23. Status: implementation plan with a measured kernel baseline; runtime mesher behavior is unchanged.

## Acceptance target

Use AI visual reasoning and engineering intent to select cleanup operations. Deterministic geometry
workers execute the operations and own acceptance. Preserve valid visible source surfaces, close only
authorized defects, and preserve functional passages and independent parts when requested.

"Visually unchanged" is an explicit measured requirement. Projected vertices lying on source
triangles, watertightness, or a high whole-object silhouette score alone cannot establish it.

Distinguish three things in both decisions and tests:

- A missing surface is an open mesh boundary requiring repair or a deliberate boundary patch.
- A functional passage can exist in an already watertight solid, such as a thick-walled tube.
- An inlet/outlet patch can close a CFD domain surface while retaining its boundary-condition role.

The current `seal_all_openings` target is a declared policy, not proof that every physical passage
is closed. Actual passage preservation depends on geometry, voxel pitch, and sealing radius.

## Milestones

### P0 — Freeze evidence and establish the baseline

1. Generate synthetic geometry with known units, reference shapes, component counts, and passage
   paths. Keep generated mesh/image/report artifacts out of Git.
2. Run existing TSDF, gated projection, exact intersection, and serialized-readback entry points.
3. Measure both directions of surface distance, including triangle interiors; inspect registered
   source/candidate images. Record failure and rejected candidates without calling them accepted.
4. Verify metric oracles with negative controls: vertices on a sphere but faces off the sphere,
   and an open duct versus a blocked duct.
5. Run a separate full-worker policy A/B using the same source and parameters, changing only a
   fingerprint-bound `cap` versus `preserve` decision. Check the actual aperture, not just JSON.

Exit: reproducible measurements and an explicit list of missing capabilities. A failed fidelity or
topology expectation is a useful baseline finding, not permission to relax its threshold.

### P1 — Connect one opening decision to one local operation

1. Reuse `repair_policy.py` stable region IDs and packet/item fingerprints. Present neutral shaded
   context and a local close-up with the same region highlighted; include exact measured size and
   connectivity. Allow the Agent to request another view or abstain.
2. Start with one unambiguous missing-cover boundary loop. Implement `cap` under the region owner;
   implement `preserve` as explicit protection from sealing, with its intended domain role recorded.
3. Make the owning geometry entry point consume the decision before geometry mutation. Preserve
   unaffected source faces directly. Define the patch and any conforming transition ring as the
   authorized edit region; avoid a new orchestration framework or decision registry.
4. Reject unsupported actions, stale decisions, and out-of-tolerance patches explicitly. Record the
   deterministic operator, affected source IDs, output region, and before/after measurements.

Exit: the same opening yields the requested geometric difference for `cap`/`preserve`, the unrelated
surface remains unchanged, and both supported outputs pass their declared topology/domain contract.
`porous_surface` stays unsupported until a tagged patch and downstream boundary handoff exist.

### P2 — Preserve appearance, thin features, and component relationships

1. Expand local repair to compatible seam stitching and measured missing-surface patches. Keep
   valid source surfaces as shape truth; use TSDF as a closure scaffold where reconstruction is needed.
2. Protect functional passages and independent parts throughout voxel sealing, reconstruction,
   projection, and final validation. A render-space visibility mask alone is insufficient.
3. Use source curvature and sharp edges for local size control. Refine geometry where the measured
   surface error exceeds budget; preserve or explicitly reconstruct constrained feature curves.
4. Validate both local patch seams and the complete candidate after every eligible mutation. Roll
   back rejected edits; preserve the original immutable input and the last valid artifact.

Exit: all mandatory synthetic fidelity, opening, and separation tests below pass. The renderer and
visual reviewer also pass; numerical gates and visual review are separately reported.

### P3 — Evaluate the AI cleanup loop and downstream usefulness

1. Give the active Agent anonymized region IDs, identical observation channels, and a written target.
   Hide expected decisions and fixture answer labels from the observation packet. Save every view
   request, rationale, decision, deterministic operation, and validation result.
2. Evaluate the same aperture under external-flow and cooling-flow intents. Include ambiguous cases
   where the correct action is another observation or a question. Validate geometry after execution.
3. Repeat with changed camera poses, rotations, and units. Track action consistency, missed
   functional openings, unnecessary edits, intervention count, latency, and model cost.
4. Add held-out public/synthetic assemblies. Use an independent human rubric for appearance and
   semantic decisions; do not infer model accuracy from protocol tests or a self-reviewed showcase.
5. Compare one external-flow scenario against GAI at equivalent geometry targets and mesh budgets.
   Verify boundary identities and file import, volume-mesh/boundary-layer generation, and then
   compare mesh-refinement trends in Cd and local Cp. No external jobs are submitted by this plan.

Exit: report geometry success, visual acceptance, AI decision quality, full-worker success, and CFD
acceptance independently. A kernel baseline cannot satisfy the later gates.

## Proposed acceptance budgets

The following numerical defaults apply only to the 10–20 mm synthetic fixtures. For a vehicle or
another engineering target, define physical tolerances and protected regions before running.

| Requirement | Proposed gate | Current baseline coverage |
| --- | --- | --- |
| Unaffected source region | Original source vertex positions and retained face connectivity unchanged; explicit edit/transition mask | Planned; current experiment globally reconstructs |
| Surface fidelity | Bidirectional area-weighted sampled p95 ≤ 0.1 mm and sampled max ≤ 0.3 mm | Measured with seven points per triangle; not a certified Hausdorff bound |
| Critical feature fidelity | Separate per-region distance/normal/feature-curve budget; no global percentile substitution | Planned |
| Appearance | Same cameras, scale, material, lighting, and normal settings; six axis and four oblique views; outside the authorized edit mask, silhouette overlap ≥ 99.9% and contour displacement ≤ 1 px at 1024 px | Planned; baseline has unscored paired isometric renders |
| Geometry validity | Zero boundary/non-manifold edges and vertices, degeneracy, winding inconsistency, exact intersection pairs; reliable outward component volume | Measured for globally closed baseline targets |
| Topological features | Expected component count and Euler characteristic; specific aperture/connection measurements | Measured counts/Euler plus five duct rays; not a complete topological-equivalence proof |
| Passage preservation | Connected intended path remains open, with cross-section and minimum-clearance change ≤ 5% | Five fixed rays measured; full path/section/clearance validation planned |
| Serialization | VTP readback preserves geometry and topology; exact intersection certification repeated on readback | Measured |
| AI execution | Correct operation on referenced current region; abstention on ambiguity; same aperture changes under `cap`/`preserve` | Full-worker injected A/B resolves both decisions but produces identical candidates; actual AI accuracy remains untested |

For authorized caps, compare against the known intended surface in synthetic tests. For real missing
geometry, disclose the assumed patch shape; the absent original surface cannot provide a ground truth.

## Test matrix

| ID | Input and intent | Expected result | Verification |
| --- | --- | --- | --- |
| G01 | Closed sphere, preserve shape | No flattening beyond tolerance | Both-direction triangle-interior distances and registered renders |
| G02 | Sharp box, preserve edges | Corners and edge lines remain within budget | Whole-surface baseline; feature-curve checks in P2 |
| G03 | Sphere with small missing top cover, repair | Close only missing region; match intact reference | Closed topology and reference distances; edit mask in P1 |
| G04 | Thick tube, 4 mm bore, preserve flow | Through-passage survives | Euler 0 and clear axial probe rays; full clearance in P2 |
| G05 | Thick tube, 1.2 mm bore, preserve flow | Narrow functional passage survives or output is rejected | Same as G04; stresses sealing radius |
| G06 | Two closed parts separated by 0.8 mm, keep separate | Two closed components; gap remains open | Component count 2, Euler sum 4, distances; gap clearance in P2 |
| G07 | 0.4 mm plate, preserve | Thickness and independent closed component survive | Topology and distance checks; local thickness in P2 |
| A01 | Same aperture, external-flow target explicitly requests cap | AI selects cap; worker creates the referenced patch | Grounded decision and actual aperture test; P1/P3 |
| A02 | Identical aperture, cooling-flow target explicitly requests preserve | AI preserves channel; worker does not seal it | Same source hash as A01, same other parameters; P1/P3 |
| A03 | Ambiguous recessed opening, no engineering purpose | Request another view or ask; no destructive edit | Recorded observation/action trace; P3 |
| A04 | Visible accessory and hidden support | Remove support only when measured evidence supports it | Hide/show comparison, source-ID edit mask, geometry revalidation; P2/P3 |
| A05 | Stale geometry hash, unknown view ID, raw coordinate edit | Reject before geometry execution | Existing `test_agent_repair_contract.py` |
| A06 | Agent accepts a geometrically invalid candidate | Deterministic gates still reject it | Existing contract tests; full-worker negative control in P1 |
| R01 | Repeat at two closure pitches; then rotate and change units | Report resolution trends; equivalent physical decisions | Two pitches measured; rigid-transform/unit equivalence in P2 |
| I01 | Boundary-tagged output handed to downstream mesher | Preserved boundary roles, valid volume mesh | Deferred end-to-end integration in P3 |

## Reproduce the baseline

From the repository root:

```bash
.venv/bin/python .agents/skills/mesh-watertight-repair/scripts/fidelity_policy_baseline.py \
  --output-dir outputs/fidelity-policy-validation-20261023-final

.venv/bin/python -m pytest -q \
  .agents/skills/mesh-watertight-repair/tests/test_fidelity_policy_baseline.py \
  .agents/skills/mesh-watertight-repair/tests/test_agent_repair_contract.py \
  .agents/skills/mesh-watertight-repair/tests/test_source_projected_contract.py \
  .agents/skills/mesh-watertight-repair/tests/test_source_projected_closure.py \
  .agents/skills/mesh-watertight-repair/tests/test_sdf_closure.py
```

Use a fresh output directory for each run. `baseline.json` records all measurements, environment
versions, and kernel failures. `baseline.html` includes the table and embedded paired renders.
Each case directory contains source/reference VTP, a candidate when produced, and its measurement.
The kernel experiment uses all source faces as candidates, bypassing the worker's Stage 1 selection.
The script then invokes the actual worker three times on the same missing-cover fixture: unreviewed,
`cap`, and `preserve`. This separately tests policy plumbing with injected decisions, not a live AI.

## Measured results on 2026-10-23

The initial matrix used closure pitches 1.0 and 0.5 mm, a requested 1.0 mm sealing radius, and the
synthetic tolerance budgets above. Thirteen of fourteen kernel experiments produced candidates;
none passed every baseline expectation. This result applies to these settings and fixtures and
does not establish a universal lower bound on the algorithm's achievable fidelity.

At the finer 0.5 mm pitch:

| Case | Worst-direction sampled p95 / max (mm) | Specific result |
| --- | --- | --- |
| Sphere | 0.0206 / 0.3613 | p95 passes; sampled max exceeds 0.3 mm |
| Sharp box | 0.3536 / 0.3536 | Corners/edges are visibly beveled |
| Missing cover | 0.3642 / 0.9308 | Strong reconstruction artifacts; Euler -582 rather than 2; readback gate fails |
| 4 mm bore | 0.0896 / 0.4400 | All five channel rays remain clear and Euler remains 0; sampled max fails |
| 1.2 mm bore | 2.2319 / 4.3906 | All five channel rays blocked; Euler becomes 2 |
| Two parts, 0.8 mm gap | 1.4080 / 3.7713 | Component count changes from 2 to 1; gap is bridged |
| 0.4 mm plate | 0.3432 / 0.5771 | Reconstructed at 0.5 mm pitch but fidelity fails; at 1.0 mm pitch zero-surface extraction fails |

For the full-worker A/B, both fingerprint-bound decisions resolve successfully, but `cap`,
`preserve`, and unreviewed runs have the same candidate geometry hash. All three runs are rejected
by the production contract; `outputs.accepted_mesh_vtp` is not populated. This demonstrates the
missing geometric effect of policy in this path, alongside functioning rejection of this defective
candidate. It is not an AI classification-accuracy result.

The retained automated suites pass 105 tests plus seven fixture subtests. These verify existing
contracts and the measurement oracles; their success does not convert the failed geometry matrix
into a mesher acceptance pass. VTK/scikit-image emitted NumPy deprecation warnings.

Prioritize P1 policy-to-operation execution and P2 passage/component/feature preservation before
an expensive vehicle-scale comparison. Keep the failing cases and their thresholds unchanged while
implementing those capabilities.

## Reused entry points and retained responsibilities

- `fidelity_policy_baseline.py:run_case` calls `build_tsdf_closure`, `project_closure_to_source`,
  `repair_projected_self_intersections`, `mesh_report`, `roundtrip_validation`, and `write_vtp`.
- `sampled_distance` is used for both distance directions and the analytical negative-control test.
- `channel_rays_clear` is used for both duct cases and the blocked-passage negative-control test.
- `synthetic_cases` owns only reproducible synthetic inputs and expected references, consumed by the
  baseline CLI and fixture/oracle tests. It owns no runtime mesher state or AI decision logic.
- `run_policy_ab` is called by the baseline CLI; it invokes the existing full worker in separate
  processes and uses `geometry_hash_from_file` to compare actual candidate geometry.
- The full worker remains `two_stage_watertight_remesh.py`; no new Agent, runtime manager, or
  geometry-repair abstraction is introduced by this validation work.
