# Mesh Watertight Repair

Build a geometry-driven watertight exterior shell from a mesh input.

The default projected-shell workflow retains certified, nonintersecting closed source shells
and reconstructs only the remaining surface set. A valid complete source bypasses voxel
reconstruction. `--stitch-tolerance T` enables bounded matching-edge sewing; zero (the default)
keeps approximate sewing disabled. `--repair-timeout-seconds 60` bounds the complete projected-shell
worker, including native calls. That main route retains complete connected shells. The separate
[local voxel operator](#bounded-local-voxel-repair) now supports reviewed patches inside a closed
component; automatic region selection and integration into the main route remain pending.

The default purpose is `watertight`. Add `--purpose cfd --cfd-max-edge-length LENGTH` to the worker
to run optional source-based conforming surface refinement after shell acceptance. `LENGTH` uses
input coordinate units. The accepted pre-refinement shell is retained when CFD quality gates fail.
See [SKILL.md](SKILL.md#optional-cfd-surface-postprocessing) for the separate-stage CLI and gates.

For source-triangle preservation, `--surface-strategy source-material --source-max-error E`
uses an error budget in input units. This optional path builds and validates closed material
surfaces with reviewed boundary decisions; it does not certify a CFD fluid region or printing
thickness. See [source-preserving material closure](SKILL.md#source-preserving-material-closure-for-arbitrary-meshes)
for the review workflow and fidelity limits.
The optional `--source-union-backend manifold` is a faster positive-shell path with the same
acceptance gates; unsupported or invalid inputs remain rejected.
The experimental `--source-union-backend local` repairs faulty components before assembly,
with bounded numerical corrections and failure rollback. See the
[component-first workflow](references/local-material-repair.md); whole-model acceptance is still required.

Use `--purpose print --print-units mm` for material-preserving 3D print preparation. Optional
`--print-size-mm 200` sets the final longest dimension; `--print-up-axis y` maps input Y-up to
printing Z. Closed holes and nested cavities are retained by default. Reviewed planar caps,
explicit `solid`/`hollow`/`thicken` modes, and Boolean part union have separate geometry gates.
See [SKILL.md](SKILL.md#print-purpose-material-repair) for parameters and limitations.

`print_report.html` embeds interactive clipping, synchronized views, original overlay, generated
surfaces, source-distance and sampled-wall fields. Opening selections export fingerprint-bound
decision JSON for a worker rerun. STL and unit-aware 3MF candidates are independently read back;
accepted file aliases stay empty on failure. `print_layers.json` contains every geometric layer.
An optional explicit PrusaSlicer-compatible FFF CLI and profile can generate/check ASCII extrusion G-code;
without them `printer_ready` remains false. Physical printing is not certified by mesh acceptance.

All commands below assume the working directory is this skill directory (`.agents/skills/mesh-watertight-repair/`).

## Skill entrypoint and worker boundary

[`SKILL.md`](SKILL.md) is the only AI orchestration entrypoint. The active Skill agent reads
deterministic reports and visual evidence, makes semantic decisions, and invokes the scripts in this
directory as geometry workers. No worker launches an AI model, accepts an `--agent-mode`, or overrides
a failed geometry gate.

The internal geometry worker for one repair round is:

```bash
python scripts/two_stage_watertight_remesh.py INPUT_MESH \
  --output-dir OUTPUT_DIR \
  --visibility-grid 720 \
  --outside-flood-grid 192 \
  --sealed-exterior-grid 192 \
  --voxel-pitch-bbox-divisor 560 \
  --sdf-band-voxels 6 \
  --max-sdf-memory-gb 4
```

This command is intentionally deterministic. It performs one geometry round and writes
`two_stage_report.json`, `ai_policy_packet.json`, visual evidence, transaction diagnostics, and the
candidate artifacts needed by the Skill agent. Direct model execution from a mesh-repair CLI is not
part of the contract.

`--group-source-gltf` is optional diagnostic metadata. It never supplies geometry and never changes
the STL/VTP triangle truth.

The TSDF sign comes from a padded six-connected far-field flood, not source winding.  Exact
point-to-triangle distance replaces the EDT magnitude in a narrow band before bounded smoothing
and zero-isosurface extraction.  `implicit_field.npz` records the field, solid mask, transform,
pitch, and SDF parameters.

Closest-point projection is a compiled PyVista/VTK batch query. Formal self-intersection acceptance
uses libigl's CGAL exact-predicate detector on every face; the worker never nudges a proxy to satisfy
an approximate scanner. Projected intersections are repaired only by restoring the implicated
projected vertices to the already certified proxy and certifying the complete candidate again.

The primary round artifacts are:

```text
OUTPUT_DIR/
  visibility_labeled_source.vtp
  stage1_exterior_candidate.vtp
  source_preserving_candidate.vtp
  implicit_field.npz
  closure_proxy.vtp
  source_projected_watertight_candidate.vtp
  ai_policy_packet.json
  two_stage_report.json
  two_stage_report.html
  run_status.json
```

`closure_proxy.vtp` is always diagnostic. Only the path named by
`outputs.accepted_mesh_vtp` represents an accepted engineering result.

## Bounded local voxel repair

`local_patch_worker.py` is a model-agnostic geometry operator for deterministic fixed-seam
replacement of one explicitly selected, source-bound region. Create the `--regions` JSON with
`local_defect_regions.extract_defect_regions()` from the same triangulated input and reviewed
defect face rows; then select one `ready` region by its `id`. The worker checks the packet's source
geometry hash before changing anything.

```bash
python scripts/local_patch_worker.py INPUT_MESH \
  --regions REGIONS.json --region REGION_ID \
  --pitch PITCH --max-error MAX_ERROR \
  --output-dir OUTPUT_DIR --timeout-seconds 20
```

`PITCH` and `MAX_ERROR` are positive values in input coordinate units. Defaults cap the run at
2,000,000 voxels, 2,000 selected faces, 100,000 patch triangles, a 20-degree seam-angle increase,
and a 20-second process-group deadline. The input context is limited to 20,000 faces. All original
points and triangles outside the selected region remain unchanged.

This operator supports one closed, consistently wound, positive-volume component and one simple
cut loop. It does not add thickness to open sheets. It checks exact-predicate intersections for
generated faces and performs VTP readback; its source-distance bounds use floating-point queries,
not an exact-arithmetic Hausdorff proof. The report writes `accepted_local_candidate_vtp` when the
local replacement passes and `accepted_mesh_vtp` only when the whole mesh also passes. Exit 0 means
whole-mesh acceptance, exit 3 means local-patch acceptance while source defects remain elsewhere,
exit 2 means rejected input or repair, and exit 124 means timeout. Local acceptance does not
establish full-model, CFD, or print readiness; `engineering_ready` remains false.

For a closed component with native triangle self-intersections, explicitly select
`--method intersection-cut` and omit `--pitch`. This runs `repair_local_intersection_cut()`:
CGAL classifies/cuts the bounded closed component, and every triangle outside the selected ROI
must survive exactly once with its original oriented coordinates. Unchanged triangles inside the
ROI are retained too. New pieces inherit their exact source-face attributes and corner normals.
The source-error, topology, seam and readback gates remain required; no thickness is introduced.
This method is selected explicitly, with `cut_intersections` or `local_intersection_cut` intent
when a decision is supplied. It does not automatically replace the default voxel method.

The voxel method can try a source-defined common contour after direct zipper rejection. This
uses shared contour nodes instead of a cross-boundary triangle strip, with a 4,096-node cap and
the same 20-degree seam constraint. Dense planar repair is covered by an end-to-end test. Sharp
folded contours can still fail; matching positions alone does not establish a valid seam.

The worker calls `repair_mesh_patch()` for attribute-preserving assembly and
`repair_local_voxel_patch()` for geometry. The latter uses `extract_defect_regions()` to bind the
cut, `build_local_sdf_patch()` for ROI-only reconstruction, and `connect_source_collar()` /
`stitch_boundary_loops()` for the fixed seam. Reconstruction reuses
`project_closure_to_source()`; acceptance reuses `geometry_gate_arrays()` and
`source_to_candidate_bound()`. The file worker reuses `roundtrip_validation()`,
`write_html_report()`, and the existing timeout supervisor's process-group stop operation.
Both geometry methods reuse `local_patch_checks` for seam, local intersection and Euler checks.

When semantic review is needed, the Skill agent may use `agent_observation.py` and
`agent_repair_contract.py` to build registered evidence, hash-bound decisions, transaction records,
and repair state. These are protocol helpers, not a second workflow entrypoint. Any requested
geometry change must be executed by a declared deterministic worker operation and re-audited.

The target is deliberately component-agnostic. The output may contain one or several connected components; component count is reported but never used as an acceptance gate.

## Target contract

```yaml
input_kind: mesh
output_kind: watertight_mesh
repair_domain: mesh
target: watertight-exterior-shell
geometry_truth: input mesh
component_count: diagnostic_only
```

Acceptance answers only whether the result is a trustworthy watertight exterior shell:

- zero boundary edges;
- zero non-manifold edges and vertices;
- zero inconsistent-winding edges and degenerate faces;
- complete all-face self-intersection audit with zero intersections;
- every naturally occurring component has a finite non-zero enclosed volume and outward orientation;
- bbox, six-axis silhouette/depth, and source-distance fidelity remain within configured limits;
- the written VTP survives readback with the same geometry and topology.

## Geometry pipeline

1. Triangulate the input and preserve `source_triangle_index`.
2. Build six-axis conservative first-hit/depth evidence.
3. Add far-field flood and sealed-exterior evidence.
4. Remove only whole contained internal or isolated flying components when the evidence agrees; never cut a low-visibility strip out of an otherwise retained component.
5. Remove only degenerate and exact-duplicate source triangles before reconstruction.
6. Build and certify the sealed-exterior TSDF closure scaffold.
7. Project safe scaffold locations back to trusted source geometry with PyVista/VTK batch locators.
8. Certify proxy and projected candidates with libigl/CGAL exact predicates; roll back only
    implicated projected vertices.
9. When visual QA finds residual scaffold grain, use topology-preserving VTK Windowed-Sinc
    smoothing and repeat complete CGAL, topology, volume, silhouette, and distance validation.
10. Serialize, read back, and re-audit every requested output format.

The source-projected candidate is eligible for acceptance when every required geometry gate passes.
The raw closure proxy remains diagnostic and can never be promoted directly.

## Boundary semantics

A free edge is not automatically a hole. Boundary inventory distinguishes:

- `small_exterior_hole`;
- `near_coincident_part_seam`;
- `large_opening_or_missing_surface`;
- `part_perimeter_or_opening_unknown`;
- `internal_or_fragment_component_perimeter`;
- `isolated_floating_fragment_perimeter`;
- `non_simple_boundary_graph`.

Large or ambiguous regions require an opening decision. AI may classify the semantic choice, but accepted geometry is always produced and validated by deterministic mesh operations.

## Output roles

```text
OUTPUT_DIR/
  visibility_labeled_source.vtp
  stage1_exterior_candidate.vtp
  source_preserving_candidate.vtp
  closure_proxy.vtp
  source_projected_watertight_candidate.vtp
  two_stage_report.json
  two_stage_report.html
  run_status.json
  debug/
  visual/
```

Only `outputs.accepted_mesh_vtp` is an accepted result. A descriptive filename is not acceptance evidence by itself.

## Full-resolution issue report

All HTML reports embed the bundled vtk.js renderer and model data for offline interactive 3D
inspection before the tables. Drag to rotate, scroll to zoom, Shift-drag to pan; each view has
mesh-edge, opacity, and reset controls. `VTK_JS_PATH` optionally overrides the bundled renderer
with a local file. Reports never download rendering code at generation or viewing time. Workflow
reports can select labeled stages with `outputs.mesh_previews: [{label, path}, ...]`.

```bash
python scripts/watertight_issue_markers.py \
  INPUT_MESH OUTPUT_DIR/source_projected_watertight_candidate.vtp \
  --output-dir OUTPUT_DIR/watertight-issues

python scripts/watertight_issue_html.py \
  OUTPUT_DIR/watertight-issues/watertight_issue_report.json \
  --output OUTPUT_DIR/watertight-issues/watertight_issue_report.html \
  --two-stage-report OUTPUT_DIR/two_stage_report.json
```

`*_surface_with_issue_arrays.vtp` contains the complete normalized triangle surface with issue arrays. `*_watertight_issue_faces.vtp` is intentionally an issue-adjacent subset, not a decimated model. `--preview-size` changes only static PNG size. HTML embeds every triangle by default; a positive `CAD_SURFACE_MESHER_VIEWER_TRIANGLES` value is an explicit opt-in preview limit.

## References

- See [report_contract.md](report_contract.md) for the machine-readable report rules.
- See [references/yu7-high-detail-vehicle-success.md](references/yu7-high-detail-vehicle-success.md)
  for the closure-first high-detail vehicle playbook, measured tradeoffs, and rejected alternatives.
