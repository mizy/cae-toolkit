# Watertight Exterior Shell Report Contract

The JSON report is authoritative. HTML is a human-readable rendering of the same facts.

Print purpose uses `print_report.json/html`. `outputs.accepted_mesh_vtp` and
`accepted_print_files` are populated only after opening execution, material topology/orientation,
CGAL intersections, physical-scale fidelity, sampled wall, layer contours, and both STL/3MF readbacks
pass. `printing.printer_ready` additionally requires an explicitly configured external slicer/profile
to finish and produce extrusion commands; this does not certify a physical print. Failed candidate
files remain diagnostic. Nested void shells have odd depth and negative signed volume, as required
by the material boundary; the exterior-only all-positive-component rule does not apply to this purpose.

The optional `source-material` strategy writes `source_material_report.json/html` with target
`watertight-material`. Its `accepted` decision covers material geometry and the recorded fidelity
checks, not CFD-domain connectivity or print readiness. `fidelity_reference.path` identifies the
geometry actually compared, while `original_pipeline_input` records the CLI input when applicable.
Candidate-to-reference has a continuous one-sided numerical bound; reference-to-candidate uses
finite samples unless `reverse_surface_bound` was requested and reports `within_budget`. That
optional floating-point diagnostic includes internal reference surfaces, and an exhausted budget
is `unproven`. Explicitly preserved functional openings receive sampled straight-ray checks only.

HTML embeds the offline vtk.js renderer and full-resolution 3D mesh data before diagnostic tables.
The default view supports rotation, zoom, pan, mesh edges, opacity, and camera reset. A positive
`CAD_SURFACE_MESHER_VIEWER_TRIANGLES` explicitly limits preview triangles without changing audited
artifacts. An optional `outputs.mesh_previews` list of `{label, path}` objects selects comparison
stages; it does not affect geometry acceptance. Unaccepted candidates must be labeled accordingly.

## Scope

```text
input_kind = mesh
output_kind = watertight_mesh
repair_domain = mesh
target = watertight-exterior-shell
component_count = diagnostic_only
```

The contract is geometry-driven and input-agnostic. Runtime and optional metadata are diagnostic only and have no acceptance authority.

## Standalone local patch worker

`local_patch_worker.py` reports both local and whole-mesh acceptance. A populated
`outputs.accepted_local_candidate_vtp` means the source-bound region passed its local geometry
checks and serialized readback. It preserves source coordinates and triangles outside the selected
region; file compaction may remove unused interior vertices. The local checks include an exact-predicate scan for intersections involving
generated faces; pre-existing intersections between retained source faces may remain. Source-distance
bounds use floating-point queries and are not an exact-arithmetic Hausdorff proof.

`accepted_mesh_vtp` is populated only when the complete candidate also passes the whole-mesh
geometry gate. The worker's exit codes are 0 for whole-mesh acceptance, 3 for local-patch acceptance
with the whole mesh still unaccepted, 2 for a rejected or invalid run, and 124 for timeout. Code 3
must not be reported as whole-model, CFD, or print acceptance; `engineering_ready` remains false.

`--method intersection-cut` explicitly selects native CGAL intersection cutting. The bounded
closed component supplies classification context; the reviewed ROI limits permitted changes.
`face_preservation` records exact oriented triangle survival outside the ROI and actual changed
source rows. `reconstruction.normal_source_face_rows` maps output faces to their native source
birth for attribute and corner-normal transport. The unchanged triangles inside the ROI also
retain their native geometry. This method adds no material thickness and requires the same
source-distance, topology, seam, global geometry, and readback checks.

## Required top-level sections

- `decision`
- `input`
- `input_truth`
- `output_contract`
- `target`
- `parameters`
- `stages`
- `inventory_before`
- `inventory_after`
- `deterministic_passes`
- `comparisons`
- `gates`
- `outputs`
- `ignored_outputs`
- `unhandled_items`
- `repair_report`

## Required geometry gates

Every required gate must contain `passed`, the measured value, its threshold, and a stable failure reason or reason code.

Required acceptance checks:

1. Candidate artifact exists and can be decoded.
2. Boundary edges equal zero.
3. Non-manifold edges and non-manifold vertices equal zero.
4. Inconsistent-winding edges equal zero.
5. Degenerate faces equal zero.
6. The all-face libigl/CGAL exact-predicate self-intersection certification completed and found zero pairs.
7. Every connected component has finite non-zero enclosed volume.

The volume change between the closure proxy and the source-projected candidate
is reported for diagnosis only. The proxy owns connectivity, not target shape,
so no fixed proxy-relative volume percentage is an acceptance gate.

The closure proxy volume is constructed by locally sealing the voxelized source
shell, flooding outside space from a padded far field, filling the complement,
and restoring the temporary dilation offset before marching cubes. Direct hole
filling of a still-leaky voxel shell is not an exterior-volume construction.

Acceptance also requires the projected candidate's reliable signed volume to
remain above the sealed volume's erosion core. The erosion radius is the
smallest integer voxel radius covering the configured maximum projection
distance. This absolute, resolution-derived lower bound prevents a
topologically closed thin coating from passing; it is not a fixed percentage
comparison with proxy volume.
8. Every connected component is outward oriented.
9. Bbox drift is within the configured dimensionless tolerance.
10. All six axis-aligned silhouette/depth comparisons are within tolerance.
11. Bidirectional source-distance evidence is complete and within bbox/local-edge normalized limits.
12. The serialized candidate preserves topology and geometry on readback.

Connected-component count is always reported but is never itself a pass/fail condition. A
legitimate disconnected exterior shell may contain more than one component.

## Geometry truth and provenance

The input mesh is the only source geometry truth. Optional GLTF/CAD metadata may label or diagnose source regions but cannot move, add, remove, or accept triangles by itself.

Source faces retain `source_triangle_index`. Generated faces record at least:

- `face_origin`;
- source/proxy consumption state;
- projection method or certified proxy fallback reason;
- parent source region where applicable.

The report may include file and geometry SHA-256 values to bind evidence to artifacts. Hashes are lineage metadata, not model-specific allowlists and not geometric acceptance gates.

## Skill-owned semantic extension

When semantic review is required, the Skill orchestration may add the following authoritative
records without weakening any geometry gate:

- `implicit_field.npz` uses `implicit_field/v1` and records the flood-signed TSDF, solid mask,
  transform, pitch, band width, and smoothing request;
- `observation_packet.json` binds every Agent decision to the current candidate geometry hash,
  stable region IDs, registered view IDs, deterministic gates, and a defect signature;
- `agent_decisions.jsonl` contains region-level semantic or parameter choices only; raw point,
  vertex, face, triangle, or coordinate edits are invalid;
- `transactions.jsonl` records deterministic round ownership and geometry hashes;
- `repair_state.json` records round budget, no-progress and oscillation detection, and the final stop reason.

An automatic review timeout, invalid schema, stale geometry hash, unknown region or view ID, missing
region decision, or unauthorized action is fail-closed. Skill-agent approval is an additional
semantic gate; it cannot turn a rejected deterministic candidate into an accepted mesh.

The raw TSDF zero surface remains a closure proxy. The Skill workflow may populate
`accepted_mesh_vtp` only with the final source-projected, readback-audited candidate after both
the deterministic gates and Skill-owned semantic gates pass.

## Output roles

- `closure_proxy_vtp`: diagnostic topology/shape guide.
- `source_preserving_candidate_vtp`: cleaned source fallback; may remain open.
- `source_projected_watertight_candidate_vtp`: main whole-shell candidate.
- `accepted_mesh_vtp`: populated only when every required gate passes.
- `rejected_candidate_vtp`: candidate retained for diagnosis after rejection.

`decision.final_output_path` and `outputs.accepted_mesh_vtp` must be identical for an accepted run. Both are null for a rejected run.

## Optional CFD purpose

`--purpose watertight` is the unchanged default path. `--purpose cfd` requires a positive finite
`--cfd-max-edge-length` in input units and runs source-based refinement after shell acceptance.
The `cfd_surface_refinement` section records metrics, sizing parameters, iterations, and gates.

- `watertight_mesh_vtp`: accepted shell before refinement.
- `cfd_surface_candidate_vtp`: optional postprocessed candidate.
- `accepted_mesh_vtp`: refined candidate only when both stage contracts pass.
- On CFD rejection, final/accepted paths are null. `source_fallback_vtp` and `mesh_result.path`
  identify the accepted watertight input to this stage, with role `accepted_watertight_fallback`.
- Cell attributes inherit through `watertight_parent_face_id`. `sizing_source_face_id` and
  `target_edge_length` record which original-source face supplied the local target.
- `cfd_*` point arrays describe this stage's projection; old input vertices remain fixed.

Refinement additionally requires preserved components and Euler characteristic, fixed old vertices,
bounded movement, every local edge target, maximum aspect ratio 40, the face budget, and exact
readback certification. CFD `engineering_ready` remains false pending downstream validation.

## Full-resolution issue evidence

The original and processed annotated VTPs retain every normalized input triangle. The water-tightness comparison HTML also embeds full-resolution geometry. The report records:

```text
triangles == original_triangles
full_resolution == true
downsampled == false
```

Issue-face VTPs are subsets by definition and must be labeled as such.

## Non-geometric diagnostics

Elapsed time, optional timeouts, atomic-write state, and output-directory locks may be reported for operations. They never change whether a decoded candidate is geometrically watertight. Unknown units or coordinate semantics are reported as metadata limitations rather than replaced with guessed values.
