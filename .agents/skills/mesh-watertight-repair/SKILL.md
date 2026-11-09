---
name: mesh-watertight-repair
description: Build, audit, and validate a geometry-driven watertight exterior shell from an arbitrary mesh input. Use when Codex needs to take a non-watertight mesh (STL, OBJ, VTP, VTK, GLB, GLTF) and produce a watertight exterior shell with zero boundary edges, zero non-manifold elements, zero self-intersections, reliable enclosed volume, and outward orientation — or to generate full-resolution watertightness issue reports with HTML viewer evidence.
---

# Mesh Watertight Repair

Build a watertight exterior shell from a mesh input using Skill-orchestrated, deterministic geometry operations.

## Entrypoint and Worker Boundary

This `SKILL.md` is the only AI orchestration entrypoint. The active Skill agent inspects evidence,
makes semantic decisions, invokes deterministic geometry workers, and owns any multi-round repair
loop. Files under `scripts/` never represent or launch an Agent, and no mesh-repair CLI may claim an
`--agent-mode`.

The geometry workers may generate observation packets, registered views, transaction records, and
validation reports for the Skill agent. Those artifacts support the Skill loop; they do not create a
second Agent entrypoint and cannot override a failed geometry gate.

## Intent, execution, and acceptance

Before choosing or executing a geometry-changing repair, establish the user's intended outcome:
CFD simulation or 3D printing. Use intent already stated in the request or conversation; do not
ask again when it is clear. If the purpose is unknown, inspect and report source geometry and
defects without executing closure, and ask which outcome is intended. `watertight` is an explicit
geometry-only target for a user who requests it, not an implicit substitute for an unknown purpose.
If the user requests both CFD and printing, make separate purpose-bound decisions and outputs.

For CFD, identify the simulation domain (external flow or internal flow, including intended
inlets, outlets and passages), the surfaces that must retain their shape, source units and required
surface sizing. Do not automatically seal a functional opening or remove interior geometry that
defines the fluid domain. Use `--purpose cfd` with an explicit edge limit in input units; if a
required input is unknown, ask or record a justified provisional value as such. Review every
detected boundary against the requested fluid domain before acceptance. The projected-shell worker
can generate an unreviewed closure for diagnosis, but CFD acceptance and refinement require all
recorded opening decisions to have actually been executed; unsupported preserved boundaries remain
unaccepted. Surface acceptance is not volume mesh, boundary-layer or solver validation.

For printing, establish units or final size, printing process, intended separate or joined parts,
functional holes and cavities, and the required wall/error parameters from the request or explicit
printer settings. Use `--purpose print` and its material-preserving pipeline. Preserve functional
passages, mating surfaces and intended cavities; add walls or caps only when the intent and resulting
geometry are supported. Geometric closure is not a slicer, fit, stiffness or print-success check.

Review each region using source geometry, views, purpose and any available reference; record the
reason, confidence and uncertainty in the existing fingerprint-bound `ai_policy_packet`. Reuse a
decision only for its matching target and run/packet/item fingerprints. Do not claim an Agent
decision, an operation, or a semantic check was executed merely because a recommendation is present.
Escalate genuinely ambiguous function or missing purpose-specific parameters to the user; continue
independent read-only inspection while waiting.

Keep this architecture when adding repair methods. AI or human review chooses region-specific
intent and records evidence, reasons, confidence, and uncertainty in the fingerprint-bound policy
packet. An interpretation is not a proof of the object's purpose. Preserve unresolved openings
for review; the Python API and main CLI both default to `review`. Explicit `wall` or `cap` settings
are caller-supplied global geometry policies and must not be presented as AI review.

Deterministic operators execute the selected repair inside its declared region and error budget.
Retain unchanged source triangles and their normals wherever possible. Choose among existing
sewing/capping, wall construction, Boolean, and local topology operations according to the actual
defect; do not introduce model names or fixed face IDs into the algorithm.

The fixed-seam local voxel operator is implemented for one reviewed, source-bound region. It
replaces only selected source faces and keeps source coordinates, triangles, and native normals
outside that region unchanged. Unused interior vertices are removed during file compaction.
It has a hard process timeout and bounded voxel, face, seam, and patch budgets.
Its supported domain is narrower than arbitrary mesh repair: one closed, consistently wound,
positive-volume component and one simple cut loop. It does not add thickness to open sheets. The
operator checks seam continuity, bounded source distance, topology, generated-face intersections,
and serialized readback. Distance bounds are floating-point checks, not an exact-arithmetic
Hausdorff proof. See [Bounded local voxel repair](#bounded-local-voxel-repair) for its CLI and
acceptance levels.

Report geometry, appearance, function, and runtime separately. Water-tightness alone does not
establish that a cup opening, vent, or internal cavity has the right purpose. Sampled straight-ray
passage checks are not a proof of full fluid connectivity. Keep SL03 unaccepted until the complete
model, opening decisions, and all applicable gates have been verified; synthetic test counts and
small-component successes do not replace that evidence.

## I/O Contract

```yaml
input_kind: mesh
output_kind: watertight_mesh
repair_domain: mesh
target: watertight-exterior-shell
geometry_truth: input mesh
component_count: diagnostic_only
```

## Quick Start

All commands below assume the working directory is this skill directory (`.agents/skills/mesh-watertight-repair/`).

### Internal deterministic geometry worker

```bash
python scripts/two_stage_watertight_remesh.py INPUT_MESH \
  --output-dir OUTPUT_DIR \
  --visibility-grid 720 \
  --outside-flood-grid 192 \
  --sealed-exterior-grid 192 \
  --voxel-pitch-bbox-divisor 560
```

The command performs one deterministic geometry round. It is called by this Skill; it is not a
separate user-facing or Agent entrypoint.

### Bounded local voxel repair

Use this deterministic worker for one explicitly reviewed defect region when the input satisfies
the supported domain above. Build the region packet from the same triangulated input with
`local_defect_regions.extract_defect_regions()` and the reviewed defect face rows. The packet's
`source_geometry_sha256` binds it to the source; select one `ready` entry by its `id`.

```bash
python scripts/local_patch_worker.py INPUT_MESH \
  --regions REGIONS.json --region REGION_ID \
  --pitch PITCH --max-error MAX_ERROR \
  --output-dir OUTPUT_DIR --timeout-seconds 20
```

`PITCH` and `MAX_ERROR` use input coordinate units and must be positive. The worker defaults to
2,000,000 voxels, 2,000 selected source faces, 100,000 patch triangles, a 20-degree maximum seam
angle increase, and a 20-second process-group deadline. The source context is limited to 20,000
faces and a cut loop to 512 edges. The output directory must not already exist.

The worker writes `local_patch_candidate.vtp`, `local_patch_report.json`, and
`local_patch_report.html`. `accepted_local_candidate_vtp` means the selected patch passed local
checks and VTP readback. `accepted_mesh_vtp` is populated only when the complete output also passes
the whole-mesh geometry gate. Exit 0 means whole-mesh acceptance; exit 3 means local-patch
acceptance while other source defects remain; exit 2 means rejection or invalid input; exit 124
means the deadline expired. Exit 3 is not whole-model, CFD, or print acceptance. The report keeps
`engineering_ready` false.

For a closed native surface with localized triangle self-intersections, the same worker accepts
`--method intersection-cut --max-error E` without `--pitch`. The default remains `voxel`.
Exact CGAL cutting uses the bounded closed component for material classification, while the
permitted modifications remain restricted to the reviewed ROI. Reject any cut that changes,
splits or removes an outside-ROI triangle. Preserve unchanged inside-ROI triangles as well, and
transport source-face attributes and native corner normals to the cut pieces. Require the same
seam, topology, bidirectional numerical distance, global geometry and readback gates. Supplied
policy intent must be `cut_intersections` or `local_intersection_cut` for this method.

### Modeled piece for a reviewed missing surface

The Skill agent may build a separate approximate piece from local geometry or a reviewed reference,
then attach it to the source. The source and piece must already be trimmed to matching simple
boundary loops in the same frame and units. This worker does not trim an intersecting solid or infer
which side is material. It currently supports one face-connected component and one boundary per
input, with one-to-one boundary vertices and the same boundary edges. It snaps only patch boundary
vertices within the explicit tolerance to source vertices; source faces and their coordinates are
retained, while unused source points are removed on import.

For an explicitly reviewed spherical cap or a single-valued quadratic surface, the optional model
worker fits source faces adjacent to the selected boundary and makes a separate piece:

```bash
python scripts/modeled_surface_patch.py INPUT.vtp --region boundary_loop_0001 \
  --model sphere --fit-rings 4 --max-fit-error E --patch-rings 12 --output-dir MODEL_DIR
python scripts/modeled_patch.py INPUT.vtp --patch MODEL_DIR/modeled_patch.vtp \
  --seam-tolerance T --max-seam-angle-deg A --output-dir ASSEMBLY_DIR
```

Select the boundary ID from `repair_inventory.build_inventory()` on the source read with
`read_surface(path, clean=False).remove_unused_points()`, after reviewing the opening. The model
worker records the source geometry fingerprint, selected fit, residual and limits. `--model
quadratic` fits a local height field and triangulates its projected simple boundary with Gmsh by
default. Gmsh preserves every boundary edge without introducing seam vertices; `--mesh-size`
sets the interior size in input units (default: median boundary edge length). The required
`--max-patch-offset` bounds distance from the boundary plane in input units. Use
`--mesh-method centroid --subdivisions N` only for the earlier simple refinement method, which
can produce poor triangle quality. Gmsh requires the repository's `gmsh` dependency; the worker
uses a private Gmsh session and rejects an already initialized Gmsh session. The triangle estimate
and 100,000-output-face limit are guards, not a hard runtime or memory bound. The agent must still
inspect triangle quality, source fit, seam and full-geometry checks.
Neither model infers absent geometry or proves fidelity: even a low fit residual can extrapolate to
the wrong shape. Inspect the separate piece and assembled model; compare to CAD, measurements, a
reviewed opposite side, or another defensible reference where available. If the supported model or
boundary cannot express the shape, retain the evidence and report it unsupported.

`modeled_patch.py` also accepts a separately constructed triangle piece. It checks the snap and seam
angle, retained source geometry, full topology, exact self-intersection, material orientation and
serialized readback. `geometry_accepted_candidate_vtp` means those geometry checks passed; it is not
final acceptance. `accepted_mesh_vtp` remains empty because shape and opening intent still require
Skill review. Inspect both `modeled_patch_report.html` and the modeled piece before final decisions.
For a part inside an assembly, pass `--context OTHER_PARTS.vtp` with the complete unchanged
surrounding geometry in the same frame and units, excluding the original part. The worker rejects
a candidate if its exact cross-context check finds any contact/intersection or is incomplete. Without
context, the report explicitly says the assembly is not checked. Contacts at intended shared seams
are conservatively treated as intersections; a zero result only covers the supplied context and
does not establish clearance, material containment, function or whole-assembly acceptance.

For a candidate produced by another worker, run the same check before considering assembly:

```bash
python scripts/assembly_intersections.py ORIGINAL_PART CANDIDATE_PART OTHER_PARTS \
  --output ASSEMBLY_CHECK.json
```

The tool uses a conservative union-of-input-bounds broad phase followed by CGAL exact predicates;
face and intersection budgets fail closed. Before/after neighbor face IDs help locate conflicts,
but the same face intersected both times does not prove the intersection is unchanged. The agent
must still audit the entire assembled mesh and its semantics.

When direct voxel zippering fails, a source-defined shared contour can be tried under the same
deadline. Its 4,096-node cap applies to shared-node fitting, while direct zippering retains the
512-node cap. The 20-degree seam constraint remains unchanged. Sharp folded contours can remain
unsupported even after positional alignment; never promote such a candidate on distance alone.

### Smooth source-projected surfaces

The default `projected-shell` route first checks the prepared source for usable closed shells.
A fully valid source bypasses voxel reconstruction. In a mixed input, individually valid closed
components are retained only after exact cross-component intersection checks; touching or
intersecting components remain in the repair set. Retained coordinates, triangles and available
native corner normals are kept. Only the remaining surface set is passed to TSDF reconstruction
and projection. The combined proxy, assembled material orientation and serialized result still
have to pass acceptance. This main route selects complete connected shells. The separate local
voxel worker supports reviewed patches within its declared domain; automatic patch selection and
integration into this route remain pending. Do not describe a whole intersecting part as safe to discard or rebuild
merely because this preservation test rejected it.

`--stitch-tolerance T` explicitly enables approximate boundary sewing in input units. It joins
only mutually matched boundary vertices supported by oppositely directed boundary-edge pairs.
It leaves closed shells without shared defective vertices unchanged, bounds the displacement,
rejects face flips/topology regressions, and checks the affected neighborhood with exact predicates.
Unmatched openings are not capped by sewing. The default `T=0` disables approximate sewing;
exact-coordinate welding remains part of mesh import. Reviewed planar caps continue to use
`opening_repair.apply_opening_decisions`; a successfully closed source then bypasses TSDF too.

The main projected-shell CLI now uses the same process-group supervisor as source-material.
`--repair-timeout-seconds` defaults to 120 seconds and stops native work on expiry. Timeout or
worker crash cannot promote a partial output. A supervisor terminated externally cannot enforce
its deadline; direct Python calls still require an external subprocess limit. Print-purpose
execution does not use this new projected-shell timeout.

For a visibly rough scaffold, use `--scaffold-smoothing-iterations 30` to apply
VTK Windowed-Sinc filtering before projection. Motion is capped at half the closure pitch;
the smoothed scaffold is globally recertified with CGAL before it becomes fallback geometry.
Use `--cross-component-jump-ratio 0.5` to allow neighbouring source panels only when the
change in projection displacement across a component boundary stays below half the original
scaffold edge length. Omitting this option retains strict source-component equality.
Use `--projection-target REVIEWED_EXTERIOR.vtp` when semantic review selects complete
native exterior parts for shape projection. The target must use the input frame and units;
keep original geometry immutable and validate the result against it. The closure still uses
the main input/extraction for topology. This prevents partial first-hit extraction or hidden
inner panels from defining the projection shape. Record the selection evidence separately.
These options apply to watertight/CFD; print mode rejects them rather than silently ignoring them.

An explicit reliable-face mask restricts the closest-point search itself; a nearer hidden
face cannot suppress a trusted exterior match. Source topology and face IDs still refer to
the complete source mesh, so visibility-mask edges do not become artificial source defects.
Distance, source reliability, normal orientation, collisions, triangle quality and exact
self-intersection checks still apply. The worker also barycentrically transfers source point normals onto projected vertices;
generated fallback patches keep geometry normals. CFD refinement keeps existing file normals
at fixed input vertices. The HTML viewer preserves saved normals and offers a switch back
to geometry-derived normals. This attribute transfer changes no positions or connectivity.
These settings improve reconstruction; they do not
establish appearance equivalence. Verify registered smooth-shaded detail views, visible-source
and candidate surface distances, and real opening apertures independently.

### Source-preserving material closure for arbitrary meshes

Use the optional `source-material` strategy when the input's native triangles should define the
surface. It accepts the same mesh formats and input-coordinate units as the default worker. The
required `--source-max-error E` is a geometric budget in those units; the worker normalizes the
scene internally and derives shell thickness, contact repair, and cleanup limits from `E`. It has
no model-name, GLTF-node, or fixed-face-ID rules. The source material closure worker preserves
closed components, builds bounded material walls for orientable open sheets, separates non-manifold
face fans, and reports an unaccepted candidate when a source region cannot meet the exact gates.
`--source-union-backend manifold` selects an experimental faster union for positive material
shells. Negative cavity shells and invalid manifold inputs are rejected. This backend does not
run global cleanup or patch recovery. It may perform one bounded pass of independent short-edge
collapses and exact-collinear retriangulations around at most 1,000 degenerate faces and 50,000
adjacent faces, keeping exact source vertices fixed. Its candidate must still pass the same geometry, sampled
source-fidelity, and serialized readback gates. The default backend remains `cgal`.
`--source-union-backend local` is an experimental component-first alternative. It audits
individual shells, retains healthy ones, and attempts local CGAL cleanup before fast positive
material assembly. For faulty generated sheet walls it first tests an eightfold smaller inward
offset, leaving the original sheet positions fixed. A final numerical-degeneracy proposal may
move only vertices incident to degenerate faces by at most `E/1024`; that proposal requires fresh
geometry and source-distance checks. This grid rounding is not CGAL iterative snap rounding.
Each repair is limited to 10,000 input faces, and a run to 64 repaired components. A failed part
stops assembly and returns an unaccepted source reference. Negative cavity shells are unsupported.
This route keeps the main CLI's subprocess timeout, full-model gates and serialized readback;
individual component success is not full-model acceptance. See
[the local repair workflow](references/local-material-repair.md) for ownership and test limits.
For an open sheet, vertex offsets are checked against every incident face normal. Conflicting
directions are corrected locally; if no common inward direction exists, incident face fans are
split without moving their source vertices. Split seams are construction intermediates, not
accepted geometry: the resulting mesh must still pass union, exact self-intersection and fidelity
checks. Per-component diagnostics report corrected and split vertex counts.

For reviewed, near-coplanar open sheets that should form one wall, the explicit
`--source-union-backend planar` projects them to a common plane, unions their 2D outlines and
extrudes the result once. It rejects non-manifold input, closed components, incompatible normals,
and projection deviation above one quarter of the constructed wall depth. Geometry is newly
triangulated and has no exact per-source-face lineage; the report records projection and wall depth
in input units. Candidate-to-source distance uses a bounded floating-point numerical subdivision;
an exhausted or unproven bound is a rejection. Source-to-candidate fidelity remains sampled.
This can reduce slivers from unioning almost coplanar 3D solids, but it does not guarantee a
particular triangle quality: inspect the result and check the surrounding assembly.

For an isolated part being repaired within an assembly, provide
`--source-assembly-context OTHER_PARTS.vtp` to the main CLI, or `--assembly-context` to the
standalone source-material worker. The context must contain all unchanged surrounding geometry,
exclude the original part, and use the same units and frame. Cross-context intersections or an
incomplete check block acceptance; without a context the report says `not_checked`. Intended shared
contact is conservatively reported as an intersection, and a context-only check is not a whole-model
topology, containment, functional or appearance certification.

```bash
python scripts/two_stage_watertight_remesh.py INPUT_MESH \
  --purpose watertight --surface-strategy source-material \
  --source-max-error E --source-boundary-mode review \
  --output-dir OUTPUT_DIR
```

`review` is the default and returns a fingerprint-bound packet plus a full-resolution 3D report
when a free boundary has unresolved semantics. Review the affected regions with AI or a human,
then rerun in a fresh output directory with that packet's per-region decisions supplied through
`--policy-decisions`. `wall` preserves reviewed rims as bounded material walls. `cap` applies the
planar cap operator to supported loops; unsupported loops remain unaccepted. Source face lineage
and native corner normals are carried through welding and CGAL union. Recovery proposals are chosen
from measured first-hit pixels and per-triangle geometric witnesses; final acceptance still requires
the exact CGAL, volume, manifold, normalized readback, and source-fidelity gates.

For an explicit `functional_opening` decision, the worker checks sampled, initially unobstructed
straight rays through the opening after construction. Unsupported boundaries or failed rays are
rejected. This is not a proof of fluid connectivity. A `wall` default alone records no functional
opening intent and does not perform this semantic check.
Oppositely oriented closed shells that intersect without a verified nesting relationship are
semantically ambiguous; the worker rejects them instead of silently reversing one shell. Source
components that fail their individual self-intersection checks are not used to filter witnesses.

This target validates a closed material mesh. It does not certify that a CFD fluid region cannot
enter a cabin or internal passage, and it does not validate printer minimum wall thickness or a
volume/flow mesh. Use the CFD or print purpose for those additional checks. These checks do not
claim a continuous Hausdorff certificate in both directions: candidate-to-reference faces have a
continuous one-sided bound, while reference-to-candidate checks use finite samples. The report names
the actual reference path separately from the original pipeline input when a projection target is
provided. Large, intersecting sheet collections can require substantial memory and computation;
the current material path has not established a scalable time or memory guarantee.
One recovery union is limited to 1,000 selected source facets by default. A larger selection
returns `recovery_facet_budget_exceeded` before building patches; callers can explicitly change
the limit with `--source-max-recovery-facets`. This bounds that operation's input, not its runtime
or memory usage, and does not bound earlier construction or exact intersection checks.
The main CLI runs `source-material` in a supervised process group with a default 120-second
wall-clock limit, configurable with `--source-timeout-seconds`. A timeout stops the worker and
records a failed run with no accepted mesh once the worker has acquired its output lease; a timeout
during startup can return code 124 before any report is created. The limit depends on the supervisor
remaining alive and does not impose a hard memory limit. The direct Python API and standalone
`source_material_repair.py` worker do not have this supervisor; callers must provide their own limit.
When the constructed candidate has exactly the same oriented triangles as the reference, fidelity
sampling and recovery are skipped; geometry, self-intersection, opening and readback gates remain.

Add `--certify-reverse-fidelity` to attempt an adaptive, bounded reverse surface check over the
entire reference, including internal surfaces. Exhausted subdivision limits yield `unproven`; a
distance failure may refer to intentionally removed interior material. This optional diagnostic
does not change the material acceptance decision. Its distance calculations use floating-point
queries and are not an exact-arithmetic proof.
The report includes both raw sample distances and distances after interior-witness filtering, in
the original coordinate units as well as the normalized working frame.

The main CLI exposes this same geometry-only strategy with `--surface-strategy source-material`,
`--source-max-error` in input units, and `--source-boundary-mode review|wall|cap`. It is available
with `--purpose watertight`; `projected-shell` remains the default strategy.

A watertight surface can enclose a solid with real through-passages. Do not classify every
part perimeter as a hole to cap. Review native source geometry and connected surrounding
surfaces before recording a region decision. A region recommendation is not an executed
repair; report the actual cap/keep transaction and aperture verification separately.

### Optional CFD surface postprocessing

The default `--purpose watertight` finishes at the accepted watertight shell without refinement.
For CFD, explicitly provide a physical edge limit in input coordinate units:

```bash
python scripts/two_stage_watertight_remesh.py INPUT_MESH \
  --output-dir OUTPUT_DIR --purpose cfd --cfd-max-edge-length 0.02
```

This runs `cfd_surface_refine.py` only after shell acceptance. Original source surfaces supply
crease/boundary sizing evidence; the accepted shell supplies closed connectivity. Conforming
shared-edge subdivision retains all old vertices, and only new vertices may project to the source
within a bounded displacement. Source creases above 30 degrees and boundary faces receive half
the requested edge length. All lengths use input units, not assumed mm.

Recheck topology, exact intersections, component count, Euler characteristic, fixed old vertices,
local edge targets, aspect ratio, and serialized readback. Maximum aspect ratio 40 is this tool's
surface threshold, not a universal CFD solver requirement. Unmet targets at the face budget fail.
Refinement does not fix inherited slivers, recover lost features, resolve functional openings, or
verify a volume/boundary-layer mesh.

CFD `outputs.watertight_mesh_vtp` retains the accepted unrefined shell. Only refinement passing
every surface gate becomes `accepted_mesh_vtp`; rejected refinement stays diagnostic. CFD
`mesh_result.engineering_ready` remains false pending downstream acceptance. Existing accepted
shells can be postprocessed without rerunning closure:

```bash
python scripts/cfd_surface_refine.py WATERTIGHT.vtp --source-mesh SOURCE_EXTERIOR.vtp \
  --output-dir CFD_OUTPUT_DIR --max-edge-length 0.02 --max-output-faces 2000000
```

### Print-purpose material repair

Use `--purpose print` for printable material geometry. This branches directly to
`print_mesh.run_print_pipeline` before exterior visibility or internal-component pruning.
Closed native surfaces and functional through-holes are retained. Unknown boundaries enter the
fingerprint-bound policy packet; a free boundary preserved without wall construction blocks acceptance.

```bash
python scripts/two_stage_watertight_remesh.py INPUT \
  --purpose print --print-units mm --print-size-mm 200 \
  --print-min-wall-mm 0.8 --print-max-error-mm 0.1 \
  --output-dir OUTPUT_DIR
```

Input units are required unless an explicit final size is supplied, or the source is glTF (metres).
`--print-up-axis x|y|z` rotates the declared input up axis to printing Z. All print lengths are mm.
Configure wall and layer thresholds for the actual printer/material; defaults are not universal.

- `--print-mode preserve` retains material cavities and parts; nested cavity shells point into voids.
- `solid` explicitly fills enclosed cavities using material Booleans, after boundary repair.
- `hollow --print-wall-mm T` creates a sampled SDF material wall; SLA rejects sealed cavities until
  drainage is separately established, or use solid mode.
- `thicken --print-wall-mm T` adds a wall of T around open sheets, or expands closed material by T/2.
  Source error, protected closed-source Euler characteristic, and shell count are rechecked.
- `--print-parts merge` Boolean-unions contacting/overlapping solids; it never invents a bridge
  between separated parts. Default `preserve` rejects intersections and retains separate parts.
- `--print-openings cap` explicitly caps all simple sufficiently planar free loops. Default `review`
  requires item decisions. Thicken mode defaults to preserving free rims in the constructed wall.
  Unsupported priors and non-simple/nonplanar boundaries remain unaccepted.

Review `print_report.html`, select and inspect opening regions, and download a decisions JSON.
Rerun with `--policy-decisions FILE` in a fresh output directory. Stale supplied decisions cannot
inherit a global cap default. Legacy shell/CFD runs now execute supplied caps and reject unsupported
preserved free boundaries before global TSDF closure; supplied choices are never silently ignored.

The report provides clipping, synchronized cameras, source overlay, generated-face/source-distance/
sampled-wall heatmaps and labeled STL/3MF downloads. Audit both file readbacks with exact intersections,
material orientation, topology and sampled geometry error. `print_layers.json` records material contours
at every layer midpoint and support/island diagnostics. This is geometric slicing, not machine G-code.

Optionally pass `--print-slicer /path/to/PrusaSlicer --print-slicer-profile printer.ini` for explicit
FFF profile-based CLI slicing. The interface requests ASCII G-code (`--no-binary-gcode`) to inspect
extrusion commands; use a CLI supporting that flag. SLA exports remain geometry preparation for a
separate resin slicer. Without external slicing `printer_ready` and `engineering_ready` remain false even when
print geometry is accepted. Wall thickness uses sampled inward normal chords, and fidelity uses seven
samples per triangle; neither is a continuous minimum/Hausdorff certificate. Wall construction fails
at `--print-max-voxels` instead of silently coarsening the requested physical accuracy.

### Skill-owned repair loop

The active Skill agent performs this loop. A worker executes one operation and returns evidence;
it does not make the engineering decision. Start with the original input, purpose, units, allowable
geometry error, functional openings and available CAD or reference geometry. Unknown units or
opening intent must remain explicit; do not infer a physical tolerance or silently seal an opening.

1. **Observe.** Audit the actual source geometry and inspect registered views of the suspect region
   and surrounding parts. Record boundary, manifold, orientation, self-intersection and volume
   evidence, as applicable. Keep input geometry immutable. The source fingerprint identifies
   the exact mesh to which a region and decision apply.
2. **Decide.** Choose one bounded defect region and document its evidence, intended material or
   opening, editable area, fixed interface, method, error budget and uncertainty. Prefer the
   smallest supported source-preserving operation: matched seam sewing for a verified seam;
   reviewed planar cap for a supported loop; local `intersection-cut` for intersections in a
   supported closed component; material walls or Boolean assembly only when the intended material
   is established. Use the corresponding CLI/API documented above. A visual gap alone is not proof
   that two boundaries should be joined. Local voxel repair is a candidate only in its stated domain.
3. **Execute.** Run one worker in a fresh directory, retaining the input, command, source-bound
   policy or region packet, output and elapsed time. For local intersections, call
   `source_projected_validation.exact_self_intersection_report` and pass the complete reported
   face pairs to `local_defect_regions.extract_defect_regions`; start with a small halo and select
   one `ready` region. If pairs are truncated or detection fails, do not treat the packet as complete.
   Run `local_patch_worker.py --method intersection-cut` with that packet and region ID. For sewing
   use `--stitch-tolerance`; for a supported reviewed cap use `--policy-decisions` and
   `opening_repair.apply_opening_decisions`. A Python call to an internal operator is a proposal,
   not a fully validated output.
4. **Inspect and choose.** Read the actual worker report, exit code, output and local before/after
   views. Verify the declared fixed region, topology, exact intersections, distance/error budget,
   normals, serialization and opening semantics. A local exit 3 is only a local candidate: further
   source defects remain. A candidate or diagnostic proxy is not accepted merely because it exists.
   The global projected-shell worker can produce a closed candidate without reviewed opening
   decisions; treat unresolved opening semantics as an independent acceptance blocker.
5. **Continue or stop.** If the candidate passes its applicable gates, continue from that exact
   geometry with a newly generated fingerprint and fresh defect evidence. If it fails, preserve the
   previous input; record the failed gate and change the region, method or budget only when evidence
   supports the change. Stop and report unresolved work for missing intent, unsupported geometry,
   exhausted time/round budget, repeated no-progress or oscillation. Do not claim whole-model,
   CFD or print acceptance from a component or local pass.

For large or nonplanar missing surfaces, do not equate an automatically generated planar cap with
the original shape. `cap_loop` only supports simple sufficiently planar loops; local voxel repair
requires a closed source component. When those operators cannot express the repair, examine source
CAD, a known opposing surface, or measured geometry and propose an explicit bounded model and its
intersection with the fixed interface. Record the supporting evidence and compare shape as well as
topology; a plausible-looking or watertight surface alone cannot prove recovery. Existing policy
shape-prior names are not implementations of mirroring or surface fitting. Without sufficient
evidence, leave the shape unresolved rather than claiming a faithful reconstruction.

For a repeated feature such as wheel spokes, the agent may use another intact feature as a modeled
piece after establishing the axis, transform and correspondence from the actual source. Verify the
full ordered boundary and adjacent face connectivity, record the measured alignment error and snap
only the piece within a declared budget. Cut out an independent piece without modifying the donor;
then run `modeled_patch.py` and compare seam angles, visual detail and retained-source geometry.
Repetition in one model is not a general symmetry guarantee; the current worker does not discover
or extract such a piece automatically. Thin sheets may instead be thickened when their intended
material and error budget are understood. Interior removal requires exterior evidence and must not
silently remove visible details or functional passages.

For every round record the input fingerprint and scope, evidence paths, selected region, intended
operation and reason, parameters and units, command, output/report paths, failed and passed checks,
decision, and next action. Keep a single final workflow report with the full sequence and limitations.
`agent_observation.py` and `agent_repair_contract.py` are protocol helpers for the existing global
projected-shell audit; their global actions and acceptance rules do not implement this local loop.

## Route Selection

Use the closure-first route for arbitrary dirty meshes. Never use global "delete every intersecting
face, then fill" as the default route; it destroys source evidence, couples unrelated defects, and
makes the replacement surface ambiguous. Enforce this ownership boundary:

1. Retain certified nonintersecting source shells. Let the sealed-exterior TSDF proxy own
   reconstructed topology and inside/outside classification for the remaining repair set.
2. Let trusted source triangles own target shape through gated closest-point projection.
3. Let all-face libigl/CGAL exact predicates own self-intersection acceptance.
4. Roll back implicated projected vertices to the certified proxy; never repair an exact-predicate
   failure by heuristic nudging or unconditional face deletion.
5. Apply local topology-preserving smoothing only to reconstructed scaffold regions, then
   repeat exact certification and every fidelity gate.

Read [references/yu7-high-detail-vehicle-success.md](references/yu7-high-detail-vehicle-success.md)
when repairing a high-detail vehicle or another large globally defective mesh, tuning TSDF
resolution, diagnosing visible voxel grain, or choosing a scalable self-intersection strategy.

## Performance Guardrails

- Use compiled PyVista/VTK batch closest-cell queries; never loop over hundreds of thousands of
  vertices with one locator call per vertex in Python.
- Use libigl/CGAL exact-predicate detection for formal all-face certification. Keep approximate or
  tolerance-based scanners diagnostic-only.
- Escalate TSDF resolution from a cheap diagnostic proxy to the lowest resolution that passes
  silhouette, source-distance, and smooth-shaded visual QA; do not promote a visibly faceted proxy.
- Run cheap decode, topology, degeneracy, and volume gates before expensive full-candidate checks,
  but repeat exact self-intersection certification after every geometry mutation eligible for
  acceptance.

### Full-resolution watertightness issue report

```bash
python scripts/watertight_issue_markers.py \
  INPUT_MESH OUTPUT_DIR/source_projected_watertight_candidate.vtp \
  --output-dir OUTPUT_DIR/watertight-issues

python scripts/watertight_issue_html.py \
  OUTPUT_DIR/watertight-issues/watertight_issue_report.json \
  --output OUTPUT_DIR/watertight-issues/watertight_issue_report.html \
  --two-stage-report OUTPUT_DIR/two_stage_report.json
```

## Pipeline

1. Triangulate input and preserve `source_triangle_index`.
2. Build six-axis conservative first-hit/depth evidence.
3. Add far-field flood and sealed-exterior evidence.
4. Remove wholly contained internal or isolated flying components when evidence agrees.
5. Remove only degenerate and exact-duplicate source triangles before reconstruction.
6. Build the sealed-exterior TSDF and extract one certified closure scaffold.
7. Project safe scaffold locations to trusted source geometry with PyVista/VTK batch locators.
8. Certify the proxy and every projected candidate with libigl/CGAL exact predicates; roll back
    only projected vertices touched by reported intersecting face pairs.
9. Use topology-preserving VTK Windowed-Sinc smoothing only when visual QA finds scaffold grain,
    then repeat CGAL certification and topology/fidelity gates.
10. Serialize, read back, and re-audit every requested output format.

## Required Geometry Gates

Every required gate must pass for acceptance:

1. Candidate artifact exists and decodes.
2. Boundary edges == 0.
3. Non-manifold edges == 0 and non-manifold vertices == 0.
4. Inconsistent-winding edges == 0.
5. Degenerate faces == 0.
6. All-face libigl/CGAL exact-predicate self-intersection certification: zero pairs.
7. Every component has finite non-zero enclosed volume.
8. Every component is outward oriented.
9. Bbox drift within tolerance.
10. All six axis-aligned silhouette/depth comparisons within tolerance.
11. Bidirectional source-distance evidence complete and within limits.
12. Serialized candidate survives readback with identical topology and geometry.

## Output Roles

Generate reports through `scripts/html_report.py:write_html_report`. HTML must include an
interactive 3D comparison before diagnostic tables, with full-resolution mesh data and the
bundled offline vtk.js renderer embedded in the file. Rotation, zoom, pan, mesh-edge display,
opacity, and camera reset are available without a CDN. `VTK_JS_PATH` is an optional local override.
For a curated workflow comparison, populate `outputs.mesh_previews` with `{label, path}` entries
for each reviewed stage; labels must preserve rejected/diagnostic status. Otherwise the report
selects geometry from existing output roles. Static screenshots supplement the 3D evidence.

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

Only `outputs.accepted_mesh_vtp` is accepted for the requested purpose. In CFD mode this is the
certified refined surface; `watertight_mesh_vtp` separately retains the accepted pre-refinement shell.

## Boundary Semantics

- `small_exterior_hole`
- `near_coincident_part_seam`
- `large_opening_or_missing_surface`
- `part_perimeter_or_opening_unknown`
- `internal_or_fragment_component_perimeter`
- `isolated_floating_fragment_perimeter`
- `non_simple_boundary_graph`

## Core Operators

- NumPy/SciPy/scikit-image sealed-exterior TSDF and marching cubes
- PyVista/VTK batch closest-cell projection and Windowed-Sinc smoothing
- libigl/CGAL exact all-face self-intersection certification
- deterministic projection rollback to certified proxy positions
- topology, volume, silhouette, source-distance, and serialized-readback gates

## Documentation

- [README.md](README.md) — detailed pipeline, semantics, and usage
- [report_contract.md](report_contract.md) — machine-readable report contract
