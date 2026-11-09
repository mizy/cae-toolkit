# Component-first material repair

The intended engineering workflow is to inspect the source, retain sound surfaces, repair each
part's defects, resolve part interactions, then certify the assembled result. AI or human review
owns opening and cavity intent through the existing fingerprint-bound policy. The worker never
chooses intent from a model name, component number, or fixed face list.

The `local` source-material backend is experimental. It is not an established optimum or a
guarantee for arbitrary meshes. `cgal` remains the default.

## Operations and constraints

1. Reuse deterministic cleanup and policy handling in `repair_source_mesh`.
2. Reuse `source_material_closure._build_components` to construct bounded material shells.
3. Retain individually valid components without modifying their geometry.
4. For a failed generated sheet shell, test a reduced inner-wall offset before local Boolean
   cleanup. Only the generated inner positions change at this construction step. Existing closed
   source components do not receive this wall adjustment.
5. If cleanup leaves numerically degenerate triangles, propose rounding just their incident
   vertices, with maximum displacement `E/1024`. Remove only resulting exactly zero-area faces,
   run the local Boolean again, and repeat the geometry gates. This is a bounded numerical repair
   proposal, not the CGAL iterative snap-rounding algorithm or a standalone topology guarantee.
6. Require local manifoldness, no degeneracy, reliable volume, exact self-intersection absence,
   material orientation, a continuous candidate-to-source bound, and seven source samples per
   triangle. Reverse sampling is not a continuous Hausdorff certificate. Failed attempts return
   the original component unchanged; no rejected component is sent to assembly.
7. Assemble accepted positive component proposals with the existing Manifold backend, then run
   the whole-model geometry, source-fidelity, opening-ray and serialization gates. Generated
   contact faces retain lineage `-1`; they must not inherit an unrelated source face or normal.

The wall depth is computational construction geometry, not a printer minimum wall thickness.
Use the print-purpose checks for printer requirements. A local Boolean can remove real source
details; the source-distance gate must reject that result even when geometry is valid.

## Runtime

Local Boolean inputs are limited to 10,000 triangles per component and at most 64 components may
be repaired. Native operations still run under the existing main-CLI subprocess supervisor:

```bash
python scripts/two_stage_watertight_remesh.py INPUT \
  --surface-strategy source-material --source-union-backend local \
  --source-max-error E --source-timeout-seconds 60 --output-dir OUTPUT
```

Use explicit reviewed policy decisions for boundaries. The default `review` mode does not repair
unresolved openings. `--source-boundary-mode wall` is an explicit global diagnostic policy,
not evidence of AI review. Direct Python API calls still require an external subprocess deadline.

## Ownership and validation

- `repair_source_mesh` owns policy, route selection and final acceptance.
- `source_material_local.build_local_material` owns component-before-assembly ordering and rollback.
- `material_cleanup` contains the reused Boolean cleanup and geometry gates, called directly by
  both the original workflow and the new component workflow.
- `mesh_topology_repair.round_degenerate_vertices` owns the bounded rounding proposal. Its caller
  in `repair_material_component` must perform the subsequent Boolean and acceptance checks.

Synthetic regressions cover healthy-component identity, real overlap repair, shape-loss rejection,
rollback, owned array lifetimes, bounded rounding, generated face lineage, protected torus topology,
and default opening review. Real vehicle results are generated under `outputs/`; those proprietary
meshes are not fixtures. Local vehicle component passes never certify the complete original GLB,
automatic interior removal, or whole-vehicle performance.

References: [SpaceClaim repair order](https://help.spaceclaim.com/v19.0/en/Content/Repair_Overview.htm)
and [CGAL numerical self-intersection handling](https://cgal.github.io/cgal-web/2025/06/13/autorefine-and-snap/).
