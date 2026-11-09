(function () {
  const raw = document.getElementById('mesh-preview-data');
  if (!raw || typeof vtk === 'undefined') {
    document.querySelectorAll('.vtk-viewer').forEach((node) => { node.textContent = 'vtk.js failed to load'; });
    return;
  }
  const meshes = JSON.parse(raw.textContent);
  const vtkGenericRenderWindow = vtk.Rendering.Misc.vtkGenericRenderWindow;
  const vtkActor = vtk.Rendering.Core.vtkActor;
  const vtkMapper = vtk.Rendering.Core.vtkMapper;
  const vtkPolyData = vtk.Common.DataModel.vtkPolyData;
  const vtkDataArray = vtk.Common.Core.vtkDataArray;
  const vtkColorTransferFunction = vtk.Rendering.Core.vtkColorTransferFunction;
  const views = [];
  let syncing = false;

  function color(view, values, name) {
    view.polyData.getCellData().setScalars(vtkDataArray.newInstance({ name, numberOfComponents: 1, values: Float32Array.from(values) }));
    let min = Infinity, max = -Infinity;
    values.forEach((value) => { if (value >= 0) { min = Math.min(min, value); max = Math.max(max, value); } });
    if (!Number.isFinite(min)) min = max = 0;
    if (['proxy_weight', 'watertight_issue', 'selected_opening', 'print_generated_face', 'projection_fallback_fraction'].includes(name)) { min = 0; max = 1; }
    const lookup = vtkColorTransferFunction.newInstance();
    lookup.addRGBPoint(-1, 0.7, 0.7, 0.7);
    const wall = name === 'print_thickness_mm';
    lookup.addRGBPoint(min, ...(wall ? [0.95, 0.26, 0.12] : name === 'watertight_issue' ? [0.55, 0.62, 0.7] : [0.22, 0.58, 0.82]));
    if (name === 'proxy_weight') lookup.addRGBPoint(0.5, 0.95, 0.7, 0.18);
    lookup.addRGBPoint(max > min ? max : min + 1e-6, ...(wall ? [0.22, 0.58, 0.82] : [0.95, 0.26, 0.12]));
    if (wall && view.minimumWall > min && view.minimumWall < max) lookup.addRGBPoint(view.minimumWall, 0.1, 0.7, 0.3);
    view.mapper.setLookupTable(lookup);
    view.mapper.setScalarModeToUseCellData();
    view.mapper.setScalarVisibility(true);
    view.mapper.setScalarRange(-1, max > min ? max : min + 1e-6);
    document.getElementById(`vtk-field-range-${view.index}`).textContent = `${min.toPrecision(4)} … ${max.toPrecision(4)}`;
    view.generic.getRenderWindow().render();
  }

  function defaultColor(view, mesh) {
    if (mesh.issue_values?.length === mesh.triangles) color(view, mesh.issue_values, 'watertight_issue');
    else if (mesh.proxy_weights?.length === mesh.triangles) color(view, mesh.proxy_weights, 'proxy_weight');
    else {
      view.mapper.setScalarVisibility(false);
      document.getElementById(`vtk-field-range-${view.index}`).textContent = '';
      view.generic.getRenderWindow().render();
    }
  }

  meshes.forEach((mesh, index) => {
    const container = document.getElementById(`vtk-viewer-${index}`);
    if (!container) return;
    const generic = vtkGenericRenderWindow.newInstance({ background: [0.98, 0.99, 1.0] });
    generic.setContainer(container);
    generic.resize();
    const polyData = vtkPolyData.newInstance();
    polyData.getPoints().setData(Float32Array.from(mesh.points.flat()), 3);
    polyData.getPolys().setData(Uint32Array.from(mesh.polys));
    const normals = vtk.Filters.Core.vtkPolyDataNormals.newInstance({ computePointNormals: true, computeCellNormals: false });
    normals.setInputData(polyData);
    const geometricNormals = mesh.geometry_normals ? vtkDataArray.newInstance({ name: 'GeometryNormals', numberOfComponents: 3, values: Float32Array.from(mesh.geometry_normals.flat()) }) : normals.getOutputData().getPointData().getNormals();
    const storedNormals = mesh.normals ? vtkDataArray.newInstance({ name: 'Normals', numberOfComponents: 3, values: Float32Array.from(mesh.normals.flat()) }) : null;
    polyData.getPointData().setNormals(storedNormals || geometricNormals);
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(polyData);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setInterpolationToPhong();
    actor.getProperty().setColor(0.55, 0.62, 0.7);
    const renderer = generic.getRenderer();
    renderer.addActor(actor);
    const camera = renderer.getActiveCamera();
    camera.setPosition(1, 0.7, 1);
    camera.setViewUp(0, 1, 0);
    renderer.resetCamera();
    const initialCamera = { position: [...camera.getPosition()], focalPoint: [...camera.getFocalPoint()], viewUp: [...camera.getViewUp()] };
    const view = { index, generic, polyData, mapper, actor, camera, minimumWall: mesh.minimum_wall_mm };
    views.push(view);
    defaultColor(view, mesh);
    camera.onModified(() => {
      if (syncing || !document.getElementById('vtk-sync-cameras').checked) return;
      syncing = true;
      try {
        views.forEach((peer) => {
          if (peer === view) return;
          peer.camera.set({ position: [...camera.getPosition()], focalPoint: [...camera.getFocalPoint()], viewUp: [...camera.getViewUp()], parallelScale: camera.getParallelScale() });
          peer.generic.getRenderer().resetCameraClippingRange();
          peer.generic.getRenderWindow().render();
        });
      } finally { syncing = false; }
    });
    document.getElementById(`vtk-reset-${index}`).addEventListener('click', () => {
      camera.set(initialCamera);
      renderer.resetCamera();
      generic.getRenderWindow().render();
    });
    document.getElementById(`vtk-edges-${index}`).addEventListener('change', (event) => {
      actor.getProperty().setEdgeVisibility(event.target.checked);
      actor.getProperty().setEdgeColor(0.12, 0.17, 0.22);
      generic.getRenderWindow().render();
    });
    document.getElementById(`vtk-opacity-${index}`).addEventListener('input', (event) => {
      actor.getProperty().setOpacity(Number(event.target.value));
      generic.getRenderWindow().render();
    });
    document.getElementById(`vtk-smooth-${index}`).addEventListener('change', (event) => {
      if (event.target.checked) actor.getProperty().setInterpolationToPhong();
      else actor.getProperty().setInterpolationToFlat();
      generic.getRenderWindow().render();
    });
    if (storedNormals) {
      document.getElementById(`vtk-source-normals-${index}`).addEventListener('change', (event) => {
        polyData.getPointData().setNormals(event.target.checked ? storedNormals : geometricNormals);
        polyData.modified();
        generic.getRenderWindow().render();
      });
    }
    document.getElementById(`vtk-field-${index}`).addEventListener('change', (event) => {
      if (event.target.value) color(view, mesh.fields[event.target.value], event.target.value);
      else defaultColor(view, mesh);
    });
    let ghostMapper;
    const originalView = views.find((item) => item.index === 0);
    if (index && originalView) {
      ghostMapper = vtkMapper.newInstance();
      ghostMapper.setInputData(originalView.polyData);
      ghostMapper.setScalarVisibility(false);
      const ghost = vtkActor.newInstance();
      ghost.setMapper(ghostMapper);
      ghost.getProperty().setColor(0.1, 0.15, 0.2);
      ghost.getProperty().setOpacity(0.18);
      ghost.setVisibility(false);
      renderer.addActor(ghost);
      document.getElementById(`vtk-overlay-${index}`).addEventListener('change', (event) => {
        ghost.setVisibility(event.target.checked);
        generic.getRenderWindow().render();
      });
    } else if (index) {
      document.getElementById(`vtk-overlay-${index}`).disabled = true;
    }
    function clip() {
      const axis = document.getElementById(`vtk-cut-axis-${index}`).value;
      [mapper, ghostMapper].filter(Boolean).forEach((item) => item.removeAllClippingPlanes());
      if (axis !== '') {
        const normal = [0, 0, 0]; normal[Number(axis)] = 1;
        const bounds = polyData.getBounds();
        const origin = [0, 1, 2].map((i) => (bounds[2*i] + bounds[2*i+1]) / 2);
        const ratio = Number(document.getElementById(`vtk-cut-position-${index}`).value);
        origin[axis] = bounds[2*axis] + ratio * (bounds[2*axis+1] - bounds[2*axis]);
        const plane = vtk.Common.DataModel.vtkPlane.newInstance({ normal, origin });
        [mapper, ghostMapper].filter(Boolean).forEach((item) => item.addClippingPlane(plane));
      }
      generic.getRenderWindow().render();
    }
    document.getElementById(`vtk-cut-axis-${index}`).addEventListener('change', clip);
    document.getElementById(`vtk-cut-position-${index}`).addEventListener('input', clip);
    if (mesh.regions?.length) {
      const selection = document.getElementById(`vtk-region-${index}`);
      const action = document.getElementById(`vtk-region-action-${index}`);
      const download = document.getElementById(`vtk-policy-export-${index}`);
      const decisions = new Map();
      download.disabled = true;
      selection.addEventListener('change', () => {
        if (selection.value === '') { defaultColor(view, mesh); return; }
        const region = mesh.regions[Number(selection.value)];
        action.value = decisions.get(region.id)?.decision || 'defer';
        const values = new Float32Array(mesh.triangles);
        region.source_triangle_ids.forEach((id) => { if (id >= 0 && id < values.length) values[id] = 1; });
        color(view, values, 'selected_opening');
        const center = region.centroid;
        const bounds = region.bbox;
        if (center && bounds?.min && bounds?.max) {
          const old = camera.getPosition(), focal = camera.getFocalPoint();
          const length = Math.hypot(...old.map((v, i) => v - focal[i]));
          const distance = Math.max(Math.hypot(...bounds.max.map((v, i) => v - bounds.min[i])) * 2.5, 1e-6);
          camera.setFocalPoint(...center);
          camera.setPosition(...center.map((v, i) => v + (old[i] - focal[i]) / length * distance));
          renderer.resetCameraClippingRange();
          generic.getRenderWindow().render();
        }
      });
      action.addEventListener('change', () => {
        if (selection.value === '') return;
        const region = mesh.regions[Number(selection.value)];
        decisions.set(region.id, { ...region.decision_template, status: 'decided', decision: action.value, semantic_label: 'unknown', shape_prior: action.value === 'cap' ? 'planar' : 'none', confidence: 1, reviewer: 'human_html_review', reason: 'Explicit selection in the 3D report; geometry execution and acceptance still required.' });
        download.disabled = false;
      });
      download.addEventListener('click', () => {
        const url = URL.createObjectURL(new Blob([JSON.stringify({ decisions: [...decisions.values()] }, null, 2)], { type: 'application/json' }));
        const link = document.createElement('a'); link.href = url; link.download = 'opening-decisions.json'; link.click();
        URL.revokeObjectURL(url);
      });
    }
    new ResizeObserver(() => generic.resize()).observe(container);
  });
  document.getElementById('vtk-sync-cameras').addEventListener('change', () => {
    if (views.length) views[0].camera.modified();
  });
})();
