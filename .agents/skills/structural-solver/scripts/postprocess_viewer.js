(() => {
  const byId = (id) => document.getElementById(id);
  const stage = byId("stage");
  const probe = byId("probe");
  try {
    const model = JSON.parse(byId("result-data").textContent);
    const decode = (value, Type) => {
      const bytes = atob(value);
      const buffer = new Uint8Array(bytes.length);
      for (let i = 0; i < bytes.length; ++i) buffer[i] = bytes.charCodeAt(i);
      return new Type(buffer.buffer);
    };
    const { vtkActor, vtkMapper, vtkColorTransferFunction, vtkCellPicker, vtkAxesActor } = vtk.Rendering.Core;
    const { vtkPolyData, vtkPlane } = vtk.Common.DataModel;
    const { vtkSphereSource, vtkLineSource } = vtk.Filters.Sources;
    const win = vtk.Rendering.Misc.vtkGenericRenderWindow.newInstance({ background: [.044, .078, .125] });
    win.setContainer(stage);
    win.resize();
    const renderer = win.getRenderer();
    const render = () => win.getRenderWindow().render();

    const poly = vtkPolyData.newInstance();
    const mapper = vtkMapper.newInstance();
    const actor = vtkActor.newInstance();
    mapper.setInputData(poly);
    actor.setMapper(mapper);
    actor.getProperty().setAmbient(.42);
    actor.getProperty().setDiffuse(.58);
    actor.getProperty().setEdgeColor(.31, .41, .52);
    renderer.addActor(actor);

    const ghostPoly = vtkPolyData.newInstance();
    const ghostMapper = vtkMapper.newInstance();
    const ghost = vtkActor.newInstance();
    ghostMapper.setInputData(ghostPoly);
    ghostMapper.setScalarVisibility(false);
    ghost.setMapper(ghostMapper);
    ghost.getProperty().setRepresentationToWireframe();
    ghost.getProperty().setOpacity(.18);
    ghost.getProperty().setColor(.68, .85, 1);
    ghost.setPickable(false);
    ghost.setVisibility(false);
    renderer.addActor(ghost);

    const markerSource = vtkSphereSource.newInstance({ thetaResolution: 18, phiResolution: 12 });
    const distanceSource = vtkLineSource.newInstance();
    function addMarker(source, color) {
      const mapper = vtkMapper.newInstance();
      mapper.setInputConnection(source.getOutputPort());
      const item = vtkActor.newInstance();
      item.setMapper(mapper);
      item.getProperty().setColor(...color);
      item.setPickable(false);
      item.setVisibility(false);
      renderer.addActor(item);
      return item;
    }
    const marker = addMarker(markerSource, [.32, .97, 1]);
    const line = addMarker(distanceSource, [.32, .97, 1]);
    line.getProperty().setLineWidth(3);

    const Widget = vtk.Interaction?.Widgets?.vtkOrientationMarkerWidget;
    if (Widget && vtkAxesActor) {
      const widget = Widget.newInstance({ actor: vtkAxesActor.newInstance(), interactor: win.getInteractor() });
      widget.setViewportCorner(Widget.Corners.BOTTOM_LEFT);
      widget.setViewportSize(.12);
      widget.setEnabled(true);
    }

    const contours = vtkPolyData.newInstance();
    const contourMapper = vtkMapper.newInstance();
    const contourActor = vtkActor.newInstance();
    contourMapper.setInputData(contours);
    contourMapper.setScalarVisibility(false);
    contourActor.setMapper(contourMapper);
    contourActor.setPickable(false);
    contourActor.setVisibility(false);
    contourActor.getProperty().setColor(.26, .98, .88);
    contourActor.getProperty().setLineWidth(2);
    renderer.addActor(contourActor);

    const picker = vtkCellPicker.newInstance({ tolerance: 0.0005 });
    picker.setPickFromList(true);
    picker.addPickList(actor);
    const camera = renderer.getActiveCamera();
    const fields = byId("field");
    const scale = byId("scale");
    const range = byId("range");
    const rangeMin = byId("range-min");
    const rangeMax = byId("range-max");
    const clip = byId("clip");
    const clipPosition = byId("clip-position");
    const sectionSelect = byId("sections");
    const measure = byId("measure");
    const animate = byId("animate");
    const clipPlane = vtkPlane.newInstance();
    const fmt = (x) => Number(x.toPrecision(5)).toString();
    let active, pts, displacement, polys, displayed, field, colorValues;
    let hit = -1, vertex = -1, measuring = false, measurements = [];
    let visibleCells = null, contourEdges = [];
    const threshold = byId("threshold");
    const thresholdValue = byId("threshold-value");
    const contour = byId("contour");
    const contourValue = byId("contour-value");
    let reversed = false, animation = 0, started = 0;
    const sections = Object.entries(model.sections).flatMap(([axis, cuts]) =>
      cuts.map((cut) => ({ ...cut, caption: `${axis.toUpperCase()} = ${fmt(cut.position)} ${model.unit}` })));
    if (sections.length) {
      byId("sections-control").hidden = false;
      sectionSelect.add(new Option("外表面", ""));
      sections.forEach(({caption}, i) => sectionSelect.add(new Option(caption, i)));
    }

    function updateProbe() {
      if (hit < 0) {
        probe.textContent = measuring ? "依次点选两个节点，按原形坐标测距。" : "点击模型读取场值。";
        return;
      }
      const x = pts.subarray(3*vertex, 3*vertex+3);
      const value = colorValues[field.association === "cell" ? hit : vertex];
      const loc = field.association === "cell" ? "单元值" : "最近节点";
      const a = measurements[0], b = measurements[1];
      const distance = a && b ? ` · 原形距离 ${fmt(Math.hypot(...a.raw.map((v,i)=>v-b.raw[i])))} ${model.unit}` : "";
      probe.textContent = `${loc} · ${field.label} ${fmt(value)} ${field.unit} · 原形坐标 (${Array.from(x, fmt).join(", ")}) ${model.unit}${distance}`;
    }

    function recolor() {
      field = active.fields[Number(fields.value)];
      colorValues = field.decoded ||= decode(field.values, Float32Array);
      filterSurface();
      range.options[0].disabled = field.signed;
      if (field.signed && range.value === "p99") range.value = "full";
      byId("range-custom").hidden = range.value !== "custom";
      const cap = field.signed ? Math.max(Math.abs(field.min), Math.abs(field.max))
        : range.value === "full" || field.p99 <= 0 ? field.max : field.p99;
      let hi = cap || 1e-12;
      let lo = field.signed ? -hi : 0;
      if (range.value === "custom") {
        if (rangeMin.value === "" || rangeMax.value === "") {
          rangeMin.value = fmt(lo);
          rangeMax.value = fmt(hi);
        }
        if (rangeMin.valueAsNumber < rangeMax.valueAsNumber) {
          lo = rangeMin.valueAsNumber;
          hi = rangeMax.valueAsNumber;
          rangeMax.removeAttribute("aria-invalid");
        } else {
          rangeMax.setAttribute("aria-invalid", "true");
        }
      }
      const colors = field.signed ? ["#244ea0","#69b2e0","#f5f5f0","#f17a50","#9d2527"]
        : ["#23104a","#4b3e9b","#bd4280","#f67b51","#ffebb4"];
      const lookup = vtkColorTransferFunction.newInstance();
      colors.forEach((hex, i) => {
        const color = Number.parseInt(hex.slice(1), 16);
        lookup.addRGBPoint(lo + (hi-lo)*i/(colors.length-1),
          (color >> 16)/255, ((color >> 8)&255)/255, (color&255)/255);
      });
      mapper.setLookupTable(lookup);
      mapper.setScalarRange(lo, hi);
      mapper.setScalarVisibility(true);
      byId("ramp").style.background = `linear-gradient(90deg,${colors.join(",")})`;
      byId("legend-title").textContent = field.unit ? `${field.label} (${field.unit})` : field.label;
      byId("range-note").textContent = `${fmt(field.min)} – ${fmt(field.max)} ${field.unit}${range.value === "p99" && cap < field.max ? " · 色标截至 P99" : ""}`;
      byId("ticks").replaceChildren(...[lo, (lo+hi)/2, hi].map((x) => {
        const tick = document.createElement("span");
        tick.textContent = fmt(x);
        return tick;
      }));
      updateProbe();
      render();
    }

    function filterSurface() {
      thresholdValue.disabled = !threshold.checked;
      contour.disabled = field.association !== "point";
      contourValue.disabled = contour.disabled || !contour.checked;
      if (thresholdValue.value === "") thresholdValue.value = fmt(field.p99);
      if (contourValue.value === "") contourValue.value = fmt((field.min + field.max)/2);
      const limit = thresholdValue.valueAsNumber;
      const filtering = threshold.checked && Number.isFinite(limit);
      visibleCells = filtering ? [] : null;
      if (filtering) {
        const shown = [];
        for (let i=0; i<polys.length; i+=4) {
          const face = i/4;
          const v = field.association === "cell" ? colorValues[face]
            : (colorValues[polys[i+1]]+colorValues[polys[i+2]]+colorValues[polys[i+3]])/3;
          if (v >= limit) {
            shown.push(3,polys[i+1],polys[i+2],polys[i+3]);
            visibleCells.push(face);
          }
        }
        poly.getPolys().setData(Uint32Array.from(shown));
      } else poly.getPolys().setData(polys);
      const cellField = field.association === "cell";
      const vals = cellField && visibleCells ? Float32Array.from(visibleCells, i => colorValues[i]) : colorValues;
      const scalars = vtk.Common.Core.vtkDataArray.newInstance({ name: field.label, numberOfComponents: 1, values: vals });
      if (cellField) {
        poly.getCellData().setScalars(scalars);
        mapper.setScalarModeToUseCellData();
      } else {
        poly.getPointData().setScalars(scalars);
        mapper.setScalarModeToUsePointData();
      }
      poly.modified();

      contourEdges = [];
      const level = contourValue.valueAsNumber;
      if (!contourValue.disabled && Number.isFinite(level)) {
        const shown = poly.getPolys().getData();
        for (let i=0; i<shown.length; i+=4) {
          const edges = [];
          for (let j=0; j<3; ++j) {
            const a = shown[i+j+1], b = shown[i+(j+1)%3+1];
            if ((colorValues[a] < level) !== (colorValues[b] < level)) {
              edges.push([a,b,(level-colorValues[a])/(colorValues[b]-colorValues[a])]);
            }
          }
          if (edges.length === 2) contourEdges.push(...edges);
        }
      }
      const indices = new Uint32Array(3*contourEdges.length/2);
      for (let i=0; i<indices.length; i+=3) indices.set([2,2*i/3,2*i/3+1],i);
      contours.getLines().setData(indices);
      updateContourGeometry();
      const note = [];
      if (visibleCells) note.push(`${visibleCells.length.toLocaleString()} / ${(polys.length/4).toLocaleString()} 面${cellField ? "" : " · 按顶点均值筛选"}`);
      if (contour.disabled) note.push("单元场可做阈值筛选，等值线需连续点场。");
      else if (contour.checked) note.push(`等值线 ${(contourEdges.length/2).toLocaleString()} 段`);
      byId("filter-note").textContent = note.join(" · ");
    }

    function updateContourGeometry() {
      contourActor.setVisibility(contourEdges.length > 0);
      if (!contourEdges.length) return;
      const points = new Float32Array(3*contourEdges.length);
      contourEdges.forEach(([a,b,t],i) => {
        for (let j=0; j<3; ++j) points[3*i+j] = displayed[3*a+j] + t*(displayed[3*b+j]-displayed[3*a+j]);
      });
      contours.getPoints().setData(points,3);
      contours.modified();
    }

    function pointAt(i) {
      return Array.from(displayed.subarray(3*i, 3*i+3));
    }
    function updateMarkers() {
      marker.setVisibility(vertex >= 0);
      if (vertex >= 0) markerSource.setCenter(...pointAt(vertex));
      line.setVisibility(measurements.length === 2);
      if (measurements.length === 2) {
        distanceSource.setPoint1(...pointAt(measurements[0].i));
        distanceSource.setPoint2(...pointAt(measurements[1].i));
      }
    }
    function updateClip() {
      mapper.removeAllClippingPlanes();
      ghostMapper.removeAllClippingPlanes();
      contourMapper.removeAllClippingPlanes();
      clipPosition.disabled = !clip.value || active !== model;
      byId("flip").disabled = clipPosition.disabled;
      if (!clipPosition.disabled) {
        const axis = Number(clip.value);
        const bounds = poly.getBounds();
        const origin = [0,1,2].map((i) => (bounds[2*i]+bounds[2*i+1])/2);
        const normal = [0,0,0];
        normal[axis] = reversed ? -1 : 1;
        origin[axis] = bounds[2*axis] + (bounds[2*axis+1]-bounds[2*axis])*Number(clipPosition.value)/100;
        clipPlane.setOrigin(...origin);
        clipPlane.setNormal(...normal);
        mapper.addClippingPlane(clipPlane);
        ghostMapper.addClippingPlane(clipPlane);
        contourMapper.addClippingPlane(clipPlane);
        clipPosition.title = `${"XYZ"[axis]} = ${fmt(origin[axis])} ${model.unit}`;
      }
      render();
    }
    function deform(amount) {
      if (!active.hasDeformation) amount = 0;
      displayed = Float32Array.from(pts, (value, i) => value + amount * displacement[i]);
      poly.getPoints().setData(displayed, 3);
      poly.modified();
      byId("view-label").textContent = `${active.caption || (clip.value ? "表面剖切" : "外表面")} · ${fmt(amount)}×`;
      updateMarkers();
      updateContourGeometry();
      renderer.resetCameraClippingRange();
      updateClip();
    }

    function view(direction, up = [0,1,0]) {
      camera.setFocalPoint(0, 0, 0);
      camera.setPosition(...direction);
      camera.setViewUp(...up);
      renderer.resetCamera();
      render();
    }
    function dataView(data) {
      const selected = field?.label || "von Mises 等效应力";
      active = data;
      pts = data.positionArray ||= decode(data.points, Float32Array);
      displacement = data.displacementArray ||= decode(data.displacement, Float32Array);
      polys = data.polyArray ||= decode(data.polys, Uint32Array);
      poly.getPolys().setData(polys);
      ghostPoly.getPoints().setData(model.positionArray, 3);
      ghostPoly.getPolys().setData(model.polyArray);
      ghostPoly.modified();
      fields.replaceChildren(...data.fields.map((f, i) => new Option(f.label,i)));
      fields.value = Math.max(0, data.fields.findIndex((f) => f.label === selected));
      hit = vertex = -1;
      measurements = [];
      contourEdges = [];
      clip.disabled = data !== model;
      const span = Math.hypot(...[0,1,2].map((i) => {
        let min = Infinity, max = -Infinity;
        for (let j=i; j<pts.length; j+=3) { min=Math.min(min,pts[j]); max=Math.max(max,pts[j]); }
        return max-min;
      }));
      markerSource.setRadius(span*.006);
      byId("mesh-info").textContent = `${data.pointsCount.toLocaleString()} 节点 · ${data.faceCount.toLocaleString()} 三角面`;
      scale.disabled = animate.disabled = !data.hasDeformation;
      deform(data.hasDeformation ? Number(scale.value) : 0);
      recolor();
    }
    function showPeak() {
      if (threshold.checked) { threshold.checked = false; filterSurface(); }
      let max = -Infinity, idx = 0;
      for (let i=0; i<colorValues.length; ++i) {
        const v = field.signed ? Math.abs(colorValues[i]) : colorValues[i];
        if (v > max) { max = v; idx = i; }
      }
      if (field.association === "point") {
        vertex = idx;
        hit = 0;
      } else {
        hit = idx;
        vertex = polys[4*idx+1];
      }
      if (clip.value && active === model) {
        clip.value = "";
        deform(Number(scale.value));
      }
      const loc = pointAt(vertex);
      const delta = camera.getPosition().map((x,i) => x-camera.getFocalPoint()[i]);
      camera.setFocalPoint(...loc);
      camera.setPosition(...loc.map((x,i)=>x+delta[i]));
      updateMarkers();
      updateProbe();
      renderer.resetCameraClippingRange();
      render();
    }
    function stopAnimation() {
      cancelAnimationFrame(animation);
      animation = started = 0;
      animate.textContent = "播放形变";
      animate.setAttribute("aria-pressed","false");
    }
    function frame(time) {
      started ||= time;
      deform(Number(scale.value) * (.5-.5*Math.cos((time-started)*Math.PI/1000)));
      animation = requestAnimationFrame(frame);
    }

    fields.addEventListener("change", () => {
      rangeMin.value = rangeMax.value = thresholdValue.value = contourValue.value = "";
      hit = vertex = -1;
      measurements = [];
      updateMarkers();
      recolor();
    });
    rangeMin.addEventListener("input", recolor);
    rangeMax.addEventListener("input", recolor);
    range.addEventListener("change", recolor);
    scale.addEventListener("input", () => {
      if (scale.validity.valid && scale.value !== "") deform(Number(scale.value));
    });
    animate.addEventListener("click", () => {
      if (animation) { stopAnimation(); deform(Number(scale.value)); return; }
      if (!scale.validity.valid || scale.value === "") return;
      animate.textContent = "暂停形变";
      animate.setAttribute("aria-pressed","true");
      animation = requestAnimationFrame(frame);
    });
    byId("representation").addEventListener("change", ({target}) => {
      const prop = actor.getProperty();
      prop.setEdgeVisibility(target.value === "edges");
      if (target.value === "wire") prop.setRepresentationToWireframe();
      else if (target.value === "points") { prop.setRepresentationToPoints(); prop.setPointSize(3); }
      else prop.setRepresentationToSurface();
      render();
    });
    for (const el of [threshold, thresholdValue, contour, contourValue]) {
      el.addEventListener("change", () => {
        hit = vertex = -1;
        measurements = [];
        filterSurface();
        updateMarkers();
        updateProbe();
        render();
      });
    }
    byId("overlay").addEventListener("change", (e) => { ghost.setVisibility(e.target.checked); render(); });
    byId("iso").addEventListener("click", () => view([1.1, .44, 1.65]));
    byId("end").addEventListener("click", () => view([0,0,1]));
    byId("side").addEventListener("click", () => view([1,0,0]));
    byId("top").addEventListener("click", () => view([0,1,0],[0,0,-1]));
    clip.addEventListener("change", () => deform(Number(scale.value)));
    clipPosition.addEventListener("input", updateClip);
    byId("flip").addEventListener("click", () => { reversed = !reversed; updateClip(); });
    sectionSelect.addEventListener("change", () => {
      const next = sectionSelect.value === "" ? model : sections[Number(sectionSelect.value)];
      dataView(next);
      if (next !== model) {
        const direction = [0,0,0];
        const axis = "XYZ".indexOf(next.caption[0]);
        direction[axis] = 1;
        view(direction, axis === 1 ? [0,0,-1] : [0,1,0]);
      } else view([1.1,.44,1.65]);
    });
    byId("peak").addEventListener("click", showPeak);
    measure.addEventListener("click", () => {
      measuring = !measuring;
      measure.setAttribute("aria-pressed", String(measuring));
      hit = vertex = -1;
      measurements = [];
      updateMarkers();
      updateProbe();
      render();
    });
    byId("snapshot").addEventListener("click", async () => {
      const button = byId("snapshot");
      button.disabled = true;
      try {
        const canvas = stage.querySelector("canvas");
        render();
        const file = document.createElement("a");
        file.href = canvas.toDataURL("image/png");
        file.download = "structural-result.png";
        file.click();
      } finally { button.disabled = false; }
    });
    byId("fullscreen").addEventListener("click", () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else stage.parentElement.requestFullscreen();
    });
    new ResizeObserver(() => win.resize()).observe(stage);
    let press;
    stage.addEventListener("pointerdown", (e) => { press = [e.clientX, e.clientY]; }, true);
    stage.addEventListener("pointerup", (e) => {
      if (e.button !== 0 || !press || Math.hypot(e.clientX-press[0], e.clientY-press[1]) > 4) return;
      const canvas = stage.querySelector("canvas");
      const rect = canvas.getBoundingClientRect();
      picker.pick([(e.clientX-rect.left)*canvas.width/rect.width, (rect.bottom-e.clientY)*canvas.height/rect.height, 0], renderer);
      const picked = picker.getCellId();
      hit = picked >= 0 && visibleCells ? visibleCells[picked] : picked;
      vertex = -1;
      if (hit >= 0) {
        const point = picker.getPickPosition();
        let nearest = Infinity;
        for (let j=1; j<=3; ++j) {
          const idx = polys[hit*4+j];
          const d = Math.hypot(...point.map((v, axis) => v-displayed[3*idx+axis]));
          if (d < nearest) { nearest = d; vertex = idx; }
        }
        if (measuring) {
          if (measurements.length === 2) measurements = [];
          measurements.push({ i: vertex, raw: Array.from(pts.subarray(3*vertex, 3*vertex+3)) });
        }
      }
      updateMarkers();
      updateProbe();
      render();
    }, true);

    dataView(model);
    view([1.1,.44,1.65]);
  } catch (error) {
    probe.textContent = "3D 结果无法载入：" + error.message;
    probe.classList.add("failure");
    console.error(error);
  }
})();
