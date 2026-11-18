(() => {
  const data = JSON.parse(document.getElementById("demo-data").textContent);
  const geometry = JSON.parse(
    document.getElementById("demo-geometry").textContent,
  );
  const comparison = data.comparison;
  const protocol = comparison.protocol;
  const anchorId = protocol.anchor_id;
  const names = ["A0", "A1", "A2"];
  const $ = (id) => document.getElementById(id);
  const fmt = (x) => Number(x).toFixed(6);
  const signed = (x) => `${x >= 0 ? "+" : ""}${fmt(x)}`;
  const rimName = (id) => `Rim${id.split("-").pop()}`;
  const querySelect = $("query");
  protocol.display_query_ids.forEach((id) => {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = `Rim0001 → ${rimName(id)}`;
    querySelect.append(option);
  });

  const rows = Object.fromEntries(
    names.map((name) => [
      name,
      Object.fromEntries(
        comparison.models[name].rows.map((row) => [row.case_id, row]),
      ),
    ]),
  );
  const bases = Object.fromEntries(
    names.map((name) => [name, comparison.models[name].anchors[0]]),
  );
  const coefficient = () => Number($("target").value);
  const modelName = () => $("model").value;
  const queryId = () => querySelect.value;
  const modelDelta = (name, id, target) =>
    rows[name][id].prediction[target] - bases[name].prediction[target];
  const trueDelta = (id, target) =>
    rows.A0[id].truth[target] - bases.A0.truth[target];

  function download(name, text, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function scatter() {
    const target = coefficient();
    const name = modelName();
    const points = comparison.models[name].rows
      .filter((r) => r.source === "drivaerrim")
      .map((r) => ({
        id: r.case_id,
        x: trueDelta(r.case_id, target),
        y: modelDelta(name, r.case_id, target),
      }));
    const extent =
      Math.max(
        ...points.flatMap((p) => [Math.abs(p.x), Math.abs(p.y)]),
        0.001,
      ) * 1.12;
    const x = (v) => 58 + ((v + extent) / (2 * extent)) * 400;
    const y = (v) => 278 - ((v + extent) / (2 * extent)) * 240;
    const ticks = [-1, -0.5, 0, 0.5, 1].map((a) => a * extent);
    const grid = ticks
      .map(
        (v) =>
          `<line x1="58" x2="458" y1="${y(v)}" y2="${y(v)}" stroke="#e9eef5"/><text x="49" y="${y(v) + 4}" text-anchor="end">${v.toFixed(3)}</text><text x="${x(v)}" y="299" text-anchor="middle">${v.toFixed(3)}</text>`,
      )
      .join("");
    const circles = points
      .map(
        (p) =>
          `<circle cx="${x(p.x)}" cy="${y(p.y)}" r="${p.id === queryId() ? 6 : 3.3}" fill="${p.id === queryId() ? "#ec8d28" : "#5885bd"}" fill-opacity=".75" stroke="${p.id === queryId() ? "#7b4c1d" : "white"}" stroke-width="${p.id === queryId() ? 2 : 0.7}"><title>${rimName(p.id)} · CFD ${signed(p.x)} · 预测 ${signed(p.y)}</title></circle>`,
      )
      .join("");
    $("scatter").innerHTML =
      `<svg viewBox="0 0 500 335" role="img" aria-label="92个轮组改型的预测差值与CFD差值散点图"><g font-size="10" fill="#8692a3">${grid}<line x1="58" y1="${y(0)}" x2="458" y2="${y(0)}" stroke="#c1cbd8"/><line x1="${x(0)}" y1="38" x2="${x(0)}" y2="278" stroke="#c1cbd8"/><line x1="58" y1="278" x2="458" y2="38" stroke="#9aa9bc" stroke-dasharray="5 5"/>${circles}<text x="258" y="325" text-anchor="middle">源 CFD Δ${target === 0 ? "Cd" : "Cl"}</text><text x="15" y="22">预测差值</text></g></svg>`;
  }

  function update() {
    const id = queryId();
    const target = coefficient();
    const name = modelName();
    const delta = trueDelta(id, target);
    const estimate = modelDelta(name, id, target);
    const scores = comparison.models[name].anchor_scores[anchorId];
    $("true-delta").textContent = signed(delta);
    $("true-relative").textContent =
      target === 0
        ? `相对基准 Cd 变化 ${delta >= 0 ? "+" : ""}${((delta / bases.A0.truth[0]) * 100).toFixed(2)}%`
        : "整车升力系数变化，保留正负号";
    $("pred-title").textContent = `${name} 预测的差值`;
    $("pred-delta").textContent = signed(estimate);
    $("error").textContent = fmt(Math.abs(estimate - delta));
    $("direction").textContent =
      Math.abs(delta) < 1e-10
        ? "源 CFD 差值为零，方向不计"
        : Math.sign(estimate) === Math.sign(delta)
          ? "本例预测方向与源 CFD 一致"
          : "本例预测方向与源 CFD 不一致";
    $("mae").textContent = fmt(scores.delta_mae[target]);
    $("direction-overall").textContent =
      `非零差值方向正确率 ${(scores.direction_accuracy_nonzero[target] * 100).toFixed(1)}%（${scores.direction_denominator[target]}例）`;
    $("base-title").textContent = `基准 · ${rimName(anchorId)}`;
    $("query-title").textContent = `改型 · ${rimName(id)}`;
    const cfd = `<tr class="cfd"><td>源 CFD</td><td>${fmt(bases.A0.truth[target])}</td><td>${fmt(rows.A0[id].truth[target])}</td><td>${signed(delta)}</td><td>—</td><td>—</td></tr>`;
    $("case-table").querySelector("tbody").innerHTML =
      cfd +
      names
        .map((n) => {
          const d = modelDelta(n, id, target);
          return `<tr class="${n === name ? "selected" : ""}"><td>${n}${n === "A0" ? " · 冻结" : n === "A1" ? " · 绝对训练" : " · 增加差值监督"}</td><td>${fmt(bases[n].prediction[target])}</td><td>${fmt(rows[n][id].prediction[target])}</td><td>${signed(d)}</td><td>${fmt(bases[n].truth[target] + d)}</td><td>${fmt(Math.abs(d - delta))}</td></tr>`;
        })
        .join("");
    $("summary-table").querySelector("tbody").innerHTML = names
      .map((n) => {
        const s = comparison.models[n].anchor_scores[anchorId];
        return `<tr class="${n === name ? "selected" : ""}"><td>${n}</td><td>${fmt(s.delta_mae[target])}</td><td>${(100 * s.direction_accuracy_nonzero[target]).toFixed(1)}%</td><td>${(s.predicted_delta_std[target] / s.true_delta_std[target]).toFixed(3)}</td></tr>`;
      })
      .join("");
    $("summary-table").querySelector("tbody").innerHTML += [
      ["zero_change", "零变化基线"],
      ["train_mean", "训练均值基线"],
    ]
      .map(([key, label]) => {
        const s = data.baselines[key];
        return `<tr><td>${label}</td><td>${fmt(s.delta_mae[target])}</td><td>${key === "zero_change" ? "不估方向" : `${(100 * s.direction_accuracy_nonzero[target]).toFixed(1)}%`}</td><td>0.000</td></tr>`;
      })
      .join("");
    const a1 = comparison.models.A1.anchor_scores[anchorId].delta_mae[target];
    const a2 = comparison.models.A2.anchor_scores[anchorId].delta_mae[target];
    const change = ((a1 - a2) / a1) * 100;
    const ci = comparison.A2_vs_A1_delta_mae_reduction_query_bootstrap_95;
    const pairChange =
      ((data.pairwise.A1.delta_mae[target] -
        data.pairwise.A2.delta_mae[target]) /
        data.pairwise.A1.delta_mae[target]) *
      100;
    $("finding").textContent =
      `主基准 Rim0001 下，A2 相比 A1 的差值 MAE ${change >= 0 ? "降低" : "上升"} ${Math.abs(change).toFixed(1)}%。主基准查询重采样的误差减少区间为 [${signed(ci[0][target])}, ${signed(ci[1][target])}]。改用全部92个查询的4186个两两改型时，MAE ${pairChange >= 0 ? "降低" : "上升"} ${Math.abs(pairChange).toFixed(1)}%。这些配对相互依赖；结果不代表 CFD 噪声或跨车型精度。`;
    $("input-image").src = data.images[id];
    const audit = data.geometryAudit.cases[id];
    const pixels = Object.values(audit.input_changes).reduce(
      (sum, v) => sum + v.changed_pixels,
      0,
    );
    $("pixel-count").textContent =
      `五视图变化像素 ${pixels} / ${5 * 256 * 256}`;
    scatter();
    if (views.length) showQuery(id);
  }

  const views = [];
  let camera;
  function decode(text, Type) {
    const raw = atob(text);
    const bytes = Uint8Array.from(raw, (c) => c.charCodeAt(0));
    return new Type(bytes.buffer);
  }
  function actor(mesh, color, mirror = false) {
    const poly = vtk.Common.DataModel.vtkPolyData.newInstance();
    poly.getPoints().setData(decode(mesh.positions, Float32Array), 3);
    poly.getPolys().setData(decode(mesh.polys, Uint32Array));
    const normals = vtk.Filters.Core.vtkPolyDataNormals.newInstance({
      computePointNormals: true,
      computeCellNormals: false,
    });
    normals.setInputData(poly);
    const mapper = vtk.Rendering.Core.vtkMapper.newInstance();
    mapper.setInputConnection(normals.getOutputPort());
    const result = vtk.Rendering.Core.vtkActor.newInstance();
    result.setMapper(mapper);
    result.getProperty().setColor(...color);
    result.getProperty().setAmbient(0.18);
    result.getProperty().setDiffuse(0.8);
    result.getProperty().setSpecular(0.15);
    if (mirror) result.setScale(1, -1, 1);
    return result;
  }
  function showQuery(id) {
    const view = views[1];
    if (view.queryId === id) return;
    view.wheels.forEach((a) => {
      view.renderer.removeActor(a);
      a.delete();
    });
    view.wheels = [];
    ["FL", "RL"].forEach((pos) =>
      [false, true].forEach((mirror) => {
        const part = actor(
          geometry.wheels[id][pos],
          [0.14, 0.43, 0.83],
          mirror,
        );
        view.renderer.addActor(part);
        view.wheels.push(part);
      }),
    );
    view.queryId = id;
    view.generic.getRenderWindow().render();
  }
  function resetCamera() {
    camera.setPosition(-6.5, 8, 3.8);
    camera.setFocalPoint(0, 0, 0.65);
    camera.setViewUp(0, 0, 1);
    camera.setClippingRange(0.1, 60);
    views[0].renderer.resetCamera();
    camera.zoom(1.35);
    views.forEach((v) => v.generic.getRenderWindow().render());
  }
  try {
    ["base-view", "query-view"].forEach((id, i) => {
      const generic = vtk.Rendering.Misc.vtkGenericRenderWindow.newInstance({
        background: [0.965, 0.977, 0.99],
      });
      generic.setContainer($(id));
      generic.resize();
      const renderer = generic.getRenderer();
      const body = actor(geometry.body, [0.73, 0.77, 0.82]);
      renderer.addActor(body);
      const wheels = [];
      ["FL", "RL"].forEach((pos) =>
        [false, true].forEach((mirror) => {
          const part = actor(
            geometry.wheels[anchorId][pos],
            [0.14, 0.43, 0.83],
            mirror,
          );
          renderer.addActor(part);
          wheels.push(part);
        }),
      );
      views.push({ generic, renderer, body, wheels });
      new ResizeObserver(() => generic.resize()).observe($(id));
    });
    camera = views[0].renderer.getActiveCamera();
    views[1].renderer.setActiveCamera(camera);
    let queued = false;
    camera.onModified(() => {
      if (queued) return;
      queued = true;
      requestAnimationFrame(() => {
        queued = false;
        views.forEach((v) => v.generic.getRenderWindow().render());
      });
    });
    resetCamera();
    $("viewer-status").textContent =
      "完整车辆预览已加载。预览简化不参与系数计算；请以完整五视图回放和源 CFD 对照核验。";
    $("mesh-count").textContent =
      `车身预览 ${geometry.body.triangles.toLocaleString()} 面；源车身 ${geometry.body.source_triangles.toLocaleString()} 面`;
    $("opacity").addEventListener("input", () =>
      views.forEach((v) => {
        v.body.getProperty().setOpacity(Number($("opacity").value));
        v.generic.getRenderWindow().render();
      }),
    );
    $("reset").addEventListener("click", resetCamera);
  } catch (error) {
    $("viewer-status").textContent =
      `3D 预览加载失败：${error.message}。下方完整视图和数值证据仍可核验。`;
    console.error(error);
  }

  const proof = comparison.training_proof;
  const maxReplay = Math.max(
    ...Object.values(data.geometryAudit.cases).flatMap((c) =>
      Object.values(c.raw_source_replay_max_abs),
    ),
  );
  $("checks").innerHTML = [
    "车身、轮罩、支撑件的5个共享源文件 SHA 已核验，固定车身保持一致。",
    "基准与3个展示查询均从原始 STL 重建，源轮组哈希与案例身份匹配。",
    `四个展示几何的完整五视图回放最大数值差：${maxReplay}。`,
    "720 train / 92 val / 92 historical test；配对两端均来自训练集。",
    `A1/A2 各 ${proof.optimizer_steps_per_arm} 步、${proof.endpoint_exposures_per_arm.toLocaleString()} 次端点曝光，实际样本轨迹一致。`,
    "展示查询按编号预先固定；本轮A1/A2权重仅按验证集选择。",
  ]
    .map((s) => `<li>${s}</li>`)
    .join("");
  $("controls-proof").innerHTML =
    names
      .map((name) => {
        const model = comparison.models[name];
        const s = model.controls;
        const reference = Object.entries(model.anchor_scores)
          .map(([id, score]) => `${rimName(id)}: ${fmt(score.delta_mae[0])}`)
          .join("；");
        const pair = data.pairwise[name];
        return `<div class="proof-row"><strong>${name} · 自检</strong><span>重复输入 ${s.same_geometry_repeat_max_abs.toExponential(1)}；反向残差 ${s.reverse_residual.toExponential(1)}；环路残差 ${s.three_geometry_loop_residual.toExponential(1)}</span></div><div class="proof-row"><strong>${name} · 换基准后的 ΔCd MAE</strong><span>${reference}</span></div><div class="proof-row"><strong>${name} · 4186个两两查询改型</strong><span>ΔCd MAE ${fmt(pair.delta_mae[0])}；ΔCl MAE ${fmt(pair.delta_mae[1])}；非零Cd差值排序 ${(100 * pair.cd_ranking["0"].accuracy).toFixed(1)}%</span></div><div class="proof-row"><strong>${name} · 缓存输入热推理中位数</strong><span>${(model.warm_cached_inference_median_seconds * 1000).toFixed(1)} ms，不含原网格处理</span></div>`;
      })
      .join("") +
    "<p>零改动、反向和环路一致性来自共享绝对头相减的结构；这些自检通过，不代表模型已学会正确的气动影响。</p>";
  $("radiant-status").textContent = data.radiant.status;
  $("download-json").addEventListener("click", () =>
    download(
      "detail-contribution-evidence.json",
      JSON.stringify(data, null, 2),
      "application/json",
    ),
  );
  $("download-csv").addEventListener("click", () => {
    const csv = [
      "case_id,model,cd_true,cd_pred,delta_cd_true,delta_cd_pred,anchored_cd,cl_true,cl_pred,delta_cl_true,delta_cl_pred,anchored_cl",
    ];
    names.forEach((name) =>
      comparison.models[name].rows
        .filter((r) => r.source === "drivaerrim")
        .forEach((r) => {
          csv.push(
            [
              r.case_id,
              name,
              r.truth[0],
              r.prediction[0],
              trueDelta(r.case_id, 0),
              modelDelta(name, r.case_id, 0),
              bases[name].truth[0] + modelDelta(name, r.case_id, 0),
              r.truth[1],
              r.prediction[1],
              trueDelta(r.case_id, 1),
              modelDelta(name, r.case_id, 1),
              bases[name].truth[1] + modelDelta(name, r.case_id, 1),
            ].join(","),
          );
        }),
    );
    download(
      "detail-contribution-queries.csv",
      csv.join("\n"),
      "text/csv;charset=utf-8",
    );
  });
  [querySelect, $("target"), $("model")].forEach((element) =>
    element.addEventListener("change", update),
  );
  update();
})();
