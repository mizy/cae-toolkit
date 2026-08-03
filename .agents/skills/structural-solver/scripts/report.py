#!/usr/bin/env python3
"""Authoritative JSON report + self-contained HTML report."""
from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any

JSON_KEYS = (
    "decision", "input", "material", "units", "load_cases",
    "results", "validation", "gates", "outputs", "runtime",
)


def build_report_json(**kw: Any) -> dict:
    rep: dict[str, Any] = {}
    for key in JSON_KEYS:
        rep[key] = kw.get(key)
    rep["generated_at"] = datetime.now().isoformat(timespec="seconds")
    return rep


def write_json(report: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _b64(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _fmt(x: float | None, unit: str, nd: int = 4) -> str:
    if x is None or not np_isfinite(x):
        return "n/a"
    return f"{x:.{nd}e} {unit}"


def np_isfinite(x: float) -> bool:
    import math
    return math.isfinite(float(x))


def _gate_rows(gates: dict[str, dict]) -> str:
    rows = []
    for name, g in gates.items():
        status = "PASS" if g.get("passed") else "FAIL"
        color = "#1a7f37" if g.get("passed") else "#cf222e"
        rows.append(
            f"<tr><td>{name}</td><td>{g.get('measured','')}</td>"
            f"<td>{g.get('threshold','')}</td>"
            f"<td style='color:{color};font-weight:bold'>{status}</td></tr>"
        )
    return "\n".join(rows)


def _bench_rows(benchmarks: list[dict]) -> str:
    rows = []
    for b in benchmarks:
        status = "PASS" if b.get("passed") else "FAIL"
        color = "#1a7f37" if b.get("passed") else "#cf222e"
        rows.append(
            f"<tr><td>{b['name']}</td>"
            f"<td>{b.get('analytical','')}</td><td>{b.get('measured','')}</td>"
            f"<td>{b.get('rel_error','')}</td><td>{b.get('extra','')}</td>"
            f"<td style='color:{color};font-weight:bold'>{status}</td></tr>"
        )
    return "\n".join(rows)


def _encode_float32(arr) -> str:
    import base64
    import numpy as np
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode("ascii")


def _encode_uint32(arr) -> str:
    import base64
    import numpy as np
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.uint32).tobytes()).decode("ascii")


VIEWER_JS = r"""
function b64ToArray(b64, Ctor) {
  var bin = atob(b64), n = bin.length;
  var arr = new Ctor(n / Ctor.BYTES_PER_ELEMENT);
  var dv = new DataView(arr.buffer);
  for (var i = 0; i < n; i++) dv.setUint8(i, bin.charCodeAt(i));
  return arr;
}
function jet(t) {
  t = Math.max(0, Math.min(1, t));
  var r = Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 3)));
  var g = Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 2)));
  var b = Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 1)));
  return 'rgb(' + (r*255|0) + ',' + (g*255|0) + ',' + (b*255|0) + ')';
}
function buildViewer(vdata) {
  var holder = document.getElementById('v3d');
  if (!holder) return;
  var W = holder.clientWidth || 900, H = 480;
  var canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  canvas.style.cursor = 'grab';
  holder.appendChild(canvas);
  var ctx = canvas.getContext('2d');
  var diag = vdata.bbox_diag;
  var cx = vdata.center[0], cy = vdata.center[1], cz = vdata.center[2];
  var pts = b64ToArray(vdata.points, Float32Array);
  var ptsDef = b64ToArray(vdata.points_def, Float32Array);
  var faces = b64ToArray(vdata.faces, Uint32Array);
  var scal = vdata.scalar ? b64ToArray(vdata.scalar, Float32Array) : null;
  var scal2 = vdata.scalar2 ? b64ToArray(vdata.scalar2, Float32Array) : null;
  var sMin = vdata.scalar_min, sMax = vdata.scalar_max;
  var sMin2 = vdata.scalar2_min, sMax2 = vdata.scalar2_max;
  var fieldIdx = 0;
  var nP = pts.length / 3, nF = faces.length / 3;
  // look-at camera: iso view matching pyvista static renders (up = +Z)
  var theta = 0.7853981634, phi = 0.9553166181, radius = diag * 1.9;
  var panX = 0, panY = 0;
  var showWire = true, showDef = true, showFix = true, showLoad = true;
  var bx = 0, by = 0, bz = 0, ux = 0, uy = 0, uz = 0, nx = 0, ny = 0, nz = 0;
  function updateBasis() {
    var ex = cx + radius * Math.sin(phi) * Math.cos(theta);
    var ey = cy + radius * Math.cos(phi);
    var ez = cz + radius * Math.sin(phi) * Math.sin(theta);
    var ddx = cx - ex, ddy = cy - ey, ddz = cz - ez;
    var dl = Math.sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
    nx = ddx/dl; ny = ddy/dl; nz = ddz/dl;
    var rxl = -ny, ryl = nx, rzl = 0; // up(0,0,1) x n -> X+ points right
    var rl = Math.sqrt(rxl*rxl + ryl*ryl + rzl*rzl) || 1;
    bx = rxl/rl; by = ryl/rl; bz = rzl/rl;
    ux = by*nz - bz*ny; uy = bz*nx - bx*nz; uz = bx*ny - by*nx;
  }
  updateBasis();
  var sx = new Float32Array(nP), sy = new Float32Array(nP), sd = new Float32Array(nP);
  var ox = new Float32Array(nP), oy = new Float32Array(nP);
  var projPts = function(P, X, Y) {
    var f = H * 1.4;
    for (var i = 0; i < nP; i++) {
      var dx = P[i*3] - cx, dy = P[i*3+1] - cy, dz = P[i*3+2] - cz;
      var px = dx*bx + dy*by + dz*bz;
      var py = dx*ux + dy*uy + dz*uz;
      var pz = dx*nx + dy*ny + dz*nz + radius;
      X[i] = W/2 + px*(f/pz) + panX;
      Y[i] = H/2 - py*(f/pz) + panY;
    }
  };
  function projectAll() {
    projPts(ptsDef, sx, sy); // deformed faces
    projPts(pts, ox, oy);    // original wireframe
  }
  var pj = new Float32Array(2);
  function projPt(x, y, z) {
    var dx = x - cx, dy = y - cy, dz = z - cz;
    var px = dx*bx + dy*by + dz*bz;
    var py = dx*ux + dy*uy + dz*uz;
    var pz = dx*nx + dy*ny + dz*nz + radius;
    var s = (H * 1.4) / pz;
    pj[0] = W/2 + px*s + panX; pj[1] = H/2 - py*s + panY;
  }
  function render(quick) {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#f4f6f8'; ctx.fillRect(0, 0, W, H);
    projectAll();
    var order = new Array(nF);
    for (var f = 0; f < nF; f++) order[f] = f;
    var depth = new Float32Array(nF);
    for (var f = 0; f < nF; f++) {
      depth[f] = (sd[faces[f*3]] + sd[faces[f*3+1]] + sd[faces[f*3+2]]) / 3;
    }
    order.sort(function(a, b) { return depth[b] - depth[a]; }); // far first
    // deformed faces with field coloring (gradient per triangle)
    if (showDef && !quick) {
      var cur = fieldIdx === 0 ? scal : scal2;
      var mn = fieldIdx === 0 ? sMin : sMin2, mx = fieldIdx === 0 ? sMax : sMax2;
      for (var oi = 0; oi < nF; oi++) {
        var f = order[oi];
        var i0 = faces[f*3], i1 = faces[f*3+1], i2 = faces[f*3+2];
        ctx.beginPath();
        ctx.moveTo(sx[i0], sy[i0]); ctx.lineTo(sx[i1], sy[i1]); ctx.lineTo(sx[i2], sy[i2]);
        ctx.closePath();
        if (cur) {
          var t0 = (cur[i0] - mn) / (mx - mn || 1), t1 = (cur[i1] - mn) / (mx - mn || 1), t2 = (cur[i2] - mn) / (mx - mn || 1);
          var g = ctx.createLinearGradient(sx[i0], sy[i0], sx[i2], sy[i2]);
          g.addColorStop(0, jet(t0)); g.addColorStop(0.5, jet(t1)); g.addColorStop(1, jet(t2));
          ctx.fillStyle = g;
          ctx.strokeStyle = 'rgba(0,0,0,0.22)'; ctx.lineWidth = 0.5;
          ctx.fill(); ctx.stroke();
        } else {
          ctx.fillStyle = 'rgba(90,140,220,0.85)'; ctx.fill();
        }
      }
    }
    // original wireframe (static reference at undeformed position)
    if (showWire) {
      ctx.strokeStyle = 'rgba(60,60,60,0.6)'; ctx.lineWidth = 0.7;
      ctx.beginPath();
      for (var f = 0; f < nF; f++) {
        var i0 = faces[f*3], i1 = faces[f*3+1], i2 = faces[f*3+2];
        ctx.moveTo(ox[i0], oy[i0]); ctx.lineTo(ox[i1], oy[i1]);
        ctx.moveTo(ox[i1], oy[i1]); ctx.lineTo(ox[i2], oy[i2]);
        ctx.moveTo(ox[i2], oy[i2]); ctx.lineTo(ox[i0], oy[i0]);
      }
      ctx.stroke();
    }
    // axes (origin at model min corner): X red, Y green, Z blue
    var axL = diag * 0.28;
    var a0x = 0, a0y = 0, a0z = 0;
    projPt(a0x, a0y, a0z);
    var axs = pj[0], ays = pj[1];
    var axisT = [['X', '#d62728', axL, 0, 0], ['Y', '#2ca02c', 0, axL, 0], ['Z', '#1f77b4', 0, 0, axL]];
    ctx.font = 'bold 14px Helvetica, Arial, sans-serif';
    for (var a = 0; a < 3; a++) {
      var at = axisT[a];
      projPt(a0x + at[2], a0y + at[3], a0z + at[4]);
      ctx.strokeStyle = at[1]; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(axs, ays); ctx.lineTo(pj[0], pj[1]); ctx.stroke();
      ctx.fillStyle = at[1];
      ctx.fillText(at[0], pj[0] + 5, pj[1] - 5);
    }
    // constraints
    if (showFix) {
      (vdata.fixes || []).forEach(function(fx) {
        var fp = b64ToArray(fx.points, Float32Array);
        ctx.fillStyle = '#d62728';
        for (var i = 0; i < fp.length / 3; i++) {
          projPt(fp[i*3], fp[i*3+1], fp[i*3+2]);
          ctx.beginPath(); ctx.arc(pj[0], pj[1], 2.4, 0, 6.2832); ctx.fill();
        }
        projPt(cx, cy + diag * 0.22, cz);
        ctx.font = 'bold 15px Helvetica, Arial, sans-serif';
        ctx.fillStyle = '#d62728';
        ctx.fillText('Fixed: ' + fx.face, pj[0] + 10, pj[1] - 10);
      });
    }
    // loads
    if (showLoad) {
      (vdata.loads || []).forEach(function(ld) {
        var L = diag * 0.16;
        var ex = ld.center[0] + ld.dir[0]*L, ey = ld.center[1] + ld.dir[1]*L, ez = ld.center[2] + ld.dir[2]*L;
        projPt(ld.center[0], ld.center[1], ld.center[2]);
        var lx0 = pj[0], ly0 = pj[1];
        projPt(ex, ey, ez);
        var tx = pj[0], ty = pj[1];
        ctx.strokeStyle = '#d62728'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(lx0, ly0); ctx.lineTo(tx, ty); ctx.stroke();
        var ang = Math.atan2(ty - oy, tx - ox);
        ctx.fillStyle = '#d62728';
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(tx - 12*Math.cos(ang - 0.4), ty - 12*Math.sin(ang - 0.4));
        ctx.lineTo(tx - 12*Math.cos(ang + 0.4), ty - 12*Math.sin(ang + 0.4));
        ctx.closePath(); ctx.fill();
        ctx.font = 'bold 15px Helvetica, Arial, sans-serif';
        ctx.fillText(ld.label, tx + 10, ty - 10);
      });
    }
  }
  var dragging = false, mode = null, lastX = 0, lastY = 0;
  canvas.addEventListener('pointerdown', function(e) {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    mode = (e.button === 2) ? 'pan' : 'rot';
    canvas.style.cursor = 'grabbing';
  });
  window.addEventListener('pointermove', function(e) {
    if (!dragging) return;
    var dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (mode === 'rot') {
      theta -= dx * 0.008;
      phi = Math.max(0.1, Math.min(Math.PI - 0.1, phi - dy * 0.008));
      updateBasis();
    } else {
      panX += dx; panY += dy;
    }
    render(true);
  });
  window.addEventListener('pointerup', function() {
    if (!dragging) return;
    dragging = false;
    canvas.style.cursor = 'grab';
    render(false);
  });
  canvas.addEventListener('wheel', function(e) {
    e.preventDefault();
    radius *= Math.exp(e.deltaY * 0.001);
    radius = Math.max(diag * 0.2, Math.min(diag * 20, radius));
    updateBasis();
    render(false);
  }, {passive: false});
  canvas.addEventListener('contextmenu', function(e) { e.preventDefault(); });
  document.getElementById('cWire').addEventListener('change', function(e) { showWire = e.target.checked; render(false); });
  document.getElementById('cDef').addEventListener('change', function(e) { showDef = e.target.checked; render(false); });
  document.getElementById('cFix').addEventListener('change', function(e) { showFix = e.target.checked; render(false); });
  document.getElementById('cLoad').addEventListener('change', function(e) { showLoad = e.target.checked; render(false); });
  var lc = document.getElementById('v3d-legend');
  function drawLegend() {
    lc.innerHTML = '';
    var cur = fieldIdx === 0 ? scal : scal2;
    if (!cur) return;
    var mn = fieldIdx === 0 ? sMin : sMin2, mx = fieldIdx === 0 ? sMax : sMax2;
    var lo = document.createElement('canvas'); lo.width = 220; lo.height = 16;
    var lx = lo.getContext('2d');
    for (var i = 0; i < 220; i++) { lx.fillStyle = jet(i / 219); lx.fillRect(i, 0, 1, 16); }
    lc.appendChild(lo);
    var sp2 = document.createElement('span');
    sp2.textContent = mn.toFixed(3) + '  ->  ' + (mx >= 100 ? mx.toFixed(1) : mx.toFixed(4)) + ' ' + (fieldIdx === 0 ? (vdata.scalar_unit || '') : (vdata.scalar2_unit || ''));
    lc.appendChild(sp2);
  }
  drawLegend();
  var sel = document.getElementById('cField');
  if (sel) sel.addEventListener('change', function(e) {
    fieldIdx = parseInt(e.target.value, 10);
    drawLegend(); render(false);
  });
  render(false);
  window.addEventListener('resize', function() {
    W = holder.clientWidth || 900;
    canvas.width = W; canvas.style.width = W + 'px';
    render(false);
  });
}
document.addEventListener('DOMContentLoaded', function() { if (typeof VDATA !== 'undefined') buildViewer(VDATA); });
"""

VIEWER2_JS = r"""
function buildAnimViewer(vdata) {
  var holder = document.getElementById('v3d-anim');
  if (!holder) return;
  var W = holder.clientWidth || 900, H = 460;
  var canvas = document.createElement('canvas');
  canvas.width = W; canvas.height = H;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  canvas.style.cursor = 'grab';
  holder.appendChild(canvas);
  var ctx = canvas.getContext('2d');
  var diag = vdata.bbox_diag;
  var cx = vdata.center[0], cy = vdata.center[1], cz = vdata.center[2];
  var pts = b64ToArray(vdata.points, Float32Array);
  var ptsDef = b64ToArray(vdata.points_def, Float32Array);
  var faces = b64ToArray(vdata.faces, Uint32Array);
  var scal = vdata.scalar ? b64ToArray(vdata.scalar, Float32Array) : null;
  var sMin = vdata.scalar_min, sMax = vdata.scalar_max;
  var nP = pts.length / 3, nF = faces.length / 3;
  var step = Math.max(1, Math.ceil(nF / 8000));
  var nFs = Math.floor(nF / step);
  var baseScale = vdata.display_scale || 1;
  var userScale = 15;
  var playing = true;
  var theta = 0.7853981634, phi = 0.9553166181, radius = diag * 1.9;
  var panX = 0, panY = 0;
  var bx = 0, by = 0, bz = 0, ux = 0, uy = 0, uz = 0, nx = 0, ny = 0, nz = 0;
  function updateBasis() {
    var ex = cx + radius * Math.sin(phi) * Math.cos(theta);
    var ey = cy + radius * Math.cos(phi);
    var ez = cz + radius * Math.sin(phi) * Math.sin(theta);
    var ddx = cx - ex, ddy = cy - ey, ddz = cz - ez;
    var dl = Math.sqrt(ddx*ddx + ddy*ddy + ddz*ddz);
    nx = ddx/dl; ny = ddy/dl; nz = ddz/dl;
    var rxl = -ny, ryl = nx, rzl = 0;
    var rl = Math.sqrt(rxl*rxl + ryl*ryl + rzl*rzl) || 1;
    bx = rxl/rl; by = ryl/rl; bz = rzl/rl;
    ux = by*nz - bz*ny; uy = bz*nx - bx*nz; uz = bx*ny - by*nx;
  }
  updateBasis();
  // interpolated vertex positions (deformation animation)
  var tx = new Float32Array(nP * 3);
  var sx = new Float32Array(nP), sy = new Float32Array(nP), sd = new Float32Array(nP);
  function setFrame(k) {
    var amp = (userScale / baseScale) * k;
    for (var i = 0; i < nP; i++) {
      tx[i*3]   = pts[i*3]   + (ptsDef[i*3]   - pts[i*3])   * amp;
      tx[i*3+1] = pts[i*3+1] + (ptsDef[i*3+1] - pts[i*3+1]) * amp;
      tx[i*3+2] = pts[i*3+2] + (ptsDef[i*3+2] - pts[i*3+2]) * amp;
    }
    var f = H * 1.4;
    for (var i = 0; i < nP; i++) {
      var dx = tx[i*3] - cx, dy = tx[i*3+1] - cy, dz = tx[i*3+2] - cz;
      var px = dx*bx + dy*by + dz*bz;
      var py = dx*ux + dy*uy + dz*uz;
      var pz = dx*nx + dy*ny + dz*nz + radius;
      sd[i] = pz;
      var s = f / pz;
      sx[i] = W/2 + px*s + panX;
      sy[i] = H/2 - py*s + panY;
    }
  }
  var order = new Array(nFs);
  var depth = new Float32Array(nFs);
  function render(quick) {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#f4f6f8'; ctx.fillRect(0, 0, W, H);
    for (var f = 0; f < nFs; f++) order[f] = f;
    for (var f = 0; f < nFs; f++) {
      var ff = f * step;
      depth[f] = (sd[faces[ff*3]] + sd[faces[ff*3+1]] + sd[faces[ff*3+2]]) / 3;
    }
    order.sort(function(a, b) { return depth[b] - depth[a]; });
    if (!quick) {
      for (var oi = 0; oi < nFs; oi++) {
        var f = order[oi] * step;
        var i0 = faces[f*3], i1 = faces[f*3+1], i2 = faces[f*3+2];
        ctx.beginPath();
        ctx.moveTo(sx[i0], sy[i0]); ctx.lineTo(sx[i1], sy[i1]); ctx.lineTo(sx[i2], sy[i2]);
        ctx.closePath();
        if (scal) {
          var t = ((scal[i0] + scal[i1] + scal[i2]) / 3 - sMin) / (sMax - sMin || 1);
          ctx.fillStyle = jet(t);
          ctx.fill();
        } else {
          ctx.fillStyle = 'rgba(90,140,220,0.9)'; ctx.fill();
        }
      }
    }
    // deformed wireframe (interpolated position), faint so field colors stay visible
    ctx.strokeStyle = 'rgba(60,60,60,0.16)'; ctx.lineWidth = 0.6;
    ctx.beginPath();
    for (var f = 0; f < nFs; f++) {
      var ff = f * step;
      var i0 = faces[ff*3], i1 = faces[ff*3+1], i2 = faces[ff*3+2];
      ctx.moveTo(sx[i0], sy[i0]); ctx.lineTo(sx[i1], sy[i1]);
      ctx.moveTo(sx[i1], sy[i1]); ctx.lineTo(sx[i2], sy[i2]);
      ctx.moveTo(sx[i2], sy[i2]); ctx.lineTo(sx[i0], sy[i0]);
    }
    ctx.stroke();
  }
  // animation loop
  var t0 = null;
  function frame(ts) {
    if (t0 === null) t0 = ts;
    var el = (ts - t0) / 1000;
    if (playing) {
      var k = 0.5 + 0.5 * Math.sin(el * 2.2);
      setFrame(k);
      render(false);
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  // controls
  var dragging = false, mode = null, lastX = 0, lastY = 0;
  canvas.addEventListener('pointerdown', function(e) {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    mode = (e.button === 2) ? 'pan' : 'rot';
    canvas.style.cursor = 'grabbing';
  });
  window.addEventListener('pointermove', function(e) {
    if (!dragging) return;
    var dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (mode === 'rot') {
      theta -= dx * 0.008;
      phi = Math.max(0.1, Math.min(Math.PI - 0.1, phi - dy * 0.008));
    } else { panX += dx; panY += dy; }
    updateBasis();
  });
  window.addEventListener('pointerup', function() {
    if (!dragging) return;
    dragging = false; canvas.style.cursor = 'grab';
  });
  canvas.addEventListener('wheel', function(e) {
    e.preventDefault();
    radius *= Math.exp(e.deltaY * 0.001);
    radius = Math.max(diag * 0.2, Math.min(diag * 20, radius));
    updateBasis();
  }, {passive: false});
  canvas.addEventListener('contextmenu', function(e) { e.preventDefault(); });
  var btn = document.getElementById('aPlay');
  if (btn) btn.addEventListener('click', function() {
    playing = !playing;
    btn.textContent = playing ? '暂停' : '播放';
  });
  var sld = document.getElementById('aScale');
  var sval = document.getElementById('aScaleVal');
  if (sld) sld.addEventListener('input', function() {
    userScale = parseInt(sld.value, 10);
    if (sval) sval.textContent = userScale + 'x';
  });
}
document.addEventListener('DOMContentLoaded', function() { if (typeof VDATA !== 'undefined') buildAnimViewer(VDATA); });
"""


def _viewer_section(viewer: dict) -> str:
    import json
    if not viewer:
        return ""
    data = {
        "points": _encode_float32(viewer["points"]),
        "points_def": _encode_float32(viewer["points_def"]),
        "faces": _encode_uint32(viewer["faces"]),
        "scalar": _encode_float32(viewer["scalar"]),
        "scalar_min": float(viewer["scalar"].min()),
        "scalar_max": float(viewer["scalar"].max()),
        "scalar2": _encode_float32(viewer["scalar2"]),
        "scalar2_min": float(viewer["scalar2"].min()),
        "scalar2_max": float(viewer["scalar2"].max()),
        "scalar_unit": viewer.get("scalar_unit", ""),
        "bbox_diag": float(viewer["bbox_diag"]),
        "center": [float(v) for v in viewer["center"]],
        "display_scale": float(viewer.get("display_scale", 1.0)),
        "fixes": [
            {"face": f["face"], "points": _encode_float32(f["points"])} for f in viewer.get("fixes", [])
        ],
        "loads": viewer.get("loads", []),
    }
    html = f"""
<h2>3D 交互查看器</h2>
<p style="color:#57606a">拖拽旋转 · 右键拖拽平移 · 滚轮缩放。红色箭头 = 载荷方向与大小,红色点 = 固定约束面。
变形网格按位移幅值着色(放大 {data['display_scale']:.1f}×,点数组里的位移为真实值)。</p>
<div id="v3d" style="width:100%;height:480px;border:1px solid #d0d7de;border-radius:8px;background:#f4f6f8"></div>
<div style="margin:.5rem 0">
  <label><input type="checkbox" id="cWire" checked> 原始网格(线框)</label>&nbsp;
  <label><input type="checkbox" id="cDef" checked> 变形网格(位移着色)</label>&nbsp;
  <label><input type="checkbox" id="cFix" checked> 约束标记</label>&nbsp;
  <label><input type="checkbox" id="cLoad" checked> 载荷标记</label>&nbsp;
  <label>着色字段:
    <select id="cField">
      <option value="0">位移幅值 |u|</option>
      <option value="1">von Mises 应力</option>
    </select>
  </label>
</div>
<div id="v3d-legend" style="display:flex;align-items:center;gap:.6rem;font-size:.85rem;color:#57606a"></div>
<h3 style="margin-top:1.4rem">变形动画视图(原始 ↔ 变形往复摆动)</h3>
<div id="v3d-anim" style="width:100%;height:460px;border:1px solid #d0d7de;border-radius:8px;background:#f4f6f8"></div>
<div style="margin:.5rem 0;display:flex;align-items:center;gap:1rem;font-size:.9rem">
  <button id="aPlay" type="button" style="padding:.3rem 1rem">暂停</button>
  <label>变形放大: <input id="aScale" type="range" min="1" max="50" value="15" style="vertical-align:middle">
  <span id="aScaleVal" style="font-weight:bold">15x</span></label>
  <span style="color:#57606a">拖拽旋转 · 右键平移 · 滚轮缩放(摆动幅度 = 放大倍数 × 摆动相位)</span>
</div>
<script>var VDATA = {json.dumps(data)};</script>
<script>{VIEWER_JS}</script>
<script>{VIEWER2_JS}</script>
"""
    return html


def write_html(report: dict, images: dict[str, Path], out_path: Path, viewer: dict | None = None) -> None:
    r = report
    res = r.get("results", {})
    unit = r.get("units", {})
    fields = res.get("field_summary", {})
    img_def = "".join(
        f'<h3>{title}</h3><img src="data:image/png;base64,{_b64(path)}" alt="{title}" '
        f'style="max-width:100%;border:1px solid #ddd;border-radius:6px"/>'
        for title, path in images.items()
    )
    load_rows = "".join(
        f"<tr><td>{lc['face']}</td><td>f = ({lc['force_total'][0]:.4g}, {lc['force_total'][1]:.4g}, "
        f"{lc['force_total'][2]:.4g}) {unit.get('force','N')}</td><td>{lc['nodes']} nodes</td></tr>"
        for lc in r.get("load_cases", {}).get("loads", [])
    )
    fix_rows = "".join(f"<tr><td>{f}</td></tr>" for f in r.get("load_cases", {}).get("fixes", []))
    gate_rows = _gate_rows(r.get("gates", {}))
    bench_rows = _bench_rows(r.get("validation", {}).get("benchmarks", []))
    outs = r.get("outputs", {})
    out_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in outs.items() if v
    )
    viewer_html = _viewer_section(viewer) if viewer else ""
    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>{r.get('decision',{}).get('title','Structural Solve Report')}</title>
<style>
body{{font-family:-apple-system,'PingFang SC','Microsoft YaHei',sans-serif;margin:2rem auto;max-width:1000px;color:#222;line-height:1.5}}
h1{{border-bottom:3px solid #0969da;padding-bottom:.3rem}} h2{{margin-top:2.2rem;color:#0969da}}
table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.92rem}}
th,td{{border:1px solid #d0d7de;padding:.45rem .6rem;text-align:left}}
th{{background:#f6f8fa}}
.cards{{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0}}
.card{{flex:1;min-width:170px;border:1px solid #d0d7de;border-radius:8px;padding:.8rem 1rem;background:#f6f8fa}}
.card b{{display:block;font-size:1.25rem;color:#0969da;margin-top:.2rem}}
.meta{{color:#57606a;font-size:.85rem}}
img{{margin:.4rem 0 1.2rem}}
.pass{{color:#1a7f37;font-weight:bold}} .fail{{color:#cf222e;font-weight:bold}}
</style></head><body>
<h1>{r.get('decision',{}).get('title','结构求解报告')}</h1>
<p class="meta">生成时间 {r.get('generated_at','')} · 模型 {r.get('input',{}).get('geometry','')} ·
材料 {r.get('material',{}).get('name','')} (E={r.get('material',{}).get('E','')}, ν={r.get('material',{}).get('nu','')}) ·
单位 {unit.get('length','m')}/{unit.get('force','N')}/{unit.get('stress','Pa')} · 形变放大 {res.get('display_scale','')}×</p>
<h2>结果总览</h2>
<div class="cards">
<div class="card">载荷点位移 δ<b>{_fmt(fields.get('load_point_delta'), unit.get('length','m'))}</b></div>
<div class="card">静刚度 k = F/δ<b>{_fmt(fields.get('stiffness'), unit.get('stiffness','N/m'))}</b></div>
<div class="card">最大位移 |u|max<b>{_fmt(fields.get('max_displacement'), unit.get('length','m'))}</b></div>
<div class="card">最大 von Mises<b>{_fmt(fields.get('max_von_mises'), unit.get('stress','Pa'))}</b></div>
<div class="card">应变能<b>{_fmt(fields.get('strain_energy'), unit.get('energy','J'))}</b></div>
</div>
{viewer_html}
<h2>形变与场数据(静态图)</h2>
{img_def}
<h2>场数据汇总</h2>
<table>
<tr><th>指标</th><th>值</th></tr>
<tr><td>最大位移 |u|max</td><td>{_fmt(fields.get('max_displacement'), unit.get('length','m'))}</td></tr>
<tr><td>载荷点位移 δ(沿载荷方向)</td><td>{_fmt(fields.get('load_point_delta'), unit.get('length','m'))}</td></tr>
<tr><td>静刚度 k = |F|/δ</td><td>{_fmt(fields.get('stiffness'), unit.get('stiffness','N/m'))}</td></tr>
<tr><td>最大 von Mises 应力</td><td>{_fmt(fields.get('max_von_mises'), unit.get('stress','Pa'))}</td></tr>
<tr><td>总应变能 (0.5 uᵀKu)</td><td>{_fmt(fields.get('strain_energy'), unit.get('energy','J'))}</td></tr>
<tr><td>单元数 / 节点数</td><td>{r.get('input',{}).get('mesh_stats','')}</td></tr>
</table>
<h2>载荷与约束</h2>
<table><tr><th>固定面</th></tr>{fix_rows}</table>
<table><tr><th>载荷面</th><th>总力</th><th>施力节点数</th></tr>{load_rows}</table>
<h2>验证</h2>
<h3>解析基准(实测 vs 解析解)</h3>
<table><tr><th>基准</th><th>解析解</th><th>FEA 实测</th><th>相对误差</th><th>补充</th><th>判定</th></tr>{bench_rows}</table>
<h3>本次运行 Gates</h3>
<table><tr><th>Gate</th><th>实测值</th><th>阈值</th><th>判定</th></tr>{gate_rows}</table>
<h2>输出文件</h2>
<table><tr><th>角色</th><th>路径</th></tr>{out_rows}</table>
</body></html>"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
