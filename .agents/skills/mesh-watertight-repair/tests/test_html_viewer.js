// Exercise report controls with real vtk objects and synthetic data; no browser or WebGL.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
global.window = global;
global.self = global;
global.screen = { width: 1280, height: 800 };
global.location = { search: '' };
const element = () => ({ style: {}, setAttribute() {}, appendChild() {}, removeChild() {}, getContext() { return null; } });
global.document = { createElement: element, querySelector: element, createTextNode: () => ({}), documentElement: { style: {} } };
require(path.resolve(__dirname, '../assets/vtk-32.12.0/vtk.js'));
const renderers = [];
let renders = 0;
vtk.Rendering.Misc.vtkGenericRenderWindow.newInstance = () => {
  const renderer = vtk.Rendering.Core.vtkRenderer.newInstance();
  renderers.push(renderer);
  return { setContainer() {}, resize() {}, getRenderer: () => renderer, getRenderWindow: () => ({ render() { renders++; } }) };
};
const nodes = new Map();
function node(id) {
  const result = { value: '', checked: false, events: {}, addEventListener(event, callback) { this.events[event] = callback; } };
  nodes.set(id, result);
  return result;
}
node('vtk-sync-cameras');
for (let i = 0; i < 2; i++) {
  for (const name of ['viewer', 'reset', 'edges', 'smooth', 'opacity', 'field', 'field-range', 'cut-axis', 'cut-position']) node(`vtk-${name}-${i}`);
  nodes.get(`vtk-cut-position-${i}`).value = '0.5';
}
node('vtk-overlay-1');
node('vtk-source-normals-1');
for (const name of ['region', 'region-action', 'policy-export']) node(`vtk-${name}-0`);
const original = { points: [[0, 0, 0], [1, 0, 0], [0, 1, 0]], polys: [3, 0, 1, 2], triangles: 1, fields: {}, regions: [{ id: 'opening_0001', source_triangle_ids: [0], centroid: [0.3, 0.3, 0], bbox: { min: [0, 0, 0], max: [1, 1, 0] }, decision_template: { item_id: 'opening_0001', run_fingerprint: 'run', packet_fingerprint: 'packet', item_fingerprint: 'item' } }] };
const candidate = { ...original, normals: [[0, 1, 0], [0, 1, 0], [0, 1, 0]], geometry_normals: [[1, 0, 0], [1, 0, 0], [1, 0, 0]], regions: [], fields: { print_source_distance_mm: [0.2] } };
nodes.set('mesh-preview-data', { textContent: JSON.stringify([original, candidate]) });
document.getElementById = (id) => { assert(nodes.has(id), `Unknown control ${id}`); return nodes.get(id); };
global.ResizeObserver = class { constructor(fn) { this.fn = fn; } observe() { this.fn(); } };
vm.runInThisContext(fs.readFileSync(path.resolve(__dirname, '../scripts/html_mesh_viewer.js'), 'utf8'));
const actor = renderers[1].getActors()[0];
assert.deepEqual(Array.from(actor.getMapper().getInputData().getPoints().getData()), original.points.flat());
assert.deepEqual(Array.from(actor.getMapper().getInputData().getPolys().getData()), original.polys);
assert.equal(actor.getMapper().getInputData().getPointData().getNormals().getNumberOfTuples(), 3);
assert.equal(actor.getProperty().getInterpolation(), 2);
assert.deepEqual(Array.from(actor.getMapper().getInputData().getPointData().getNormals().getData()), candidate.normals.flat());
nodes.get('vtk-source-normals-1').events.change({ target: { checked: false } });
assert.deepEqual(Array.from(actor.getMapper().getInputData().getPointData().getNormals().getData()), candidate.geometry_normals.flat());
nodes.get('vtk-source-normals-1').events.change({ target: { checked: true } });

nodes.get('vtk-smooth-1').events.change({ target: { checked: false } });
assert.equal(actor.getProperty().getInterpolation(), 0);
nodes.get('vtk-smooth-1').events.change({ target: { checked: true } });
assert.equal(actor.getProperty().getInterpolation(), 2);
nodes.get('vtk-edges-1').events.change({ target: { checked: true } });
nodes.get('vtk-opacity-1').events.input({ target: { value: '0.4' } });
assert.equal(actor.getProperty().getEdgeVisibility(), true);
assert.equal(actor.getProperty().getOpacity(), 0.4);
nodes.get('vtk-field-1').events.change({ target: { value: 'print_source_distance_mm' } });
assert(Math.abs(actor.getMapper().getInputData().getCellData().getScalars().getData()[0] - 0.2) < 1e-6);
nodes.get('vtk-cut-axis-1').value = '0';
nodes.get('vtk-cut-position-1').value = '0.25';
nodes.get('vtk-cut-axis-1').events.change();
assert.equal(actor.getMapper().getClippingPlanes()[0].getOrigin()[0], 0.25);
nodes.get('vtk-overlay-1').events.change({ target: { checked: true } });
assert.equal(renderers[1].getActors()[1].getVisibility(), true);
nodes.get('vtk-sync-cameras').checked = true;
renderers[0].getActiveCamera().setPosition(3, 4, 5);
assert.deepEqual(renderers[1].getActiveCamera().getPosition(), [3, 4, 5]);
nodes.get('vtk-region-0').value = '0';
nodes.get('vtk-region-0').events.change();
assert.equal(renderers[0].getActors()[0].getMapper().getInputData().getCellData().getScalars().getData()[0], 1);
nodes.get('vtk-region-action-0').value = 'cap';
nodes.get('vtk-region-action-0').events.change();
assert.equal(nodes.get('vtk-policy-export-0').disabled, false);
let exported;
URL.createObjectURL = (blob) => { exported = blob; return 'blob:synthetic-test'; };
URL.revokeObjectURL = () => {};
document.createElement = () => ({ click() {} });
nodes.get('vtk-policy-export-0').events.click();
exported.text().then((text) => {
  const decision = JSON.parse(text).decisions[0];
  assert.equal(decision.decision, 'cap');
  assert.equal(decision.item_fingerprint, 'item');
  assert.equal(decision.packet_fingerprint, 'packet');
  assert(renders > 5);
  console.log('PASS: field colors, clipping, overlay, camera sync, region focus, and fingerprint-bound decision export (no WebGL rendering).');
}).catch((error) => { console.error(error); process.exitCode = 1; });
