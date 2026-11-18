// Both browser and native entry points call structural-core; no JS solver copy.
import init, { solve_json } from './pkg/structural_core.js';
let initialized;
export async function solvePCG(system, backend = 'auto') {
  initialized ||= init();
  await initialized;
  return JSON.parse(await solve_json(JSON.stringify(system), backend));
}
