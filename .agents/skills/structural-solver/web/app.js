import { solvePCG } from './gpu-solver.js';

const file = document.querySelector('#input');
const solve = document.querySelector('#solve');
const download = document.querySelector('#download');
const output = document.querySelector('#result');
let data, result;

function loadCase(value, name) {
  if (!Array.isArray(value.rowPtr) || !Array.isArray(value.colIdx) ||
      !Array.isArray(value.values) || !Array.isArray(value.rhs)) throw new Error('缺少 CSR 方程数据');
  data = value; result = null; download.disabled = true; solve.disabled = false;
  const meta = data.metadata || {};
  const forces = (meta.loads || []).map(load => `${load.face}: [${load.force_total.join(', ')}] N`).join('；');
  output.textContent = `已载入 ${name}\n几何：${meta.geometry || '未提供'}\n材料：${meta.material || '未提供'}\n固定：${meta.fixes?.join('、') || meta.fix || '未提供'}\n载荷：${forces || '未提供'}\n自由度：${data.rhs.length}\n矩阵非零项：${data.values.length}`;
}
document.querySelector('#example').addEventListener('click', async () => {
  try {
    const select = document.querySelector('#case');
    const response = await fetch(`./${select.value}`);
    if (!response.ok) throw new Error(`请先按照使用说明生成 ${select.value}`);
    loadCase(await response.json(), select.selectedOptions[0].textContent);
  } catch (error) { output.textContent = `无法载入示例：${error.message}`; }
});

file.addEventListener('change', async () => {
  data = result = null;
  solve.disabled = download.disabled = true;
  if (!file.files.length) return;
  try {
    loadCase(JSON.parse(await file.files[0].text()), file.files[0].name);
  } catch (error) { output.textContent = `无法读取：${error.message}`; }
});

solve.addEventListener('click', async () => {
  solve.disabled = true;
  file.disabled = true;
  document.querySelector('#example').disabled = true;
  document.querySelector('#case').disabled = true;
  download.disabled = true;
  output.textContent = '正在初始化 GPU 并求解，较大的模型可能需要一些时间……';
  await new Promise(requestAnimationFrame);
  try {
    const tolerance = Number(document.querySelector('#tolerance').value);
    const maxIterations = Number(document.querySelector('#iterations').value);
    if (!(tolerance >= 1e-7 && tolerance <= 0.1) || !Number.isInteger(maxIterations) || maxIterations < 1 || maxIterations > 10000) throw new Error('请检查残差目标和迭代次数');
    const preconditioner = document.querySelector('#preconditioner').value;
    result = await solvePCG({ ...data, tolerance, maxIterations, preconditioner, onProgress: (iteration, residual) => {
      output.textContent = `正在求解：第 ${iteration} 次迭代，递推相对残差 ${residual.toExponential(3)}…`;
    } });
    let text = `${result.converged ? '已收敛' : '未收敛'}\n迭代次数：${result.iterations}\n真实相对残差：${result.relativeResidual.toExponential(4)}\n求解时间：${(result.timingMs / 1000).toFixed(3)} s`;
    if (data.metadata?.displacement_unit === 'm' && result.solution.length % 3 === 0) {
      let max = 0;
      for (let i = 0; i < result.solution.length; i += 3) max = Math.max(max, Math.hypot(result.solution[i], result.solution[i+1], result.solution[i+2]));
      text += `\n最大位移：${max.toExponential(5)} m`;
    }
    if (Array.isArray(data.reference) && data.reference.length === result.solution.length) {
      let diff = 0, norm = 0;
      for (let i = 0; i < data.reference.length; i++) { diff += (result.solution[i] - data.reference[i]) ** 2; norm += data.reference[i] ** 2; }
      if (norm > 0) text += `\n相对参考位移 L2 误差：${(Math.sqrt(diff / norm) * 100).toPrecision(4)}%`;
    }
    output.textContent = text;
    download.disabled = !result.converged;
  } catch (error) { output.textContent = `求解失败：${error.message}`; }
  finally { solve.disabled = false; file.disabled = false; document.querySelector('#example').disabled = false; document.querySelector('#case').disabled = false; }
});

download.addEventListener('click', () => {
  const blob = new Blob([JSON.stringify({ coordinates: data.coordinates, metadata: data.metadata, displacement: Array.from(result.solution), iterations: result.iterations, relativeResidual: result.relativeResidual })], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url; link.download = 'displacement.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
