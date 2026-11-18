import { solvePCG } from './gpu-solver.js';
const $ = id => document.getElementById(id);
let data, result;
const value = id => $(id).value;

function busy(state) {
  for (const input of document.querySelectorAll('form input, form select, form button')) input.disabled = state;
  $('download').disabled = state || !result?.converged;
  if (!state && value('geometry') !== 'box') {
    for (const id of ['length','width','height']) $(id).disabled = true;
    $('units').disabled = value('geometry') === 'hinge';
  }
}
async function run(prepare) {
  result = null; busy(true);
  const started = performance.now();
  try {
    if (prepare) {
      $('result').textContent = '正在本机准备模型、生成 T10 网格和有限元方程……';
      const geometry = value('geometry') === 'box' ? `box:${value('length')},${value('width')},${value('height')}` : value('geometry');
      const model = { geometry, units: value('units') };
      if (geometry === 'stl') {
        const file = $('stl').files[0];
        if (!file || file.size > 10*1024*1024) throw new Error('请选择不超过 10 MB 的封闭 STL');
        model.stl_base64 = await new Promise((resolve,reject) => {
          const reader = new FileReader(); reader.onload = () => resolve(reader.result.split(',')[1]); reader.onerror = reject; reader.readAsDataURL(file);
        });
      }
      const conditions = { material:value('material'), fixes:[value('fix')], loads:[] };
      if (['fx','fy','fz'].some(id => Number(value(id)) !== 0)) conditions.loads = [`${value('face')}: fx=${value('fx')},fy=${value('fy')},fz=${value('fz')}`];
      if ($('self-weight').checked) { conditions.gravity = [0,0,Number(value('gravity-z'))]; conditions.density = Number(value('density')); }
      const response = await fetch('./prepare', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
        model, conditions,
      }) });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || '模型准备失败');
      data = body;
    }
    if (!data?.rhs?.length) throw new Error('请先准备或导入方程');
    $('result').textContent = `方程已准备：${data.rhs.length} 自由度，正在求解……`;
    const time = performance.now();
    result = await solvePCG(data, value('backend'));
    let text = `${result.converged ? '已收敛' : '未收敛'}\n后端：${result.backend}\n自动单元：${data.metadata?.representation || `T${data.metadata?.order === 1 ? 4 : 10}`}\n自由度：${data.rhs.length}\n迭代次数：${result.iterations}\n真实相对残差：${result.relativeResidual.toExponential(4)}\n求解耗时：${((performance.now()-time)/1000).toFixed(3)} s\n本次总耗时：${((performance.now()-started)/1000).toFixed(3)} s`;
    if (data.reference?.length === result.solution.length) {
      let error = 0, norm = 0;
      for (let i = 0; i < data.reference.length; i++) { error += (result.solution[i]-data.reference[i])**2; norm += data.reference[i]**2; }
      if (norm > 0) text += `\n相对参考位移 L2 误差：${(Math.sqrt(error/norm)*100).toPrecision(4)}%`;
    }
    const gravity = data.metadata?.integrated_loads?.gravity;
    if (gravity) text += `\n质量：${gravity.mass_kg.toPrecision(6)} kg\n自重：[${gravity.total_force_n.map(v=>v.toPrecision(5)).join(', ')}] N`;
    if (result.fallbackReason) text += `\n已自动转为 CPU：${result.fallbackReason}`;
    $('result').textContent = text;
  } catch (error) { $('result').textContent = `失败：${error.message || error}`; }
  finally { busy(false); }
}
$('analysis').addEventListener('submit', event => { event.preventDefault(); run(true); });
$('prepared').addEventListener('click', () => run(false));
$('input').addEventListener('change', async () => {
  result = data = null; $('download').disabled = true;
  try { data = JSON.parse(await $('input').files[0].text()); $('result').textContent = `已载入方程：${data.rhs.length} 自由度`; }
  catch (error) { $('result').textContent = `导入失败：${error.message}`; }
});
$('geometry').addEventListener('change', () => {
  const hinge = value('geometry') === 'hinge';
  $('stl-field').hidden = value('geometry') !== 'stl';
  for (const id of ['length','width','height']) $(id).disabled = value('geometry') !== 'box';
  $('units').value = 'mm'; $('units').disabled = hinge;
});
$('download').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify({model:data.model,conditions:data.conditions,coordinates:data.coordinates,metadata:data.metadata,...result})], {type:'application/json'});
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href=url; link.download='structural-result.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url),1000);
});
