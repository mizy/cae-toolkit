#!/usr/bin/env python3
"""Reproducible STL bearing/pressure and gravity validation on synthetic models."""
import argparse
import base64
import html
import json
from pathlib import Path
import subprocess
import time

import gmsh
import numpy as np
import trimesh
from assembly import MATERIALS, assemble, elasticity_D
from boundary import face_nodes
from export_web_case import export_case
from loads import cylinder_load,gravity_load,boundary_faces
from mesh_tet import build_tet_mesh
from stl_volume import mesh_stl

ROOT=Path(__file__).resolve().parents[1]


def make_plate(path):
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal',0)
        box=gmsh.model.occ.addBox(0,0,0,.08,.04,.008)
        bore=gmsh.model.occ.addCylinder(.05,.02,0,0,0,.008,.006)
        gmsh.model.occ.cut([(3,box)],[(3,bore)])
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber('Mesh.MeshSizeMin',.005)
        gmsh.option.setNumber('Mesh.MeshSizeMax',.005)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    surface=trimesh.load_mesh(path,process=True)
    # Gmsh surface export can leave inconsistent winding; orient this synthetic
    # single connected closed solid before it becomes the STL input fixture.
    surface.fix_normals()
    surface.export(path)
    return surface


def check_case(name, mesh, output, *, cylinder=None, gravity=None, density=None):
    path=output/f'{name}.json'
    meta=export_case(name,'m',.005,2,'steel','x_min',[],path,mesh=mesh,
                     cylinder=cylinder,gravity=gravity,density=density)
    case=json.loads(path.read_text())
    rhs=np.zeros(3*len(mesh.nodes))
    if cylinder: rhs+=cylinder_load(mesh.nodes,mesh.elements,2,cylinder)[0]
    if gravity is not None: rhs+=gravity_load(mesh.nodes,mesh.elements,2,density,gravity)[0]
    mat=MATERIALS['steel']
    K,_=assemble(mesh.nodes,mesh.elements,elasticity_D(mat.E,mat.nu),2)
    fixed=face_nodes(mesh.nodes,'x_min')
    dofs=(3*fixed[:,None]+np.arange(3)).ravel()
    results={}
    binary=ROOT/'core/target/release/structural-equations'
    for backend in ['cpu','gpu']:
        started=time.perf_counter()
        run=subprocess.run([str(binary),str(path),backend],capture_output=True,text=True,timeout=180)
        if not run.stdout:
            results[backend]={'error':run.stderr,'passed':False}
            continue
        result=json.loads(run.stdout)
        u=np.asarray(result['solution'])
        reaction=(K@u-rhs)[dofs].reshape(-1,3).sum(axis=0)
        total=rhs.reshape(-1,3).sum(axis=0)
        scale=np.linalg.norm(rhs.reshape(-1,3),axis=1).sum()
        balance=float(np.linalg.norm(reaction+total)/scale)
        ref=np.asarray(case['reference'])
        error=float(np.linalg.norm(u-ref)/np.linalg.norm(ref))
        tip=face_nodes(mesh.nodes,'x_max')
        results[backend]={'backend':result['backend'],'converged':result['converged'],
            'iterations':result['iterations'],'residual':result['relativeResidual'],
            'seconds':time.perf_counter()-started,'displacement_error':error,
            'reaction_n':reaction.tolist(),'force_balance':balance,'mean_tip_z_m':float(u.reshape(-1,3)[tip,2].mean()),
            'passed':bool(result['converged'] and balance<1e-3 and error<0.01)}
    return {'metadata':meta,'results':results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    surface=make_plate(out/'plate-with-hole.stl')
    mesh=mesh_stl(out/'plate-with-hole.stl','m',.005)
    cylinder=dict(type='bearing',center=[.05,.02,.004],axis=[0,0,1],radius=.006,tolerance=.0002,force=[1000,0,0])
    report={'input_kind':'synthetic STL','output_kind':'T10 volume + CSR',
        'geometry_policy':'Preserve the through bore and source surface facets; no hole filling',
        'source':{'watertight':bool(surface.is_watertight),'consistent_winding':bool(surface.is_winding_consistent),
                  'components':len(surface.split()),'volume_m3':float(surface.volume),'faces':len(surface.faces)}}
    faces,_=boundary_faces(mesh.elements,2)
    remeshed=trimesh.Trimesh(mesh.nodes,faces[:,:3],process=True)
    remeshed.remove_unreferenced_vertices()
    volume=gravity_load(mesh.nodes,mesh.elements,2,1,[0,0,0])[1]['volume_m3']
    counts=np.bincount(remeshed.edges_unique_inverse)
    report['volume_mesh']={'nodes':len(mesh.nodes),'elements':len(mesh.elements),
        'boundary_edges':int(np.sum(counts==1)),'nonmanifold_edges':int(np.sum(counts>2)),
        'surface_euler_number':int(remeshed.euler_number),'source_euler_number':int(surface.euler_number),
        'volume_m3':volume,'relative_volume_change':abs(volume-surface.volume)/surface.volume}
    report['bearing']=check_case('bearing',mesh,out,cylinder=cylinder)
    cylinder={**cylinder,'type':'pressure','pressure_pa':1e6}
    report['uniform_bore_pressure']=check_case('pressure',mesh,out,cylinder=cylinder)
    beam=build_tet_mesh('box:0.1,0.01,0.01',.005,2,'m')
    report['self_weight']=check_case('self-weight',beam,out,gravity=[0,0,-9.80665],density=7850)
    E=MATERIALS['steel'].E;I=.01*.01**3/12
    exact=7850*.01*.01*9.80665*.1**4/(8*E*I)
    reference=json.loads((out/'self-weight.json').read_text())
    u=np.asarray(reference['reference']).reshape(-1,3)
    observed=-u[face_nodes(beam.nodes,'x_max'),2].mean()
    report['self_weight']['euler_bernoulli']={'tip_m':exact,'reference_tip_m':observed,
        'relative_error':float(abs(observed-exact)/exact),'passed':bool(abs(observed-exact)/exact<.05)}
    pressure_force=np.asarray(report['uniform_bore_pressure']['metadata']['integrated_loads']['total_force_n'])
    report['uniform_bore_pressure']['zero_resultant_passed']=bool(np.linalg.norm(pressure_force)<1e-6)
    report['stl_preservation_passed']=bool(remeshed.is_watertight and remeshed.euler_number==surface.euler_number and abs(volume-surface.volume)/surface.volume<1e-6)
    report['passed']=report['stl_preservation_passed'] and report['uniform_bore_pressure']['zero_resultant_passed'] and all(result['passed'] for name in ['bearing','uniform_bore_pressure','self_weight'] for result in report[name]['results'].values()) and report['self_weight']['euler_bernoulli']['passed']
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    write_report(report,mesh,beam,out)
    print(json.dumps(report,indent=2))


def write_report(report,mesh,beam,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(12,4.8))
    ax=fig.add_subplot(121,projection='3d')
    faces,_=boundary_faces(mesh.elements,2)
    xyz=mesh.nodes[faces[:,:3]]*1000
    ax.add_collection3d(Poly3DCollection(xyz,facecolor='#b9d4e6',edgecolor='#506478',linewidth=.15,alpha=.22))
    spec=dict(type='bearing',center=[.05,.02,.004],axis=[0,0,1],radius=.006,tolerance=.0002,force=[1000,0,0])
    force,info=cylinder_load(mesh.nodes,mesh.elements,2,spec)
    idx=np.flatnonzero(np.linalg.norm(force.reshape(-1,3),axis=1)>0)
    points=mesh.nodes[idx]*1000; directions=force.reshape(-1,3)[idx]
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    ax.quiver(*points.T,*directions.T,length=4,color='#c53b32',linewidth=.6)
    fixed=mesh.nodes[face_nodes(mesh.nodes,'x_min')]*1000
    ax.scatter(*fixed.T,color='#1766ad',s=4)
    ax.set(xlim=(0,80),ylim=(0,40),zlim=(0,8),xlabel='X / mm',ylabel='Y / mm',zlabel='Z / mm',title='STL bore: 1000 N bearing load; fixed left end')
    ax.set_box_aspect((80,40,20));ax.view_init(35,-65)
    ax=fig.add_subplot(122)
    x=np.linspace(0,.1,150);q=7850*.01*.01*9.80665;EI=210e9*.01*.01**3/12
    deflection=q*x*x*(6*.1*.1-4*.1*x+x*x)/(24*EI)
    case=json.loads((out/'self-weight.json').read_text());u=np.asarray(case['reference']).reshape(-1,3)
    centers=np.linspace(0,.1,12);values=[]
    for x0 in centers:
        keep=abs(beam.nodes[:,0]-x0)<.002
        values.append(-u[keep,2].mean() if keep.any() else np.nan)
    ax.plot(x*1000,deflection*1e6,label='Euler-Bernoulli uniform gravity')
    ax.plot(centers*1000,np.asarray(values)*1e6,'o',label='FEM section mean')
    ax.set(xlabel='X / mm',ylabel='Downward displacement / micrometers',title='Steel cantilever under self weight')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout(w_pad=4);image=out/'validation.png';fig.savefig(image,dpi=160);plt.close(fig)
    rows=''.join(f"<tr><td>{name}</td><td>{backend}</td><td>{r['converged']}</td><td>{r['residual']:.3e}</td><td>{100*r['displacement_error']:.5g}%</td><td>{r['force_balance']:.3e}</td></tr>" for name in ['bearing','uniform_bore_pressure','self_weight'] for backend,r in report[name]['results'].items() if 'error' not in r)
    encoded=base64.b64encode(image.read_bytes()).decode()
    document=f"""<!doctype html><meta charset='utf-8'><title>STL and gravity validation</title>
<style>body{{font:16px/1.6 system-ui;margin:2rem auto;max-width:1100px;padding:0 20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ccc;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>STL 孔壁载荷与自重验证</h1><p>合成板件：80×40×8 mm，孔径12 mm；左端固定。钢 E=210 GPa、ν=0.3。自重梁：100×10×10 mm，ρ=7850 kg/m³，g=9.80665 m/s²。</p>
<p>总检查：{'PASS' if report['passed'] else 'FAIL'}。STL 通孔与原表面保留。已知圆柱参数选区，未测试任意模型自动识孔、交互拾取、摩擦、预紧或接触。</p>
<img style='width:100%' src='data:image/png;base64,{encoded}' alt='STL bearing region and gravity beam comparison'>
<table><tr><th>算例</th><th>后端</th><th>收敛</th><th>真实残差</th><th>相对 SciPy 位移误差</th><th>合力平衡误差</th></tr>{rows}</table>
<p>线性求解器比较使用同一有限元矩阵。自重另与梁解析解对照；孔边峰值应力及网格独立性尚未验证。</p><details><summary>原始数值与几何检查</summary><pre>{html.escape(json.dumps(report,indent=2))}</pre></details>"""
    (out/'validation.html').write_text(document)
    (out/'browser-check.html').write_text("""<!doctype html><meta charset='utf-8'><title>Load validation</title><h1>WebGPU load validation</h1><pre id='result'>Running…</pre><script type='module'>
import {solvePCG} from '../../web/gpu-solver.js';
const results=[];const output=document.querySelector('#result');
try {for(const name of ['bearing','pressure','self-weight']) {
 const response=await fetch(`./${name}.json`);const data=await response.json();
 const start=performance.now();const r=await solvePCG(data,'gpu');let diff=0,norm=0;
 for(let i=0;i<r.solution.length;i++){diff+=(r.solution[i]-data.reference[i])**2;norm+=data.reference[i]**2;}
 const error=Math.sqrt(diff/norm);
 results.push({name,backend:r.backend,iterations:r.iterations,residual:r.relativeResidual,displacementError:error,seconds:(performance.now()-start)/1000,passed:r.converged&&error<.01});
 output.textContent=JSON.stringify(results,null,2);
} output.textContent+='\\nCOMPLETE';} catch(e){output.textContent+='\\nERROR '+e;}
</script>""")


if __name__=='__main__':main()
