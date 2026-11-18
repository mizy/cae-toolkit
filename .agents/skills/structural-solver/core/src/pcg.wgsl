struct Size { n:u32, stride:u32, pad0:u32, pad1:u32 }
@group(0) @binding(0) var<storage, read> rows:array<u32>;
@group(0) @binding(1) var<storage, read> cols:array<u32>;
@group(0) @binding(2) var<storage, read> vals:array<f32>;
// packed x, r, p, A*p; scalar rr, alpha, beta, active, iterations
@group(0) @binding(3) var<storage, read_write> v:array<f32>;
@group(0) @binding(4) var<storage, read_write> s:array<f32>;
@group(0) @binding(5) var<uniform> size:Size;
var<workgroup> partial:array<f32,128>;

@compute @workgroup_size(128)
fn spmv(@builtin(global_invocation_id) id:vec3<u32>) {
 let i=id.x+id.y*size.stride;
 if i>=size.n || s[3]!=1.0 {return;}
 var sum=0.0;
 for(var j=rows[i];j<rows[i+1u];j++){sum+=vals[j]*v[2u*size.n+cols[j]];}
 v[3u*size.n+i]=sum;
}
@compute @workgroup_size(128)
fn alpha(@builtin(local_invocation_id) id:vec3<u32>) {
 let k=id.x;
 var sum=0.0;
 for(var i=k;i<size.n;i+=128u){sum+=v[2u*size.n+i]*v[3u*size.n+i];}
 partial[k]=sum;
 workgroupBarrier();
 for(var d=64u;d>0u;d/=2u){if k<d {partial[k]+=partial[k+d];} workgroupBarrier();}
 if k==0u && s[3]==1.0 {
   if partial[0]<=0.0 {s[3]=-1.0;} else {s[1]=s[0]/partial[0];s[4]+=1.0;}
 }
}
@compute @workgroup_size(128)
fn update(@builtin(global_invocation_id) id:vec3<u32>) {
 let i=id.x+id.y*size.stride;
 if i>=size.n || s[3]!=1.0 {return;}
 v[i]+=s[1]*v[2u*size.n+i];
 v[size.n+i]-=s[1]*v[3u*size.n+i];
}
@compute @workgroup_size(128)
fn beta(@builtin(local_invocation_id) id:vec3<u32>) {
 let k=id.x;var sum=0.0;
 for(var i=k;i<size.n;i+=128u){sum+=v[size.n+i]*v[size.n+i];}
 partial[k]=sum;workgroupBarrier();
 for(var d=64u;d>0u;d/=2u){if k<d {partial[k]+=partial[k+d];} workgroupBarrier();}
 if k==0u && s[3]==1.0 {s[2]=partial[0]/s[0];s[0]=partial[0];if partial[0]<1e-8 {s[3]=0.0;}}
}
@compute @workgroup_size(128)
fn direction(@builtin(global_invocation_id) id:vec3<u32>) {
 let i=id.x+id.y*size.stride;
 if i>=size.n || s[3]!=1.0 {return;}
 v[2u*size.n+i]=v[size.n+i]+s[2]*v[2u*size.n+i];
}
