// Experimental hybrid PCG: WebGPU computes CSR SpMV in f32, while the
// iteration and reductions run on the CPU. CSR must already include fixed BCs.
const shader = `
struct Params { n: u32, stride: u32, _pad0: u32, _pad1: u32 }
@group(0) @binding(0) var<storage, read> row: array<u32>;
@group(0) @binding(1) var<storage, read> col: array<u32>;
@group(0) @binding(2) var<storage, read> val: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * params.stride;
    if (i >= params.n) { return; }
    var sum = 0.0;
    for (var j = row[i]; j < row[i+1u]; j++) { sum += val[j] * x[col[j]]; }
    y[i] = sum;
}`;

/** Solve an SPD CSR system using a Jacobi-preconditioned conjugate gradient method. */
export async function solvePCG({rowPtr, colIdx, values, rhs, tolerance = 1e-5, maxIterations = 2000, preconditioner = 'jacobi', onProgress}) {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable in this browser.');
    const n = rhs.length;
    if (!n || rowPtr.length !== n + 1 || colIdx.length !== values.length || rowPtr[0] !== 0 ||
        rowPtr[n] !== values.length || !(tolerance > 0) || !Number.isInteger(maxIterations) || maxIterations < 1) {
        throw new Error('Invalid CSR matrix or solver settings.');
    }
    if (!['jacobi', 'block-jacobi'].includes(preconditioner) || (preconditioner === 'block-jacobi' && n % 3 !== 0)) {
        throw new Error('Block Jacobi requires three displacement DOFs per node.');
    }
    const diag = new Float64Array(n);
    const block = preconditioner === 'block-jacobi' ? new Float64Array(n * 3) : null;
    let bnorm2 = 0, hasLoad = false;
    for (let i = 0; i < n; i++) {
        if (!Number.isFinite(rhs[i]) || !Number.isInteger(rowPtr[i + 1]) || rowPtr[i] > rowPtr[i + 1] || rowPtr[i + 1] > values.length) throw new Error('Invalid CSR row or right-hand side.');
        bnorm2 += rhs[i] * rhs[i];
        hasLoad ||= rhs[i] !== 0;
        for (let j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
            if (!Number.isInteger(colIdx[j]) || colIdx[j] < 0 || colIdx[j] >= n || !Number.isFinite(values[j]) || !Number.isFinite(Math.fround(values[j])) || (values[j] !== 0 && Math.fround(values[j]) === 0)) throw new Error('Invalid CSR column or f32 value.');
            if (colIdx[j] === i) diag[i] += values[j];
            if (block && Math.floor(colIdx[j] / 3) === Math.floor(i / 3)) block[i * 3 + colIdx[j] % 3] += values[j];
        }
        if (!(diag[i] > 0) || !Number.isFinite(diag[i])) throw new Error('Non-positive or missing diagonal; check constraints and material.');
    }
    // Factor the nodal 3x3 diagonal blocks once; each is SPD for an SPD matrix.
    const factors = block ? new Float64Array(n * 2) : null;
    if (block) for (let i = 0; i < n; i += 3) {
        const k = i * 3, f = i * 2;
        const symmetryTolerance = 1e-8 * Math.max(block[k], block[k + 4], block[k + 8]);
        if (Math.abs(block[k + 1] - block[k + 3]) > symmetryTolerance ||
            Math.abs(block[k + 2] - block[k + 6]) > symmetryTolerance ||
            Math.abs(block[k + 5] - block[k + 7]) > symmetryTolerance) {
            throw new Error('Non-symmetric nodal block; check the matrix.');
        }
        const l00 = Math.sqrt(block[k]), l10 = block[k + 3] / l00, l20 = block[k + 6] / l00;
        const l11 = Math.sqrt(block[k + 4] - l10 * l10);
        const l21 = (block[k + 7] - l20 * l10) / l11;
        const l22 = Math.sqrt(block[k + 8] - l20 * l20 - l21 * l21);
        if (!(l00 > 0 && l11 > 0 && l22 > 0) || !Number.isFinite(l22)) throw new Error('Nodal block is not positive definite.');
        factors.set([l00, l10, l11, l20, l21, l22], f);
    }
    if (!Number.isFinite(bnorm2) || (bnorm2 === 0 && hasLoad)) throw new Error('Load magnitude is outside the supported numeric range.');
    const start = performance.now();
    if (bnorm2 === 0) return {solution: new Float64Array(n), iterations: 0, relativeResidual: 0, converged: true, timingMs: 0};
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter available.');
    const device = await adapter.requestDevice();
    const buffers = [];
    const upload = (data, usage) => {
        const buffer = device.createBuffer({size: Math.max(4, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST});
        buffers.push(buffer);
        device.queue.writeBuffer(buffer, 0, data);
        return buffer;
    };
    try {
        const bytes = Math.max((n + 1) * 4, values.length * 4, n * 4);
        if (bytes > device.limits.maxStorageBufferBindingSize || bytes > device.limits.maxBufferSize) {
            throw new Error('Matrix exceeds the WebGPU buffer limit for this device.');
        }
        const row = upload(new Uint32Array(rowPtr), GPUBufferUsage.STORAGE);
        const col = upload(new Uint32Array(colIdx), GPUBufferUsage.STORAGE);
        const val = upload(new Float32Array(values), GPUBufferUsage.STORAGE);
        const input = upload(new Float32Array(n), GPUBufferUsage.STORAGE);
        const output = device.createBuffer({size: n * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
        const read = device.createBuffer({size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
        buffers.push(output, read);
        const workgroups = Math.min(Math.ceil(n / 128), device.limits.maxComputeWorkgroupsPerDimension);
        const height = Math.ceil(n / (workgroups * 128));
        if (height > device.limits.maxComputeWorkgroupsPerDimension) throw new Error('Matrix has too many rows for one dispatch.');
        const params = upload(new Uint32Array([n, workgroups * 128, 0, 0]), GPUBufferUsage.UNIFORM);
        const pipeline = await device.createComputePipelineAsync({layout: 'auto', compute: {module: device.createShaderModule({code: shader}), entryPoint: 'main'}});
        const bind = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [row, col, val, input, output, params].map((buffer, binding) => ({binding, resource: {buffer}}))});
        const x = new Float64Array(n), r = Float64Array.from(rhs), z = new Float64Array(n), p = new Float64Array(n), p32 = new Float32Array(n);
        const precondition = () => {
            if (!factors) { for (let i = 0; i < n; i++) z[i] = r[i] / diag[i]; return; }
            for (let i = 0; i < n; i += 3) {
                const f = i * 2, a = factors[f], b = factors[f + 1], c = factors[f + 2], d = factors[f + 3], e = factors[f + 4], g = factors[f + 5];
                const y0 = r[i] / a, y1 = (r[i + 1] - b * y0) / c, y2 = (r[i + 2] - d * y0 - e * y1) / g;
                z[i + 2] = y2 / g;
                z[i + 1] = (y1 - e * z[i + 2]) / c;
                z[i] = (y0 - b * z[i + 1] - d * z[i + 2]) / a;
            }
        };
        const trueResidual = () => {
            let norm2 = 0;
            for (let i = 0; i < n; i++) {
                let ax = 0;
                for (let j = rowPtr[i]; j < rowPtr[i + 1]; j++) ax += values[j] * x[colIdx[j]];
                r[i] = rhs[i] - ax;
                norm2 += r[i] * r[i];
            }
            return Math.sqrt(norm2 / bnorm2);
        };
        let rz = 0;
        precondition();
        for (let i = 0; i < n; i++) { p[i] = z[i]; rz += r[i] * z[i]; }
        let iterations = 0, relativeResidual = 1, restarts = 0;
        while (iterations < maxIterations && relativeResidual > tolerance) {
            p32.set(p);
            device.queue.writeBuffer(input, 0, p32);
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(workgroups, height); pass.end();
            encoder.copyBufferToBuffer(output, 0, read, 0, n * 4);
            device.queue.submit([encoder.finish()]);
            await read.mapAsync(GPUMapMode.READ);
            const ap = new Float32Array(read.getMappedRange());
            let pap = 0;
            for (let i = 0; i < n; i++) pap += p[i] * ap[i];
            if (!(pap > 0) || !Number.isFinite(pap)) { read.unmap(); throw new Error('PCG breakdown: matrix may not be positive definite or single precision is insufficient.'); }
            const alpha = rz / pap;
            let nextRz = 0, norm2 = 0;
            for (let i = 0; i < n; i++) {
                x[i] += alpha * p[i]; r[i] -= alpha * ap[i]; norm2 += r[i] * r[i];
            }
            read.unmap();
            precondition();
            for (let i = 0; i < n; i++) nextRz += r[i] * z[i];
            iterations++;
            relativeResidual = Math.sqrt(norm2 / bnorm2);
            if (relativeResidual <= tolerance) {
                relativeResidual = trueResidual();
                if (onProgress) onProgress(iterations, relativeResidual);
                if (relativeResidual <= tolerance) break;
                // Recursive residual drifted: restart PCG from the actual f64 residual.
                rz = 0;
                precondition();
                for (let i = 0; i < n; i++) { p[i] = z[i]; rz += r[i] * z[i]; }
                restarts++;
                continue;
            }
            if (onProgress && iterations % 10 === 0) onProgress(iterations, relativeResidual);
            const beta = nextRz / rz;
            for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
            rz = nextRz;
        }
        relativeResidual = trueResidual();
        return {solution: x, iterations, restarts, relativeResidual, converged: relativeResidual <= tolerance, timingMs: performance.now() - start};
    } finally {
        for (const buffer of buffers) buffer.destroy();
        device.destroy();
    }
}
