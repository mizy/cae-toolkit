use crate::{norm, LinearSystem, Solution};
use wgpu::util::DeviceExt;

pub async fn solve(
    a: &LinearSystem,
    diagonal: &[f64],
    tolerance: f64,
    max: usize,
) -> Result<Solution, String> {
    let n = a.rhs.len();
    let bnorm = norm(&a.rhs);
    if bnorm == 0.0 || !bnorm.is_finite() {
        return Err("GPU solve requires a finite nonzero load".into());
    }
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map_err(|e| format!("GPU adapter unavailable: {e}"))?;
    let name = format!("wgpu-{:?}", adapter.get_info().backend);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .map_err(|e| e.to_string())?;
    let limit = device.limits().max_storage_buffer_binding_size as usize;
    if n * 16 > limit || a.values.len() * 4 > limit || a.row_ptr.len() * 4 > limit {
        return Err("Matrix exceeds device buffer limits".into());
    }
    let scale: Vec<f64> = diagonal.iter().map(|d| 1.0 / d.sqrt()).collect();
    let mut values = Vec::with_capacity(a.values.len());
    for i in 0..n {
        for j in a.row_ptr[i] as usize..a.row_ptr[i + 1] as usize {
            let v = a.values[j] * scale[i] * scale[a.col_idx[j] as usize];
            let f = v as f32;
            if !f.is_finite() || (v != 0.0 && f == 0.0) {
                return Err("Matrix cannot be represented by GPU f32".into());
            }
            values.push(f);
        }
    }
    let storage = wgpu::BufferUsages::STORAGE;
    let upload = |label, data: &[u8], usage| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage,
        })
    };
    let rows = upload("rows", bytemuck::cast_slice(&a.row_ptr), storage);
    let cols = upload("cols", bytemuck::cast_slice(&a.col_idx), storage);
    let vals = upload("values", bytemuck::cast_slice(&values), storage);
    let vectors = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PCG vectors"),
        size: (n * 16) as u64,
        usage: storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let scalars = upload(
        "PCG scalars",
        bytemuck::cast_slice(&[1f32, 0., 0., 1., 0.]),
        storage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    let width = (n as u32).div_ceil(128).min(65535);
    let height = (n as u32).div_ceil(width * 128);
    let params = upload(
        "size",
        bytemuck::cast_slice(&[n as u32, width * 128, 0, 0]),
        wgpu::BufferUsages::UNIFORM,
    );
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("batch readback"),
        size: (n * 4 + 20) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shared PCG"),
        source: wgpu::ShaderSource::Wgsl(include_str!("pcg.wgsl").into()),
    });
    // All entry points share an explicit bind layout, including unused bindings.
    let entries: Vec<_> = (0..6)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if i == 5 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage { read_only: i < 3 }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipelines: Vec<_> = ["spmv", "alpha", "update", "beta", "direction"]
        .iter()
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        })
        .collect();
    if let Some(error) = errors.pop().await {
        return Err(error.to_string());
    }
    let entries: Vec<_> = [&rows, &cols, &vals, &vectors, &scalars, &params]
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &entries,
    });
    let mut x = vec![0.0; n];
    let mut iterations = 0;
    let mut residual = 1.0;
    while iterations < max && residual > tolerance {
        let r = a.residual(&x);
        let scaled: Vec<f64> = r.iter().zip(&scale).map(|(r, s)| r * s).collect();
        let factor = norm(&scaled);
        if !factor.is_finite() || factor == 0.0 {
            return Err("Invalid correction magnitude".into());
        }
        let mut state = vec![0f32; 4 * n];
        for i in 0..n {
            state[n + i] = (scaled[i] / factor) as f32;
            state[2 * n + i] = state[n + i];
        }
        queue.write_buffer(&vectors, 0, bytemuck::cast_slice(&state));
        queue.write_buffer(&scalars, 0, bytemuck::cast_slice(&[1f32, 0., 0., 1., 0.]));
        let batch = (max - iterations).min(512);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &bind, &[]);
            for _ in 0..batch {
                for (i, pipeline) in pipelines.iter().enumerate() {
                    pass.set_pipeline(pipeline);
                    if i == 1 || i == 3 {
                        pass.dispatch_workgroups(1, 1, 1);
                    } else {
                        pass.dispatch_workgroups(width, height, 1);
                    }
                }
            }
        }
        encoder.copy_buffer_to_buffer(&vectors, 0, &read, 0, (n * 4) as u64);
        encoder.copy_buffer_to_buffer(&scalars, 0, &read, (n * 4) as u64, 20);
        queue.submit(Some(encoder.finish()));
        let slice = read.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        #[cfg(not(target_arch = "wasm32"))]
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.await
            .map_err(|e| e.to_string())?
            .map_err(|e| e.to_string())?;
        let mapped = slice.get_mapped_range().map_err(|e| e.to_string())?;
        let data: &[f32] = bytemuck::cast_slice(&mapped);
        let valid = data[..n].iter().all(|v| v.is_finite())
            && data[n + 3] >= 0.0
            && data[n + 4].is_finite()
            && data[n + 4] > 0.0;
        if !valid {
            return Err("GPU PCG breakdown; try CPU f64".into());
        }
        for i in 0..n {
            x[i] += data[i] as f64 * factor * scale[i];
        }
        iterations += data[n + 4] as usize;
        drop(mapped);
        read.unmap();
        residual = norm(&a.residual(&x)) / bnorm;
        if !residual.is_finite() {
            return Err("Nonfinite true residual".into());
        }
    }
    Ok(Solution {
        solution: x,
        iterations,
        relative_residual: residual,
        converged: residual <= tolerance,
        backend: name,
        fallback_reason: None,
    })
}
