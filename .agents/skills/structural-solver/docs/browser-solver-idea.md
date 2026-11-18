# 浏览器端结构求解器构想

记录日期：2026-11-06。

## 目标

探索使用 Rust 和 WebGPU，在用户设备上完成线性静力分析。长期工作流为导入体网格、设置材料与约束/载荷、求解、查看位移及应力，并导出结果。原生计算仍可用于本机运行及精度对照。

第一阶段先验证真实 FEM 线性方程在浏览器中的求解，不将简单 GPU 算子跑分视为结构求解器性能。优先选择已有求解器生成的约束后稀疏系统，对比浏览器 WebGPU 与原生双精度参考解的位移、残差和时间；检查不收敛、约束不足、输入不合法等失败情况。浏览器中的 GPU 浮点精度、预条件及收敛性需要单独验证。

第一阶段不承诺任意 CAD 输入、压力载荷、接触、塑性或生产级应力精度。现有外部网格输入、载荷分配和应力验证的限制也需要在扩展完整流程前处理。

## 初步性能线索

2026-11-06 在 Apple M2 Pro 上用相同的 f32 邻点计算、每轮 20 次分派及一次结果读回，分别测试 Swift 直接 Metal、原生 Rust/wgpu 和 Chrome WebGPU。每组以最后三轮中位数计时，单位 ms：

| 数组元素 | 直接 Metal | Rust/wgpu | Chrome WebGPU |
|---:|---:|---:|---:|
| 1,000,000 | 1.42 | 2.59 | 1.70 |
| 10,000,000 | 10.99 | 10.11 | 13.70 |
| 30,000,000 | 32.51 | 30.30 | 33.20 |

三种路径先后运行，样本少，内存分配、命令编码和同步细节不完全一致；仅抽查首、中、末输出。Chrome 未提供可读的适配器名称。这个规则访问微基准没有稀疏矩阵、归约、预条件、收敛和网格组装，不能作为 FEM 求解速度或精度结论。首次运行包含额外准备开销，实际产品应分别记录冷启动、重复运行和端到端时间。

## 验证方向

1. 使用相同的真实 FEM 矩阵、右端项和边界条件，比较浏览器与原生结果。
2. 报告残差、迭代次数、位移误差和耗时，覆盖拉伸杆、悬臂梁及更大网格。
3. 比较 f32 的可用精度、预条件效果、设备限制和不收敛表现，必要时保留 CPU f64 路径。
4. 在数值可靠性建立后，再扩展模型导入、交互式边界条件、云图及导出。

## 2026-11-11 architecture update

Inputs are split into model (geometry, units) and conditions (material,
constraints, loads). Automatic initial T10 meshing and equation preparation
currently support the existing box/one-body hinge models. The shared Rust
`core/` now supplies both native wgpu and WASM/WebGPU equation solves. GPU vectors,
dot products and updates stay resident for batches; Rust checks the original f64
residual and applies corrections. The browser has no separate JS solver.
Native FEM assembly/stress recovery and faer CPU LU remain in `native/`;
Python owns current meshing, equation export and reporting.

Future work: migrate remaining FEM operations where justified, add supported
model formats and mesh convergence, then separately validate shell/beam elements
and geometry extraction. Automatic midsurface extraction is not available.
