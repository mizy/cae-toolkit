# 结构求解器(structural-solver)落地计划

## 1. 目标

在现有 CAD 网格仓库内新增独立 skill `structural-solver`:

- 输入:水密表面网格(VTP,来自 `mesh-watertight-repair` / `surface-tessellation`)
- 求解:线性静力结构分析(静刚度),各向同性弹性材料
- 输出:
  - **形变结果**:变形后的网格 **VTP**(几何已按位移场更新,场数据以点数组内嵌)
  - **场数据**:位移矢量/幅值、von Mises 应力、应变能密度
  - **静刚度**:k = F / δ(载荷点沿载荷方向的线位移刚度)
  - **最终 HTML 报告**:内嵌"形变前后对比图 + 场数据表 + 实测基准对比",人眼可直接确认是否发生形变
  - **JSON 报告**:权威数据,HTML 只是同一事实的可读渲染(沿用 `mesh-watertight-repair/report_contract.md` 的约定)

## 2. 范围与边界(v1)

| 做 | 不做(v1) |
|---|---|
| 线性小变形弹性静力 | 大变形 / 几何非线性 |
| 各向同性材料(E、ν) | 各向异性 / 超弹性 / 塑性 |
| 固定约束 + 力/压力载荷 | 接触、热-结构耦合、模态 |
| T4/T10 四面体体积网格(gmsh) | 壳单元、梁单元、六面体 |
| 静刚度(k = F/δ)与应变能 | 拓扑优化、疲劳、屈曲 |

- **单位约定**:默认 米/N/Pa;`--units mm` 切到 毫米/N/MPa。轴约定沿用仓库惯例(X 长、Y 宽、Z 高,顶部 +Z)。
- 求解失败或网格质量差时**给出可读警告/拒绝**,绝不静默输出错误结果。

## 3. 总体流程

```text
水密表面 VTP ──► gmsh 四面体体积网格(T4/T10) ──► 刚度矩阵组装 K
      │                                                  │
      └── 约束/载荷定义(fix 面、力/压力) ──► scipy 稀疏求解 K u = f
                                                        │
                  ┌─────────────────────────────────────┘
                  ▼
      后处理:位移场 / von Mises / 应变能 / 反力校验 / 静刚度
                  │
                  ▼
  输出:deformed_mesh.vtp + mesh.vtp + report.json + report.html(内嵌形变图)
```

## 4. Skill 目录结构

```text
.agents/skills/structural-solver/
├── SKILL.md
├── docs/
│   └── plan.md                 # 本文档
├── scripts/
│   ├── solve.py                # 主入口: mesh → 求解 → 报告/渲染
│   ├── mesh_tet.py             # 水密表面 → gmsh 四面体
│   ├── assembly.py             # T4/T10 单元刚度、整体组装
│   ├── boundary.py             # 约束/载荷解析(轴对齐面、半径选点、命名区域)
│   ├── postprocess.py          # 位移/应力/刚度提取 + pyvista 渲染
│   └── report.py               # report.json + report.html 生成
├── tests/
│   ├── test_tension_bar.py     # 拉伸杆基准
│   ├── test_cantilever.py      # 悬臂梁基准
│   └── test_convergence.py     # 网格收敛研究
└── references/
    └── units_and_formulas.md   # 单位、公式、材料表(E=210 GPa 钢等)
```

依赖复用仓库现有 `requirements.txt`(gmsh、scipy、numpy、pyvista、vtk),不新增重型依赖。

## 5. 实测基准(Validation,验收核心)

每个基准都跑真实求解,把"实测值 vs 解析解 vs 相对误差"写进 HTML 报告。材料默认钢 E=210 GPa、ν=0.3。

### 5.1 拉伸杆(Tension Bar)—— P1 验收,误差 < 1%

- 模型:1 m × 0.1 m × 0.1 m 长方体,一端固定,另一端轴向力 F = 1000 N
- 解析解:δ = FL/(EA) = 1000 × 1 / (210e9 × 0.01) ≈ **4.762e-7 m**;σ = F/A = 100 kPa
- 验收:位移与应力相对误差 < 1%(T10),反力平衡 |ΣR − F|/F < 0.5%

### 5.2 悬臂梁(Cantilever Beam)—— P2 验收,误差 < 5%

- 模型:1 m × 0.1 m × 0.1 m,一端固定,另一端横向力 F = 1000 N
- 解析解:δ = FL³/(3EI),I = bh³/12 = 8.333e-6 m⁴ → δ ≈ **1.905e-4 m**;k = 3EI/L³ ≈ **5.25e6 N/m**
- 验收:挠度相对误差 < 5%(T10 细网格收敛后),刚度 k 偏差同量级

### 5.3 收敛性研究(实测网格依赖)

- 同一模型按 2×/4×/8× 加密四面体,记录 δ、σ、误差随网格尺寸的变化
- 结论写入报告:给出"达到 <1% 误差所需的大致网格规模",防止用户拿粗网格结果当精确值

### 5.4 实测数据呈现

HTML 报告"验证"章节必须包含表格:

| 基准 | 解析解 δ | FEA 实测 δ | 相对误差 | 反力平衡 | 判定 |
|---|---|---|---|---|---|
| 拉伸杆 | 4.762e-7 m | <1% 内实测 | ... | <0.5% | PASS/FAIL |

## 6. 输出契约

### 6.1 VTP 结果(核心要求)

- **`deformed_mesh.vtp`**:几何为**位移后的实际坐标**(Undeformed → displaced),点数组:
  - `displacement`(float64 vector3,米/毫米)
  - `displacement_magnitude`(float64)
  - `von_mises_stress`(float64,Pa/MPa)
  - `strain_energy_density`(float64)
- **`mesh.vtp`**:原始几何 + 同一套场数据(便于 ParaView 对比、查看形变放大)
- **`mesh_tet.msh`**:gmsh 四面体网格(中间产物,诊断用)
- 所有 VTP 用 `pyvista.save_mesh_file` 写出,读回校验拓扑与场数组完整(读回审计,沿用仓库 gate 风格)

### 6.2 `report.json`(权威数据)

顶层章节:`decision`、`input`、`material`、`units`、`load_cases`、`results`(δ、k、σ_vm_max、应变能)、`validation`(实测 vs 解析表)、`gates`(每个 gate 含 `passed`/实测值/阈值/失败原因)、`outputs`(各文件路径)、`runtime`(仅诊断)。

### 6.3 `report.html`(最终报告,人读确认)

必须包含:

1. **形变对比图**(内嵌 base64 PNG):原始网格半透明线框 + 按位移场 warp(默认 20× 放大,`--scale` 可调)的彩色形变网格,同一视角并排/叠加;
2. **场数据图**:位移幅值云图 + von Mises 应力云图;
3. **场数据表**:max|u|、载荷点位移 δ、方向、von Mises max、应变能、**静刚度 k = F/δ**;
4. **载荷与约束清单**:固定面、力的位置/方向/大小;
5. **实测验证表**(见 §5.4);
6. 运行参数、单位、网格规模、求解器信息。

报告自包含(图片内嵌),可直接 `open report.html` 确认"是否发生形变、场数据长什么样"。

**3D 交互查看器(2026-08-03 新增,Canvas2D 版)**:报告内嵌纯 Canvas2D 渲染器(零依赖、**不需要 WebGL**、
任何浏览器离线可用),手写轨道控制(拖拽旋转/右键平移/滚轮缩放),包含:原始网格线框(可切换)、
按位移幅值着色的变形网格、红色载荷箭头+力标签(方向/大小/作用面)、红色固定约束标记、
坐标轴与图例。数据以 base64(Float32/Uint32)内嵌,合页报告约 1.1MB。
headless Chrome 截图验证:无 JS 错误,红色标记与位移着色面均实际渲染。
(v1 曾用 three.js WebGL,在禁用 GPU 的环境 WebGL context 创建失败导致空白,已弃用)

## 7. CLI 草案

```bash
cd .agents/skills/structural-solver
python scripts/solve.py solve model.vtp \
  --material steel \
  --fix "z_min" \
  --load "force x=+1000 @ point=0.5,0.05,1.0" \
  --output-dir out/ --scale 20
```

约束/载荷支持:轴对齐面(`z_min`/`z_max`/`x_min`...)、半径选点、可选 CAD 命名区域(若有 `Name` 数组,与 `cad-surface-mesher` 打通)。

## 8. 里程碑与验收

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| P0 | skill 骨架、CLI、单位约定、本文档落地 | `solve --help` 可用 |
| P1 | gmsh 四面体 + T10 组装 + 求解闭环 | 拉伸杆误差 < 1%,反力平衡 < 0.5% |
| P2 | 约束/载荷解析(固定面、力、压力) | 悬臂梁挠度误差 < 5% |
| P3 | 后处理 + VTP 输出 + JSON/HTML 报告 + 形变渲染 | 报告内可见形变图与场数据表,deformed_mesh.vtp 读回完整 |
| P4 | 收敛研究、多基准回归、README 示例 | pytest 全绿,示例命令可复现 |

## 9. 风险与假设

- T4 偏刚、精度差:默认 T10,`--order 4` 开关;收敛研究给出误差依据。
- gmsh 需输入为水密表面:输入非水密时给出明确报错并提示先跑 `mesh-watertight-repair`。
- 静刚度定义仅指"载荷点线位移刚度",多方向刚度作为后续扩展。
- 单位错误是最大风险源:输入有 `units` 元数据时优先采用,否则 `--units` 强制声明。

## 10. 实施状态(2026-08-03)

### 已实现

- P0-P4 全部完成:CLI、T4/T10 求解、约束/载荷、后处理、VTP/VTU 输出、JSON/HTML 报告、收敛研究、pytest 基准回归(4/4 通过)。
- 输出契约按 §6 落地:deformed VTP 为放大后几何 + 真实 `displacement` 数组;另附 `volume.vtu` 体网格。

### 实测结果(钢 E=210 GPa)

| 基准 | 解析 | FEA 实测 | 相对误差 |
|---|---|---|---|
| 拉伸杆 δ=FL/(EA) | 4.762e-6 m | 4.806e-6 m | 0.92% |
| 悬臂梁 δ=FL³/(3EI) | 1.905e-3 m | 1.904e-3 m | 0.02% |

合页 case(1.5mm T10、固定叶片1 x_min、叶片2 端面 Fz=-100 N):

- δ = 1.58 mm,k = 63.3 kN/m,max von Mises = 143.5 MPa,应变能 = 0.079 J
- 收敛:3.0mm vs 1.5mm δ 变化 0.93%(gate < 15%)
- Gates:反力平衡 0.0000%、能量一致性 0.0000%、VTP 读回审计、Jacobian 全部 PASS
- 报告:`outputs/hinge/hinge_report.html`(内嵌形变对比图/位移云图/应力云图 + 基准表 + gates 表)

### 工程教训(偏离原计划的实现决策)

1. **gmsh OCC 布尔卡死**:带孔叶片/销轴等圆柱面相交在 `fragment`/`generate(3)` 上会无限挂起;
   合页 v1 改用确定性 numpy 网格生成器(矩形卷耳块截面 + y 向拉伸拆 tet,T10 全局去重),
   0.4s 生成 9k 单元,网格规则、体积精确。原计划的"销轴+内孔+螺丝孔"几何留待后续。
2. **VTP 格式**:pyvista 的体网格只能写 `.vtu`;`.vtp` 是表面 PolyData 格式。
   交付 = `*_deformed.vtp`(表面+场数组,ParaView 直接看)+ `*_volume.vtu`(完整体网格)。
3. **单元公式**:gmsh tet10 边中点顺序最后两条边(2-3,1-3)与 VTK 约定相反,读取时需交换;
   Jacobian 矩阵 `J = dNᵀ·X` 是转置约定,物理梯度必须用 `dN @ inv(J).T`。

## 11. Rust 原生求解器(2026-08-03)

数值核心已用 Rust 重写并接入:`native/` crate(纯 Rust,依赖 faer 0.24 稀疏 LU + rayon):

- 模块:`mesh.rs`(MSH 2.2 解析、tet10 边序重排)、`fem.rs`(T4/T10 单元刚度、
  4 点高斯积分、COO 组装、Dirichlet 消去、faer LU、节点应力恢复)、`main.rs`(CLI)。
- 与 Python 参考求解器**逐位一致**:同一高斯点/形状函数/Jacobian 转置约定,
  合页 δ 对比 1.579746225... vs 1.579746225... (1e-10 级)。
- 集成:`solve.py` 检测 release 二进制 → 序列化 MSH+BC → 子进程求解 → 读回
  u/sigma 二进制;失败自动回退 Python(scipy)。
- 性能(合页 1.5mm,9120 T10 / 2.7 万 dof):Rust 纯求解 ~0.2 s vs Python ~20 s
  (组装 10-20 s + LU 2-5 s),**约 100×**;端到端(含网格/报告)约 10×。
- 正确性:patch test(单单元能量精确)、拉伸杆(<2%)、悬臂梁(<5%)cargo test 全绿;
  pytest 4/4 走 native 路径通过。
- 构建:`cd native && cargo build --release`;clippy 无错误(少量风格警告)。
