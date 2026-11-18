# 共享 Cd/Cl 预测头对照（2026-11-16）

## 结论和此前完成范围

**此前没有完成统一车辆数据上的预测头结构对照，本轮已补做。** 上一轮 `absolute / features / balanced / features-balanced` 改变了优化目标、内部特征标准化及梯度分配，四组仍使用同一个 `168→64→16→2` 神经头；更早的线性/RBF探针只针对Rim单来源，属于表示诊断。

本轮固定整套编码、投影、工况编码和融合，比较原头续训与五种校正头。RBF由验证集选中，在历史test上同时改善初始模型的Cd、Cl绝对MAE，但仍未超过上一轮保留的绝对MAE续训候选。扩大MLP没有取得稳定收益。**默认权重未替换，上一轮探索性候选保留。**

这些结果支持“简单增加预测头容量暂未解决当前误差”。单种子、固定特征、固定学习率的结果不足以证明所有预测头结构都已穷尽，也不能证明编码器或数据是唯一误差来源。

## 数据与控制条件

- 数据沿用 `outputs/vehicle-label-completion-20261116/prepared/`：1819条train、211条val、227条test，共2257条记录。
- 来源为DrivAerML、AhmedML、WindsorML、DrivAerRim及26辆内置车。训练中1819个Cd标签、1793个Cl标签；26辆内置车均为train-only，Cl缺失掩码继续生效。
- Re估计和来源工况的不确定性沿用[标签补全记录](learning-label-completion.md)，本轮没有修改标签或将假设变成已核实数据。飞机不参与本轮车辆头对照。
- 所有组从 `outputs/vehicle-detail-contribution-20261116/A1/model.pt` 出发，使用同一份完整五视图得到的168维融合特征。整个CNN、投影层、工况编码和attention融合均冻结。
- 训练集单独计算特征均值、标准差，标准差下限0.05；验证和测试不参与统计。
- 五个神经组均为种子42、Adam、学习率1e-4、40轮，每组1160步、72760次抽样曝光。抽样轨迹和逐样本曝光计数逐项一致，每条训练记录至少被抽到10次。
- 原头组只训练原头；校正组冻结原头并训练新增分支，最后一层零初始化。所有方法起始函数一致；与大批次参考输出的最大标准化输出差约1.19e-7，来自浮点批次计算。
- 所有神经组损失仅包含原系数单位的Cd、Cl绝对MAE。选择规则为val上四个CFD来源等权的Cd MAE + Cl MAE，且两项各自不得劣于初始模型。epoch0保留为回退。
- RBF使用训练标签拟合源加权核岭回归，Cd/Cl分别只使用已观测目标；采用同一验证选择规则。它有30个核超参数候选，求解目标为平方残差加正则，计算预算与神经组不同，不应视为同预算架构排名。

## 实际比较的头

令 `z` 为冻结的168维融合表示，`f₀(z)` 为原共享头。除原头续训外，输出均为 `f₀(z) + g(standardize(z))`。这里的残差是单个几何的绝对系数预测误差；输入不需要第二辆车或参考车。

| 组 | 训练部分 | 选中轮次/参数 |
| --- | --- | --- |
| original | 原共享头168→64→16→2 | 34 |
| linear | 线性校正168→2 | 0，验证双指标门槛均未通过 |
| mlp | 残差MLP，168→64→2，GELU | 1 |
| wide | 加宽加深残差MLP，168→128→64→2，GELU | 0，验证双指标门槛均未通过 |
| normalized | 168→64→LayerNorm→GELU→2 | 3 |
| rbf | Gaussian RBF核残差校正，共1819个训练参考向量 | scale=0.3、alpha=1.0 |

linear和wide都实际完成40轮训练。epoch0表示验证规则最终回退到原函数，不代表跳过训练。LayerNorm只在每个样本的64维隐藏向量内归一化，预测不依赖其他查询行。RBF按特征维数平均平方距离，双精度计算核；保存的权重可通过专用完整几何预测入口读取。

## 绝对 MAE

以下均为各CFD来源先分别计算MAE，再对四个来源等权平均；不是按227条记录直接加权的整体平均。train包含26辆内置车，不能把训练拟合视作新车泛化。历史test反复用于开发对比，不能称为新盲测。

| 方法 | val Cd MAE | val Cl MAE | test Cd MAE | test Cl MAE |
| --- | ---: | ---: | ---: | ---: |
| 初始A1 | 0.010045 | 0.042621 | 0.008609 | 0.040658 |
| 原头续训 | 0.009766 | 0.042401 | 0.008314 | 0.040792 |
| 线性校正，回退 | 0.010045 | 0.042621 | 0.008609 | 0.040658 |
| 残差MLP | 0.010021 | 0.042620 | 0.008593 | 0.040542 |
| 加宽加深MLP，回退 | 0.010045 | 0.042621 | 0.008609 | 0.040658 |
| LayerNorm残差MLP | 0.009903 | 0.042598 | 0.008534 | 0.040511 |
| **本轮val选中的RBF** | **0.009985** | **0.042034** | **0.008517** | **0.040128** |
| 上一轮保留的absolute候选 | 0.009838 | 0.042177 | 0.008325 | 0.040120 |

上一轮absolute候选同时更新投影、工况编码、融合和原头，训练自由度不同；本轮只改头。它此前已在历史test检查后保留为探索性候选，不能将跨轮比较包装为新盲测选择。两者Cl差约8e-6，没有充分证据赋予这个微小差异统计意义。

RBF相对初始A1的分来源test结果：

| 来源 | 数量 | 初始Cd MAE | RBF Cd MAE | 初始Cl MAE | RBF Cl MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| AhmedML | 50 | 0.009249 | 0.009279 | 0.043447 | 0.042585 |
| DrivAerML | 50 | 0.006510 | 0.006127 | 0.027363 | 0.028222 |
| DrivAerRim | 92 | 0.003004 | 0.002889 | 0.015492 | 0.015370 |
| WindsorML | 35 | 0.015672 | 0.015772 | 0.076330 | 0.074334 |

总体改善仍伴随来源退步：Ahmed/Windsor的Cd、DrivAer的Cl变差。已有基础车型族的几何变体精度不能等同于任意新车型精度。

## 核验和入口

输出目录：`outputs/vehicle-head-variants-20261116/`。

- `protocol.json`：冻结的输入、特征统计、数据/代码哈希、超参数及验证规则。
- `{original,linear,mlp,wide,normalized,rbf}/model.pt`、`training_report.json`：各组可读取的选中权重、完整训练历史或30候选网格。
- `comparison.json`：全部2257条记录的预测及按来源/划分的绝对误差。
- `decision.json`：验证选择、与上一轮候选的差异和未替换默认的决定。
- `verification.json`：50份原产物哈希未变；训练集统计、缺失Cl掩码、曝光次数、选轮正确；五个来源加DrivAer11的完整五视图前向与缓存融合特征对账，系数差最大约3.30e-7。
- `rbf/raw-prediction.json`：从DrivAer11原始STL只输入Re及无量纲工况，Cd=0.2974379748、Cl=0.0963067422，与缓存完整视图前向逐值一致。这是接口核验，不是新增独立精度样本。
- `tests.log`：车辆与飞机相关265项测试通过。新增三项覆盖初始函数/冻结、归一化查询独立性、RBF计算及权重保存读取；Ruff和格式检查通过。

复现命令（仓库根目录；prepare需使用新的输出目录）：

```bash
export PYTHONPATH="$PWD/.agents/skills/predict-vehicle-cd/scripts:$PWD/.agents/skills/predict-aircraft-coefficients/scripts"
.venv/bin/python -m vehicle_cd_learning.head_experiment prepare outputs/vehicle-absolute-readout-20261116 outputs/vehicle-head-variants-new
.venv/bin/python -m vehicle_cd_learning.head_experiment train outputs/vehicle-head-variants-new original
# 依次对 linear、mlp、wide、normalized、rbf 运行同一 train 入口
.venv/bin/python -m vehicle_cd_learning.head_experiment evaluate outputs/vehicle-head-variants-new
.venv/bin/python -m vehicle_cd_learning.coefficient_head outputs/vehicle-head-variants-20261116/rbf/model.pt outputs/vehicle-dimensionless-20261116/re-only-case.json --output outputs/head-raw-prediction.json
```

新头权重使用 `vehicle_coefficient_head_v1` 专用合同，旧dimensionless读取器拒绝误读；原模型和上一轮可兼容旧入口的候选均保留。内置假设工况的完整几何CLI继续只对外返回Cd。

复用入口及真实调用点：`head_experiment.prepare` 调用已有 `readout_experiment.load/PooledCases` 和 `CoefficientModel.encode_pooled`；训练调用既有 `coefficient_loss`，评分调用 `absolute_scores`。新增 `CoefficientHead` 统一拥有校正权重/标准化，在缓存特征训练的 `predict_features` 和完整视图推理的 `forward` 两个实际调用点使用。完整几何CLI复用 `render_geometry`、`resample_views`、`dimensionless.parameters/predict_views`，没有新增几何处理或工况编码层。

后续若继续，只优先检验投影/融合表示的可恢复信息与训练泛化，并用新的几何组保留集确认；本轮没有证据支持直接换更大的预测头。多种子、学习率扫描、端到端重训均尚未完成。
