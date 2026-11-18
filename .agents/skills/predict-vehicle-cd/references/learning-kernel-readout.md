# 固定基准的压缩前特征验证（2026-11-16）

后续已完成下文所提的三种子整网对照，并增加针对尺度差异的多尺度读出，见[整网修正与Radiant同案例对照](learning-end-to-end-radiant.md)。本文保留前轮快照，“尚未执行”指该快照时的状态。

## 当前结论

**已直接检验此前“约27%”所指的压缩前特征路线。全体车辆的Cd/Cl绝对MAE仅小幅改善，尚未确认稳定收益，更没有证明全体车辆能改善20%以上。** 后续工作只围绕一个固定基准验收，避免换基准、换数据范围后混报结果。

当前工作基准为 `outputs/vehicle-absolute-readout-20261116/candidate/model.pt`，SHA256为 `799ce27f8bae0e2518cb3c68cac97ddb8d0b20e08589ab9a6cbb4ce011e00f4e`。该权重此前作为探索性候选保留，未替换旧默认。所有本轮结果都与这一文件比较。

| 同一227条历史test记录，四来源等权 | Cd MAE | Cl MAE |
| --- | ---: | ---: |
| 固定基准 | 0.008325242 | 0.040119790 |
| 压缩后168维读出对照 | 0.008258612 | 0.040267612 |
| **验证集选中的压缩前5120维读出** | **0.008296547** | **0.039144798** |

压缩前候选相对基准，Cd MAE减少0.000028695，Cl MAE减少0.000974992。对测试案例按来源分别进行5000次配对重采样，“候选MAE减基准MAE”的95%区间为：

- Cd：[-0.000498430, +0.000419689]。
- Cl：[-0.002872256, +0.001028641]。

两者都包含零，且存在分来源退步，因此候选只留作实验权重，没有提升为新的工作基准或默认。这些区间只描述现有样本条件下的不确定性，不包括反复使用开发测试集造成的选择偏差，也不能推断陌生车族的精度。

## “降低二十多%”的准确范围

近期[特征诊断](learning-radiant-detail-diagnosis.md)中的27%，来自 **DrivAerRim的92条test记录**：原A2 Cd MAE为0.003170457，使用压缩前5120维表示的专用核回归探针为0.002311133。它是同一Estateback车身的轮组变化诊断，使用额外监督拟合和不同模型形式。它说明这批特征仍有预测信息，不能直接承诺跨来源整车模型会获得相同收益。

更早[无量纲与扰动实验](experiment-history-and-error-summary-20261113.md)还记录过Cd改善23.4%、Cl改善30.2%，相应基准是0.011860/0.062105，候选为0.009080/0.043372。当前模型沿用了那条路线，不能再把这些百分比当作下一轮可重复叠加的收益。

如果以后要声称“相对当前固定基准，两项误差均降低20%”，在完全相同的评分集合与权重下，至少需要达到Cd MAE≤0.006660193、Cl MAE≤0.032095832；同时还要检查重复实验和独立数据证据。这里仅解释数值含义，不作可达性承诺，不把百分比加入损失函数。

## 本轮如何验证

1. **固定数据。** 共2257条记录，1819 train、211 val、227 test；来源为DrivAerML、AhmedML、WindsorML、DrivAerRim和26辆内置车。26辆内置车均只在train中，提供Cd，Cl缺失继续掩码；训练实际使用1819个Cd与1793个Cl标签。Re与工况假设沿用既有来源记录，本轮未补造标签。
2. **只改读出位置。** 复用同一冻结CNN和完整五视图。一个分支读取融合后168维，另一个读取投影前5120维；两者都附加既有36项无量纲数值/缺失掩码，没有车辆分类或数据来源输入。所有已有神经权重逐值保持不变。
3. **只用训练集拟合。** 特征均值、标准差、核参考向量、拟合系数全部来自train。每维标准差下限0.05，几何和工况两个块分别取均方距离再相加。源权重与原采样份额一致。缺失Cl不参与拟合。
4. **只用验证集选参数。** 每个分支固定72个候选：6个Gaussian带宽×6个正则强度×直接预测/残差校正两种形式。Cd/Cl共享核及超参数；选择四来源等权Cd MAE+Cl MAE最小，且两项各自不得劣于固定基准。未用比例、升阻比、设计差值或排序选模型。
5. **先封存选择，再读test。** `selection.json`在test评分前写入，包含选中的模型和权重哈希。227条test只用于本轮完成后的开发评估。每个来源先计算MAE再等权平均；没有靠Rim样本较多改变总体权重。

kernel拟合是确定性的源加权核岭回归，优化平方残差加正则，验证选择只用绝对MAE。反复使用不同随机种子运行相同核求解不会提供新的重复训练证据；本轮没有将这种重复包装为多种子验证。核求解与神经训练的计算预算不同。

压缩前分支选中带宽1.0、alpha=0.1、残差校正。它的val Cd/Cl MAE为0.009819483/0.041028734，固定基准为0.009838293/0.042177007。残差是单个几何的预测误差校正，推理无需用户提供参考车。

## 分来源结果

| test来源 | 样本数 | 基准Cd MAE | 新Cd MAE | 基准Cl MAE | 新Cl MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| AhmedML | 50 | 0.009104 | 0.009585 | 0.042278 | 0.041572 |
| DrivAerML | 50 | 0.006644 | 0.006047 | 0.028366 | 0.022883 |
| DrivAerRim | 92 | 0.003120 | 0.002879 | 0.015418 | 0.015238 |
| WindsorML | 35 | 0.014433 | 0.014675 | 0.074418 | 0.076887 |

DrivAer与Rim两项都有改善；Ahmed的Cd和Windsor的Cd/Cl退步。当前证据支持继续研究压缩前表示的利用方式，尚不足以将其视为解决全体误差的方法。来源中的基础车型数量有限，2257条记录不能解释成2257种独立量产车型。

## 下一步只做一个训练对照

以下为**计划，尚未执行**，与本轮已完成的确定性核读出验证分开记录。

比较当前原结构与“压缩前完整特征直接接到共享Cd/Cl预测分支”的结构，进行整网联合续训。两组从同一工作基准初始化，固定种子42/43/44、相同抽样序列和训练预算。这样检验编码器能否一起学会利用细节；不再只比较冻结特征后的另一种读出器。

验收顺序固定：

1. 用211条val选轮次，三个种子的两组全部完成后封存权重，再评分历史test。
2. 报告三次训练的Cd/Cl绝对MAE、均值和波动，以及每个来源的结果。单独挑一次最好的种子不能证明稳定提升；Cd改善伴随Cl退步也不算双指标目标完成。
3. 只有双指标改善在配对训练中重复出现、样本不确定性和分来源退步可解释后，才将新结构提升为工作候选。
4. 最终泛化验收需要未参与开发的新几何组；若改用留一车族验证，所有学习到的组件必须在排除该车族后重新训练。直接拿已见过该车族的编码器换划分，会使评估失真。

当前历史test只负责开发对账，无法通过换一个称呼变成新盲测。新增真实几何/工况覆盖仍待补充；本轮不声称解决了数据不足。

## 交付、核验与复现

输出目录：`outputs/vehicle-kernel-readout-20261116/`。

- `protocol.json`：冻结的数据/代码/工作基准哈希、候选网格和验证规则。
- `fused/fit.json`、`pooled/fit.json`：各72个候选及选择结果；同目录`model.pt`可直接加载完整几何推理。
- `selection.json`、`comparison.json`、`decision.json`：验证选择、2257条记录的原始预测、绝对误差和未提升默认的决定。
- `verification.json`：60份原产物哈希未变；训练统计和参考向量仅来自train；缺失Cl权重为零；所存系数满足加权核岭方程（最大残差约5.3e-10）。
- 五个来源各一例加DrivAer11的完整五视图与缓存特征对账通过，系数差最大约4.33e-7。
- DrivAer11从原始STL推理，Cd=0.300863049、Cl=0.092599310，与缓存完整视图对账；单次包含模型读取和渲染约1.93秒，只说明这一案例和机器的实测。
- 车辆和飞机相关269项测试通过；新增四项覆盖压缩前后完整前向、工况向量保留、原权重冻结和两种预测模式保存读取；Ruff、格式检查通过。

```bash
export PYTHONPATH="$PWD/.agents/skills/predict-vehicle-cd/scripts:$PWD/.agents/skills/predict-aircraft-coefficients/scripts"
.venv/bin/python -m vehicle_cd_learning.kernel_experiment prepare outputs/vehicle-absolute-readout-20261116 outputs/vehicle-absolute-readout-20261116/candidate/model.pt outputs/vehicle-kernel-new
.venv/bin/python -m vehicle_cd_learning.kernel_experiment fit outputs/vehicle-kernel-new fused
.venv/bin/python -m vehicle_cd_learning.kernel_experiment fit outputs/vehicle-kernel-new pooled
.venv/bin/python -m vehicle_cd_learning.kernel_experiment evaluate outputs/vehicle-kernel-new
.venv/bin/python -m vehicle_cd_learning.kernel_readout outputs/vehicle-kernel-readout-20261116/pooled/model.pt outputs/vehicle-dimensionless-20261116/re-only-case.json --output outputs/kernel-raw-prediction.json
```

复用入口及实际调用点：`kernel_experiment.prepare`复用`readout_experiment.load/PooledCases`和已有`CoefficientModel.encode_pooled`；拟合复用`sampling_weights`与`detail_kernel.distances`，评分复用`absolute_scores`。新增`KernelReadout`拥有核参考、归一化和拟合系数，`predict_features`用于缓存拟合/核验，`forward`用于完整五视图推理。CLI复用`render_geometry`、`resample_views`、`dimensionless.parameters/predict_views`；推理为一次前向和核矩阵乘法，不进行CFD迭代。
