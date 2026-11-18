# Re 与 Cl 缺失补值记录

2026-11-16。按用户要求搜索公开资料，并允许明确标注的经验补值。另存新版车辆数据，保留原数据、原权重和历史评估。

## 实际产物

`outputs/vehicle-label-completion-20261116/prepared/dataset.json`保留2257条记录及1819/211/227切分；新增补值来源记录。完整五视图逐文件复制并校验。`completed-values.csv`包含全部记录的可用值和来源状态；`builtin-estimates.csv`单列26辆参考车。

| 字段 | 本次补值 | 来源与训练处理 |
|---|---:|---|
| DrivAerRim 的 Re | 904条，均约11,912,966 | 源车速140 km/h、已审计车长4.612811267 m；空气性质按假设，状态为estimated；进入数值输入 |
| 内置车的 Re | 26条，约9,773,908～13,099,875 | 用户约定120 km/h、逐车已审计车长及原有假设空气性质；状态为estimated；进入数值输入 |
| 内置车的 Cl | 26条模型伪标签，约−0.2894～+0.0965 | 保存在`target_estimates.cl`和CSV估值列；监督`targets[1]`仍为null |
| 可信 Cl 标签 | 2231条，保持原值 | 未新增匹配本地几何、工况和系数定义的可信Cl标签 |

原1327条已存在Re保持原值。补齐后输入Re缺失为0，但其中930条仍属于估算工况。没有增加独立车型或训练样本，没有重训、替换权重或刷新精度指标。

### 飞机与完整联合池

进一步核查完整汽车/飞机池后，COCOANet还有1497条Re未知。已按源速度、源高度和逐几何机身X向长度，加上显式ISA/Sutherland假设计算，另存完整联合版本`outputs/vehicle-label-completion-20261116/joint/prepared/dataset.json`。

联合池仍为5554条、4121/614/819切分，原Cd/Cl和完整视图保持不变；统一18项数值＋18项状态位，删除原20项中的独立log(U)、log(L)。新合同为`aerodynamic_dimensionless_v1`。原车辆候选及原飞机7维检查点继续使用各自原合同；本次没有重训新联合模型。

- COCOANet估算Re约2,909,709～40,158,338；高度约8.67～9998.98 m，全部落在本次0～11000 m标准对流层公式范围内。
- 明确假设干空气、标准日、无温度偏移，源高度近似作为位势高度；T=288.15−0.0065h，p=101325(T/288.15)^[g/(R·0.0065)]，rho=p/(RT)，R=287.05287、g=9.80665。Sutherland采用mu0=1.716e−5 Pa·s、T0=273.15 K、S=110.4 K。
- [NASA标准大气介绍](https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html)和[Sutherland/雷诺数说明](https://www.grc.nasa.gov/www/k-12/airplane/viscosity.html)支持这种按高度估算流体性质的方法；本次公式和常数明确写入代码，计算后的空气性质保留在逐例审计，不声明已还原COCOANet实际求解器配置。
- 检查固定源数据的`case_1/param_4.json`后，确认其速度、高度、迎角与当前低速记录一致，仍未找到该文件中的实际流体性质；公开数据卡也没有补足这一项。
- HiLift已有1800条Re继续沿用源Re_MAC按参考长度换算的Re_body，不覆盖。

因此完整联合池新增**2427条估算Re**：汽车930、飞机1497；原3127条已有Re保留，Re缺失为0。真实Cl仍为5528条，26条参考车Cl继续掩码。飞机估算值单列在`joint/cocoanet-estimates.csv`，全量值表为`joint/completed-values.csv`。

## Re 计算和证据边界

统一用车身总长度定义`Re_body = rho * U * L_body / mu`，模型只读取log(Re)，U、L、rho、mu写在审计元数据，不增加独立数值输入。

- DrivAerRim假设rho=1.204 kg/m³、mu=1.813e−5 Pa·s。
- 内置车沿用原假设rho=1.204 kg/m³、mu=1.81e−5 Pa·s；未把120 km/h声明成源风洞实测条件。
- [DrivAerRim当前数据卡](https://huggingface.co/datasets/BeyondXia1212/DrivAerRim)及本地固定版本`1e7e6a246f348c14db5fdf625432ceea559f4b4d`仅明确部分工况。搜索索引中的旧数据卡列出mu=1.813e−5，但当前卡和已检查的历史版本没有该项，[源论文](https://arxiv.org/html/2611.08842v1)也未核实黏度。因此本次930条全部标为estimated，不升级为已核实的源Re。

`log_re_body_known=1`表示已提供数值，不能解释为该值已获原始CFD核验。更换输入缺失状态会影响已有模型预测；新版数据供后续重新训练/对照使用，原检查点和原基线数据继续配套保留。

## Cl 搜索结果及补值策略

| 对应参考车 | 搜到的公开Cl信息 | 可用范围 |
|---|---|---|
| Model 3 | [PerfECT论文Table 2](https://www.mdpi.com/2032-6653/14/12/337)列Cz=0.09、Cd=0.23、迎风面积2.22 m² | 车型层面参考；本轮由原论文搜索抽取表核对，全文/PDF直取403，尚未确认本地几何、速度及轴定义完全一致 |
| DrivAer sedan/notchback | [轮前导流板研究](https://link.springer.com/article/10.1186/s42774-021-00086-7)综述中，同为带镜、光滑底板及运动轮设置，列出TUM −0.028、Audi +0.004 | 文献自身展示风洞/设置差异；不能选一个值覆盖当前Cd=0.2719704的项目模板 |
| AeroSUV | [原工程团队报告的baseline](https://www.enginsoft.com/bootstrap5/images/expertise/2025/futurities2025-2-rbf.pdf)为Cd=0.302、Cl=−0.013，140 km/h | 与当前项目Cd=0.282及120 km/h假设不一致，保留为候选参考 |

其他搜索命中包含改型/概念Cybertruck、SU7 Ultra及其他代际车型，没有补为当前网格真值。公开来源值和模型值分别记录在输出目录`literature-candidates.json`中，均不新增监督标签。

26条伪标签来自冻结的无量纲A候选`vehicle-dimensionless-20261116/model/model.pt`，使用相同完整五视图及补Re后的标称工况。另存原缺失Re输入下的Cl、补值前后变化及教师权重SHA。没有把变动量当作置信区间。

其中Model 3伪标签约−0.07575，与上述文献+0.09存在符号差异；本次不能从该差异判定哪套工况的物理结果正确，但它说明当前伪标签尚不足以作为新的监督真值。26辆车仅补Re便造成Cl预测最多约0.04635变化，同样需要后续条件对照。已有模型反过来给自身提供标签不会增加独立的气动证据。

现有`SceneCases`和损失函数只读取`targets`，自然排除`target_estimates`；均值、方差及指标仍按原有2231条Cl标签计算。未把Cl缺失补为零，也未修改原skill对内置假设工况只输出Cd的规则。

## 重现与所有权

```bash
export PYTHONPATH="$PWD/.agents/skills/predict-vehicle-cd/scripts:$PWD/.agents/skills/predict-aircraft-coefficients/scripts"
python -m vehicle_cd_learning.complete_labels \
  outputs/vehicle-dimensionless-20261116/prepared \
  --builtin-audit outputs/predict-vehicle-cd/all-builtin-training/run26-upright/orientation-audit.json \
  --rim-prepared outputs/aerodynamic-expansion-20261109/prepared/drivaerrim \
  --checkpoint outputs/vehicle-dimensionless-20261116/model/model.pt \
  --output-dir /new/vehicle-label-completion
```

`complete_labels.complete`复用`dimensionless.read_prepared/load_model/predict_views`完成源文件校验及教师推断。`fill_re`只处理两类已授权来源的缺失Re；`write_table`直接读取真实标签或`target_estimates.cl`，输出各值状态。未新增模型结构或训练管理层。

联合补值入口复用相同数值字段及CSV输出；后续训练真实消费者为已有`scene_experiment freeze/train`和`SceneCases`，模式A不使用场景输入。这里只准备和核验数据，不启动预设A/B/C对照或多种子训练：

```bash
python -m aircraft_learning.complete_reynolds \
  outputs/aerodynamic-builtin-assumptions-20261112/prepared \
  --vehicles outputs/vehicle-label-completion-20261116/prepared \
  --aircraft outputs/predict-aircraft-coefficients/cocoanet-20261109/prepared \
  --output-dir /new/joint-label-completion
```

针对性测试覆盖Re公式和参考长度、已知Re保留、无效长度/未经授权来源拒绝、新版数据源文件不变，以及Cl伪标签继续被实际`SceneCases`训练边界掩码。全量案例核验和源产物SHA记录见输出目录`verification.json`。
