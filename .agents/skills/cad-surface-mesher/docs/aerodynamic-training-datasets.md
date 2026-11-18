# 气动系数预测：实际训练数据清单

2026-11-17：完整补Re后的车辆＋固定翼联合池已用于[全训练池在线图像扰动对照](../../predict-vehicle-cd/references/learning-online-jitter-full-pool.md)。原B共享CNN接辅助绝对系数读出，两组单种子42各6轮，全部4121条train每轮恰好参与一次，另保留720个Rim真实改型对。26条缺失Cl继续掩码，未增加新标签、记录或类别输入。真实重渲染采样收益不稳定、部分来源退化，所有新候选未通过联合验证门槛；原B及既有飞机/联合A默认保留。以下“尚未重训”措辞保留其各自记录日期的历史状态。

2026-11-16绝对误差优化：另做四组各40轮的冻结CNN读出续训，复用现有2257条车辆记录（1819/211/227），没有新增数据或修改切分，26辆参考车缺失Cl仍掩码。只按Cd/Cl原始单位MAE训练/选择，飞机权重未改；候选及逐来源退步见[绝对误差记录](../../predict-vehicle-cd/references/learning-absolute-readout.md)。重复训练不增加样本数量，旧默认保留。

原始核验日期：2026-11-09；场景对照更新：2026-11-11。此清单以本地数据manifest、已保存检查点的`train_ids`、标签计数及训练报告为准。它记录实际参与训练的数据，不把官网宣称规模、下载清单或多次重复训练相加为新增样本。原预测器仍为独立来源模型；汽车/飞机A/B/C联合实验的九次40轮训练和评估已完成，结果另见[场景条件化记录](aerodynamic-scene-conditioning-plan.md)。

2026-11-12补充、2026-11-13更新：按用户指定的内置车标称零偏航低阻工况另建联合数据版本；见第2节。汽车切片已用于八组各40轮的误差诊断，完整汽车/飞机联合池尚未重训，既有默认权重和历史实验保持原样。诊断结果见[几何与工况对照](../../predict-vehicle-cd/references/learning-geometry-diagnosis.md)。

2026-11-09新增公开来源的下载、几何/工况核验、训练结果和申请入口见[数据扩展记录](aerodynamic-data-expansion-20261109.md)。HiLift独立训练已完成并计入下表；尚未训练的下载来源不计入训练总量。

2026-11-16：车辆无量纲输入完成18数值＋18状态位的等价迁移，原权重预测保持一致。后续按用户要求另存[缺失补值数据](../../predict-vehicle-cd/references/learning-label-completion.md)：汽车930条、COCOANet飞机1497条Re按显式假设计算，完整联合数据也统一删除独立U/L输入。26条Cl仅保存为模型伪标签和文献候选，未新增可信Cl监督或训练样本，补值版尚未重训。原始来源的黏度/工况未核实限制继续保留。

## 1. 当前实际用于系数训练的全部来源

| 来源 | 训练记录 | 验证记录 | 测试记录 | 训练标签 | 几何覆盖 |
|---|---:|---:|---:|---|---|
| [DrivAerML](https://huggingface.co/datasets/neashton/drivaerml) | 400 | 34 | 50 | Cd、Cl | DrivAer notchback 基础车族的参数化变体 |
| [AhmedML](https://huggingface.co/datasets/neashton/ahmedml) | 399 | 50 | 50 | Cd、Cl | 简化 Ahmed 车体变体，非完整量产汽车 |
| [WindsorML](https://huggingface.co/datasets/neashton/windsorml) | 274 | 35 | 35 | Cd、Cl | 简化 Windsor 车体变体 |
| 原 skill 内置参考车库 | 26 | 0 | 0 | 仅 Cd；Cl 缺失并掩码 | 26 份有对应三维模型的参考车 |
| [COCOANet](https://huggingface.co/datasets/ShapeBench/COCOANet) 低速子集 | 1,042 | 223 | 232 | Cd、Cl | CCA 飞机构型族；分别为 275/59/59 个几何 |
| [HiLiftAeroML](https://huggingface.co/datasets/nvidia/HiLiftAeroML) | 1,260 | 180 | 360 | Cd、Cl | CRM-HL；分别126/18/36个几何；三个独立模型均完整40轮 |
| [DrivAerRim](https://huggingface.co/datasets/BeyondXia1212/DrivAerRim) | 720 | 92 | 92 | Cd、Cl | 同一Estateback车身的904个完整轮组组合；独立模型40轮 |
| **按记录相加** | **4,121** | **614** | **819** | **训练 Cd 4,121；训练 Cl 4,095** | **原独立模型与新A/B/C实验复用这些数据；多次训练不增加样本数** |

“记录”代表几何与工况配对。车辆联合集共 1,353 条记录，几何哈希各异；COCOANet所用 1,497 条工况记录对应 393 个几何，HiLift的1800条对应180个几何，均不能把工况条数当成飞机种类数。同一条记录重复用于不同模型、种子、轮数或实验，不增加数据规模。

## 2. 车辆数据：实际工况与标签处理

原车辆独立模型输入为前、后、右、上、下五个完整视图的数值深度/掩码，以及宽长比、高长比、log(Re)、偏航 sin/cos 和四个来源工况 profile 编码。联合A使用20项物理/几何数值及20个可用状态位，不输入profile。原始几何、标签、预处理与切分保持可追溯。

| 来源 profile | 当前保留的主要工况 | 标签及限制 |
|---|---|---|
| `drivaerml` | 38.889 m/s、零偏航、静止轮胎；静止地面分上游滑移和下游无滑移；hybrid RANS/LES | 使用逐几何迎风参考面积的平均 Cd/Cl。模型 Re 按车身长度换算；源速度固定，不构成同车多速度扫描 |
| `ahmedml` | 1 m/s、零偏航、无车轮、静止地面；SA-DDES | 399/50/50 条配对数据；平均窗口等源描述存在需保留的限制，详见源 manifest |
| `windsorml` | 40 m/s、2.5°偏航、无车轮、静止地面；WMLES | 标签保留源模型轴约定，不能混同任意来流轴定义 |
| `builtin_assumed` | 暂按 120 km/h、零偏航、移动地面、旋转轮胎 | **工况为假设，未核实**；Cd 来自混合公开参考，Cl 为 null，损失和指标掩码；该工况预测接口仅输出 Cd |

### 2026-11-12：内置车标称工况补充，供下一轮训练

用户再次确认内置26车按正常姿态、零偏航的低阻工况使用。`aircraft_learning.prepare_scenes.vehicle_records`将已有120 km/h预设、0°迎角/偏航、移动地面、无滑移滚动车轮作为显式假设输入；地面速度/来流速度及车轮圆周速度/来流速度均假设为1。`source_evidence.assumptions_not_verified_physics`记录假设依据及字段，`source_conditions_verified=false`；模型的known位表示提供了训练输入，不表示源工况经过核实。

未核实的Re、Mach、参考面积、离地间隙等继续为None，不沿用旧车辆数据按假设空气性质计算的Re。Cd、缺失Cl掩码、几何、切分和采样策略保持不变；没有新增或复制标签，也没有新增同车多偏航记录。

新版准备数据：[dataset.json](../../../../outputs/aerodynamic-builtin-assumptions-20261112/prepared/dataset.json)，[补充与核验记录](../../../../outputs/aerodynamic-builtin-assumptions-20261112/README.md)。车辆train中0°记录由1519条已声明CFD工况加26条用户假设组成，共1545条；Windsor仍为274条固定+2.5°。

已完成A/B/C及32/128实验实际使用的旧联合数据中，内置26车的工况输入仍是None、mask=0。旧实验保持冻结，其分数不能描述这次补充后的效果。2026-11-13完成的新汽车诊断切片为1819 train /211 val /227历史test；五个主对照均实际采样全部26车，其中views-legacy刻意恢复旧缺失工况作控制；三个Ahmed来源留出组按协议排除假设车。完整联合池尚未重训，默认权重未替换。

三个公开 CFD 池主要来自三个基础车族，验证/历史测试主要检验族内未见几何。26 车全部在训练集，其拟合误差没有独立车型泛化含义。车辆历史测试集已在此前被查看，不应再称为全新盲测；后续调参使用开发验证时需披露其反复使用的事实。

下载数量与最终训练数量的差异：

- DrivAerML：本地 484 个配对几何/工况记录。
- AhmedML：本地 499 个配对记录；完整来源的可用变参考面积系数记录不足 500，未把缺失记录补造。
- WindsorML：本地 350 份 STL，其中 6 个精确重复几何被去重，最终 344 个记录；另 5 个设计没有发布对应网格。重复记录按历史 test > val > train 的优先级保留，避免同几何跨组。
- 原始 STL、去重记录和旧产物均保留，未把未配对几何当作新增可训练标签。

车辆数据证据：

- [联合数据 manifest](../../../../outputs/predict-vehicle-cd/builtin-inclusive-20261030/prepared/dataset.json)，SHA-256 `95af777a26aadb92b88666738f1848d317cdebaec1d1e588cb06b9b62c273355`。
- [26 车实际训练核验证据](../../../../outputs/predict-vehicle-cd/builtin-inclusive-20261030/final-inclusion-proof.json)。
- [Windsor 去重与缺失清单](../../../../outputs/predict-vehicle-cd/public-data/windsorml/full-source-deduplicated.json)。
- [来源工况、参考面积和导入约定](../../predict-vehicle-cd/references/public-datasets.md)。

## 3. 飞机数据：实际筛选与几何留出

COCOANet 固定 revision 为 `210068b62a4678a3bcfb9a87926bcec44e6416f9`，源元数据标注 CC BY 4.0。当前固定 CSV 实读 **3,513 行、401 个几何 ID**，与数据卡宣称的 3,570 行有差异；清单采用实读文件，不推断差额原因或补造记录。

首轮筛选源速度 **≤80 m/s**，保留 1,497 条、393 个几何。其余 2,016 条高速记录未参与本轮训练/验证/测试；另 8 个几何没有满足此筛选的记录。这里按源速度划定首轮范围，没有把它替换成已核验的 Mach/Re 阈值。

| 集合 | 几何数 | 工况数 |
|---|---:|---:|
| train | 275 | 1,042 |
| val | 59 | 223 |
| test | 59 | 232 |

按几何 ID 和完整设计参数向量分组后固定切分（seed 20261109）；同一几何的全部低速工况位于同一集合，并检查字节相同的 STL 不跨组。精确分组不证明近重复消除或跨机型独立性。源数据中的 CFx/CFz 按迎角旋转后与修正 CD/CL 对账（atol 1e-7）。所需 STL、面积/弦长参考文件和全部完整五视图已实际准备。

飞机模型输入包含完整五视图、翼展/机长、高/机长、log机长、log速度、高度以及迎角 sin/cos。仅支持数据中的零侧滑。直接采用源速度/高度，不在源大气/黏度约定未核实时构造 Re 或 Mach。主模型没有车辆监督预训练，按飞机训练侧计算归一化。

三种已训练模型使用**同一组 COCOANet 样本和切分**：

| 模型 | 实际输入 | 训练预算与选中轮次 |
|---|---|---|
| 完整五视图 CNN（主模型） | STL 五视图＋几何比例与飞行条件 | 40 轮；val 选第 27 轮 |
| 设计参数 MLP（参照） | 源 16 项设计参数＋速度/高度/迎角 | 40 轮；val 选第 40 轮 |
| 仅工况 MLP（参照） | 速度/高度/迎角，不接收几何 | 40 轮；val 选第 32 轮 |

三个模型不能重复计为三份数据。测试按预先固定的主模型与参照设置，在各自训练完成后评估；不根据测试调参。当前结果限定在同一 CCA 构型族内未见几何，尚无跨运输机、其他无人机构型、旋翼机或任意物体的泛化证据。

飞机数据证据：

- [准备后的数据 manifest](../../../../outputs/predict-aircraft-coefficients/cocoanet-20261109/prepared/dataset.json)，SHA-256 `6b851b59b8628cc4a27ad1352ab6b2ec4c008825e6981f59b18196c8b95cf5e4`。
- [冻结的模型和切分协议](../../../../outputs/predict-aircraft-coefficients/cocoanet-20261109/protocol.json)。
- [训练、测试与泛化报告](../../../../outputs/predict-aircraft-coefficients/cocoanet-20261109/README.md)。

## 4. 哪些实验复用了这些数据

| 已有实验 | 实际数据来源 | 状态与计数方式 |
|---|---|---|
| 车辆原联合模型 | 上述三组 CFD＋内置 26 车，共 1,099 条 train | 80 轮，val 选第 47 轮 |
| 车辆扰动训练 seed 42/43/44 | 同一 1,099 条 train | seed42 原计划 80 轮，约 40 轮后中断，保留第39轮最佳权重；seed43/44各完整40轮，选第36/28轮；不是新增数据 |
| 车辆每视图 32/128 维对照（2026-11-09，已停止） | 同一 1,099 条 train | 32维完整40轮；128维被用户停止，仅中间权重；不能对该次维度收益下最终结论 |
| 车辆弱监督场 A/B | 同一 1,099 条记录监督系数；其中 32 个 DrivAer train 几何额外参加力积分/代理方程/边界约束；34个DrivAer val几何评估 | **没有真实 CFD 压力/速度/剪切场标签**；不是新增32辆，也不是完整场监督数据集 |
| 飞机五视图与两个参照 | COCOANet 低速子集同一1,042条 train | 主模型随机初始化；并未与汽车合并训练 |
| CRM-HL五视图与两个参照 | HiLift同一1260条train、126个几何 | 各40轮，分别选32/21/39轮；CNN仅复用COCOANet图像编码器，无跨家族测试结论 |
| DrivAerRim五视图 | 720条train，92条val，92条test | 完整40轮，选38轮；仅复用旧车辆图像编码器，固定工况独立实验 |
| 场景A/B/C联合对照 | 上述全部4121条train，614条val，819条历史test | 各3种子×40轮，共9跑；统一20项数值＋20个known位，Windsor转风轴、旧车缓存统一256；随机配对初始化，无预训练留出几何泄漏；不重复增加训练样本数 |
| A每视图32/128维单种子对照（2026-11-12） | 同一4121/614/819联合池 | 128维seed42完整40轮、选39轮；32维复用已核验初始化/采样的A-seed42；Cd改善伴随车辆Cl/升阻比变差，默认32维保留 |
| 汽车几何/工况/损失诊断 | 汽车切片1819/211/227；来源留出折1394/161/50 | 八组seed42各40轮；五视图、3D点云、数值参照、名义工况前后、来源尺度和绝对尺度输入对照；不是新增标签，不替换默认A |
| 汽车工况/细节分配第二轮诊断（2026-11-13） | 同一汽车切片；Rim监督720/92/92 | 八组seed42各40轮，加Rim专门训练160轮预算对照；仅已有样本重训。完整视图核回归只用720条Rim train拟合、92条val选参数。默认A未替换 |

早期 DrivAer/Ahmed/Windsor 小样本试验、单来源专家模型和内置车交叉验证，只是上述来源的子集或重训，不能在总量中再加一遍。人工制造解、二维通道流及 synthetic 车辆 fixture 用于实现/物理算子验证，属于历史独立验证实验，没有混入当前车辆或飞机系数数据。

## 5. 已调研或下载，但没有加入当前训练的数据

2026-11-14：按用户要求，将同车小改型、多工况、独立车型族和后续表面贡献验证的数据需求记录在[待补数据清单](../../predict-vehicle-cd/references/learning-data-gaps.md)，后续再补充，不计入当前训练总量。

| 候选/来源 | 本地或访问状态 | 未参与训练的说明 |
|---|---|---|
| MIT Car Drag Prediction | Cd 标签表及两辆配对 OBJ 已取到 | 仍需确认单位/轴/参考量、部分工况和上游授权；镜像及宽度变体须按原车分组 |
| ShapeNetCar | 单样本 Cd/表面/速度文件与固定版本目录清单 | 参考面积、物理尺度/地面及上游授权仍需核对；889对目录不等于已导入889条 |
| DrivAerNet++ | Cd 表8,121唯一ID，面积表8,007 ID | 网格访问登记/包获取及Cl文件配对尚未完成；114条在该面积表找不到对应项 |
| DrivAerStar | 来源和访问条件已查 | 需要申请、实际配对和许可核验未完成 |
| SHIFT-SUV / SHIFT-Truck | 来源说明已查，主库受访问条件限制 | 实际几何/系数未导入，需核对基准参考面积、工况和适用许可 |
| SuperWing | 几何、系数及工况全部核验 | 4239机翼/28856工况；跨声速独立机翼池，未混入低速整机训练 |
| BlendedNet / 其他飞机候选 | 文献和访问途径调研 | 尚未导入训练 |
| Flow360车辆测试账号案例 | 只读发现与界面核查 | 没有导入当前训练集；用户已决定暂不用这些车辆案例 |
| Flow360自有低Re飞机案例 | 用户本轮重新授权；首批3个实际计算表面、9条UAT记录已取到并核验 | 6候选（其中2条短重启历史待补），3条网格敏感性记录；独立池，尚未混入系数训练。与此前取消的车辆测试账号采集不同 |

[车辆新增数据审计记录](../../../../outputs/predict-vehicle-cd/vehicle-data-audit-20261106/README.md)保存具体限制、文件和哈希。表中的数量不加进第1节。

## 6. 当前类型标签决定与长期目标

2026-11-11补充：用户进一步要求比较无场景、场景拼接、场景FiLM三种联合模型，场景值仅为车辆外流/固定翼整机外流。该实验允许场景提示，未添加SUV等子类、分类输出或默认预测器切换；不将来源profile直接作为新场景ID。以下2026-11-09决定保留为历史记录。

2026-11-09 用户决定：**暂不新增汽车/飞机、SUV/轿车或飞机子类的输入标签和分类输出**。车辆参考库中虽然保存 `category` 元数据，它没有作为当前监督分类任务；飞机 `family_id` 也是数据分组元数据。车辆已有的四个 `condition_profile` 输入编码保留的是来源/工况区别，并不等同于新设计的汽车/飞机类型分类。

长期目标是使系数预测覆盖更广的几何和工况。当前尚无“可准确预测一切物体 Cd”的证据。扩展时须保留速度/黏度/Re与长度定义、迎角/侧滑、地面/车轮等边界以及参考面积和力轴；同一个物体在不同条件或不同系数参考定义下，标签不必相同。类型标签省略不意味着这些物理条件可以省略。

## 7. 核验与维护

核验快照记录9份车辆/飞机检查点，均已核对 `train_ids` 与对应 manifest 的训练集合相等，并确认数据 SHA 一致；记录在 [机器可读核验快照](../../../../outputs/training-data-inventory-20261109.json)。另外核对了 A 场实验的几何子集和无场标签声明。现场未发现系数训练进程仍在运行。文档是本次快照，后续增加来源或改变筛选/切分时需重新核对，不能仅修改总数。

本轮另核对HiLift三份及DrivAerRim一份新检查点的训练ID与数据SHA；见扩展产物`hilift-summary.json`、`drivaerrim-summary.json`。原9份检查点及原两个manifest重新核对哈希，均未改变。SuperWing和Flow360候选池的下载/配对核验不计为训练成员。

维护时以 `vehicle_cd_learning.combine_builtin/load_dataset/train_model` 与 `aircraft_learning.prepare/AircraftCases/train` 的实际输出为依据；分类元数据、下载目录或官网总数都不能替代最终训练成员核验。本文没有新增数据缓存、运行时或注册层。

## 附录：内置26车（均为train-only Cd标签）

以下数值为当前参考库的训练标签快照，不能解释为同一已核实工况下的实测。每辆的来源和可信性说明以 [reference_cars.json](../../predict-vehicle-cd/assets/reference-library/reference_cars.json) 的 `cd_source` / `cd_confidence` 为准。

| ID | 显示名称 | Cd 标签 |
|---|---|---:|
| `drivaer-sedan` | DrivAer Sedan | 0.2719704 |
| `model3` | Model 3 | 0.23 |
| `aerosuv` | Aero SUV | 0.282 |
| `mpv` | Zeekr 009 MPV | 0.27 |
| `pickup` | Hyundai Santa Cruz Pickup | 0.37 |
| `ford-f150-raptor-r-2024` | Ford F-150 Raptor R 2024 | 0.56 |
| `tesla-cybertruck` | Tesla Cybertruck | 0.384 |
| `toyota-hilux-2022` | Toyota Hilux 2022 | 0.394 |
| `rivian-r1t-2022` | Rivian R1T 2022 | 0.3 |
| `toyota-fortuner-2021` | Toyota Fortuner 2021 | 0.38 |
| `toyota-auris-wagon` | Toyota Auris Touring Sports | 0.28 |
| `deepal-sl03` | Deepal SL03 | 0.23 |
| `xiaomi-su7` | Xiaomi SU7 | 0.195 |
| `xiaomi-yu7` | Xiaomi YU7 | 0.245 |
| `nio-et5` | NIO ET5 | 0.24 |
| `toyota-rav4` | Toyota RAV4 Plug-in Hybrid | 0.32 |
| `model-y` | Tesla Model Y | 0.23 |
| `buick-regal` | Buick Regal | 0.27 |
| `volkswagen-caddy-2017` | Volkswagen Caddy 2017 | 0.33 |
| `changan-cs95` | Changan CS95 | 0.294 |
| `byd-seal` | BYD Seal | 0.219 |
| `byd-atto-3` | BYD Atto 3 | 0.29 |
| `nio-et7` | NIO ET7 | 0.208 |
| `nio-es7` | NIO ES7 | 0.263 |
| `xpeng-p7` | XPeng P7 | 0.236 |
| `changan-ford-mondeo-st-line` | Changan Ford Mondeo ST-Line | 0.293 |
