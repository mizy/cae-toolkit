# 汽车与固定翼飞机数据扩展：2026-11-09

本轮只获取几何、系数、工况、许可说明及核验所需小型元数据，不获取压力、速度、剪切场、切片或体网格。原有工作区、COCOANet、车辆联合数据集及检查点保留。源数据与运行产物位于忽略目录 `outputs/aerodynamic-expansion-20261109/`。

## 可直接获取的来源

| 来源 | 固定 revision | 几何/记录 | 文件选择与状态 |
|---|---|---|---|
| [DrivAerRim](https://huggingface.co/datasets/BeyondXia1212/DrivAerRim) | `1e7e6a246f348c14db5fdf625432ceea559f4b4d` | 904 个轮辋方案，1 个共用 DrivAer Estateback 车体 | 1,824文件、70,857,555,918字节（其中1,813份STL），全部通过源哈希/大小核验 |
| [HiLiftAeroML](https://huggingface.co/datasets/nvidia/HiLiftAeroML) | `bbec30bcfc6103309c1375c5228b3ad0a586bfaf` | 180 个 CRM-HL 几何、1,800 个工况 | 372 文件、36,778,087,508 字节，全部通过源哈希/大小核验 |
| [SuperWing](https://huggingface.co/datasets/yunplus/SuperWing) | `1cd417b5d4036b5ae70f6befce2204c4b5ccb15c` | 4,239 个独立机翼、28,856 个工况 | 6 文件、3,404,636,771 字节，全部通过源哈希/大小核验；独立跨声速机翼池 |

HiLift 的1,800份STL在源仓库只有180个不同SHA-256，每个几何选取一份，保存原始目录清单中的十个路径与哈希；十个迎角并未被丢弃。SuperWing仅下载`origingeom.npy`几何坐标、`index.npy`系数/条件、`configs.dat`、`running_info.jsonl`、训练索引及README，未下载名称相似的`data.npy`、`data_surf`或`data_vol`。

SuperWing已对全部28,856条记录核对原始求解日志，迎角、Mach、求解器Cd/Cl逐项差值为0；Re为2×10⁷、温度300 K。几何数组顺序ID与`configs.dat`第一列原始运行ID有区别，已正确映射。官方训练25,985条/3,815几何，测试2,871条/424几何，几何交集0。它是Ma0.75–0.90的独立机翼，已检查首例参考曲面从10%半翼展开始，含部分展向截取；不能称为完整飞机。此轮低速整机训练不直接混入此池。

## 几何与工况核验

HiLift源几何单位英寸、机头-X、竖直+Z，半模型位于Y≥0。沿Y=0镜像后生成五个完整视图；半模型的力和半参考面积同时翻倍，因此原Cd/Cl保持不变。参考面积为297,360 in²（半机）、参考弦长275.8 in；Ma=0.2、Re_MAC=1.6×10⁶、迎角4–22°、零侧滑。使用包含六个已记录表面积分修正的源汇总系数，不重新下载流场。压力/黏性分量与总Cd/Cl最大偏差均为10⁻⁶。

HiLift采用官方`geometry_train/val/test`：126/18/36个几何、1,260/180/360条记录；不使用会使同一几何跨集合的`full`案例级随机切分。源STL哈希和完整八项设计向量也不能跨集合。全机视图与两种输入基准（仅工况、源设计参数＋工况）使用同一切分。

DrivAerRim源几何单位米、机头-X、竖直+Z。共用车体、前/后左轮、轮罩、支架组合，并沿Y=0镜像左侧部件恢复完整车辆。所有组件在同一个完整车辆相机坐标系合成深度缓冲，仅保存完整五视图，不生成局部裁剪图。已用直接合并网格的渲染结果验证缓冲合成等价。

DrivAerRim标签保留源固定参考面积2.16 m²、密度1.204 kg/m³、140 km/h、零偏航、移动无滑移地面、轮辋MRF和轮胎旋转壁面；求解为STAR-CCM+稳态realizable k-ε RANS，标签平均最后500次稳态迭代。尚未核验源黏度，固定工况训练不伪造Re，不与旧Re输入合同直接拼接。官方轮辋ID切分720/92/92。

来源疑点：前左轮0501/0503字节相同，后左轮0501/0502字节相同。三者完整前后轮组合仍不同，全部904个完整组合哈希唯一；局部部件重复不能直接等同整车跨集合泄漏。保留源完整配对进行实验，并在`drivaerrim-source-audit.json`明确标记这三个索引待源侧确认；本地文件核验不证明作者的CFD配对绝对正确，结果应按此限制解读。

本轮先独立验证新来源。CNN初始化可复用已有模型的图像编码器，参数头重新训练；原预测入口及权重保留。新来源族内结果不能证明旧车型精度提高或跨家族泛化。尚未增加汽车/飞机类型或子类标签。

两个新来源均使用256×256的完整五视图。轮辋实验未做分辨率、编码器容量等归因对照，因此不能单凭本轮结果认定具体架构问题。

## 需要申请或接受访问条款

本轮未提交申请，也未代用户接受条款。以下状态来自公开HF API的`gated`字段，证据见`applications-needed.json`。

| 来源 | 入口 | 状态 |
|---|---|---|
| SHIFT-SUV | [申请](https://huggingface.co/datasets/luminary-shift/SUV) | manual；CC BY-NC 4.0，另保留AeroSUV原始几何限制 |
| SHIFT-Truck | [申请](https://huggingface.co/datasets/luminary-shift/Truck) | manual；CC BY-NC 4.0及GTU原始模型条款 |
| DrivAerStar两部分 | [Part 1](https://huggingface.co/datasets/drivaerstar000/drivaerstar)、[Part 2](https://huggingface.co/datasets/drivaerstar002/drivaerstar) | 均为manual；许可文本需与论文再核对 |
| UAV-AID-500 | [申请](https://huggingface.co/datasets/george-erfem/UAV-AID-500) | manual；CC BY 4.0，属于升力面而非完整整机 |
| SHIFT-WING | [申请](https://huggingface.co/datasets/luminary-shift/WING) | manual；CC BY-NC 4.0，独立跨声速机翼 |
| SHIFT-CCA | [申请](https://huggingface.co/datasets/luminary-shift/CCA) | manual；custom/other，需查看具体条款及与COCOANet的几何重合 |
| SHIFT公开样例 | [SUV-sample](https://huggingface.co/datasets/luminary-shift/SUV-sample)、[Wing-sample](https://huggingface.co/datasets/luminary-shift/Wing-sample) | auto门槛，仍需登录并接受条款，当前没有直接下载 |

另有技术访问阻碍，不能混称为已证实需要申请：BlendedNet和DrivAerNet++ STL的Harvard Dataverse API本次均返回403；保留响应记录，不绕过访问控制。BlendedNet++确认有论文但尚未取得可核验的完整公开包。MIT Car Drag、ShapeNetCar沿用此前的尺度/参考面积/来源授权待核验状态，不纳入本轮训练。

## Flow360自有案例

用户于本轮授权向“解释低雷诺数下的升阻比”会话索取fangzhen与UAT固定翼案例交接。首批UAT交接已取到：3个实际计算表面、9条记录，其中6条为源审计的训练候选、3条仅作网格敏感性比较。只复制实际UGRID导出的几何数组，不以打印用CAD STL替换计算表面。服务地址、case ID及非公开几何只保存在忽略的本地产物目录，不写入Git。

`aircraft_learning.import_flow360_handoff`独立校验源表面/导出文件哈希、力和残差末步一致，按最后10%迭代范围重算Cd/Cl，核对力轴旋转、mm²参考面积和源物理/translated Re。9条均通过这些配对检查。条件为6m/s、零侧滑、参考Re约23471；同源3个相关几何、9条工况不能算9种飞机。

两条warm-start/restart候选仅有7/15个保存的尾部样本，保留短历史告警，需拼接父算例后再提高标签置信度；残差停止不等于网格独立、非定常收敛或实验真实性。网格敏感性3条不重复计训练几何。该池含Spalart–Allmaras、Ncrit=9转捩模型及流体重力，当前公开模型输入合同没有这些独立物理设置；加上只有3个相关几何，本轮先独立保留，未直接混训。fangzhen原目录尚待找回，不凭会话描述补造标签。

核验产物为`flow360-verified/pool.json`及三个完整五视图，源首批交接、后续总清单由原会话维护。用户自有参数化设计仅用于本地个人研究，不将客户、专有模型或未核验的第三方资产打包发布。

## 已完成训练：HiLift

三个模型均完整训练40轮、seed42，验证集选择权重；预先固定设置后才评估测试集。五视图主模型仅复用COCOANet图像编码器，条件编码器、共享Cd/Cl头及归一化重新训练。1260个实际训练ID及数据SHA已与三个检查点逐一核对一致。

| 模型 | 最佳轮次 | 测试Cd MAE | 测试Cl MAE | 测试Cl/Cd MAE | 测试Cd MAPE |
|---|---:|---:|---:|---:|---:|
| 五视图CNN | 32 | 0.013171 | 0.038743 | 0.400989 | 5.25% |
| 仅工况MLP | 21 | 0.036133 | 0.132147 | 0.761776 | 12.86% |
| 设计参数＋工况MLP | 39 | 0.009947 | 0.031690 | 0.266419 | 3.21% |

五视图在此几何留出集上优于仅工况基准，源设计参数仍更准确。数据中所有测试Cl均为正，零符号错误不证明跨正负Cl的判断能力。36几何簇bootstrap（10000次）得到五视图Cd/Cl/升阻比MAE的95%区间约[0.01220,0.01413]/[0.03409,0.04404]/[0.36872,0.43129]，只反映这一来源与一个训练种子的测试抽样波动。

另取测试几何LHC006，从源STL重新读取、镜像、渲染并推理，与缓存评估结果最大差为0；本机单例约2.70秒，不构成所有输入的延迟保证。报告见`hilift-results.md`、`hilift-summary.json`、`hilift-raw-replay.json`，权重位于`models/hilift-views-seed42/model.pt`。旧COCOANet部署CLI保持原输入合同，暂不直接接受这个Re/Mach合同的检查点。

## 已完成训练：DrivAerRim

904个完整几何组合均通过渲染/标签配对，4520个数值视图张量均有限、深度在[0,1]、掩码二值且非空，904组完整视图数值各异。按官方720/92/92完整组合切分，无完整组合哈希跨组。原车辆sampling-jitter模型仅提供图像编码器初值，共享Cd/Cl头、条件编码器和归一化重训；40轮、seed42，验证集选择第38轮。

| 模型 | 测试Cd MAE | 测试Cl MAE | 测试Cl/Cd MAE |
|---|---:|---:|---:|
| 五视图CNN | 0.002736 | 0.014643 | 0.060896 |
| 训练集均值基准 | 0.002961 | 0.015598 | 0.065765 |

Cd MAPE为1.11%，但均值基准已很接近；Cd MAE仅改善约7.6%，Cl约6.1%。92个测试组合配对bootstrap（10000次）得到Cd MAE改善的95%区间约[-0.000045,0.000516]，包含0，尚无稳定Cd收益的充分证据。该数据车身固定、标签分布窄，不能将小绝对误差解释为任意车辆的高泛化精度。保留实验模型，不替换原预测器。

协议、完整评估、成员/数据SHA核验和来源别名告警分别见`drivaerrim-protocol.json`、`models/drivaerrim-views-seed42/evaluation.json`、`drivaerrim-summary.json`与`drivaerrim-source-audit.json`。

源组合STL重读、整车重建、渲染及推理的单例重放与缓存预测最大差为0。此例未缓存共用车身，端到端约12.55秒，其中CNN CPU前向约0.017秒；高面数原始几何预处理仍有开销，不能将纯网络耗时当成完整输入延迟。证据为`drivaerrim-raw-replay.json`。

## 实现入口与证据

选择性下载使用本地可选依赖`huggingface_hub`、`hf_xet`、`requests`。示例命令从项目根目录执行，`discovery`中的清单是下载前保存的HF API固定revision快照：

```bash
PYTHONPATH=.agents/skills/predict-vehicle-cd/scripts .venv/bin/python \
  .agents/skills/cad-surface-mesher/scripts/download_aerodynamic_data.py hilift \
  --catalogue outputs/aerodynamic-expansion-20261109/discovery/nvidia--HiLiftAeroML.json \
  --output-dir outputs/aerodynamic-expansion-20261109/raw/hilift
```

- `cad-surface-mesher/scripts/download_aerodynamic_data.py`拥有本轮来源白名单和下载清单，实际调用既有`vehicle_cd_learning.collect_bluff_bodies._download`做大小/哈希核验，LFS使用HF官方客户端下载与续传。
- `aircraft_learning.prepare_hilift`调用既有`aircraft_learning.prepare.render`及完整五视图保存函数；`vehicle_cd_learning.prepare_drivaerrim`调用既有`_rasterize`，相机统一后合成完整视图。
- `aircraft_learning.train.AircraftCases/train/evaluate`沿用既有`CoefficientModel`与`vehicle_cd_learning.train.train_model`。显式`condition_parameters/condition_names`供新来源的条件基准读取，默认仍兼容COCOANet七项参数合同；无新增类型分类或注册层。
- 下载计划、逐文件日志、`inventory-geometry.json`、`superwing-audit.json`、训练协议与报告均保留于本轮产物目录。没有用下载条目数代替实际训练成员数。
