# 车辆无量纲系数输入

2026-11-16。新入口`vehicle_cd_learning.dimensionless`实际移除独立绝对长度、速度及两者的可用位：18项几何/无量纲条件＋18个缺失状态位，共36项数值输入。Re直接由调用者提供，不从默认流体性质反推，不要求独立输入U、L或黏度。

几何宽长比、高长比、Re、可用Mach、角度sin/cos、面积/长度²、离地间隙比、地面/轮周速度比等保留。上传网格仍需明确单位和轴；物理长度在预处理内用于规范化和几何比例，不作为独立数值特征送入网络。原始速度等可保留在来源审计元数据，未进入新`parameters`向量。

## 当前可用产物

`outputs/vehicle-dimensionless-20261116/prepared/dataset.json`为车辆切片的新输入合同，案例、标签、分组和完整视图保持原样。2257条记录中930条Re未知，继续缺失；26辆参考车的Cl继续null和损失掩码。

同日后续按用户要求另存[缺失补值版本](learning-label-completion.md)：`outputs/vehicle-label-completion-20261116/prepared/dataset.json`将930条Re以显式假设补齐，另存26条Cl伪标签。原数据和下述等价迁移检查点保持原样；补值版尚未重训，Cl估值不进入监督损失。

`outputs/vehicle-dimensionless-20261116/model/model.pt`由已训练的`vehicle-conditioning-diagnosis-20261113/dimensionless-jitter/model.pt`等价迁移。该源模型训练时两个绝对尺度字段及其known位始终为零；迁移仅切除首层矩阵四个闲置输入列和对应归一化列，90173→90109参数，不重新优化其余权重。普通全输入旧权重不允许直接去列。

此权重沿用单种子汽车候选的精度范围，不把结构迁移视为一次精度提升；旧实验与旧权重继续保留。

## 调用方式

从项目根目录设置两套已有脚本路径：

```bash
export PYTHONPATH="$PWD/.agents/skills/predict-vehicle-cd/scripts:$PWD/.agents/skills/predict-aircraft-coefficients/scripts"
```

预测JSON示例：

```json
{
  "geometry": {
    "path": "/path/to/car.stl",
    "unit_scale_to_m": 1.0,
    "front_axis": "-x",
    "up_axis": "+z"
  },
  "conditions": {
    "re_body": 10000000,
    "alpha_deg": 0,
    "beta_deg": 0,
    "ground_present": 1,
    "ground_speed_over_velocity": 1,
    "wheels_present": 1,
    "wheel_rotating": 1,
    "wheel_rotation_speed_over_velocity": 1
  }
}
```

条件值仅为用户输入格式示例，不代表某车已验证的源工况。`re_body`采用车身长度定义；如果参考面积比例已知，可显式提供`reference_area_over_length2`，未提供保持未知。缺失条件不会被静默补成零；实际零值会作为已提供的条件编码。`condition_profile=builtin_assumed`作为输出审计标记时只返回Cd，不把Cl头的内部数值当作有标签支持的结果。

```bash
python -m vehicle_cd_learning.dimensionless predict \
  outputs/vehicle-dimensionless-20261116/model/model.pt \
  /path/to/case.json --output /path/to/prediction.json
```

所有查询统一采用完整512视图后按原相机公式重采样至256，以衔接原车辆缓存；Rim原数据的256直接光栅化缓存保留原样，不能宣称任意来源原网格回放都与该缓存逐值相同。本次原始网格回放验证选用DrivAer案例。`source_render_config`与网络实际`render_config`分别持有这两个真实使用点。

未来准备/训练也使用同一合同：

```bash
python -m vehicle_cd_learning.dimensionless prepare \
  /path/to/audited-twenty-field/prepared --output-dir /new/dimensionless/prepared
python -m vehicle_cd_learning.dimensionless train \
  /new/dimensionless/prepared --output-dir /new/model --epochs 40 --device mps
```

`prepare`另存车辆数据，不改冻结源文件；新训练直接使用36维，复用既有训练器、完整视图A、采样扰动和缺失Cl掩码。`export`只服务已有无量纲候选的等价迁移，不用于把依赖绝对尺度的模型伪装成新合同。

## 范围和所有权

车辆旧5/9维入口原本已经使用Re，不需要速度/黏度重复数值输入。这里修复的是后来20＋20维联合实验的输入合同，并为车辆候选打通准备、训练和预测。COCOANet旧飞机速度/高度检查点和预测接口保持原样。同日按用户允许估算的后续要求，另存了带ISA估算Re的联合36维数据，详见[补值记录](learning-label-completion.md)；该联合数据尚未重训，不声称已核实源大气或完成新飞机推理部署。

`dimensionless.parameters`拥有明确的18项字段，`normalize_parameters`负责预测边界，与已有`SceneCases`训练数值逐项核验。`prepare/read_prepared`供训练、导出使用；`export_checkpoint/load_model`供等价迁移和预测加载；`predict_case`复用`render_geometry/resample_views/validate_views`，`train`复用`SceneCases/normalization/sampling_weights/train_model`。未增加运行时或模型注册管理层。

预测输出保留训练支持范围提示。Re、角度和工况进入模型不等于相应变化已经被配对CFD验证；未知Re固定工况样本不提供自由改变Re的准确性保证。
