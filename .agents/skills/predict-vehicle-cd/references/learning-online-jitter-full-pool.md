# 全训练池在线图像扰动与Rim差值对照（2026-11-17）

Date: 2026-11-17

Experiment: `outputs/aerodynamic-online-jitter-20261117`

Status: completed. Retain online epoch5 as the user-selected continuation candidate; original B remains the comparison baseline. Automatic validation results remain recorded.

## 当前保留版本

按用户决定，**在线扰动第5轮作为后续实验起点保留**，提供固定入口`outputs/aerodynamic-online-jitter-20261117/retained-online.pt`，与`online/epoch-05.pt`字节一致。第5轮来自原有验证采样分数排序，没有重新按测试挑选。第6轮及在线组全部中间权重继续归档，原B保留作对照。

`retention.json`记录这次保留决定、源权重、SHA和加载核验。既有`selection.json`、`decision.json`及评估产物保留当时的自动验收结论；采样/来源退化记录继续有效。现有`joint_sampling.load_checkpoint`已实际加载固定入口，并核验配对网络与辅助读出的全部张量和源权重相同。

## 结论与对比

本次完整车辆＋固定翼训练已完成：两组各6轮、单种子42，每轮实际使用全部4121条训练记录。**没有新权重同时通过采样稳定性、真实改型响应和各来源绝对误差门槛。现按用户决定保留在线第5轮继续探索，原B用作对照。** 两组纯训练合计31.1分钟，完整准备、训练、选轮及测试约35.4分钟。

下面展示按验证采样分数预先选出的在线第5轮诊断权重及同轮无扰动对照，未按测试挑轮次。测试集上，主水平+0.25像素条件下，在线组同几何假ΔCd比原B降低16.5%，假ΔCl增大83.7%；同轮无扰动对照的假ΔCl也明显增大。真实改型的平均误差稍有改善，但幅度相对同轮对照有减弱，不能据此替换B。

### 真实几何重渲染的采样误差

92个历史test几何，512输入，水平+0.25像素。真实几何和物理条件保持不变。假差值及同一改型的预测漂移理想值均为0；数值越小越好。

| 模型 | 同几何假ΔCd MAE | 同几何假ΔCl MAE | 两端共同换采样：Cd漂移MAE | Cl漂移MAE |
| --- | --- | --- | --- | --- |
| 原B | 0.005806 | 0.005715 | 0.002052 | 0.003341 |
| 无扰动续训，第5轮 | 0.004634 | 0.010305 | 0.002226 | 0.004016 |
| 在线扰动，第5轮（诊断，未采用） | 0.004850 | 0.010500 | 0.001847 | 0.003343 |

相对同轮control，在线组同几何假ΔCd/ΔCl分别增大4.7%/1.9%；共同改型漂移分别降低17.0%/16.8%。因此有改善的模式，也有未改善的模式。无扰动组的Cl敏感性也上升，联合续训本身参与了退化，无法把全部退化只归因于图像扰动。

| 模型 | 仅查询换采样：Cd漂移 | Cl漂移 | 仅参考换采样：Cd漂移 | Cl漂移 |
| --- | --- | --- | --- | --- |
| 原B | 0.005967 | 0.005514 | 0.005666 | 0.005639 |
| 无扰动续训，第5轮 | 0.004750 | 0.010199 | 0.004486 | 0.010265 |
| 在线扰动，第5轮（诊断，未采用） | 0.004993 | 0.010305 | 0.004691 | 0.010451 |

24个测试几何的水平+0.125/+0.5、竖直+0.25三种附加相位，在线组同几何假ΔCd/ΔCl MAE为0.003939/0.005060，原B为0.012714/0.005960。这批附加相位有所改善，但不能掩盖主相位Cl的退化。线上图像扰动没有获得跨相位一致的收益，所有采样门槛仍未通过。

![真实重渲染采样误差](../../../../outputs/aerodynamic-online-jitter-20261117/sampling-robustness.png)

### 真实改型的误差、方向和幅度

先按既有主口径：三个固定训练参考×92个test查询，共276个相关配对。方向排除数值零差值；幅度为预测Δ对真实Δ的过原点斜率，理想值1。

本表方向与幅度按276个配对合并计算；不对三个参考各自的斜率先取均值。

| 模型 | ΔCd MAE | ΔCl MAE | Cd方向 | Cl方向 | Cd幅度斜率 | Cl幅度斜率 |
| --- | --- | --- | --- | --- | --- | --- |
| 原B | 0.002659 | 0.018515 | 75.0% | 76.4% | 0.569 | 0.258 |
| 无扰动续训，第5轮 | 0.002495 | 0.017670 | 76.8% | 78.2% | 0.645 | 0.319 |
| 在线扰动，第5轮（诊断，未采用） | 0.002608 | 0.018034 | 76.8% | 79.6% | 0.584 | 0.284 |

在线组相对原B的固定参考平均指标略有改善；相对同轮control，两个平均误差都稍高、Cd/Cl幅度分别减弱约9.5%/11.1%。测试逐参考检查仍发现Rim0003的Cd方向及Rim0004的Cd幅度未守住门槛，不能只依据合并平均数判定没有退化。

再检查两端都为未见几何的4186个无序配对。它们共享92个端点，不能当作4186个独立样本。

| 模型 | ΔCd MAE | ΔCl MAE | Cd方向 | Cl方向 | Cd幅度斜率 | Cl幅度斜率 |
| --- | --- | --- | --- | --- | --- | --- |
| 原B | 0.003379 | 0.021948 | 71.3% | 57.7% | 0.481 | 0.076 |
| 无扰动续训，第5轮 | 0.003248 | 0.022042 | 72.7% | 57.8% | 0.528 | 0.087 |
| 在线扰动，第5轮（诊断，未采用） | 0.003254 | 0.021900 | 72.8% | 58.0% | 0.471 | 0.081 |

在线组相对原B的两两配对ΔCd/ΔCl MAE降低约3.7%/0.2%，方向略升，Cd幅度略降；相对同轮control，Cd幅度降低10.7%、Cl幅度降低5.9%。Cl幅度仍仅约0.08，远低于理想值1，任意未见轮组对的Cl细节响应继续很弱。仅查询端换采样后，在线组Cd/Cl方向为56.5%/56.7%，尚未达到稳定排序微小改型的要求。

![真实改型预测幅度](../../../../outputs/aerodynamic-online-jitter-20261117/delta-response.png)

### 全池辅助系数读出：车辆与飞机

这组绝对值来自原B共享CNN加新暖启动辅助读出，未与此前通用A模型做等预算对比。历史test为227条车辆和592条飞机。下表采样漂移采用256图像插值，和前述512原始几何重渲染分开解读。

| 领域 | 辅助读出 | Cd MAE | Cl MAE | Cd图像平移漂移 | Cl图像平移漂移 |
| --- | --- | --- | --- | --- | --- |
| 车辆 | 原B CNN＋暖启动辅助头 | 0.008865 | 0.046712 | 0.004369 | 0.028368 |
| 车辆 | 无扰动，第5轮 | 0.008317 | 0.048812 | 0.004362 | 0.027252 |
| 车辆 | 在线扰动，第5轮 | 0.008067 | 0.046471 | 0.003098 | 0.022056 |
| 固定翼 | 原B CNN＋暖启动辅助头 | 0.009722 | 0.070279 | 0.009358 | 0.048193 |
| 固定翼 | 无扰动，第5轮 | 0.009861 | 0.073393 | 0.011974 | 0.064515 |
| 固定翼 | 在线扰动，第5轮 | 0.009279 | 0.074035 | 0.012489 | 0.083811 |

车辆辅助读出的图像平移漂移有所改善，飞机Cl则恶化，不能把车辆收益推广到固定翼。逐来源也有取舍：在线组COCOANet Cd/Cl MAE为0.006861/0.105862，同轮control为0.006472/0.098831；HiLift名义Cl MAE从0.057000改善到0.053524，但图像平移Cl漂移从0.081669增大到0.116795。逐来源完整预测与分母见`comparison.json`及`test-absolute-*.npz`。

### 结论的适用范围与下一步依据

本轮说明，全训练池在线图像扰动可以带来部分平均误差或部分采样模式的改善，但没有同时保住跨相位、跨来源和真实改型幅度。它也没有重现上一轮32个几何真实重渲染增强对主相位假差值的改善程度；两轮同时改变了监督分配及增强形式，不能只按样本数归因。

可能原因包括图像插值与三角面重渲染的表示差异，以及全池绝对监督改变共享CNN后的任务取舍。它们各自的因果份额尚未通过独立消融确认。后续更有依据的实验是把已有真实重渲染相位与在线扰动混合，并单独对照全池辅助监督的影响；继续保留原B和同轮control的真实改型门槛。本轮未执行该后续实验。

## Scope and dataset

The audited joint dataset has 4,121 train, 614 validation, and 819 test records. Training includes 1,819 vehicle and 2,302 fixed-wing aircraft records across seven sources:

| Domain | Source | Train records |
|---|---|---:|
| Vehicle | DrivAerML | 400 |
| Vehicle | AhmedML | 399 |
| Vehicle | WindsorML | 274 |
| Vehicle | DrivAerRim | 720 |
| Vehicle | `builtin_assumed` | 26 |
| Fixed-wing | CoCoAeroNet | 1,042 |
| Fixed-wing | HiLiftAeroML | 1,260 |

The 4,121 training records contain 2,220 unique geometries because aircraft geometries have multiple operating points. Twenty-six `builtin_assumed` records have a finite Cd label and masked Cl; every other training record has finite Cd and Cl. Across all 5,554 records, estimated Re is an input value for 2,427 records: COCOANet 1,497, DrivAerRim 904, and `builtin_assumed` 26. Of these estimates, 1,788 belong to train. These estimates enter the numeric `log_re_body` input and are marked known; their estimated status remains audit metadata, not a model input. In particular, the source air properties for Rim and the ISA/Sutherland aircraft estimates are assumptions, not verified source CFD conditions.

All full-pool cached views are existing 256×256 depth/mask images. The Rim paired branch uses the original genuine 512×512 cache. No 256-to-512 upscaling or full-pool rerender is performed. The absolute and delta branches therefore expose the shared CNN to different raster resolutions; compare their metrics within their declared evaluation paths.

## Model and training

The original Rim B delta model has 54,506 parameters. It retains its 18 physical values plus 18 known masks, five depth/mask views, antisymmetric pair prediction, and zero self-delta property. No source, scene, or category input was added. The new `AbsoluteReadout` has 79,059 parameters and consumes B's pooled image features plus the same 36 numeric fields. Total joint-training parameters are 133,565. The auxiliary head is not part of the existing Rim delta inference structure.

Every finite full-pool Cd/Cl label supervises `AbsoluteReadout`; the masked `builtin_assumed` Cl contributes no loss. This loss backpropagates through the shared B CNN. Only the original matched-condition Rim pairs supervise B's delta head; no arbitrary cross-source or aircraft-condition pairs are labeled as geometry modifications. Source weights affect loss weighting only and are not model inputs.

The paired arms use seed 42, six epochs, batch size 16, 258 optimizer updates per epoch, and 1,548 updates total. Original B initializes both arms with the same frozen B tensor hash; both also start from the same auxiliary readout. The fixed warm start freezes B and fits only the readout for 20 epochs on train records and cached pooled features. It covered all 4,121 train records / 2,220 unique geometries, took 69.9 seconds on MPS, and did not use validation or test data.

Each epoch visits every full-pool training record once and uses all 720 frozen Rim training pairs. The `control` arm uses original images. The `online` arm translates the original images afresh by continuous offsets sampled uniformly from [-0.5, +0.5] pixels on each axis; each record's five views share the same translation. Bilinear sampling retains a soft mask and normalizes depth by mask coverage. For Rim pair endpoints, half the sampled pairs share one translation and half use independent endpoint translations.

Frozen loss weights from `protocol.json`:

| Loss | Weight |
|---|---:|
| Absolute nominal / augmented / consistency | 0.25 / 0.25 / 0.10 |
| Rim nominal / augmented / self / consistency / teacher | 1.00 / 0.50 / 0.25 / 0.25 / 1.00 |

Within each step, control reuses identical original-image CNN features for nominal and augmented loss terms. The encoder uses per-sample GroupNorm and has no stochastic batch layers, so reuse preserves the duplicated forward values and gradients. Both arms have the same optimizer updates, record exposures, pair draws, and loss objectives. This is not equal-FLOP training: online encodes original and perturbed inputs, and the report records logical exposures separately from actual CNN image encodings.

## Selection and held-out evaluation

Validation selection uses all 614 full-pool validation records for per-source Cd/Cl absolute-MAE guards, with a 5% tolerance against the warmed initial readout; online is also checked against control at the same epoch. Rim nominal selection uses all 92 validation queries with three fixed training anchors; an additional 12 predeclared validation geometries supply four true-rendered phases and all 66 pair combinations. Gates require nominal MAE within 5%, amplitude slope at least 90% of reference, direction accuracy within 3 percentage points, and normalized sampling-drift MAE/P95 no greater than 0.05/0.10 of each target's delta standard deviation. The phase-specific selection therefore uses only 12 geometries.

After validation selection is sealed, Rim test evaluation covers all 92 held-out geometries and their 4,186 unordered pairs at the main +0.25 px x phase, plus 24 predeclared cases at three additional raw-rerender phases. Separately, full-pool absolute auxiliary predictions cover all 819 test records with synthetic image shifts at +0.25 px x, -0.25 px x, and +0.25 px y. These synthetic 256-pixel shifts are not raw geometry rerenders and are reported separately from Rim's 512-pixel phase evaluation. Test scores do not select or rerank checkpoints. If no candidate passes every validation gate, retain original B; failed candidates remain diagnostics.

## Verification and outcome

- Scoped test run: 373 passed; 2,656 warnings, primarily existing VTK/NumPy deprecations. Ruff checks and formatting on eight scoped files passed; output is saved in `lint.log`.
- The inherited 116 protected artifact hashes were rechecked against disk and all match. The current protected-artifact manifest contains 157 entries; all 157 hashes match disk (41 additions over the inherited list).
- Protocol, dataset, all 2,830 view-cache hashes, draws, initial tensors and actual offset/exposure traces were verified. The external `test-source-audit.json` rechecked 20 source/model/artifact hashes immediately after sealed validation selection; all matched.
- `verification.json` records complete training membership, masked labels and actual execution. `decision.json` retains original B; online ep5 is a rejected validation-selected diagnostic. No old default or source checkpoint was replaced.

## Reused entry points and reproduction

`PairedDelta.encode` / `predict_pair` in `paired_delta.py` owns the shared CNN and Rim delta path; the real caller is `joint_sampling_train.rim_backward`. `scene_experiment.normalization`, `sampling_weights`, and `verify_data` retain the full-pool dataset conventions; `joint_sampling.freeze` calls them. `SamplingCases` / `batch_views` retain the frozen genuine-512 Rim inputs; the real caller is `joint_sampling_train.rim_backward`. `AbsoluteReadout` is the only added head; `full_backward` trains it through the shared CNN and `joint_sampling_evaluation.evaluate_full` scores it. `translate_views` owns depth/mask translations and is called by both training branches and the synthetic full-pool evaluation.

Output root: `outputs/aerodynamic-online-jitter-20261117`. To reproduce in a fresh output directory, use the frozen full-pool manifest and original B sampling experiment:

```bash
export PYTHONPATH="$PWD/.agents/skills/predict-vehicle-cd/scripts:$PWD/.agents/skills/predict-aircraft-coefficients/scripts${PYTHONPATH:+:$PYTHONPATH}"
OUT=outputs/aerodynamic-online-jitter-<fresh-id>
.venv/bin/python -m vehicle_cd_learning.joint_sampling outputs/vehicle-label-completion-20261116/joint/prepared outputs/vehicle-sampling-consistency-20261117 "$OUT"
.venv/bin/python -m vehicle_cd_learning.joint_sampling_evaluation prepare "$OUT"
.venv/bin/python -m vehicle_cd_learning.joint_sampling_train train "$OUT" --arm control
.venv/bin/python -m vehicle_cd_learning.joint_sampling_train train "$OUT" --arm online
.venv/bin/python -m vehicle_cd_learning.joint_sampling_evaluation select "$OUT"
.venv/bin/python -m vehicle_cd_learning.joint_sampling_evaluation test "$OUT"
```
