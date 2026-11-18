"""Render the completed geometry diagnostic metrics as plots and a review report."""

import numpy as np


def plot(experiment, runs, results):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    names = ("views", "points", "parameters", "views-source-loss")
    for run, _, report in runs:
        if run["name"] in names:
            axes[0, 0].plot(
                [h["epoch"] for h in report["history"]],
                [h["val_mse"] for h in report["history"]],
                label=run["name"],
            )
    axes[0, 0].set(
        title="Validation loss (same criterion)", xlabel="Epoch", yscale="log"
    )
    axes[0, 0].legend(fontsize=8)
    sources = ("drivaerml", "ahmedml", "windsorml", "drivaerrim")
    for i, name in enumerate(names):
        scores = results[name]["metrics"]
        axes[0, 1].bar(
            np.arange(4) + (i - 1.5) * 0.2,
            [scores[f"test/source/{s}"]["cd_mae"] for s in sources],
            width=0.2,
            label=name,
        )
    axes[0, 1].set(title="Held-out Cd MAE", xticks=np.arange(4), xticklabels=sources)
    axes[0, 1].legend(fontsize=8)
    for name in names:
        rows = [
            r
            for r in results[name]["rows"]
            if r["split"] == "test" and r["source"] == "drivaerrim"
        ]
        axes[1, 0].scatter(
            [r["truth"][0] for r in rows],
            [r["prediction"][0] for r in rows],
            s=9,
            label=name,
        )
        axes[1, 1].plot(
            sources,
            [results[name]["ranking"][s]["0.003"]["accuracy"] for s in sources],
            marker="o",
            label=name,
        )
    axes[1, 0].plot([0.23, 0.26], [0.23, 0.26], color="gray", ls="--")
    axes[1, 0].set(
        title="Rim: does prediction track wheel changes?",
        xlabel="CFD Cd",
        ylabel="Predicted Cd",
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set(
        title="Within-source ordering, |CFD delta| >= 0.003",
        ylabel="Direction accuracy",
        ylim=(0, 1),
    )
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(experiment / "comparison.png", dpi=160)
    plt.close(fig)


def write_report(experiment, result):
    lines = [
        "# 汽车系数误差诊断：同数据几何基线",
        "",
        "单种子42，每组40轮；主对照1819 train /211 val /227历史test。所有主对照同一几何、工况、共享Cd/Cl头、采样序列、优化器与验证选权重规则；不加场景。共同诊断训练不增强，与旧联合A的采样扰动/飞机数据训练不同。",
        "",
        "PointNet-basic采用公开结构的缩小版条件回归适配；不是工业论文成绩复现。先验证八跑完整预算、成员、初始化和实际采样一致性，再读取测试分数。旧测试曾被查看，本轮不是新盲测。",
        "",
        "## 主对照（各来源等权）",
        "",
        "| Run | Parameters | Best epoch | Cd MAE | Cl MAE | Cl/Cd MAE | Cached forward ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in result["runs"].items():
        if r["run"]["fold"] != "primary":
            continue
        m = r["metrics"]["test_source_macro/vehicle"]
        lines.append(
            f"| {name} | {r['parameter_count']} | {r['best_epoch']} | {m['cd_mae']:.6f} | {m['cl_mae']:.6f} | {m['signed_cl_over_cd_mae']:.6f} | {r['latency']['median_ms']:.3f} |"
        )
    lines += [
        "",
        "views-legacy只将内置26车的工况输入恢复为缺失；views使用用户指定的标称工况。views-source-loss只改变训练误差的来源内尺度，输入、网络和验证准则与views相同。parameters包含几何比例/参考量及工况，不是纯工况模型。",
        "",
        "## 各来源Cd误差与改型排序",
        "",
        "排序为同来源历史test几何的两两比较，真值|ΔCd|≥0.003；并非单一部件改动，也不同于论文的基准车DPA。配对共享样本，不能把配对数当独立试验数。",
        "",
        "| Source | train median Cd MAE | views Cd MAE / order | points Cd MAE / order | numeric Cd MAE | source-loss Cd MAE / order |",
        "|---|---:|---|---|---:|---|",
    ]
    for s in ("drivaerml", "ahmedml", "windsorml", "drivaerrim"):
        key = f"test/source/{s}"
        values = []
        for name in ("views", "points", "views-source-loss"):
            r = result["runs"][name]
            values.append(
                f"{r['metrics'][key]['cd_mae']:.6f} / {r['ranking'][s]['0.003']['accuracy']:.1%}"
            )
        median = result["constant_baselines"]["primary"]["median"]["metrics"][key][
            "cd_mae"
        ]
        numeric = result["runs"]["parameters"]["metrics"][key]["cd_mae"]
        lines.append(
            f"| {s} | {median:.6f} | {values[0]} | {values[1]} | {numeric:.6f} | {values[2]} |"
        )
    lines += [
        "",
        "均值/中位数参照只使用训练标签，但知道来源ID，仅用于诊断；不能部署为任意新车预测器。完整train/val/test分来源误差、预测幅度、R²、不同ΔCd阈值、原始预测在comparison.json。",
        "",
        "## 拟合能力与细节响应幅度",
        "",
        "| Source | views train / test Cd MAE | points train / test Cd MAE | source-loss train / test Cd MAE |",
        "|---|---|---|---|",
    ]
    for source in ("drivaerml", "ahmedml", "windsorml", "drivaerrim"):
        cells = []
        for name in ("views", "points", "views-source-loss"):
            m = result["runs"][name]["metrics"]
            cells.append(
                f"{m['train/source/' + source]['cd_mae']:.6f} / {m['test/source/' + source]['cd_mae']:.6f}"
            )
        lines.append(f"| {source} | {' | '.join(cells)} |")
    lines += [
        "",
        "32个Rim训练几何、仅用该子集归一化、固定256步记忆探针。数值输入在该子集内恒定，差异由几何输入表达；下表仅训练拟合，不代表泛化。",
        "",
    ]
    for name, r in result["small_train_fit"].items():
        m = r["final_eval_metrics"]["train/vehicle"]
        lines.append(
            f"- {name}: Cd MAE {m['cd_mae']:.6f}, Cl MAE {m['cl_mae']:.6f}; training objective {r['initial_training_loss']:.5f} → {r['final_training_loss']:.5f}."
        )
    lines += [
        "",
        "## 来源与车族联合留出",
        "",
        "所有Ahmed训练/验证标签从该折中排除，内置假设车也排除；1394 train /161 val，测试原50个Ahmed test几何。它同时改变车型、速度/Re/求解器等分布，结果只能说明来源与车族联合外推，不能单独归因几何。",
        "",
    ]
    for name in ("views-ahmed-holdout", "points-ahmed-holdout", "views-ahmed-no-scale"):
        m = result["runs"][name]["metrics"]["test/vehicle"]
        lines.append(f"- {name}: Cd MAE {m['cd_mae']:.6f}; Cl MAE {m['cl_mae']:.6f}.")
    lines += [
        "",
        "views-ahmed-no-scale只屏蔽绝对长度与速度输入，保留Re、几何比例和其余无量纲条件。源数据事实不变；comparison.json的input_extrapolation量化各特征在留出集超出训练范围的程度。",
        "",
        "配对MAE区间为固定单种子模型下、已知来源内按几何重采样的条件区间，不包含训练随机性或新车族不确定性。train_bias_correction只用训练残差检查校准偏差，不修改任何模型权重。",
        "",
        "## 边界与复用",
        "",
        "没有改造后视镜的CFD真值，没有同车多偏航配对，本轮无法核实后视镜减阻方向或侧风精度。完整表面点云仍有8192点采样和PointNet全局池化的限制；负结果不能否定所有3D模型。所有比较保留共享Cd/Cl头，不切换默认A。",
        "",
        "复用SceneCases/normalization/sampling_weights准备视图、输入及样本分布；复用CoefficientModel构造A和数值参照；复用masked_mse_loss/move_batch/evaluate计算训练与验证；复用grouped_scores输出物理系数指标。新PointCases拥有点云读取，PointRegressor拥有点输入编码；独立geometry_experiment训练循环服务三种输入及来源尺度损失，避免修改原训练器的图像/检查点合同。prepare_points的surface_cdf复用于完整表面及Rim共享车身，避免904次重复处理约900万三角形。",
        "",
        "![comparison](comparison.png)",
        "",
    ]
    (experiment / "README.md").write_text("\n".join(lines))
