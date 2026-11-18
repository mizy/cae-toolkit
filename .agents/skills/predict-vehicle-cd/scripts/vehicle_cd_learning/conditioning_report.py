"""Present the prespecified condition/allocation diagnostics."""

import json

import numpy as np


def plot(experiment, result):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    primary = ("full", "dimensionless", "full-jitter", "dimensionless-jitter")
    for name in primary:
        history = json.loads((experiment / name / "training_report.json").read_text())[
            "history"
        ]
        axes[0, 0].plot(
            [h["epoch"] for h in history], [h["val_mse"] for h in history], label=name
        )
    axes[0, 0].set(title="Primary validation criterion", xlabel="Epoch", yscale="log")
    axes[0, 0].legend(fontsize=8)
    sources = ("drivaerml", "ahmedml", "windsorml", "drivaerrim")
    for i, name in enumerate(primary):
        m = result["runs"][name]["selected"]["metrics"]
        axes[0, 1].bar(
            np.arange(4) + (i - 1.5) * 0.2,
            [m[f"test/source/{s}"]["cd_mae"] for s in sources],
            width=0.2,
            label=name,
        )
    axes[0, 1].set(
        title="Primary test Cd MAE", xticks=np.arange(4), xticklabels=sources
    )
    axes[0, 1].legend(fontsize=8)
    for name in ("full", "rim-global", "rim-local", "rim-local-jitter"):
        rows = [
            r
            for r in result["runs"][name]["rim_selected"]["rows"]
            if r["split"] == "test"
        ]
        axes[1, 0].scatter(
            [r["truth"][0] for r in rows],
            [r["prediction"][0] for r in rows],
            s=10,
            label=name,
        )
        history = json.loads((experiment / name / "training_report.json").read_text())[
            "history"
        ]
        axes[1, 1].plot(
            [h["epoch"] for h in history],
            [h["rim_val_mse"] for h in history],
            label=name,
        )
    axes[1, 0].plot([0.23, 0.26], [0.23, 0.26], ls="--", color="gray")
    axes[1, 0].set(
        title="Rim: same validation selection criterion",
        xlabel="CFD Cd",
        ylabel="Predicted Cd",
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].set(
        title="Rim validation, fixed global target units", xlabel="Epoch", yscale="log"
    )
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(experiment / "comparison.png", dpi=160)
    plt.close(fig)


def write_report(experiment, result):
    lines = [
        "# 无量纲工况、细节训练分配与泛化诊断",
        "",
        "seed42；八组各40轮；同一完整五视图A与共享Cd/Cl头，保持原数据、旧权重和默认预测器。历史测试已被查看；本轮是开发诊断，不是新盲测。",
        "",
        "## 主切分：无量纲输入与采样扰动的交叉对照",
        "",
        "1819 train /211 val /227 test，表内各来源等权。无量纲组只屏蔽绝对长度与速度，保留Re、可用Mach和所有比例/角度；缺失物理值及Cl继续掩码。",
        "",
        "| Run | Best epoch | Cd MAE | Cl MAE | Cl/Cd MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("full", "dimensionless", "full-jitter", "dimensionless-jitter"):
        r = result["runs"][name]["selected"]
        m = r["metrics"]["test_source_macro/vehicle"]
        lines.append(
            f"| {name} | {r['epoch']} | {m['cd_mae']:.6f} | {m['cl_mae']:.6f} | {m['signed_cl_over_cd_mae']:.6f} |"
        )
    lines += [
        "",
        "## 轮组：同曝光、步数、损失分母与验证选择",
        "",
        "Rim720 train /92 val /92 test。各组重放完整训练池的同一采样批次；仅Rim梯度组将其他来源监督梯度移除，损失分母保留全批观测标签数。全局归一化仍可使用完整训练池统计；local只替换目标均值/标准差。",
        "",
        "global训练基线也按同一Rim验证集和全局目标单位选一次权重；另报告各组固定40轮，避免把checkpoint选择差异当训练分配收益。",
        "",
        "| Run / selection | epoch | train Cd MAE | test Cd MAE | test Cl MAE | ranking | predicted Cd std |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("full", "rim-global", "rim-local", "rim-local-jitter"):
        for role in ("rim_selected", "last_rim"):
            r = result["runs"][name][role]
            m = r["metrics"]
            ranking = r["ranking"]["drivaerrim"]["0.003"]["accuracy"]
            std = r["spread"]["test/drivaerrim"]["cd"]["prediction_std"]
            lines.append(
                f"| {name} / {role} | {r['epoch']} | {m['train/vehicle']['cd_mae']:.6f} | {m['test/vehicle']['cd_mae']:.6f} | {m['test/vehicle']['cl_mae']:.6f} | {ranking:.1%} | {std:.6f} |"
            )
    lines += [
        "",
        "排序是同来源测试几何两两比较，|真值ΔCd|≥0.003；相关配对不能当作独立试验数，也不是后视镜单部件改动验证。Rim专门训练组不能部署成任意新车预测器。",
        "",
        "## 绝对速度还是长度",
        "",
        "同一Ahmed来源留出；几何、来源、工况一起外推，非纯几何因果。",
        "",
        "| Input mask | Cd MAE | Cl MAE |",
        "|---|---:|---:|",
    ]
    for label, r in (
        ("无（上一轮）", result["prior_holdout"]["views-ahmed-holdout"]),
        ("只屏蔽绝对速度", result["runs"]["holdout-no-velocity"]["selected"]),
        (
            "屏蔽绝对速度和长度（上一轮）",
            result["prior_holdout"]["views-ahmed-no-scale"],
        ),
    ):
        m = r["metrics"]["test/vehicle"]
        lines.append(f"| {label} | {m['cd_mae']:.6f} | {m['cl_mae']:.6f} |")
    k = result["kernel"]
    lines += [
        "",
        "## 完整视图的信息与覆盖探针",
        "",
        f"训练样本内、跨全部完整视图的无监督变化像素筛选保留{k['variable_features']}维；没有按标签或人工部件位置选像素。线性/RBF岭回归仅用Rim训练集拟合，超参数只用验证集选择：{k['selected']}。",
        "",
        f"历史测试Cd/Cl MAE：{k['test_mae'][0]:.6f} / {k['test_mae'][1]:.6f}。它是专用凸回归信息探针，训练预算和归纳偏置不同，不作为通用A替代或严格算法排名。",
        "",
        "投影距离与误差分箱在comparison.json的kernel.coverage中；距离来自标准化投影像素，不是独立三维/气动距离，相关性不能单独证明数据覆盖因果。",
        "",
        "## 完成性与边界",
        "",
        f"重现上一轮full基线的验证曲线最大差异：{result['reproduction']['max_val_curve_difference']:.9g}；最佳轮数相同：{result['reproduction']['same_best_epoch']}。",
        "",
        "实际样本/监督曝光、Rim批次时序、优化步数、初始权重、增强RNG及各checkpoint选择规则已核验。配对区间仅条件于当前模型和已知几何族，不包含训练种子/新车型的不确定性。没有新的多偏航/多速度配对或修改后视镜CFD真值。",
        "",
        "复用SceneCases、normalization、sampling_weights、CoefficientModel/make_model、jitter_views、evaluate、train_epoch、predict_rows和grouped_scores。conditioning_experiment仅拥有本轮干预与额外checkpoint选择；conditioning_diagnostics验证和评估；detail_kernel只服务Rim完整视图信息探针；conditioning_report由评估入口生成报告。旧实现与协议未修改。",
        "",
        "![comparison](comparison.png)",
        "",
    ]
    (experiment / "README.md").write_text("\n".join(lines))
