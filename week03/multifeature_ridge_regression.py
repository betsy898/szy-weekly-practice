"""第三周实践一：单特征、多特征线性回归与 Ridge 对照。

使用 diabetes 表格回归数据集。模型输入只包含数据集的特征列，绝不包含 target 或行号。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
RIDGE_ALPHAS = (1, 100)
RESULTS_DIR = Path(__file__).parent / "results"


def metrics_row(model: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算一行 MAE、RMSE 与 MSE 指标。"""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "model": model,
        "split": split,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "MSE": mse,
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # 读取原始特征。X 只含 10 个输入特征；target 不会进入 X。
    dataset = load_diabetes(as_frame=True, scaled=False)
    feature_names = list(dataset.feature_names)
    x_all = dataset.frame[feature_names]
    y_all = dataset.target

    # 先以样本索引留出 40%，再均分为验证集和测试集，固定约 60/20/20 的划分。
    all_indices = np.arange(len(x_all))
    train_indices, holdout_indices = train_test_split(
        all_indices, test_size=0.4, random_state=RANDOM_STATE
    )
    validation_indices, test_indices = train_test_split(
        holdout_indices, test_size=0.5, random_state=RANDOM_STATE
    )
    x_train, y_train = x_all.iloc[train_indices], y_all.iloc[train_indices]
    x_validation, y_validation = x_all.iloc[validation_indices], y_all.iloc[validation_indices]
    x_test, y_test = x_all.iloc[test_indices], y_all.iloc[test_indices]

    pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "samples": [len(train_indices), len(validation_indices), len(test_indices)],
        }
    ).to_csv(RESULTS_DIR / "split_info.csv", index=False, encoding="utf-8-sig")

    # 所有 10 列的标准化统计量均只从训练集 fit，再应用于验证集与测试集。
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)

    bmi_index = feature_names.index("bmi")
    x_train_bmi = x_train_scaled[:, [bmi_index]]
    x_validation_bmi = x_validation_scaled[:, [bmi_index]]
    x_test_bmi = x_test_scaled[:, [bmi_index]]

    # 训练目标均值是完全不使用特征的参考基线。
    train_mean = y_train.mean()
    baseline_predictions = {
        "train": np.full(len(y_train), train_mean),
        "validation": np.full(len(y_validation), train_mean),
        "test": np.full(len(y_test), train_mean),
    }

    # 同一训练样本、同一标准化特征上的单特征与多特征线性回归。
    single_feature_model = LinearRegression().fit(x_train_bmi, y_train)
    multi_feature_model = LinearRegression().fit(x_train_scaled, y_train)
    ridge_models = {
        alpha: Ridge(alpha=alpha).fit(x_train_scaled, y_train) for alpha in RIDGE_ALPHAS
    }

    split_data = {
        "train": (x_train_bmi, x_train_scaled, y_train),
        "validation": (x_validation_bmi, x_validation_scaled, y_validation),
        "test": (x_test_bmi, x_test_scaled, y_test),
    }

    # 五组模型的训练、验证结果。此时尚未读取测试集指标来做配置选择。
    metric_rows = []
    for split in ("train", "validation"):
        x_bmi, x_full, y_true = split_data[split]
        metric_rows.append(metrics_row("Mean baseline", split, y_true, baseline_predictions[split]))
        metric_rows.append(
            metrics_row("Single-feature LinearRegression (bmi)", split, y_true, single_feature_model.predict(x_bmi))
        )
        metric_rows.append(
            metrics_row("Multi-feature LinearRegression", split, y_true, multi_feature_model.predict(x_full))
        )
        for alpha, model in ridge_models.items():
            metric_rows.append(metrics_row(f"Ridge alpha={alpha}", split, y_true, model.predict(x_full)))

    train_validation_metrics = pd.DataFrame(metric_rows)

    # 严格只用验证 RMSE 选择 alpha；若正好相同，固定选较小 alpha。
    ridge_validation = train_validation_metrics[
        (train_validation_metrics["split"] == "validation")
        & train_validation_metrics["model"].str.startswith("Ridge")
    ].copy()
    ridge_validation["alpha"] = ridge_validation["model"].str.extract(r"alpha=(\d+)").astype(int)
    selected_alpha = int(ridge_validation.sort_values(["RMSE", "alpha"]).iloc[0]["alpha"])
    selected_ridge_model = ridge_models[selected_alpha]

    # 只在 alpha 已由验证集确定后，报告四组要求的最终测试指标。
    x_test_bmi, x_test_full, y_test_values = split_data["test"]
    metric_rows.extend(
        [
            metrics_row("Mean baseline", "test", y_test_values, baseline_predictions["test"]),
            metrics_row(
                "Single-feature LinearRegression (bmi)",
                "test",
                y_test_values,
                single_feature_model.predict(x_test_bmi),
            ),
            metrics_row(
                "Multi-feature LinearRegression",
                "test",
                y_test_values,
                multi_feature_model.predict(x_test_full),
            ),
            metrics_row(
                f"Selected Ridge alpha={selected_alpha}",
                "test",
                y_test_values,
                selected_ridge_model.predict(x_test_full),
            ),
        ]
    )
    all_metrics = pd.DataFrame(metric_rows)
    all_metrics.to_csv(
        RESULTS_DIR / "metrics_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )

    # 三组多特征模型的标准化输入权重。截距单列保存，便于完整核对模型参数。
    weights = pd.DataFrame(
        {
            "feature": feature_names,
            "LinearRegression_coefficient": multi_feature_model.coef_,
            "Ridge_alpha_1_coefficient": ridge_models[1].coef_,
            "Ridge_alpha_100_coefficient": ridge_models[100].coef_,
        }
    )
    intercept_row = pd.DataFrame(
        {
            "feature": ["intercept"],
            "LinearRegression_coefficient": [multi_feature_model.intercept_],
            "Ridge_alpha_1_coefficient": [ridge_models[1].intercept_],
            "Ridge_alpha_100_coefficient": [ridge_models[100].intercept_],
        }
    )
    weights = pd.concat([weights, intercept_row], ignore_index=True)
    weights.to_csv(RESULTS_DIR / "multifeature_weights.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    # 分组柱状图：仅展示训练与验证 RMSE，不让测试数据参与选择。
    rmse_models = [
        "Mean baseline",
        "Single-feature LinearRegression (bmi)",
        "Multi-feature LinearRegression",
        "Ridge alpha=1",
        "Ridge alpha=100",
    ]
    plot_metrics = train_validation_metrics.pivot(index="model", columns="split", values="RMSE").reindex(rmse_models)
    positions = np.arange(len(plot_metrics))
    bar_width = 0.36
    plt.figure(figsize=(11, 5.5))
    plt.bar(positions - bar_width / 2, plot_metrics["train"], bar_width, label="Train RMSE", color="#4C78A8")
    plt.bar(positions + bar_width / 2, plot_metrics["validation"], bar_width, label="Validation RMSE", color="#F58518")
    plt.xticks(positions, ["Mean\nbaseline", "Single feature\n(bmi)", "Multi-feature\nLinearRegression", "Ridge\nalpha=1", "Ridge\nalpha=100"])
    plt.ylabel("RMSE")
    plt.title("Training and validation RMSE comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "train_validation_rmse_comparison.png", dpi=150)
    plt.close()

    # 权重图不含截距，因为 Ridge 的正则化约束的是特征权重；横轴为 10 个特征。
    feature_positions = np.arange(len(feature_names))
    bar_width = 0.25
    plt.figure(figsize=(12, 5.5))
    plt.bar(feature_positions - bar_width, multi_feature_model.coef_, bar_width, label="LinearRegression", color="#4C78A8")
    plt.bar(feature_positions, ridge_models[1].coef_, bar_width, label="Ridge alpha=1", color="#F58518")
    plt.bar(feature_positions + bar_width, ridge_models[100].coef_, bar_width, label="Ridge alpha=100", color="#54A24B")
    plt.xticks(feature_positions, feature_names)
    plt.xlabel("Feature (standardized using training statistics)")
    plt.ylabel("Coefficient")
    plt.title("Multi-feature model coefficient comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "multifeature_weight_comparison.png", dpi=150)
    plt.close()

    summary = "\n".join(
        [
            "Week 03: multi-feature regression and Ridge comparison",
            "=" * 55,
            "Data: sklearn load_diabetes(as_frame=True, scaled=False).",
            f"Fixed random seed: {RANDOM_STATE}; samples: train={len(y_train)}, validation={len(y_validation)}, test={len(y_test)}.",
            "Inputs: bmi only for the single-feature model; all 10 dataset features for multi-feature models.",
            "Preprocessing: StandardScaler was fitted on the training features only.",
            f"Baseline: training-target mean = {train_mean:.6f}.",
            f"Ridge candidates: alpha={RIDGE_ALPHAS[0]} and alpha={RIDGE_ALPHAS[1]}.",
            f"SELECTED RIDGE ALPHA (by validation RMSE only): {selected_alpha}",
            "The test set was evaluated only after alpha selection and was not used for tuning.",
            "Weights in multifeature_weights.csv correspond to standardized input features; they are not causal effects.",
        ]
    )
    print(summary)
    (RESULTS_DIR / "selection_summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
