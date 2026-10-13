"""第四周实践：PCA 回归对照与 K-means 聚类。

所有预处理仅在训练集拟合。PCA 重构误差在标准化特征空间计算；
K-means 在 10 维标准化训练特征上运行，二维图仅为 PCA 投影展示。
"""

import os
from pathlib import Path

# Windows + MKL 下小样本 K-means 的推荐设置，避免无关的并行警告。
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
PCA_COMPONENTS = (2, 5)
K_VALUES = (2, 3)
KMEANS_N_INIT = 10
RESULTS_DIR = Path(__file__).parent / "results"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """返回回归 MAE 和 RMSE。"""
    mse = mean_squared_error(y_true, y_pred)
    return mean_absolute_error(y_true, y_pred), float(np.sqrt(mse))


def reconstruction_mse(x_scaled: np.ndarray, pca: PCA) -> float:
    """在标准化特征空间计算 PCA 投影再重构的平均平方误差。"""
    projected = pca.transform(x_scaled)
    reconstructed = pca.inverse_transform(projected)
    return float(np.mean((x_scaled - reconstructed) ** 2))


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # 加载原始特征；target 不进入输入 X，也不用于 PCA 或 K-means 拟合。
    dataset = load_diabetes(as_frame=True, scaled=False)
    feature_names = list(dataset.feature_names)
    x_all = dataset.frame[feature_names]
    y_all = dataset.target

    # 先按样本索引留出 40%，再平分得到验证集和测试集：约 60%/20%/20%。
    all_indices = np.arange(len(x_all))
    train_indices, holdout_indices = train_test_split(
        all_indices, test_size=0.4, random_state=RANDOM_STATE
    )
    validation_indices, test_indices = train_test_split(
        holdout_indices, test_size=0.5, random_state=RANDOM_STATE
    )
    x_train_original = x_all.iloc[train_indices]
    x_validation_original = x_all.iloc[validation_indices]
    x_test_original = x_all.iloc[test_indices]
    y_train = y_all.iloc[train_indices].to_numpy()
    y_validation = y_all.iloc[validation_indices].to_numpy()
    y_test = y_all.iloc[test_indices].to_numpy()

    pd.DataFrame(
        {"split": ["train", "validation", "test"], "samples": [len(y_train), len(y_validation), len(y_test)]}
    ).to_csv(RESULTS_DIR / "split_info.csv", index=False, encoding="utf-8-sig")

    # 标准化器只从训练集 10 个特征得到每列统计量，随后用于验证与测试。
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_original)
    x_validation_scaled = scaler.transform(x_validation_original)
    x_test_scaled = scaler.transform(x_test_original)

    # PCA(2)、PCA(5) 均只在标准化训练特征上学习投影方向。
    pca_models: dict[int, PCA] = {}
    pca_data: dict[int, dict[str, np.ndarray]] = {}
    pca_rows = []
    for components in PCA_COMPONENTS:
        pca = PCA(n_components=components, svd_solver="full", whiten=False)
        pca_models[components] = pca
        train_projected = pca.fit_transform(x_train_scaled)
        validation_projected = pca.transform(x_validation_scaled)
        test_projected = pca.transform(x_test_scaled)
        pca_data[components] = {
            "train": train_projected,
            "validation": validation_projected,
            "test": test_projected,
        }
        pca_rows.append(
            {
                "n_components": components,
                "cumulative_explained_variance_ratio": pca.explained_variance_ratio_.sum(),
                "train_reconstruction_MSE_standardized_space": reconstruction_mse(x_train_scaled, pca),
                "validation_reconstruction_MSE_standardized_space": reconstruction_mse(x_validation_scaled, pca),
            }
        )
    pd.DataFrame(pca_rows).to_csv(
        RESULTS_DIR / "pca_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )

    # 四组预测模型：基线、原始 10 维标准化特征、PCA 2 维、PCA 5 维。
    train_mean = float(y_train.mean())
    full_linear_model = LinearRegression().fit(x_train_scaled, y_train)
    pca_regression_models = {
        components: LinearRegression().fit(pca_data[components]["train"], y_train)
        for components in PCA_COMPONENTS
    }

    model_specs = [
        ("Mean baseline", None),
        ("LinearRegression (10 standardized features)", None),
        ("PCA + LinearRegression", 2),
        ("PCA + LinearRegression", 5),
    ]
    metric_rows = []
    validation_rmse_by_components = {}
    for model_name, components in model_specs:
        row = {"model": model_name, "pca_components": components, "selected_pca_configuration": False}
        for split, x_scaled, y_true in (
            ("train", x_train_scaled, y_train),
            ("validation", x_validation_scaled, y_validation),
        ):
            if model_name == "Mean baseline":
                prediction = np.full(len(y_true), train_mean)
            elif components is None:
                prediction = full_linear_model.predict(x_scaled)
            else:
                prediction = pca_regression_models[components].predict(pca_data[components][split])
            mae, rmse = regression_metrics(y_true, prediction)
            row[f"{split}_MAE"] = mae
            row[f"{split}_RMSE"] = rmse
            if split == "validation" and components is not None:
                validation_rmse_by_components[components] = rmse
        # 按任务要求，另一组 PCA 不在测试集做评价，所以对应列留空。
        row["test_MAE"] = np.nan
        row["test_RMSE"] = np.nan
        metric_rows.append(row)

    # 只根据验证 RMSE 选择 PCA 维度，若并列则选 2 维。
    selected_components = min(PCA_COMPONENTS, key=lambda n: (validation_rmse_by_components[n], n))
    for row in metric_rows:
        if row["pca_components"] == selected_components:
            row["selected_pca_configuration"] = True

    # 在选定维数后才计算规定模型的测试指标；不以测试结果重新选择维数。
    for row in metric_rows:
        model_name, components = row["model"], row["pca_components"]
        if model_name == "Mean baseline":
            test_prediction = np.full(len(y_test), train_mean)
        elif components is None:
            test_prediction = full_linear_model.predict(x_test_scaled)
        elif components == selected_components:
            test_prediction = pca_regression_models[components].predict(pca_data[components]["test"])
        else:
            continue
        row["test_MAE"], row["test_RMSE"] = regression_metrics(y_test, test_prediction)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    # 使用所有 10 个主成分的 PCA 仅绘制 1 至 10 维的累计解释方差趋势。
    pca_all = PCA(n_components=len(feature_names), svd_solver="full", whiten=False).fit(x_train_scaled)
    cumulative_variance = np.cumsum(pca_all.explained_variance_ratio_)

    # 训练/验证 RMSE 图：四组模型使用相同样本划分。
    plot_labels = ["Mean\nbaseline", "10 features\nLinearRegression", "PCA 2 +\nLinearRegression", "PCA 5 +\nLinearRegression"]
    positions = np.arange(len(metrics))
    width = 0.36
    plt.figure(figsize=(10.5, 5.5))
    plt.bar(positions - width / 2, metrics["train_RMSE"], width, label="Train RMSE", color="#4C78A8")
    plt.bar(positions + width / 2, metrics["validation_RMSE"], width, label="Validation RMSE", color="#F58518")
    plt.xticks(positions, plot_labels)
    plt.ylabel("RMSE")
    plt.title("Regression RMSE before and after PCA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rmse_comparison.png", dpi=150)
    plt.close()

    # 训练数据的累计解释方差曲线，并标记需要比较的 2、5 维位置。
    dimensions = np.arange(1, len(feature_names) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(dimensions, cumulative_variance, marker="o", color="#4C78A8", label="Cumulative explained variance")
    for components, color in [(2, "#F58518"), (5, "#54A24B")]:
        variance = cumulative_variance[components - 1]
        plt.axvline(components, linestyle="--", color=color)
        plt.scatter(components, variance, color=color, zorder=3, label=f"{components} components: {variance:.3f}")
    plt.xticks(dimensions)
    plt.xlabel("Number of retained principal components")
    plt.ylabel("Cumulative explained variance ratio")
    plt.title("PCA cumulative explained variance (training features)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pca_variance.png", dpi=150)
    plt.close()

    # K-means：仅在训练样本的 10 维标准化特征上拟合，target 不参与。
    kmeans_models: dict[int, KMeans] = {}
    cluster_metric_rows = []
    cluster_summary_rows = []
    for k in K_VALUES:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
        labels = kmeans.fit_predict(x_train_scaled)
        kmeans_models[k] = kmeans
        cluster_metric_rows.append(
            {
                "K": k,
                "silhouette_score": silhouette_score(x_train_scaled, labels, metric="euclidean"),
                "random_state": RANDOM_STATE,
                "n_init": KMEANS_N_INIT,
                "fit_feature_dimensions": x_train_scaled.shape[1],
            }
        )
        original_with_labels = x_train_original[["bmi", "bp"]].copy()
        original_with_labels["cluster"] = labels
        grouped = original_with_labels.groupby("cluster", sort=True).agg(
            sample_count=("cluster", "size"),
            mean_bmi_original=("bmi", "mean"),
            mean_bp_original=("bp", "mean"),
        )
        if grouped["sample_count"].sum() != len(y_train):
            raise ValueError("Cluster sample counts do not match the training sample count.")
        for cluster_number, values in grouped.iterrows():
            cluster_summary_rows.append(
                {
                    "K": k,
                    "cluster": cluster_number,
                    "sample_count": int(values["sample_count"]),
                    "mean_bmi_original": values["mean_bmi_original"],
                    "mean_bp_original": values["mean_bp_original"],
                }
            )
    pd.DataFrame(cluster_metric_rows).to_csv(
        RESULTS_DIR / "cluster_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )
    pd.DataFrame(cluster_summary_rows).to_csv(
        RESULTS_DIR / "cluster_summary.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )

    # 二维图复用实践一的 PCA(2) 投影。中心先在 10 维空间，再投影至同一坐标系。
    display_pca = pca_models[2]
    train_pc2 = pca_data[2]["train"]
    center_projections = {k: display_pca.transform(model.cluster_centers_) for k, model in kmeans_models.items()}
    all_display_points = np.vstack([train_pc2, *center_projections.values()])
    x_margin = (all_display_points[:, 0].max() - all_display_points[:, 0].min()) * 0.08
    y_margin = (all_display_points[:, 1].max() - all_display_points[:, 1].min()) * 0.08
    x_limits = (all_display_points[:, 0].min() - x_margin, all_display_points[:, 0].max() + x_margin)
    y_limits = (all_display_points[:, 1].min() - y_margin, all_display_points[:, 1].max() + y_margin)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    colors = plt.get_cmap("tab10")
    for axis, k in zip(axes, K_VALUES):
        labels = kmeans_models[k].labels_
        for cluster_number in range(k):
            mask = labels == cluster_number
            axis.scatter(
                train_pc2[mask, 0], train_pc2[mask, 1], s=28, alpha=0.72,
                color=colors(cluster_number), label=f"Cluster {cluster_number}",
            )
        centers = center_projections[k]
        axis.scatter(
            centers[:, 0], centers[:, 1], s=180, marker="X", color="black", edgecolor="white", linewidth=0.8, label="Projected center",
        )
        axis.set_title(f"K={k}: clustered in 10D, displayed in PCA 2D")
        axis.set_xlabel("PC1")
        axis.set_xlim(x_limits)
        axis.set_ylim(y_limits)
        axis.legend(fontsize=9)
    axes[0].set_ylabel("PC2")
    fig.suptitle("K-means comparison: 10D clustering, 2D PCA display")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cluster_comparison.png", dpi=150)
    plt.close(fig)

    summary = "\n".join(
        [
            "Week 04: PCA regression and K-means clustering",
            "=" * 52,
            "Data: sklearn load_diabetes(as_frame=True, scaled=False), 10 input features and target.",
            f"Fixed split seed={RANDOM_STATE}: train={len(y_train)}, validation={len(y_validation)}, test={len(y_test)}.",
            "StandardScaler and all PCA models were fitted on the training features only.",
            "PCA settings: svd_solver='full', whiten=False; reconstruction MSE is in standardized feature space.",
            f"SELECTED PCA COMPONENTS (by validation RMSE only): {selected_components}",
            "The test set was not used to choose the number of components.",
            f"K-means settings: K={K_VALUES}, random_state={RANDOM_STATE}, n_init={KMEANS_N_INIT}; fitted on 10D standardized training features.",
            "Cluster plot uses PCA(2) only to display the 10D K-means assignments and projected centers.",
        ]
    )
    print(summary)
    (RESULTS_DIR / "run_summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
