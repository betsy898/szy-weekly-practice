"""单特征 BMI 线性回归：NumPy 梯度下降、基线和 LinearRegression 对照。

数据是普通表格数据。本脚本使用随机划分，而不是把样本行号视为时间。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
LEARNING_RATES = (0.01, 0.1)
EPOCHS = 500
OUTPUT_DIR = Path(__file__).parent / "results"


def gradient_descent(x: np.ndarray, y: np.ndarray, learning_rate: float, epochs: int):
    """从 w=b=0 开始，用全量训练数据拟合 y = w*x + b。"""
    w, b = 0.0, 0.0
    losses = []
    for _ in range(epochs):
        # 预测步骤：x 和 y 都是一维数组，避免广播产生二维误差矩阵。
        prediction = w * x + b
        error = prediction - y
        mse = np.mean(error**2)
        losses.append(mse)

        # 梯度计算和同时更新步骤；两个梯度均来自更新前的 w、b。
        dw = 2 * np.mean(error * x)
        db = 2 * np.mean(error)
        w = w - learning_rate * dw
        b = b - learning_rate * db

    return w, b, np.asarray(losses)


def predict_gradient_descent(x: np.ndarray, w: float, b: float) -> np.ndarray:
    """使用梯度下降得到的参数进行预测。"""
    return w * x + b


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """返回回归任务常用的 MAE、RMSE 和 MSE。"""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mse),
        "MSE": mse,
    }


def metric_row(model: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    """组织为便于保存成 CSV 的一行指标。"""
    return {"model": model, "split": split, **calculate_metrics(y_true, y_pred)}


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 读取原始特征，仅选择 bmi 作为本周实验唯一的输入特征。
    dataset = load_diabetes(as_frame=True, scaled=False)
    x = dataset.frame[["bmi"]].to_numpy()
    y = dataset.target.to_numpy()

    # 先留出 40%，再把它均分为验证、测试：约为 60%/20%/20%。
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.4, random_state=RANDOM_STATE
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
    )

    # 仅在训练集 fit；验证集、测试集只能使用训练集得到的均值与标准差 transform。
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).ravel()
    x_validation_scaled = scaler.transform(x_validation).ravel()
    x_test_scaled = scaler.transform(x_test).ravel()

    split_info = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "samples": [len(y_train), len(y_validation), len(y_test)],
        }
    )
    split_info.to_csv(OUTPUT_DIR / "split_info.csv", index=False, encoding="utf-8-sig")

    # 不看任何输入特征的均值基线；均值只从训练目标计算。
    train_target_mean = y_train.mean()
    baseline_predictions = {
        "train": np.full_like(y_train, train_target_mean, dtype=float),
        "validation": np.full_like(y_validation, train_target_mean, dtype=float),
        "test": np.full_like(y_test, train_target_mean, dtype=float),
    }

    # 两个学习率共享完全相同的初始化和迭代轮数。
    gd_results = {}
    for learning_rate in LEARNING_RATES:
        w, b, losses = gradient_descent(
            x_train_scaled, y_train, learning_rate=learning_rate, epochs=EPOCHS
        )
        gd_results[learning_rate] = {"w": w, "b": b, "losses": losses}

    # sklearn 的普通最小二乘解只作为对照，不是本脚本的梯度下降实现。
    sklearn_model = LinearRegression().fit(x_train_scaled.reshape(-1, 1), y_train)

    metric_rows = []
    for split, features, labels in (
        ("train", x_train_scaled, y_train),
        ("validation", x_validation_scaled, y_validation),
    ):
        metric_rows.append(metric_row("Mean baseline", split, labels, baseline_predictions[split]))
        for learning_rate, result in gd_results.items():
            predictions = predict_gradient_descent(features, result["w"], result["b"])
            metric_rows.append(metric_row(f"GD lr={learning_rate}", split, labels, predictions))
        sklearn_predictions = sklearn_model.predict(features.reshape(-1, 1))
        metric_rows.append(metric_row("LinearRegression", split, labels, sklearn_predictions))

    train_validation_metrics = pd.DataFrame(metric_rows)
    train_validation_metrics.to_csv(
        OUTPUT_DIR / "train_validation_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )

    # 仅依据验证集 RMSE，在两个梯度下降设置中选择配置；不查看测试集后再调参。
    validation_gd_rows = train_validation_metrics[
        (train_validation_metrics["split"] == "validation")
        & train_validation_metrics["model"].str.startswith("GD")
    ]
    selected_model_name = validation_gd_rows.loc[validation_gd_rows["RMSE"].idxmin(), "model"]
    selected_learning_rate = float(selected_model_name.split("=")[1])
    selected_gd = gd_results[selected_learning_rate]

    # 在测试集上只评价：均值基线、已选定的梯度下降模型、库函数模型。
    selected_gd_test_prediction = predict_gradient_descent(
        x_test_scaled, selected_gd["w"], selected_gd["b"]
    )
    sklearn_test_prediction = sklearn_model.predict(x_test_scaled.reshape(-1, 1))
    test_metrics = pd.DataFrame(
        [
            metric_row("Mean baseline", "test", y_test, baseline_predictions["test"]),
            metric_row(f"Selected GD lr={selected_learning_rate}", "test", y_test, selected_gd_test_prediction),
            metric_row("LinearRegression", "test", y_test, sklearn_test_prediction),
        ]
    )
    test_metrics.to_csv(
        OUTPUT_DIR / "test_metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f"
    )

    parameter_comparison = pd.DataFrame(
        [
            {"model": f"GD lr={lr}", "weight_w": result["w"], "intercept_b": result["b"],
             "final_train_MSE": result["losses"][-1]}
            for lr, result in gd_results.items()
        ]
        + [{"model": "LinearRegression", "weight_w": sklearn_model.coef_[0],
            "intercept_b": sklearn_model.intercept_,
            "final_train_MSE": mean_squared_error(y_train, sklearn_model.predict(x_train_scaled.reshape(-1, 1)))}]
    )
    parameter_comparison.to_csv(
        OUTPUT_DIR / "parameter_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.8f"
    )

    # 损失曲线：同一初始点和轮数下，展示两种学习率的训练过程。
    plt.figure(figsize=(8, 5))
    for learning_rate, result in gd_results.items():
        plt.plot(range(1, EPOCHS + 1), result["losses"], label=f"learning rate = {learning_rate}")
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE loss")
    plt.title("Gradient descent loss for two learning rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curves_learning_rates.png", dpi=150)
    plt.close()

    # 选定梯度下降模型的测试集真实值—预测值散点图，y=x 是完美预测参考线。
    lower = min(y_test.min(), selected_gd_test_prediction.min())
    upper = max(y_test.max(), selected_gd_test_prediction.max())
    plt.figure(figsize=(6.5, 6))
    plt.scatter(y_test, selected_gd_test_prediction, alpha=0.75, color="#F58518", label="Test samples")
    plt.plot([lower, upper], [lower, upper], "--", color="black", label="y = x (perfect prediction)")
    plt.xlabel("Actual target value")
    plt.ylabel("Predicted target value")
    plt.title(f"Test predictions: gradient descent (lr={selected_learning_rate})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_predictions_scatter.png", dpi=150)
    plt.close()

    # 留出几条测试样本，方便核对真实值、预测值和误差。
    examples = pd.DataFrame(
        {
            "bmi_original": x_test.ravel(),
            "actual_target": y_test,
            "predicted_target": selected_gd_test_prediction,
            "error_prediction_minus_actual": selected_gd_test_prediction - y_test,
        }
    ).head(5)
    examples.to_csv(OUTPUT_DIR / "test_prediction_examples.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    report = "\n".join(
        [
            "Single-feature diabetes regression experiment",
            "=" * 45,
            "Data: sklearn load_diabetes(as_frame=True, scaled=False); feature: bmi; target: target.",
            f"Fixed random seed: {RANDOM_STATE}. Split sizes: train={len(y_train)}, validation={len(y_validation)}, test={len(y_test)}.",
            f"StandardScaler fitted on train bmi only: mean={scaler.mean_[0]:.6f}, scale={scaler.scale_[0]:.6f}.",
            f"Mean baseline value (calculated from y_train only): {train_target_mean:.6f}.",
            f"Selected GD setting by validation RMSE: learning_rate={selected_learning_rate}.",
            "The test set was not used to select the learning rate.",
            "\nFiles saved in results: split_info.csv, train_validation_metrics.csv, test_metrics.csv,",
            "parameter_comparison.csv, test_prediction_examples.csv, loss_curves_learning_rates.png,",
            "and test_predictions_scatter.png.",
        ]
    )
    print(report)
    (OUTPUT_DIR / "run_summary.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
