"""只接收已计算的结果并保存 CSV/图形；不拟合或训练模型。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).parent / "results"


def save_step_trace(trace):
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(trace).to_csv(RESULTS_DIR / "step_trace.csv", index=False, encoding="utf-8-sig", float_format="%.8f")


def save_mlp_results(history_records, metric_records, best_epochs):
    RESULTS_DIR.mkdir(exist_ok=True)
    history = pd.DataFrame(history_records)
    metrics = pd.DataFrame(metric_records)
    history.to_csv(RESULTS_DIR / "history.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, name in zip(axes, ("MLP + SGD", "MLP + Adam")):
        rows = history[history["configuration"] == name]
        axis.plot(rows["epoch"], rows["train_MSE_standardized_target"], label="Train MSE")
        axis.plot(rows["epoch"], rows["validation_MSE_standardized_target"], label="Validation MSE")
        epoch = best_epochs[name]
        val_mse = rows.loc[rows["epoch"] == epoch, "validation_MSE_standardized_target"].iloc[0]
        axis.axvline(epoch, linestyle="--", color="black", alpha=0.65, label=f"Best epoch: {epoch}")
        axis.scatter([epoch], [val_mse], color="black", s=25, zorder=3)
        axis.set_title(name)
        axis.set_xlabel("Epoch")
        axis.legend()
    axes[0].set_ylabel("MSE (standardized target)")
    fig.suptitle("Training and validation curves — standardized target")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)
    plt.close(fig)

    positions = np.arange(len(metrics))
    width = 0.36
    plt.figure(figsize=(10, 5.5))
    plt.bar(positions - width / 2, metrics["train_RMSE"], width, label="Train RMSE", color="#4C78A8")
    plt.bar(positions + width / 2, metrics["validation_RMSE"], width, label="Validation RMSE", color="#F58518")
    plt.xticks(positions, metrics["model"])
    plt.ylabel("RMSE (original target units)")
    plt.title("Train and validation RMSE (MLP at best validation epoch)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rmse_comparison.png", dpi=150)
    plt.close()
