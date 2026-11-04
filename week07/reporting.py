"""保存实验数据与绘制结果，不承担模型训练。"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS = Path(__file__).parent / "results"
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def save_trace(rows):
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "rnn_trace.csv", index=False, encoding="utf-8-sig")


def save_forecast_results(history, metrics, predictions, best_epochs, chosen):
    RESULTS.mkdir(exist_ok=True)
    history.to_csv(RESULTS / "history.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULTS / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS / "predictions.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    for ax, name in zip(axes, ("lstm_h16", "lstm_h32")):
        rows = history[history.model == name]
        ax.plot(rows.epoch, rows.train_mse, label="Train MSE")
        ax.plot(rows.epoch, rows.val_mse, label="Validation MSE")
        ax.axvline(best_epochs[name], linestyle="--", color="black", label=f"Best epoch {best_epochs[name]}")
        ax.set_xlabel("Epoch")
        ax.set_title(name)
        ax.legend()
    axes[0].set_ylabel("MSE (standardized OT)")
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curves.png", dpi=150)
    plt.close(fig)

    start, stop = pd.Timestamp("2017-06-26"), pd.Timestamp("2017-07-03")
    week = predictions[(predictions.split == "validation") &
                       (predictions.target_time >= start) & (predictions.target_time < stop)]
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in ("last_value", "lag_linear", chosen):
        rows = week[week.model == name].sort_values("target_time")
        if len(rows) != 168:
            raise ValueError(f"{name} 验证首周不是 168 条")
        if name == "last_value":
            ax.plot(rows.target_time, rows.y_true, color="black", linewidth=1.8, label="Observed OT")
        ax.plot(rows.target_time, rows.y_pred, linewidth=1.1, label=name)
    ax.set_xlabel("Target time")
    ax.set_ylabel("油温 OT（原始数据尺度）")
    ax.set_title("First validation week: one hour ahead prediction")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS / "validation_predictions.png", dpi=150)
    plt.close(fig)
