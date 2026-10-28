"""保存已生成的记录，并将时间序列与评价结果绘制成图。"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import TEST_START, VALIDATION_START


RESULTS = Path(__file__).parent / "results"
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def save_window_examples(rows):
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "window_examples.csv", index=False, encoding="utf-8-sig")


def save_tables(tuning, metrics, predictions):
    RESULTS.mkdir(exist_ok=True)
    tuning.to_csv(RESULTS / "tuning.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULTS / "metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS / "predictions.csv", index=False, encoding="utf-8-sig")


def plot_tuning(tuning):
    labels = ["lag_linear" if row.model == "lag_linear" else f"ridge α={row.param_value:g}"
              for row in tuning.itertuples()]
    x = np.arange(len(tuning))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, tuning["train_rmse"], width, label="Train RMSE", color="#4C78A8")
    ax.bar(x + width / 2, tuning["val_rmse"], width, label="Validation RMSE", color="#F58518")
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE (OT original scale)")
    ax.set_title("Lag linear and Ridge configurations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "tuning_comparison.png", dpi=150)
    plt.close(fig)


def plot_series_split(frame, window_size):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(frame["date"], frame["OT"], linewidth=0.7, color="#4C78A8", label="OT")
    ax.axvspan(frame["date"].iloc[0], frame["date"].iloc[window_size - 1],
               color="#54A24B", alpha=0.32, label="Initial history for first sample")
    ax.axvline(VALIDATION_START, linestyle="--", color="#F58518", label="Validation start")
    ax.axvline(TEST_START, linestyle="--", color="#E45756", label="Test start")
    ax.set_xlabel("Date")
    ax.set_ylabel("油温 OT（原始数据尺度）")
    ax.set_title("ETTh1 first 14,400 hours: time based split")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS / "series_split.png", dpi=150)
    plt.close(fig)


def plot_validation_week(predictions, frame):
    start = VALIDATION_START
    end = start + pd.Timedelta(days=7)
    subset = predictions[(predictions["split"] == "validation") &
                         (predictions["target_time"] >= start) &
                         (predictions["target_time"] < end)].copy()
    observed = subset[subset["model"] == "last_value"].sort_values("target_time")
    if len(observed) != 168:
        raise ValueError("验证集首周不是完整的 168 小时")
    load = frame[(frame["date"] >= start) & (frame["date"] < end)].sort_values("date")
    if len(load) != 168:
        raise ValueError("HUFL 首周不是完整的 168 小时")

    fig, (ax_ot, ax_load) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                          gridspec_kw={"height_ratios": [2, 1]})
    ax_ot.plot(observed["target_time"], observed["y_true"], color="black", linewidth=1.7, label="Observed OT")
    colors = {"last_value": "#4C78A8", "window_mean": "#F58518",
              "lag_linear": "#54A24B", "ridge_tuned": "#B279A2"}
    for model, color in colors.items():
        rows = subset[subset["model"] == model].sort_values("target_time")
        if len(rows) != 168:
            raise ValueError(f"{model} 首周预测记录不完整")
        ax_ot.plot(rows["target_time"], rows["y_pred"], linewidth=1.05,
                   alpha=0.84, label=model, color=color)
    ax_ot.set_ylabel("油温 OT（原始数据尺度）")
    ax_ot.set_title("First validation week: one hour ahead OT forecasts")
    ax_ot.legend(ncol=3)

    ax_load.plot(load["date"], load["HUFL"], color="#E45756", linewidth=1.1, label="HUFL")
    ax_load.set_ylabel("HUFL（原始数据尺度）")
    ax_load.set_xlabel("Date / hour")
    ax_load.legend()
    ax_load.xaxis.set_major_locator(mdates.DayLocator())
    ax_load.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS / "validation_predictions.png", dpi=150)
    plt.close(fig)
