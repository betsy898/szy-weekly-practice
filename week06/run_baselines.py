"""实践二入口：24 小时单步预测、Ridge 调参、时间验证与结果保存。"""

import numpy as np
import pandas as pd

from data import (DATA_PATH, END, TEST_START, VALIDATION_START, load_etth1,
                  make_windows, split_by_target_time)
from evaluation import check_saved_predictions, regression_metrics
from models import (fit_lag_linear, fit_ridge, predict_last_value,
                    predict_window_mean)
from reporting import (plot_series_split, plot_tuning, plot_validation_week,
                       save_tables)


WINDOW_SIZE = 24
HOURS_USED = 14_400
RIDGE_ALPHAS = (0.1, 10.0, 1000.0)
MODEL_ORDER = ("last_value", "window_mean", "lag_linear", "ridge_tuned")


def _tuning_row(model_name, alpha, model, groups):
    train = groups["train"]
    validation = groups["validation"]
    train_metrics = regression_metrics(train["y"], model.predict(train["X"]))
    val_metrics = regression_metrics(validation["y"], model.predict(validation["X"]))
    return {"model": model_name, "param_name": "alpha" if alpha is not None else None,
            "param_value": alpha, "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"], "train_rmse": train_metrics["rmse"],
            "val_rmse": val_metrics["rmse"],
            "rmse_gap": val_metrics["rmse"] - train_metrics["rmse"],
            "chosen_candidate": False}


def main():
    frame = load_etth1(DATA_PATH, limit=HOURS_USED)
    if frame["date"].iloc[0] != pd.Timestamp("2016-07-01 00:00") or frame["date"].iloc[-1] != END - pd.Timedelta(hours=1):
        raise ValueError("前 14,400 小时与任务给出的划分边界不一致")
    series = frame["OT"].to_numpy(dtype=float)
    x, y, target_index = make_windows(series, WINDOW_SIZE)
    groups = split_by_target_time(x, y, target_index, frame["date"].to_numpy())
    counts = {name: len(group["y"]) for name, group in groups.items()}
    if sum(counts.values()) != HOURS_USED - WINDOW_SIZE:
        raise ValueError("可用目标数量与窗口数量不符")
    print("目标样本数：", counts)
    for name in ("validation", "test"):
        first = groups[name]
        t = int(first["target_index"][0])
        print(f"{name} 首条: 输入 {frame['date'].iloc[t-WINDOW_SIZE]} 至 {frame['date'].iloc[t-1]}; "
              f"目标 {frame['date'].iloc[t]}; 最近 OT={first['X'][0, -1]:.3f}, 实测 OT={first['y'][0]:.3f}")
    assert groups["validation"]["target_time"][0] == VALIDATION_START
    assert groups["test"]["target_time"][0] == TEST_START
    assert np.array_equal(groups["validation"]["X"][0], series[groups["validation"]["target_index"][0]-WINDOW_SIZE:
                                                                   groups["validation"]["target_index"][0]])

    train = groups["train"]
    lag_linear = fit_lag_linear(train["X"], train["y"])
    tuning_rows = [_tuning_row("lag_linear", None, lag_linear, groups)]
    ridge_candidates = {}
    for alpha in RIDGE_ALPHAS:
        ridge = fit_ridge(train["X"], train["y"], alpha)
        ridge_candidates[alpha] = ridge
        tuning_rows.append(_tuning_row("ridge", alpha, ridge, groups))
    chosen_row = min(tuning_rows[1:], key=lambda row: (row["val_rmse"], row["param_value"]))
    chosen_row["chosen_candidate"] = True
    chosen_alpha = chosen_row["param_value"]
    tuned = ridge_candidates[chosen_alpha]
    tuning = pd.DataFrame(tuning_rows)

    predictors = {
        "last_value": predict_last_value,
        "window_mean": predict_window_mean,
        "lag_linear": lag_linear.predict,
        "ridge_tuned": tuned.predict,
    }
    # 只根据完整验证集 RMSE 选最终方法，同分按 MODEL_ORDER 先后决定。
    validation_rmse = {
        name: regression_metrics(groups["validation"]["y"], fn(groups["validation"]["X"]))["rmse"]
        for name, fn in predictors.items()
    }
    selected = min(MODEL_ORDER, key=lambda name: validation_rmse[name])

    metric_rows = []
    prediction_frames = []
    for name in MODEL_ORDER:
        for split in ("train", "validation", "test"):
            group = groups[split]
            prediction = np.asarray(predictors[name](group["X"]), dtype=float).reshape(-1)
            stats = regression_metrics(group["y"], prediction)
            metric_rows.append({"model": name, "split": split, "n_samples": len(group["y"]),
                                **stats, "selected": name == selected})
            prediction_frames.append(pd.DataFrame({
                "model": name, "split": split, "target_index": group["target_index"],
                "target_time": group["target_time"], "y_true": group["y"], "y_pred": prediction,
            }))
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    check_saved_predictions(predictions, metrics)

    save_tables(tuning, metrics, predictions)
    # 与磁盘上的 CSV 再核对一次，确认序列化后仍可由逐行预测还原指标。
    saved_predictions = pd.read_csv("results/predictions.csv")
    saved_metrics = pd.read_csv("results/metrics.csv")
    check_saved_predictions(saved_predictions, saved_metrics)
    plot_tuning(tuning)
    plot_series_split(frame, WINDOW_SIZE)
    plot_validation_week(predictions, frame)

    print(f"Ridge 候选 {RIDGE_ALPHAS}; 验证 RMSE 选 alpha={chosen_alpha:g}")
    print(f"四种最终方法中，验证 RMSE 选 {selected}")
    print(tuning.to_string(index=False))
    print(metrics.to_string(index=False))
    example_time = pd.Timestamp("2017-06-26 03:00")
    example = predictions[(predictions["target_time"] == example_time) &
                          (predictions["split"] == "validation")]
    print(f"\n{example_time} 的逐模型核对：")
    print(example[["model", "target_index", "y_true", "y_pred"]].to_string(index=False))


if __name__ == "__main__":
    main()
