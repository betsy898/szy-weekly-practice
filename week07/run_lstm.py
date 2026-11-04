"""实践二：两种 LSTM 宽度与上周基线的同目标时间对照。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import fit_lag_linear, predict_last_value, prepare_series, train_loader
from evaluation import predict_raw, regression_metrics
from models import OilLSTM, fit_lstm
from reporting import RESULTS, save_forecast_results


def main():
    torch.set_num_threads(2)
    _, groups, mean, std = prepare_series()
    train, validation = groups["train"], groups["validation"]
    assert {key: len(value["y"]) for key, value in groups.items()} == {
        "train": 8616, "validation": 2880, "test": 2880}
    print("示例窗口：", groups["validation"]["target_time"][0],
          "输入最后一小时 OT=", validation["X"][0, -1], "目标 OT=", validation["y"][0])

    lag_linear = fit_lag_linear(train["X"], train["y"])
    trained, best_epochs, history_rows, shapes = {}, {}, [], {}
    for hidden_size in (16, 32):
        torch.manual_seed(42)
        model = OilLSTM(hidden_size)
        name = f"lstm_h{hidden_size}"
        shapes[name] = model.inspect(train["x_tensor"][:128])
        loader = train_loader(train, seed=42, batch_size=128)
        history, best_epoch = fit_lstm(model, train, validation, loader, epochs=40)
        trained[name], best_epochs[name] = model, best_epoch
        history_rows.extend(history)
    history = pd.DataFrame(history_rows)

    predictors = {"last_value": lambda g: predict_last_value(g["X"]),
                  "lag_linear": lambda g: lag_linear.predict(g["X"])}
    for name, model in trained.items():
        predictors[name] = lambda g, model=model: predict_raw(model, g, mean, std)
    validation_rmse = {name: regression_metrics(validation["y"], predictors[name](validation))["rmse"]
                       for name in ("lstm_h16", "lstm_h32")}
    chosen = min(("lstm_h16", "lstm_h32"), key=lambda name: validation_rmse[name])

    metric_rows, prediction_parts = [], []
    for name, predict in predictors.items():
        for split in ("train", "validation", "test"):
            if split == "test" and name.startswith("lstm") and name != chosen:
                continue
            group = groups[split]
            y_pred = np.asarray(predict(group), dtype=float).reshape(-1)
            stats = regression_metrics(group["y"], y_pred)
            metric_rows.append({"model": name, "split": split, "n_samples": len(y_pred),
                                "hidden_size": trained[name].hidden_size if name in trained else np.nan,
                                "best_epoch": best_epochs[name] if name in trained else np.nan,
                                "mae": stats["mae"], "rmse": stats["rmse"],
                                "chosen_lstm": name == chosen})
            prediction_parts.append(pd.DataFrame({"model": name, "split": split,
                    "target_index": group["target_index"], "target_time": group["target_time"],
                    "y_true": group["y"], "y_pred": y_pred}))
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    if predictions.duplicated(["model", "split", "target_index"]).any():
        raise ValueError("预测目标重复")
    for row in metrics.itertuples():
        subset = predictions[(predictions.model == row.model) & (predictions.split == row.split)]
        actual = regression_metrics(subset.y_true, subset.y_pred)
        assert len(subset) == row.n_samples
        assert np.isclose(actual["mae"], row.mae, atol=1e-8)
        assert np.isclose(actual["rmse"], row.rmse, atol=1e-8)

    RESULTS.mkdir(exist_ok=True)
    torch.save(trained[chosen].state_dict(), RESULTS / "lstm_best.pt")
    metadata = {"model": chosen, "hidden_size": trained[chosen].hidden_size,
                "best_epoch": best_epochs[chosen], "ot_train_mean": mean,
                "ot_train_std_ddof0": std, "window_size": 24, "seed": 42}
    (RESULTS / "lstm_best_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_forecast_results(history, metrics, predictions, best_epochs, chosen)
    print("形状与参数数量:", shapes)
    print("最佳轮次:", best_epochs, "选定:", chosen)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
