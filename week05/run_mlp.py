"""实践二入口：配置实验并连接数据、模型、训练、评价与保存组件。"""

import copy

import numpy as np
import torch

from data import make_train_loader, prepare_regression_data
from evaluation import evaluate_baseline_raw, evaluate_mlp_raw
from models import create_mlp, forward_shapes
from optimizers import create_optimizer
from reporting import save_mlp_results
from training import fit_baselines, train_mlp


def main():
    seed, epochs, batch_size = 42, 200, 32
    data = prepare_regression_data(seed=seed)
    splits = data["splits"]

    # 两组都深复制同一随机初始网络，之后独立创建优化器与同种子 DataLoader。
    torch.manual_seed(seed)
    initial_model = create_mlp(hidden_dim=16)
    models = {"MLP + SGD": copy.deepcopy(initial_model), "MLP + Adam": copy.deepcopy(initial_model)}
    print("Hidden width 16 forward shapes:", forward_shapes(initial_model, splits["train"]["x"][:batch_size]))
    width_8_model = create_mlp(hidden_dim=8)
    print("Hidden width 8 forward shapes:", forward_shapes(width_8_model, splits["train"]["x"][:batch_size]))

    baseline, linear = fit_baselines(splits["train"])
    configurations = (("MLP + SGD", "sgd", 0.01), ("MLP + Adam", "adam", 0.001))
    trained = {}
    all_history = []
    for name, optimizer_name, learning_rate in configurations:
        model = models[name]
        optimizer = create_optimizer(model, optimizer_name, learning_rate)
        loader = make_train_loader(splits["train"], batch_size=batch_size, seed=seed)
        result = train_mlp(model, optimizer, loader, splits["train"], splits["validation"],
                           name, learning_rate, epochs=epochs)
        trained[name] = {**result, "learning_rate": learning_rate}
        all_history.extend(result["history"])

    rows = []
    for name, model, is_mlp in (("Mean baseline", baseline, False), ("LinearRegression", linear, False),
                                ("MLP + SGD", trained["MLP + SGD"]["model"], True),
                                ("MLP + Adam", trained["MLP + Adam"]["model"], True)):
        row = {"model": name, "learning_rate": trained[name]["learning_rate"] if is_mlp else np.nan,
               "best_epoch": trained[name]["best_epoch"] if is_mlp else np.nan,
               "selected_mlp": False, "test_MAE": np.nan, "test_RMSE": np.nan}
        for split in ("train", "validation"):
            if is_mlp:
                result = evaluate_mlp_raw(model, splits[split], data["y_mean"], data["y_std"])
            else:
                result = evaluate_baseline_raw(model, splits[split])
            row[f"{split}_MAE"] = result["MAE"]
            row[f"{split}_RMSE"] = result["RMSE"]
        rows.append(row)

    # 仅用验证 RMSE 选 MLP；同分保留列表中较早的 SGD。
    selected = min(rows[2:], key=lambda row: row["validation_RMSE"])["model"]
    for row in rows:
        name = row["model"]
        if name == "Mean baseline":
            result = evaluate_baseline_raw(baseline, splits["test"])
        elif name == "LinearRegression":
            result = evaluate_baseline_raw(linear, splits["test"])
        elif name == selected:
            row["selected_mlp"] = True
            result = evaluate_mlp_raw(trained[name]["model"], splits["test"], data["y_mean"], data["y_std"])
        else:
            continue
        row["test_MAE"], row["test_RMSE"] = result["MAE"], result["RMSE"]

    best_epochs = {name: result["best_epoch"] for name, result in trained.items()}
    save_mlp_results(all_history, rows, best_epochs)
    print(f"Split sizes: train={len(splits['train']['y'])}, validation={len(splits['validation']['y'])}, test={len(splits['test']['y'])}")
    print(f"Training target mean={data['y_mean']:.6f}, std(ddof=0)={data['y_std']:.6f}")
    print("Best epochs:", best_epochs, "; selected by validation RMSE:", selected)
    for row in rows:
        print(row)
    print("Saved results/history.csv, metrics.csv, loss_curves.png, rmse_comparison.png")


if __name__ == "__main__":
    main()
