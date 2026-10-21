"""一次更新、逐批 MLP 训练、最佳验证状态保存与基线拟合。"""

import copy

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from torch import nn

from evaluation import scaled_mse


def _step_snapshot(stage, model, loss=None):
    weight, bias = model.weight, model.bias
    return {"stage": stage, "weight": weight.item(), "bias": bias.item(),
            "weight_grad": None if weight.grad is None else weight.grad.item(),
            "bias_grad": None if bias.grad is None else bias.grad.item(), "loss": loss}


def run_single_update(model, optimizer, x, y):
    """对两个样本做一次 SGD 更新，并记录清梯度、反传、更新三个阶段。"""
    loss_fn = nn.MSELoss()
    optimizer.zero_grad()
    pred = model(x)  # 当前参数前向计算。
    assert pred.shape == y.shape
    loss = loss_fn(pred, y)  # MSE 损失。
    trace = [_step_snapshot("after_zero_grad", model, loss.item())]
    old_weight, old_bias = model.weight.item(), model.bias.item()
    loss.backward()  # 自动求导，得到 weight.grad 和 bias.grad。
    trace.append(_step_snapshot("after_backward", model))
    weight_grad, bias_grad = model.weight.grad.item(), model.bias.grad.item()
    optimizer.step()  # SGD 根据当前梯度更新参数。
    with torch.no_grad():
        new_loss = loss_fn(model(x), y).item()  # 新参数要重新前向。
    trace.append(_step_snapshot("after_step", model, new_loss))
    lr = optimizer.param_groups[0]["lr"]
    assert np.isclose(trace[0]["loss"], 10.0)
    assert np.isclose(model.weight.item(), old_weight - lr * weight_grad, atol=1e-6)
    assert np.isclose(model.bias.item(), old_bias - lr * bias_grad, atol=1e-6)
    return trace


def fit_baselines(train_split):
    """训练均值基线与标准化输入上的线性回归，标签保持原始尺度。"""
    mean_baseline = float(np.mean(train_split["y_raw"]))
    linear = LinearRegression().fit(train_split["x"].cpu().numpy(), train_split["y_raw"].reshape(-1))
    return mean_baseline, linear


def train_mlp(model, optimizer, loader, train_split, validation_split, config_name, learning_rate, epochs=200):
    """每批更新，每轮对完整训练/验证集评价，并恢复验证 MSE 最佳的参数。"""
    loss_fn = nn.MSELoss()
    best_mse = float("inf")
    best_epoch = None
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            prediction = model(xb)  # 仅对当前批次做前向计算。
            assert prediction.shape == yb.shape
            loss = loss_fn(prediction, yb)
            loss.backward()  # 链式法则将最终损失的梯度传到所有层。
            optimizer.step()

        # 重新评价整个训练集与验证集，而不是使用最后一个 batch 的 loss。
        train_mse = scaled_mse(model, train_split)
        validation_mse = scaled_mse(model, validation_split)
        history.append({"configuration": config_name, "learning_rate": learning_rate,
                        "epoch": epoch, "train_MSE_standardized_target": train_mse,
                        "validation_MSE_standardized_target": validation_mse})
        if validation_mse < best_mse:  # 严格小于：相等时保留较早轮次。
            best_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_epoch": best_epoch,
            "best_validation_mse": best_mse}
