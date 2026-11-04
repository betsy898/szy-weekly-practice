"""无参数更新的整集 MSE、原始尺度预测与指标。"""

import numpy as np
import torch


def regression_metrics(y_true, y_pred):
    """按逐行预测在原始 OT 尺度计算 MAE 和 RMSE。"""
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError("预测与目标形状不一致，或预测含非有限值")
    error = pred - true
    return {"mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2)))}


def predict_scaled(model, group, batch_size=1024):
    model.eval()
    with torch.no_grad():
        parts = [model(xb).cpu().numpy() for xb in group["x_tensor"].split(batch_size)]
    return np.concatenate(parts).reshape(-1)


def full_mse(model, group):
    error = predict_scaled(model, group) - group["y_tensor"].numpy().reshape(-1)
    return float(np.dot(error, error) / len(error))


def predict_raw(model, group, mean, std):
    return predict_scaled(model, group) * std + mean
