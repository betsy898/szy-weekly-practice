"""不更新参数的模型评价：标准化 MSE 与原始尺度上的 MAE/RMSE。"""

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def predict_scaled(model, x):
    model.eval()
    with torch.no_grad():
        return model(x).cpu().numpy()


def scaled_mse(model, split):
    prediction = predict_scaled(model, split["x"])
    return float(np.mean((prediction - split["y"].cpu().numpy()) ** 2))


def raw_metrics(y_raw, prediction_raw):
    y = np.asarray(y_raw).reshape(-1)
    pred = np.asarray(prediction_raw).reshape(-1)
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred)))}


def evaluate_mlp_raw(model, split, y_mean, y_std):
    prediction_raw = predict_scaled(model, split["x"]) * y_std + y_mean
    return raw_metrics(split["y_raw"], prediction_raw)


def evaluate_baseline_raw(model, split):
    """均值基线用数值表示，线性回归使用已拟合的 sklearn 模型。"""
    if isinstance(model, float):
        prediction = np.full_like(split["y_raw"], model)
    else:
        prediction = model.predict(split["x"].cpu().numpy())
    return raw_metrics(split["y_raw"], prediction)
