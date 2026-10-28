"""原始油温尺度的评价以及保存预测后的完整性核对。"""

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred):
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape or not np.isfinite(pred).all():
        raise ValueError("预测与目标形状不一致，或预测含非有限值")
    error = pred - true
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def check_saved_predictions(predictions, metrics):
    """从逐行预测重新计算汇总指标，且检查目标下标唯一与模型覆盖。"""
    if predictions.duplicated(["model", "split", "target_index"]).any():
        raise ValueError("预测表存在重复模型、分组、目标下标")
    expected = metrics.set_index(["model", "split"])
    for (model, split), rows in predictions.groupby(["model", "split"], sort=False):
        summary = expected.loc[(model, split)]
        recomputed = regression_metrics(rows["y_true"], rows["y_pred"])
        if len(rows) != summary["n_samples"] or any(
            not np.isclose(recomputed[key], summary[key], rtol=0, atol=1e-8)
            for key in ("mae", "rmse")
        ):
            raise ValueError(f"保存的预测与汇总指标不一致: {model}/{split}")
    if len(predictions.groupby(["model", "split"])) != len(metrics):
        raise ValueError("预测表缺少某组模型或数据部分")
