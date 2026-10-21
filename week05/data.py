"""数据读取、固定划分、训练集标准化，以及张量与批次准备。"""

import numpy as np
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def prepare_step_data():
    """实践一：两个形状均为 (2, 1) 的 float32 张量。"""
    return torch.tensor([[1.0], [2.0]], dtype=torch.float32), torch.tensor([[2.0], [4.0]], dtype=torch.float32)


def prepare_regression_data(seed=42):
    """固定划分 diabetes 样本，返回标准化特征、标准化标签及原始标签。"""
    dataset = load_diabetes(as_frame=True, scaled=False)
    features = dataset.frame[dataset.feature_names].to_numpy(dtype=np.float64)
    target = dataset.target.to_numpy(dtype=np.float64)
    indices = np.arange(len(target))
    train_idx, holdout_idx = train_test_split(indices, test_size=0.4, random_state=seed)
    val_idx, test_idx = train_test_split(holdout_idx, test_size=0.5, random_state=seed)

    scaler = StandardScaler().fit(features[train_idx])
    y_mean = float(target[train_idx].mean())
    y_std = float(target[train_idx].std(ddof=0))
    if y_std <= 0:
        raise ValueError("The training target standard deviation must be positive.")

    splits = {}
    for name, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        x_scaled = scaler.transform(features[idx])
        y_raw = target[idx].reshape(-1, 1)
        y_scaled = (y_raw - y_mean) / y_std
        splits[name] = {
            "x": torch.tensor(x_scaled, dtype=torch.float32),
            "y": torch.tensor(y_scaled, dtype=torch.float32),
            "y_raw": y_raw,
            "indices": idx,
        }
    return {"splits": splits, "scaler": scaler, "y_mean": y_mean, "y_std": y_std}


def make_train_loader(train_split, batch_size=32, seed=42):
    """每个配置独立建立同种子的生成器，保持批次顺序可比较。"""
    dataset = TensorDataset(train_split["x"], train_split["y"])
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                      num_workers=0, generator=generator)
