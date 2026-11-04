"""复用第六周 ETTh1 数据读取与时间划分，准备 LSTM 张量。"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from week06.data import load_etth1, make_windows, split_by_target_time  # noqa: E402
from week06.models import fit_lag_linear, predict_last_value  # noqa: E402


def prepare_series(window_size=24, limit=14_400):
    frame = load_etth1(limit=limit)
    series = frame["OT"].to_numpy(dtype=float)
    x, y, indices = make_windows(series, window_size)
    groups = split_by_target_time(x, y, indices, frame["date"].to_numpy())
    train_end = int(groups["validation"]["target_index"][0])
    mean = float(series[:train_end].mean())
    std = float(series[:train_end].std(ddof=0))
    if std <= 0:
        raise ValueError("训练时段 OT 标准差须大于零")
    for group in groups.values():
        group["x_tensor"] = torch.tensor(((group["X"] - mean) / std)[:, :, None], dtype=torch.float32)
        group["y_tensor"] = torch.tensor(((group["y"] - mean) / std)[:, None], dtype=torch.float32)
        assert group["x_tensor"].shape[1:] == (window_size, 1)
        assert group["y_tensor"].shape[1:] == (1,)
    return frame, groups, mean, std


def train_loader(group, seed=42, batch_size=128):
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(group["x_tensor"], group["y_tensor"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                      num_workers=0, generator=generator)
