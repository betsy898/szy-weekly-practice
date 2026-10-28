"""读取 ETTh1、检查小时序列、生成窗口并按目标时间划分。"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).parent / "data" / "ETTh1.csv"
TRAIN_START = pd.Timestamp("2016-07-01 00:00")
VALIDATION_START = pd.Timestamp("2017-06-26 00:00")
TEST_START = pd.Timestamp("2017-10-24 00:00")
END = pd.Timestamp("2018-02-21 00:00")


def load_etth1(path=DATA_PATH, limit=None):
    """解析时间并检查顺序、间隔与本周使用的数值列。"""
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "OT", "HUFL"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV 缺少列: {required - set(frame.columns)}")
    if limit is not None:
        frame = frame.iloc[:limit].copy()
    if frame["date"].isna().any() or not frame["date"].is_monotonic_increasing or frame["date"].duplicated().any():
        raise ValueError("时间列缺失、逆序或重复")
    if not frame["date"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("记录并非逐小时连续")
    if not np.isfinite(frame[["OT", "HUFL"]].to_numpy(dtype=float)).all():
        raise ValueError("OT 或 HUFL 含非有限值")
    return frame.reset_index(drop=True)


def make_windows(series, window_size):
    """一行 X 对应 series[t-L:t]，标签 y 为 series[t]，下标 t 为目标。"""
    values = np.asarray(series, dtype=float).reshape(-1)
    if not 1 <= window_size < len(values):
        raise ValueError("window_size 必须在 1 与序列长度之间")
    targets = np.arange(window_size, len(values), dtype=int)
    x = np.stack([values[t - window_size:t] for t in targets])
    y = values[targets]
    assert x.shape == (len(values) - window_size, window_size)
    assert np.array_equal(x[:, -1], values[targets - 1])
    return x, y, targets


def split_by_target_time(x, y, target_index, times):
    """所有完整窗口先构造，再由目标所属时段分组。"""
    target_times = pd.DatetimeIndex(times[target_index])
    boundaries = {
        "train": (TRAIN_START, VALIDATION_START),
        "validation": (VALIDATION_START, TEST_START),
        "test": (TEST_START, END),
    }
    groups = {}
    membership = np.zeros(len(target_index), dtype=int)
    for name, (start, stop) in boundaries.items():
        mask = np.asarray((target_times >= start) & (target_times < stop))
        membership += mask.astype(int)
        groups[name] = {"X": x[mask], "y": y[mask], "target_index": target_index[mask],
                        "target_time": target_times[mask]}
    if not np.all(membership == 1):
        raise ValueError("可用目标存在遗漏或重复归组")
    return groups
