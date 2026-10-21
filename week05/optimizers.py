"""按名称与学习率创建无权重衰减的 SGD 或 Adam。"""

import torch


def create_optimizer(model, name, learning_rate):
    if name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0, weight_decay=0)
    if name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    raise ValueError(f"Unsupported optimizer: {name}")
