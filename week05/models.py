"""实践一的单层线性模型与实践二的单隐藏层 MLP。"""

import torch
from torch import nn


def create_step_model():
    model = nn.Linear(1, 1)
    # 只有一次更新的小例子使用零初始化，MLP 保持正常随机初始化。
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    return model


def create_mlp(hidden_dim=16):
    """hidden_dim 同时控制第一层输出和第二层输入，保证维数匹配。"""
    return nn.Sequential(nn.Linear(10, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))


def forward_shapes(model, batch):
    """仅检查各层形状，不训练、不改变参数。"""
    shapes = [tuple(batch.shape)]
    model.eval()
    with torch.no_grad():
        for layer in model:
            batch = layer(batch)
            shapes.append(tuple(batch.shape))
    return shapes
