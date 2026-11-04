"""RNN 五步演示、LSTM 预测器、基线与 LSTM 训练。"""

import copy

import torch
from torch import nn

from evaluation import full_mse


def rnn_five_steps(z, order, source_times, source_values):
    """同一 wx、wh、b 反复作用于五个输入，记录时不截断计算图。"""
    wx = torch.tensor(0.6, requires_grad=True)
    wh = torch.tensor(0.5, requires_grad=True)
    bias = torch.tensor(0.0, requires_grad=True)
    h = torch.tensor(0.0)
    trace = []
    for step, source_index in enumerate(order, start=1):
        x = z[source_index]
        input_term = wx * x
        memory_term = wh * h
        next_h = torch.tanh(input_term + memory_term + bias)
        trace.append({"step": step, "source_index": source_index,
                      "source_time": source_times[source_index],
                      "source_ot": source_values[source_index],
                      "z": x.item(), "h_prev": h.item(),
                      "input_term": input_term.item(), "memory_term": memory_term.item(),
                      "h": next_h.item()})
        h = next_h
    prediction = 0.8 * h + 0.1
    return prediction, trace, (wx, wh, bias)


class OilLSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.recurrent = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1,
                                 batch_first=True, dropout=0, proj_size=0)
        self.readout = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, (h_n, c_n) = self.recurrent(x)  # 默认每个窗口零 h/c，窗口间不传状态。
        prediction = self.readout(output[:, -1, :])
        return prediction

    def inspect(self, x):
        self.eval()
        with torch.no_grad():
            output, (h_n, c_n) = self.recurrent(x)
            prediction = self.readout(output[:, -1, :])
        assert torch.allclose(output[:, -1, :], h_n[-1])
        return {"input": tuple(x.shape), "output": tuple(output.shape),
                "h_n": tuple(h_n.shape), "c_n": tuple(c_n.shape),
                "prediction": tuple(prediction.shape),
                "parameters": sum(p.numel() for p in self.parameters())}


def fit_lstm(model, train, validation, loader, epochs=40):
    """每批更新，每轮整集评价，恢复验证 MSE 最低的参数。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    loss_fn = nn.MSELoss()
    best_mse, best_epoch, best_state = float("inf"), None, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            prediction = model(xb)
            loss = loss_fn(prediction, yb)
            loss.backward()
            optimizer.step()
        train_mse = full_mse(model, train)
        val_mse = full_mse(model, validation)
        history.append({"model": f"lstm_h{model.hidden_size}", "hidden_size": model.hidden_size,
                        "epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})
        if val_mse < best_mse:
            best_mse, best_epoch = val_mse, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history, best_epoch
