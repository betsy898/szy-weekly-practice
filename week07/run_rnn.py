"""实践一：五步共享参数 RNN 的前向、反向与顺序对照。"""

import torch

from data import load_etth1
from models import rnn_five_steps
from reporting import save_trace


def main():
    frame = load_etth1(limit=6)
    ot = frame.OT.to_numpy(dtype=float)
    z = torch.tensor((ot[:5] - ot[0]) / 10, dtype=torch.float64, requires_grad=True)
    target = torch.tensor((ot[5] - ot[0]) / 10, dtype=torch.float64)
    source_times = frame.date.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

    pred, original, (wx, wh, bias) = rnn_five_steps(z, [0, 1, 2, 3, 4], source_times, ot)
    loss = (pred - target) ** 2
    loss.backward()
    for row in original:
        row["case"] = "original"
        row["input_grad"] = float(z.grad[row["source_index"]])

    # 对照只做前向；新序列仍从 h0=0 开始，保持完全相同的参数数值。
    reordered_z = z.detach()
    reordered_pred, reordered, _ = rnn_five_steps(reordered_z, [0, 3, 2, 1, 4], source_times, ot)
    for row in reordered:
        row["case"] = "reordered_1_4_3_2_5"
        row["input_grad"] = None
    save_trace(original + reordered)

    print(f"目标 z6={target.item():.9f}; 原顺序预测={pred.item():.9f}, 损失={loss.item():.9f}")
    print(f"共享参数梯度: wx={wx.grad.item():.9f}, wh={wh.grad.item():.9f}, b={bias.grad.item():.9f}")
    print("五个输入梯度:", [round(float(v), 9) for v in z.grad])
    print(f"重排顺序预测={reordered_pred.item():.9f}; 第二步状态原顺序={original[1]['h']:.9f}, 重排={reordered[1]['h']:.9f}")


if __name__ == "__main__":
    main()
