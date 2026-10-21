# 第五周：MLP 与 PyTorch 训练基础

本目录包含两个可独立运行的实践：一次 PyTorch 线性回归更新，以及使用 diabetes 回归数据比较 MLP + SGD、MLP + Adam 与基线。数据由 scikit-learn 自带的 `load_diabetes(as_frame=True, scaled=False)` 加载，无需额外数据文件。

## 环境与运行

实际运行环境为 Python 3.13.5、PyTorch 2.14.1+cpu；其他依赖和版本见 `requirements.txt`。例如在 Windows PowerShell 进入 `week05` 后：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_step.py
.\.venv\Scripts\python.exe run_mlp.py
```

本机的独立 `.venv` 使用 `--system-site-packages` 创建以复用已有依赖；上面示例则创建隔离环境并安装全部依赖。运行入口会重新生成 `results/` 中对应的结果。CPU 即可，无需 GPU。

## 代码职责

- `data.py`：准备两个简单样本；读取 diabetes、固定划分、标准化、生成张量和 DataLoader。
- `models.py`：定义实践一的零初始化线性模型与可设隐藏层宽度的 MLP，并检查各层形状。
- `optimizers.py`：按配置创建 SGD 或 Adam。
- `training.py`：执行前向、损失、反向传播和更新；逐轮记录完整数据的指标、恢复最佳验证轮次，并拟合基线。
- `evaluation.py`：不更新参数地计算预测和指标，MLP 预测恢复原始目标尺度。
- `reporting.py`：保存已有记录、绘制图形，不进行训练。
- `run_step.py`：连接各组件运行一次梯度更新。
- `run_mlp.py`：连接各组件执行基线、两组 MLP 和隐藏层形状检查。

入口只在直接运行时启动，导入组件不会自动训练。可以从 `run_mlp.py` 按数据 → 模型 → 优化器 → 训练 → 评价 → 输出的调用顺序阅读。

## 实验设置

实践一使用 `x=[[1],[2]]`、`y=[[2],[4]]`、零初始化线性模型和学习率 0.1 的 SGD。`training.py` 中 `loss.backward()` 自动计算梯度，`optimizer.step()` 更新参数；更新后重新前向计算损失。

实践二采用随机种子 42 的约 60%/20%/20% 划分（265/88/89）。特征 `StandardScaler` 和目标均值、标准差 `std(ddof=0)` 都仅来自训练集；MLP 输入/标准化目标分别为 `(N, 10)` 和 `(N, 1)` 的 float32 张量。

两组实验从相同随机初始参数出发，各自使用种子 42 的 DataLoader，batch size 32、200 轮。SGD 学习率为 0.01、Adam 为 0.001。每轮完成后重新评价完整训练集和验证集的标准化目标 MSE；恢复各自验证 MSE 最小的轮次，再比较原始目标尺度的 MAE/RMSE。只以验证 RMSE 在两组 MLP 中选一组，相同时选 SGD；测试集不参与选择。均值基线使用训练目标均值，线性基线使用标准化输入与原始目标。

## 隐藏层宽度检查

`models.py` 的 `hidden_dim` 同时决定第一层 `Linear(10, hidden_dim)` 的输出维数和第二层 `Linear(hidden_dim, 1)` 的输入维数。`run_mlp.py` 中正式模型宽度为 16，另建宽度 8 的临时模型只做前向检查、不训练。运行时观察到：

```text
宽度 16：(32, 10) → (32, 16) → (32, 16) → (32, 1)
宽度 8： (32, 10) → (32, 8)  → (32, 8)  → (32, 1)
```

## 结果位置

- `results/step_trace.csv`：清梯度后、反传后、更新后的参数、梯度与适用的损失。
- `results/history.csv`：两组各 200 轮的训练/验证 MSE，目标为标准化尺度。
- `results/metrics.csv`：四组模型的训练/验证指标、规定模型的测试指标、最佳轮次和选定标记，指标为原始目标单位。
- `results/loss_curves.png`：两组训练/验证曲线和最佳轮次。
- `results/rmse_comparison.png`：四组模型训练/验证 RMSE，MLP 使用各自最佳轮次参数。
