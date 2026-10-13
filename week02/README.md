# 实践二：单特征梯度下降与模型评价

本练习使用 scikit-learn 自带的 diabetes 回归练习数据集，采用 `load_diabetes(as_frame=True, scaled=False)` 保留原始特征。只使用一个特征 `bmi` 拟合线性关系 `预测值 = w × x + b`，不把样本行号当作时间，也不训练多特征模型。

## 文件说明

- `single_feature_gradient_descent.py`：完整实验脚本，包含随机划分、仅训练集标准化、均值基线、NumPy 梯度下降、`LinearRegression` 对照、评价指标及绘图。
- `requirements.txt`：实际使用的 Python 和第三方库版本。
- `results/`：运行脚本产生的 CSV、文本报告和 PNG 图，不依赖外部数据文件。

## 环境与运行

实际运行环境为 Python 3.13.5。进入本目录后运行：

```powershell
python -m pip install -r requirements.txt
python single_feature_gradient_descent.py
```

## 实验流程

1. 以随机种子 42 先留出 40% 数据，再将其平分为验证集和测试集，得到训练/验证/测试约 60%/20%/20%的同一份划分。
2. 对 `bmi` 使用 `StandardScaler`：只在训练集 `fit`，再将同一均值和标准差用于验证集、测试集的 `transform`。目标值不标准化。这样不会让验证集或测试集的信息泄露进训练过程。
3. 均值基线仅计算训练集目标值的均值，对每个样本输出此常数；它虽不使用特征，却能判断模型是否至少优于一个简单的参考方案。
4. NumPy 梯度下降从 `w=0, b=0` 出发，按全量训练样本的 MSE 梯度同步更新参数。固定训练 500 轮，分别尝试学习率 0.01、0.1。
5. 在验证集 RMSE 上从两种梯度下降配置中选择一项；测试集不参与选择。随后在测试集比较均值基线、选定梯度下降模型和 `LinearRegression`。

## 结果文件

- `results/split_info.csv`：三个数据集的样本数量。
- `results/train_validation_metrics.csv`：均值基线、两个梯度下降设置及 `LinearRegression` 在训练集和验证集上的 MAE、RMSE、MSE。
- `results/test_metrics.csv`：均值基线、验证集选定的梯度下降模型和 `LinearRegression` 的测试指标比较。
- `results/parameter_comparison.csv`：两种梯度下降和 `LinearRegression` 的权重、截距和训练 MSE。
- `results/loss_curves_learning_rates.png`：0.01 与 0.1 两种学习率的训练损失曲线。
- `results/test_predictions_scatter.png`：选定梯度下降模型的测试集真实值—预测值散点图；虚线 `y=x` 表示完美预测。
- `results/test_prediction_examples.csv`：前 5 个测试样本的原始 BMI、真实值、预测值和误差，供人工核对。
- `results/run_summary.txt`：划分、标准化统计量、基线和学习率选择的简要记录。

学习率更大不一定总是更好：在这个简单实验中它可能更快靠近最优值，但过大时会越过最优点、振荡甚至发散。训练损失下降也不保证新数据上误差同样较小，所以需要独立的验证集和测试集。
