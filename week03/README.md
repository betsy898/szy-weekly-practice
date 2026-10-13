# 第三周实践一：多特征回归与 Ridge 对照

本目录完成 diabetes 表格回归数据上的线性回归与 Ridge 对照。数据由 scikit-learn 自带的 `load_diabetes(as_frame=True, scaled=False)` 在运行时加载，使用原始输入特征；目标为 `target`。本实验不把目标列或样本行号作为输入，也不把表格数据当作时序数据。

## 文件说明

- `multifeature_ridge_regression.py`：实际运行的完整回归脚本。
- `requirements.txt`：运行所需的 Python 与第三方库版本。
- `results/`：脚本生成的指标、权重、选择记录和图形。

## 运行环境与方法

实际运行环境：Python 3.13.5。进入 `week03` 后执行：

```powershell
python -m pip install -r requirements.txt
python multifeature_ridge_regression.py
```

## 关键设置

- 随机种子：`42`；样本按约 60%/20%/20% 划分为训练集、验证集、测试集（实际为 265/88/89）。
- 标准化：`StandardScaler` 只在训练集的 10 个特征上 `fit`，验证集和测试集只做 `transform`；目标值不标准化。
- 模型：训练集目标均值基线、`bmi` 单特征 `LinearRegression`、全特征 `LinearRegression`、全特征 `Ridge(alpha=1)` 和 `Ridge(alpha=100)`。
- 配置选择：仅根据两组 Ridge 的**验证集 RMSE**选择 `alpha`；脚本明确记录选定值，测试集从不参与选择。

## 结果位置

- `results/metrics_comparison.csv`：五组模型的训练和验证 MAE/RMSE/MSE，以及基线、单特征、多特征和选定 Ridge 的最终测试指标。
- `results/multifeature_weights.csv`：全特征线性回归、Ridge `alpha=1`、Ridge `alpha=100` 的 10 个标准化特征权重及截距。
- `results/train_validation_rmse_comparison.png`：五组模型的训练/验证 RMSE 分组柱状图。
- `results/multifeature_weight_comparison.png`：三组多特征模型的标准化特征权重比较图。
- `results/selection_summary.txt`：固定划分、预处理和**选定的 Ridge alpha**等关键信息。
- `results/split_info.csv`：各数据集样本数。

增加特征或施加 Ridge 正则化不保证验证、测试误差一定更低。Ridge 的 `alpha` 是权重 L2 惩罚强度，不是梯度下降的学习率；权重大小也不能单独解释为特征的重要程度或因果作用。
