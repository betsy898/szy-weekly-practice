# 第四周：PCA 降维与 K-means 聚类

本目录完成两项实践：以 PCA 降维前后的线性回归对照，以及 K-means 无监督聚类。两者使用 scikit-learn 自带的 diabetes 回归练习数据集，通过 `load_diabetes(as_frame=True, scaled=False)` 在运行时加载，因此不依赖外部数据文件。

## 文件说明

- `pca_regression_and_kmeans.py`：实际运行的完整脚本，包含数据划分、标准化、PCA 回归对照、重构误差、K-means、轮廓系数、组内统计和图形导出。
- `requirements.txt`：实际使用的 Python 与依赖版本。
- `results/`：脚本生成的 4 份 CSV、3 张 PNG 和辅助记录。

## 运行方法

实际运行环境为 Python 3.13.5。进入 `week04` 后执行：

```powershell
python -m pip install -r requirements.txt
python pca_regression_and_kmeans.py
```

## 共同数据处理

随机种子固定为 `42`，先留出 40% 样本，再均分为验证集和测试集，得到训练/验证/测试 265/88/89 个样本。`StandardScaler` 只在训练集的 10 个输入特征上拟合；验证集和测试集只使用该统计量变换，`target` 从不进入 PCA 或 K-means 的输入。

## 实践一：PCA 回归对照

比较均值基线、10 个标准化特征的线性回归、PCA 2 维线性回归和 PCA 5 维线性回归。PCA 设置均为 `svd_solver="full"`、`whiten=False`，只用训练特征学习投影方向。根据两个 PCA 模型的**验证集 RMSE**选择保留维数；若相同则选择 2 维。测试集仅在选择后用于最终评价，另一 PCA 配置的测试指标留空。

`pca_summary.csv` 中的重构 MSE 以“标准化特征 → PCA 投影 → inverse_transform 恢复”的结果与原标准化特征相减后计算。它衡量输入特征的近似恢复程度，不是目标 `target` 的预测误差。

## 实践二：K-means 聚类

K-means 使用训练集的 **10 维标准化特征**，分别以 `K=2`、`K=3` 独立拟合，参数为 `random_state=42`、`n_init=10`。轮廓系数评价当前特征空间和欧氏距离下的聚类结构，并不是分类准确率。聚类图将同一批结果投影到 PCA 2 维平面显示；组号只作为标签，不能直接理解为正常、异常或真实类别。

## 结果文件

- `results/metrics.csv`：PCA 回归的训练/验证 MAE、RMSE，规定模型的测试 MAE、RMSE，以及选定 PCA 配置标记。
- `results/pca_summary.csv`：PCA 2、5 维的累计解释方差比例与训练/验证重构 MSE。
- `results/rmse_comparison.png`：四组回归模型的训练/验证 RMSE。
- `results/pca_variance.png`：训练数据上 1–10 个主成分的累计解释方差曲线，标出 2、5 维。
- `results/cluster_metrics.csv`：K=2、K=3 的轮廓系数和 K-means 设置。
- `results/cluster_summary.csv`：各组样本数以及原始 BMI、BP 的组内均值。
- `results/cluster_comparison.png`：K=2、K=3 的双面板图；在 10 维空间聚类、在 PCA 二维空间展示，叉号为投影后的中心。
- `results/run_summary.txt`：选定 PCA 维数及关键设置。
