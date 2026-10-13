# 实践二：数据读取与绘图

本目录使用 scikit-learn 自带的 `diabetes` 回归练习数据集完成第一周实践二。数据由 `load_diabetes(as_frame=True, scaled=False)` 在运行时加载，不需要下载外部文件；`scaled=False` 表示使用原始特征。

## 文件说明

- `diabetes_exploration.py`：读取数据、输出数据概览和统计信息，并绘制图形。
- `requirements.txt`：实际运行时的 Python 与库版本。
- `results/data_summary.txt`：脚本运行后保存的数据概览、缺失值统计、描述性统计和文字结论。
- `results/target_distribution_20_bins.png`：目标值的 20 分箱直方图。
- `results/bmi_vs_target_scatter.png`：BMI 特征与目标值的散点图。

## 数据说明

数据共有 442 行、11 列。`age`、`sex`、`bmi`、`bp`、`s1` 到 `s6` 为 10 个特征列；`target` 是要预测的连续数值目标列。它是表格回归数据，因此图中的横轴使用特征 `bmi`，不将样本行号当作时间。

## 运行方式

建议在 Python 3.13 环境中执行：

```powershell
python -m pip install -r requirements.txt
python diabetes_exploration.py
```

运行完成后，在 `results` 文件夹中查看统计报告和两张 PNG 图。

## 小修改与检查

初始设想是采用 10 个分箱绘制目标值直方图。本次将分箱数改为 20，目的是查看更细粒度的分布形状。预期柱子会更窄、能显示更多局部起伏，但单个柱中的样本数会更少；生成的 `target_distribution_20_bins.png` 符合这一预期。

## 主要步骤和结论

脚本先加载数据并取得 DataFrame，再用 `shape`、`columns`、`head`、`isna().sum()` 和 `describe()` 了解数据；之后用 Matplotlib 分别保存直方图和散点图。

一个观察是：目标值主要集中在中间范围，且 BMI 和目标值呈总体正向关系，但点并不排列成一条直线。仅凭这些图不能断言 BMI 导致目标值变化，也不能判断任何模型在新数据上的预测表现。
