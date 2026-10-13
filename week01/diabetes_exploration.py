"""读取 sklearn diabetes 回归数据，并生成基础统计和探索性图形。

本练习只认识数据，不训练任何模型。运行后，所有结果会写入同级 results 文件夹。
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes


OUTPUT_DIR = Path(__file__).parent / "results"


def main() -> None:
    """加载数据、打印并保存统计信息，再绘制并保存两张图。"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # scaled=False 保留数据集的原始特征取值。
    dataset = load_diabetes(as_frame=True, scaled=False)
    df = dataset.frame
    feature_columns = dataset.feature_names
    target_column = dataset.target.name

    shape_text = f"行列数: {df.shape[0]} 行, {df.shape[1]} 列"
    column_text = f"列名: {', '.join(df.columns)}"
    feature_text = f"特征列（模型输入）: {', '.join(feature_columns)}"
    target_text = f"目标列（要预测的连续数值）: {target_column}"

    missing_values = df.isna().sum()
    numeric_summary = df.describe().T

    report_parts = [
        "Diabetes 数据读取与绘图结果",
        "=" * 32,
        shape_text,
        column_text,
        feature_text,
        target_text,
        "\n前 5 行：",
        df.head().to_string(),
        "\n\n各列缺失值数量：",
        missing_values.to_string(),
        "\n说明：所有列的缺失值数量均为 0。",
        "\n\n数值列基本统计（count、mean、std、min、25%、50%、75%、max）：",
        numeric_summary.to_string(),
        "\n\n小修改：",
        "目的：把目标值直方图的分箱数从 10 改为 20，以查看较细粒度的分布形状。",
        "预期：20 个分箱会比 10 个分箱显示更多局部起伏，但每个柱子的样本数会减少。",
        "检查：已输出 target_distribution_20_bins.png，可看到更细的柱形划分。",
        "\n观察：目标值大致集中在中间范围，且 bmi 与目标值呈现总体正向关系，但散点并不落在一条直线上。",
        "不能仅凭这些图得出的结论：不能断言 bmi 会导致目标值变化，也不能据此判断任何预测模型的泛化性能。",
    ]
    report = "\n".join(report_parts)
    print(report)
    (OUTPUT_DIR / "data_summary.txt").write_text(report, encoding="utf-8")

    # 修改后的直方图：20 个分箱，而非较粗的 10 个分箱。
    plt.figure(figsize=(8, 5))
    plt.hist(df[target_column], bins=20, color="#4C78A8", edgecolor="white")
    plt.xlabel("Target value (disease progression measurement)")
    plt.ylabel("Number of samples")
    plt.title("Distribution of diabetes target values (20 bins)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "target_distribution_20_bins.png", dpi=150)
    plt.close()

    # 选 bmi 是为了观察一个原始特征与目标之间的关系；横轴不是样本行号或时间。
    plt.figure(figsize=(8, 5))
    plt.scatter(df["bmi"], df[target_column], alpha=0.7, color="#F58518")
    plt.xlabel("BMI (body mass index)")
    plt.ylabel("Target value (disease progression measurement)")
    plt.title("BMI and diabetes target value")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bmi_vs_target_scatter.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
