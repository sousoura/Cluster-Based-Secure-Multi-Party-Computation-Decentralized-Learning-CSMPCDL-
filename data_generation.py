import numpy as np
import pandas as pd


def generate_non_linear_data(num_samples_per_class=5000, num_features=10, random_seed=42):
    """
    生成用于逻辑回归分类的非线性二分类数据集。

    参数:
    - num_samples_per_class: 每个类别的样本数量
    - num_features: 每个样本的特征数量
    - random_seed: 随机种子，确保结果可重复

    返回:
    - train_df: 包含训练数据的DataFrame
    - test_df: 包含测试数据的DataFrame
    """
    np.random.seed(random_seed)

    # 类别1：内部类，特征值基于平方和小于某阈值
    mean1 = 0
    cov1 = 1
    class1 = np.random.randn(num_samples_per_class, num_features) * cov1 + mean1
    labels1 = np.ones((num_samples_per_class, 1))  # 标签为1

    # 类别2：外部类，特征值基于平方和大于某阈值
    mean2 = 0
    cov2 = 1
    class2 = np.random.randn(num_samples_per_class, num_features) * cov2 + mean2
    labels2 = -np.ones((num_samples_per_class, 1))  # 标签为-1

    # 合并数据
    data = np.vstack((class1, class2))
    labels = np.vstack((labels1, labels2))

    # 定义一个非线性边界，例如平方和的阈值
    threshold = num_features  # 阈值可以根据需要调整

    for i in range(len(data)):
        if np.sum(data[i] ** 2) > threshold:
            labels[i] = -1
        else:
            labels[i] = 1

    # 添加一些噪声以增加复杂性
    noise_ratio = 0.1
    num_noisy = int(noise_ratio * len(labels))
    noisy_indices = np.random.choice(len(labels), num_noisy, replace=False)
    labels[noisy_indices] *= -1  # 翻转标签

    # 创建DataFrame
    feature_columns = [f'feature{i + 1}' for i in range(num_features)]
    df = pd.DataFrame(data, columns=feature_columns)
    df['label'] = labels

    # 划分训练集和测试集
    train_df = df.sample(frac=0.8, random_state=random_seed)
    test_df = df.drop(train_df.index)

    # 保存为CSV文件
    train_df.to_csv('data.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    print("数据已生成并保存为 data.csv 和 test.csv")


if __name__ == "__main__":
    generate_non_linear_data()
