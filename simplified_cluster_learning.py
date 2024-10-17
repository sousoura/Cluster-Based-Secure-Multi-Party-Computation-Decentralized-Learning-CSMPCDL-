import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import random

class SimplifiedClusterLogisticRegression:
    def __init__(self, num_features, learning_rate=0.01, lambda_param=0.01):
        """
        初始化逻辑回归模型参数。

        参数:
        - num_features: 特征数量
        - learning_rate: 学习率
        - lambda_param: 正则化参数
        """
        self.w = np.zeros(num_features)
        self.b = 0.0
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_prob(self, X):
        return self.sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X):
        probabilities = self.predict_prob(X)
        return np.where(probabilities >= 0.5, 1, -1)

    def compute_gradient(self, x, y):
        z = np.dot(self.w, x) + self.b
        y_prob = self.sigmoid(z)
        y_binary = 1 if y == 1 else 0  # 转换标签为0和1

        error = y_prob - y_binary  # 注意这里是 y_prob - y_binary

        # 梯度计算
        dw = x * error + 2 * self.lambda_param * self.w
        db = error
        return dw, db

    def update(self, dw, db):
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

def calculate_error(model, X_test, y_test):
    predictions = model.predict(X_test)
    incorrect = np.sum(predictions != y_test)
    error_rate = incorrect / len(y_test)
    return error_rate

def load_data(train_file='data.csv', test_file='test.csv', degree=2):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    feature_columns = [col for col in train_df.columns if col.startswith('feature')]

    X_train = train_df[feature_columns].values
    y_train = train_df['label'].values
    X_test = test_df[feature_columns].values
    y_test = test_df['label'].values

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 多项式特征转换
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)

    # 打乱训练数据
    shuffled_indices = np.random.permutation(len(X_train))
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    return X_train, y_train, X_test, y_test

def split_data(X_train, y_train, n_learners=15):
    """
    将数据集分成n_learners份，模拟n_learners个成员节点。
    """
    X_splits = np.array_split(X_train, n_learners)
    y_splits = np.array_split(y_train, n_learners)
    return X_splits, y_splits

def generate_clusters(n_learners=15, n_clusters=5, min_cluster_size=3):
    """
    随机生成团（clusters），确保每个成员至少属于一个团。

    参数:
    - n_learners: 成员节点数量
    - n_clusters: 团的数量
    - min_cluster_size: 每个团的最小成员数量

    返回:
    - clusters: 列表，每个元素是一个团的成员列表
    - member_clusters: 字典，键为成员ID，值为该成员所属的团列表
    """
    learners = list(range(n_learners))
    clusters = []
    member_clusters = {i: [] for i in learners}

    for _ in range(n_clusters):
        cluster_size = random.randint(min_cluster_size, n_learners // n_clusters + 1)
        cluster_members = random.sample(learners, cluster_size)
        clusters.append(cluster_members)
        for member in cluster_members:
            member_clusters[member].append(len(clusters) - 1)

    # 确保每个成员至少属于一个团
    for member in learners:
        if not member_clusters[member]:
            cluster_id = random.randint(0, n_clusters - 1)
            clusters[cluster_id].append(member)
            member_clusters[member].append(cluster_id)

    return clusters, member_clusters

def average_models(cluster_models):
    """
    计算团内模型的平均值。

    参数:
    - cluster_models: 团内成员的模型列表

    返回:
    - averaged_w: 平均后的权重向量
    - averaged_b: 平均后的偏置
    """
    total_w = sum([model.w for model in cluster_models])
    total_b = sum([model.b for model in cluster_models])
    k = len(cluster_models)
    averaged_w = total_w / k
    averaged_b = total_b / k
    return averaged_w, averaged_b

def simulate_simplified_cluster_learning(X_splits, y_splits, X_test, y_test, clusters, member_clusters, learning_rate=0.01, lambda_param=0.01):
    """
    模拟简化的团学习过程，不考虑多方安全聚合。

    参数:
    - X_splits: 每个成员的训练数据
    - y_splits: 每个成员的标签数据
    - X_test: 测试集特征
    - y_test: 测试集标签
    - clusters: 团的成员列表
    - member_clusters: 每个成员所属的团列表
    - learning_rate: 学习率
    - lambda_param: 正则化参数

    返回:
    - errors: 合并后的误差列表，按照学习者的顺序排列
    """
    n_learners = len(X_splits)
    learners = [SimplifiedClusterLogisticRegression(X_splits[i].shape[1], learning_rate, lambda_param) for i in range(n_learners)]
    errors = [[] for _ in range(n_learners)]  # 存储每个成员的误差

    max_steps = max(len(X_splits[i]) for i in range(n_learners))

    for step in range(max_steps):
        # 每个成员本地训练
        for learner_id in range(n_learners):
            if step < len(X_splits[learner_id]):
                x = X_splits[learner_id][step]
                y = y_splits[learner_id][step]
                dw, db = learners[learner_id].compute_gradient(x, y)
                learners[learner_id].update(dw, db)

        # 团内成员计算平均模型并更新
        for cluster_id, cluster_members in enumerate(clusters):
            # 收集团内成员的模型
            cluster_models = [learners[i] for i in cluster_members]
            # 计算平均模型
            averaged_w, averaged_b = average_models(cluster_models)
            # 更新团内成员的模型
            for member_id in cluster_members:
                learners[member_id].w = averaged_w.copy()
                learners[member_id].b = averaged_b

        # 记录误差
        for learner_id in range(n_learners):
            error = calculate_error(learners[learner_id], X_test, y_test)
            errors[learner_id].append(error)

        # 打印进度
        if (step+1) % 100 == 0 or step == max_steps - 1:
            print(f"Step {step+1}/{max_steps} completed.")

    # 合并误差数据，按照绘图规则排列
    merged_errors = []
    for i in range(len(errors[0])):
        for j in range(n_learners):
            if i < len(errors[j]):
                merged_errors.append(errors[j][i])

    return merged_errors

def plot_errors(errors, n_learners=15):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(errors) + 1), errors, label='Simplified Cluster Learning Error Rate')
    plt.xlabel('Training Steps (across learners)')
    plt.ylabel('Error Rate')
    plt.title(f'Error Reduction in Simplified Cluster Learning')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('simplified_cluster_learning_error_curve.png')
    plt.show()

if __name__ == "__main__":
    TRAIN_FILE = 'data.csv'
    TEST_FILE = 'test.csv'
    LEARNING_RATE = 0.01
    LAMBDA_PARAM = 0.01
    DEGREE = 2  # 多项式特征的度数
    N_LEARNERS = 15  # 成员节点数量
    N_CLUSTERS = 5   # 团的数量

    # 加载数据
    X_train, y_train, X_test, y_test = load_data(TRAIN_FILE, TEST_FILE, degree=DEGREE)

    # 分割数据集，模拟 n_learners 个成员
    X_splits, y_splits = split_data(X_train, y_train, n_learners=N_LEARNERS)

    # 生成团
    clusters, member_clusters = generate_clusters(n_learners=N_LEARNERS, n_clusters=N_CLUSTERS)
    print(f"Generated {N_CLUSTERS} clusters with {N_LEARNERS} learners.")
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx}: Members {cluster}")

    # 模拟简化的团学习
    errors = simulate_simplified_cluster_learning(X_splits, y_splits, X_test, y_test, clusters, member_clusters, learning_rate=LEARNING_RATE, lambda_param=LAMBDA_PARAM)

    # 绘制误差曲线
    plot_errors(errors, n_learners=N_LEARNERS)
