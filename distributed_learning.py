import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import networkx as nx


class DistributedLogisticRegression:
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

        error = y_binary - y_prob

        # 梯度计算
        dw = -x * error + 2 * self.lambda_param * self.w
        db = -error
        return dw, db

    def update(self, dw, db):
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def average_models(self, neighbor_models):
        """
        与邻居的模型参数进行平均。

        参数:
        - neighbor_models: 邻居的模型列表
        """
        total_w = self.w.copy()
        total_b = self.b
        for model in neighbor_models:
            total_w += model.w
            total_b += model.b
        num_models = len(neighbor_models) + 1  # 包含自身
        self.w = total_w / num_models
        self.b = total_b / num_models


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


def split_data(X_train, y_train, n_splits=15):
    """
    将数据集分成n_splits份，模拟n_splits个学习者。

    参数:
    - X_train: 训练集特征
    - y_train: 训练集标签
    - n_splits: 学习者的数量

    返回:
    - X_splits: 列表，包含每个学习者的训练数据
    - y_splits: 列表，包含每个学习者的标签数据
    """
    X_splits = np.array_split(X_train, n_splits)
    y_splits = np.array_split(y_train, n_splits)
    return X_splits, y_splits


def generate_topology(n_learners=15, topology_type='random'):
    """
    生成学习者之间的邻居拓扑结构。

    参数:
    - n_learners: 学习者数量
    - topology_type: 拓扑类型（'ring', 'line', 'random', 'star' 等）

    返回:
    - G: NetworkX 图对象，表示学习者之间的邻居关系
    """
    if topology_type == 'ring':
        G = nx.cycle_graph(n_learners)
    elif topology_type == 'line':
        G = nx.path_graph(n_learners)
    elif topology_type == 'star':
        G = nx.star_graph(n_learners - 1)
    elif topology_type == 'fully_connected':
        G = nx.complete_graph(n_learners)
    elif topology_type == 'random':
        G = nx.erdos_renyi_graph(n_learners, p=0.3, seed=42)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(n_learners, p=0.3, seed=np.random.randint(1000))
    else:
        raise ValueError("Unsupported topology type")
    return G


def simulate_distributed_learning(X_splits, y_splits, X_test, y_test, topology, learning_rate=0.01, lambda_param=0.01):
    n_learners = len(X_splits)
    learners = [DistributedLogisticRegression(X_splits[i].shape[1], learning_rate, lambda_param) for i in range(n_learners)]
    errors = [[] for _ in range(n_learners)]

    max_steps = max(len(X_splits[i]) for i in range(n_learners))

    for step in range(max_steps):
        # Local update
        for learner_id in range(n_learners):
            if step < len(X_splits[learner_id]):
                x = X_splits[learner_id][step]
                y = y_splits[learner_id][step]
                dw, db = learners[learner_id].compute_gradient(x, y)
                learners[learner_id].update(dw, db)

        # Model averaging with neighbors
        for learner_id in range(n_learners):
            neighbors = list(topology.neighbors(learner_id))
            neighbor_models = [learners[n_id] for n_id in neighbors]
            learners[learner_id].average_models(neighbor_models)

        # Record errors
        for learner_id in range(n_learners):
            error = calculate_error(learners[learner_id], X_test, y_test)
            errors[learner_id].append(error)

        if (step+1) % 100 == 0 or step == max_steps - 1:
            print(f"Step {step+1}/{max_steps} completed.")

    merged_errors = []
    for i in range(len(errors[0])):
        for j in range(n_learners):
            if i < len(errors[j]):
                merged_errors.append(errors[j][i])

    return merged_errors

def plot_errors(errors, n_learners=15):
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(errors) + 1), errors, label='Distributed Learning Error Rate')
    plt.xlabel('Training Steps (across learners)')
    plt.ylabel('Error Rate')
    plt.title(f'Error Reduction in Distributed Learning')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('distributed_learning_error_curve.png')
    plt.show()

def run_distributed_learning(TOPOLOGY_TYPE="random"):
    TRAIN_FILE = 'data.csv'
    TEST_FILE = 'test.csv'
    LEARNING_RATE = 0.01
    LAMBDA_PARAM = 0.01
    DEGREE = 2
    N_LEARNERS = 15

    X_train, y_train, X_test, y_test = load_data(TRAIN_FILE, TEST_FILE, degree=DEGREE)

    X_splits, y_splits = split_data(X_train, y_train, n_splits=N_LEARNERS)

    topology = generate_topology(N_LEARNERS, TOPOLOGY_TYPE)
    print(f"Generated {TOPOLOGY_TYPE} topology with {N_LEARNERS} learners.")

    errors = simulate_distributed_learning(X_splits, y_splits, X_test, y_test, topology,
                                           learning_rate=LEARNING_RATE, lambda_param=LAMBDA_PARAM)
    return errors

if __name__ == "__main__":
    errors = run_distributed_learning()
    plot_errors(errors)