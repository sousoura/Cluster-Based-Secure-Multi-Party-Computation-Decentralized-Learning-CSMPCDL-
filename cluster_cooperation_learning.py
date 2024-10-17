# cluster_cooperation_learning.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import random

class ClusterCooperativeLogisticRegression:
    def __init__(self, num_features, learning_rate=0.01, lambda_param=0.01):
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
        y_binary = 1 if y == 1 else 0

        error = y_prob - y_binary

        dw = x * error + 2 * self.lambda_param * self.w
        db = error
        return dw, db

    def update(self, dw, db):
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

def average_models(cluster_models):
    total_w = sum([model.w for model in cluster_models])
    total_b = sum([model.b for model in cluster_models])
    k = len(cluster_models)
    averaged_w = total_w / k
    averaged_b = total_b / k
    return averaged_w, averaged_b

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

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)

    shuffled_indices = np.random.permutation(len(X_train))
    X_train = X_train[shuffled_indices]
    y_train = y_train[shuffled_indices]

    return X_train, y_train, X_test, y_test

def split_data(X_train, y_train, n_learners=15):
    X_splits = np.array_split(X_train, n_learners)
    y_splits = np.array_split(y_train, n_learners)
    return X_splits, y_splits

def generate_clusters(n_learners=15, n_clusters=5, min_cluster_size=3):
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

def secure_aggregation(cluster_models):
    k = len(cluster_models)
    num_features = cluster_models[0].w.shape[0]

    shares_w = []
    shares_b = []
    for model in cluster_models:
        w_shares = []
        b_shares = []
        # 生成 k-1 个随机份额
        for _ in range(k - 1):
            w_share = np.random.randn(num_features)
            b_share = np.random.randn()
            w_shares.append(w_share)
            b_shares.append(b_share)
        # 最后一个份额确保所有份额之和等于原始模型参数
        w_last_share = model.w - sum(w_shares)
        b_last_share = model.b - sum(b_shares)
        w_shares.append(w_last_share)
        b_shares.append(b_last_share)
        shares_w.append(w_shares)
        shares_b.append(b_shares)

    received_shares_w = [[] for _ in range(k)]
    received_shares_b = [[] for _ in range(k)]
    for i in range(k):
        for j in range(k):
            # 成员 i 将份额 shares_w[i][j] 发送给成员 j
            received_shares_w[j].append(shares_w[i][j])
            received_shares_b[j].append(shares_b[i][j])

    partial_ws = []
    partial_bs = []
    for i in range(k):
        partial_w = sum(received_shares_w[i])
        partial_b = sum(received_shares_b[i])
        partial_ws.append(partial_w)
        partial_bs.append(partial_b)

    total_w = sum(partial_ws)
    total_b = sum(partial_bs)

    aggregated_w = total_w / k
    aggregated_b = total_b / k

    return aggregated_w, aggregated_b

def simulate_cluster_cooperative_learning(X_splits, y_splits, X_test, y_test, clusters, member_clusters, learning_rate=0.01, lambda_param=0.01):
    n_learners = len(X_splits)
    learners = [ClusterCooperativeLogisticRegression(X_splits[i].shape[1], learning_rate, lambda_param) for i in range(n_learners)]
    errors = [[] for _ in range(n_learners)]

    max_steps = max(len(X_splits[i]) for i in range(n_learners))

    for step in range(max_steps):
        # Local training
        for learner_id in range(n_learners):
            if step < len(X_splits[learner_id]):
                x = X_splits[learner_id][step]
                y = y_splits[learner_id][step]
                dw, db = learners[learner_id].compute_gradient(x, y)
                learners[learner_id].update(dw, db)

        # Secure aggregation within clusters
        for cluster_id, cluster_members in enumerate(clusters):
            cluster_models = [learners[i] for i in cluster_members]
            aggregated_w, aggregated_b = secure_aggregation(cluster_models)
            for member_id in cluster_members:
                learners[member_id].w = aggregated_w.copy()
                learners[member_id].b = aggregated_b

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
    plt.plot(range(1, len(errors) + 1), errors, label='Cluster Cooperative Learning Error Rate')
    plt.xlabel('Training Steps (across learners)')
    plt.ylabel('Error Rate')
    plt.title(f'Error Reduction in Cluster Cooperative Learning')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cluster_cooperative_learning_error_curve.png')
    plt.show()

def run_cluster_cooperative_learning():
    TRAIN_FILE = 'data.csv'
    TEST_FILE = 'test.csv'
    LEARNING_RATE = 0.01
    LAMBDA_PARAM = 0.01
    DEGREE = 2
    N_LEARNERS = 15
    N_CLUSTERS = 5

    X_train, y_train, X_test, y_test = load_data(TRAIN_FILE, TEST_FILE, degree=DEGREE)

    X_splits, y_splits = split_data(X_train, y_train, n_learners=N_LEARNERS)

    clusters, member_clusters = generate_clusters(n_learners=N_LEARNERS, n_clusters=N_CLUSTERS)
    print(f"Generated {N_CLUSTERS} clusters with {N_LEARNERS} learners.")
    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx}: Members {cluster}")

    errors = simulate_cluster_cooperative_learning(X_splits, y_splits, X_test, y_test, clusters, member_clusters,
                                                   learning_rate=LEARNING_RATE, lambda_param=LAMBDA_PARAM)
    return errors

if __name__ == "__main__":
    errors = run_cluster_cooperative_learning()
    plot_errors(errors)
