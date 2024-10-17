# individual_learning.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class IncrementalLogisticRegression:
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

def simulate_individual_learning(X_splits, y_splits, X_test, y_test, learning_rate=0.01, lambda_param=0.01):
    n_learners = len(X_splits)
    models = [IncrementalLogisticRegression(X_splits[i].shape[1], learning_rate, lambda_param) for i in range(n_learners)]
    errors = [[] for _ in range(n_learners)]

    max_steps = max(len(X_splits[i]) for i in range(n_learners))

    for step in range(max_steps):
        for learner_id in range(n_learners):
            if step < len(X_splits[learner_id]):
                x = X_splits[learner_id][step]
                y = y_splits[learner_id][step]
                dw, db = models[learner_id].compute_gradient(x, y)
                models[learner_id].update(dw, db)

        for learner_id in range(n_learners):
            error = calculate_error(models[learner_id], X_test, y_test)
            errors[learner_id].append(error)

        if (step + 1) % 100 == 0 or step == max_steps - 1:
            print(f"Individual: Step {step + 1}/{max_steps} completed.")

    merged_errors = []
    for i in range(len(errors[0])):
        for j in range(n_learners):
            if i < len(errors[j]):
                merged_errors.append(errors[j][i])

    return merged_errors

def run_individual_learning():
    TRAIN_FILE = 'data.csv'
    TEST_FILE = 'test.csv'
    LEARNING_RATE = 0.01
    LAMBDA_PARAM = 0.01
    DEGREE = 2
    N_LEARNERS = 15

    X_train, y_train, X_test, y_test = load_data(TRAIN_FILE, TEST_FILE, degree=DEGREE)
    X_splits, y_splits = split_data(X_train, y_train, n_learners=N_LEARNERS)
    errors = simulate_individual_learning(X_splits, y_splits, X_test, y_test, learning_rate=LEARNING_RATE, lambda_param=LAMBDA_PARAM)
    return errors

if __name__ == "__main__":
    errors = run_individual_learning()
    # 可选：绘制单独的误差曲线
