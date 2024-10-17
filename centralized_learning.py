# centralized_learning.py

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

    def update(self, x, y):
        z = np.dot(self.w, x) + self.b
        y_prob = self.sigmoid(z)
        y_binary = 1 if y == 1 else 0

        error = y_prob - y_binary

        dw = x * error + 2 * self.lambda_param * self.w
        db = error

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

def plot_error(error_list):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(error_list) + 1), error_list, label='Test Error Rate')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs. Number of Training Samples')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('centralized_error_curve.png')
    plt.show()

def centralized_learning(train_file='data.csv', test_file='test.csv', learning_rate=0.01, lambda_param=0.01, degree=2):
    X_train, y_train, X_test, y_test = load_data(train_file, test_file, degree)
    num_features = X_train.shape[1]
    model = IncrementalLogisticRegression(num_features, learning_rate, lambda_param)

    error_list = []

    for i in range(len(X_train)):
        x = X_train[i]
        y = y_train[i]
        model.update(x, y)

        # 每训练一个样本，记录一次误差
        error = calculate_error(model, X_test, y_test)
        error_list.append(error)
        if (i+1) % 100 == 0 or i == len(X_train)-1:
            print(f"Trained on {i + 1} samples, Test Error Rate: {error:.4f}")

    return error_list

def run_centralized_learning():
    TRAIN_FILE = 'data.csv'
    TEST_FILE = 'test.csv'
    LEARNING_RATE = 0.01
    LAMBDA_PARAM = 0.01
    DEGREE = 2

    errors = centralized_learning(train_file=TRAIN_FILE, test_file=TEST_FILE,
                                  learning_rate=LEARNING_RATE, lambda_param=LAMBDA_PARAM, degree=DEGREE)
    return errors

if __name__ == "__main__":
    errors = run_centralized_learning()
    plot_error(errors)
