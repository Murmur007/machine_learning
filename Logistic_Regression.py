from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weight = None
        self.bias = None

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.weight = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iters):
            linear_pred = np.dot(X_train, self.weight) + self.bias
            y_pred = 1 / (1 + np.exp(-linear_pred))

            # 计算梯度
            dw = (1/n_samples) * np.dot(X_train.T,(y_pred - y_train))
            db = (1/n_samples) * np.sum(y_pred - y_train)
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X_test):
        linear_pred = np.dot(X_test, self.weight) + self.bias
        y_pred = 1 / (1 + np.exp(-linear_pred))
        return [1 if p >= 0.5 else 0 for p in y_pred]

if __name__ == '__main__':
    # 生成二分类数据
    X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 初始化模型
    model = LogisticRegression(learning_rate=0.01, n_iters=1000)

    # 训练
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.2f}")  # 输出示例: Accuracy: 0.87