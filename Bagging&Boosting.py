import numpy as np
from collections import Counter
from DecisionTreeClassifer import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report

class BaggingClassifier:
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0):
        self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier(max_depth=3)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.estimators_ = []

    def fit(self, X, y):
        n_samples = X.shape[0]    # 行数等于样本数
        sub_sample_size = int(n_samples * self.max_samples)    # 抽取的样本数
        self.estimators_ = []    # 基学习器
        for _ in range(self.n_estimators):
            """自主采样：从 [0, 1, ..., n_samples-1] 中随机抽取 sub_sample_size 个索引，允许重复采样"""
            indices = np.random.choice(n_samples, sub_sample_size, replace=True)
            X_subset = X[indices]
            y_subset = y[indices]

            """训练基分类器"""
            estimator = self.base_estimator
            estimator.fit(X_subset, y_subset)
            self.estimators_.append(estimator)

    def predict(self, X):
        """predictions维度是 n_estimators × X.shape[0]"""
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        """多数投票"""
        return np.array([Counter(predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])

class AdaBoostClassifier:
    def __init__(self, base_estimator=None, n_estimators=50):
        self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier(max_depth=2)
        self.n_estimators = n_estimators
        self.estimators_ = []
        self.estimators_weights_ = []
        self.y_classes_ = None

    def fit(self, X, y):
        self.y_classes_ = np.unique(y)
        y = np.where(y == self.y_classes_[0], -1, 1)
        n_samples = X.shape[0]
        sample_weights = np.ones(n_samples) / n_samples    # 初始化权重均为 1/n

        self.estimators_ = []
        self.estimators_weights_ = []

        for _ in range(self.n_estimators):
            """训练器分类器"""
            estimator = self.base_estimator
            estimator.fit(X, y, sample_weight=sample_weights)
            self.estimators_.append(estimator)
            prediction = estimator.predict(X)    # 训练数据进行预测，找到分类错误的样本

            """计算加权错误率"""
            incorrect = prediction != y
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)

            """计算分类器权重"""
            error = np.clip(error, 1e-10, 1 - 1e-10)  # 防止 error=0 或 error=1
            estimator_weight = np.log((1-error) / error) / 2
            self.estimators_weights_.append(estimator_weight)

            """更新样本权重"""
            for i in range(n_samples):
                # 分类错的样本
                if incorrect[i]:
                    sample_weights[i] *= np.exp(estimator_weight)
                # 分类对的样本
                else:
                    sample_weights[i] *= np.exp(-estimator_weight)
            """样本权重归一化"""
            sample_weights /= np.sum(sample_weights)

    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        """加权投票"""
        weighted_votes = np.dot(self.estimators_weights_, predictions)
        return np.where(weighted_votes >= 0, self.y_classes_[1], self.y_classes_[0])  # 还原原始标签

if __name__ == '__main__':
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    """Bagging模型"""
    bagging = BaggingClassifier(n_estimators=50, max_samples=0.8)
    bagging.fit(X_train, y_train)
    y_pred_bagging = bagging.predict(X_test)
    print(classification_report(y_test, y_pred_bagging))

    """AdaBoost模型"""
    adaboost = AdaBoostClassifier(n_estimators=50)
    adaboost.fit(X_train, y_train)
    y_pred_adaboost = adaboost.predict(X_test)
    print(classification_report(y_test, y_pred_adaboost))