import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.metrics import classification_report

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, sample_weight=None):
        """构建决策树"""
        if sample_weight is None:
            sample_weight = np.ones(len(y)) / len(y)
        self.n_features = X.shape[1]    # X的列数对应特征种类，也就是决策树节点数
        self.tree = self._grow_tree(X, y, sample_weight)

    def _grow_tree(self, X, y, sample_weight, depth=0):
        """递归构建决策树"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))    # 葡萄酒的种类数

        # 停止条件
        if self.max_depth and depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            """返回当前数据子集中出现最频繁的类别标签作为叶子节点类别"""
            """加权投票"""
            class_counts = {}
            total_weight = 0
            for cls, w in zip(y, sample_weight):
                class_counts[cls] = class_counts.get(cls, 0) + w
                total_weight += w
            majority_class = max(class_counts.items(), key=lambda x: x[1])[0]
            return {'class': majority_class}

        # 寻找最佳分隔
        best_feature, best_threshold = self._best_split(X, y, sample_weight)

        # 如果没有找到有效分隔，返回叶节点
        if best_feature is None:
            """返回当前数据子集中出现最频繁的类别标签作为叶子节点类别"""
            """加权投票"""
            class_counts = {}
            total_weight = 0
            for cls, w in zip(y, sample_weight):
                class_counts[cls] = class_counts.get(cls, 0) + w
                total_weight += w
            majority_class = max(class_counts.items(), key=lambda x: x[1])[0]
            return {'class': majority_class}

        # 递归构造左右子树
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold
        left = self._grow_tree(X[left_idx], y[left_idx], sample_weight[left_idx], depth+1)
        right = self._grow_tree(X[right_idx], y[right_idx], sample_weight[right_idx], depth+1)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left,
            'right': right
        }

    def _best_split(self, X, y, sample_weight):
        """寻找最佳分隔特征和阈值"""
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature in range(self.n_features):    # 遍历所有特征
            """
                thresholds:阈值
            """
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:    # 遍历该特征下所有样本，找到使基尼系数最小的那个样本的该特征值作为阈值
                left_idx = X[:, feature] <= threshold    # left_idx是判断其他样本该特征是否比阈值小的布尔类型的列表
                gini = self._gini_impurity(y[left_idx], y[~left_idx], sample_weight[left_idx], sample_weight[~left_idx])

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _gini_impurity(self, left_y, right_y, left_weights, right_weights):
        """加权基尼不纯度"""
        total_weight = np.sum(left_weights) + np.sum(right_weights)
        p_left = np.sum(left_weights) / total_weight
        p_right = np.sum(right_weights) / total_weight

        def _gini(y, weights):
            classes = np.unique(y)
            gini = 1.0
            for c in classes:
                p = np.sum(weights[y == c]) / np.sum(weights)
                gini -= p**2
            return gini
        return p_left * _gini(left_y, left_weights) + p_right * _gini(right_y, right_weights)

    def predict(self, X):
        """预测类别"""
        return np.array([self._predict(x, self.tree) for x in X])

    def _predict(self, x, node):
        """递归预测单个样本"""
        if 'class' in node:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict(x, node['left'])
        else:
            return self._predict(x, node['right'])

if __name__ == '__main__':
    wine = load_wine()
    X, y = wine['data'], wine['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    print(classification_report(y_test, y_pred))