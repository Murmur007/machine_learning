import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class PCA:
    def __init__(self, n_components=2):
        """
        n_components: 要保留的主成分数量
        """
        self.n_components = n_components
        self.components = None    # 主成分（特征向量）
        self.mean = None         # 均值（用于中心化）

    def fit(self, X):
        """训练 PCA 模型"""
        """1.中心化数据"""
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        """2.计算协方差矩阵"""
        cov_matrix = np.cov(X_centered, rowvar=False)

        """3.计算特征值和特征向量"""
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        """4.按特征值降序排序特征向量"""
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        """5.选择前n_components个主成分"""
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        """将数据投影到主成分空间"""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """训练并立即转换数据"""
        self.fit(X)
        return self.transform(X)

if __name__ == '__main__':
    data = load_digits()
    X_data = data.data
    y_data = data.target

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_data, test_size=0.2, random_state=42)
    pca = PCA(n_components=30)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    clf = MLPClassifier((100, 50))
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率：{accuracy:.4f}")
