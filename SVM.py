import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class KernelSVM:
    def __init__(self, C=1.0, kernel='rbf', gamma=0.1, max_iter=1000, tol=1e-3):
        """
        初始化非线性SVM

        参数:
        C (float): 正则化参数
        kernel (str): 核函数类型 ('rbf'或'linear')
        gamma (float): RBF核的参数
        max_iter (int): 最大迭代次数
        tol (float): 收敛容差
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = None
        self.b = 0
        self.X_sv = None
        self.y_sv = None

    def _kernel_function(self, x1, x2):
        """核函数计算"""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("不支持的核函数类型")

    def fit(self, X, y):
        """训练SVM模型"""
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)  # 确保标签为±1

        # 计算核矩阵
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])

        # 初始化拉格朗日乘子
        self.alpha = np.zeros(n_samples)

        # 使用序列最小优化(SMO)算法
        for _ in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)

            for i in range(n_samples):
                # 计算预测误差
                Ei = np.sum(self.alpha * y * K[:, i]) + self.b - y[i]

                # 选择违反KKT条件的样本
                if (y[i] * Ei < -self.tol and self.alpha[i] < self.C) or \
                        (y[i] * Ei > self.tol and self.alpha[i] > 0):

                    # 随机选择另一个样本j ≠ i
                    j = np.random.choice([x for x in range(n_samples) if x != i])
                    Ej = np.sum(self.alpha * y * K[:, j]) + self.b - y[j]

                    # 保存旧值
                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                    # 计算边界L和H
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    # 计算eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # 更新alpha[j]
                    self.alpha[j] -= y[j] * (Ei - Ej) / eta

                    # 裁剪到边界
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    # 检查变化是否显著
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # 更新alpha[i]
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    # 计算b
                    b1 = self.b - Ei - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] \
                         - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - Ej - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] \
                         - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

            # 检查收敛
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break

        # 保存支持向量
        sv_indices = self.alpha > 1e-5
        self.X_sv = X[sv_indices]
        self.y_sv = y[sv_indices]
        self.alpha_sv = self.alpha[sv_indices]

    def predict(self, X):
        """预测类别"""
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for alpha, y_sv, x_sv in zip(self.alpha_sv, self.y_sv, self.X_sv):
                s += alpha * y_sv * self._kernel_function(X[i], x_sv)
            y_pred[i] = np.sign(s + self.b)
        return np.where(y_pred == -1, 0, 1)  # 转换回0/1标签


# 生成非线性可分数据集
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练非线性SVM（使用RBF核）
svm = KernelSVM(C=1.0, kernel='rbf', gamma=0.5, max_iter=1000)
svm.fit(X_train, y_train)

# 预测并评估
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")