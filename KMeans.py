import numpy as np
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None    # 聚类中心
        self.labels = None      # 每个样本的聚类标签

    def _initialize_centroids(self, X):
        """随机初始化聚类中心"""
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def _compute_distance(self, centroids, X):
        """计算每个样本到所有聚类中心的距离"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)
        return distances

    def fit(self, X):
        """训练KMeans模型"""
        self.centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            """分配样本到最近的聚类中心"""
            distances = self._compute_distance(self.centroids, X)
            new_labels = np.argmin(distances, axis=1)
            """检查是否收敛"""
            if hasattr(self, 'labels') and np.all(new_labels == self.labels):
                break
            self.labels = new_labels
            """更新聚类中心"""
            new_centroids = np.zeros((self.n_clusters, X.shape[1]))
            for k in range(self.n_clusters):
                new_centroids[k] = X[new_labels == k].mean(axis=0)
            """处理空聚类"""
            empty_clusters = np.where(np.isnan(new_centroids).any(axis=1))[0]
            for k in empty_clusters:
                new_centroids[k] = X[np.random.randint(X.shape[0])]
            self.centroids = new_centroids

    def predict(self, X):
        """预测样本所属聚类"""
        distances = self._compute_distance(self.centroids, X)
        return np.argmin(distances, axis=1)

if __name__ == '__main__':
    # 生成示例数据集
    X, y_true = make_blobs(
        n_samples=300,  # 样本数量
        centers=3,  # 聚类中心数量
        cluster_std=0.8,  # 聚类标准差
        random_state=42  # 随机种子
    )

    # 训练K-Means模型
    kmeans = KMeans(n_clusters=3, max_iter=100, random_state=42)
    kmeans.fit(X)

    # 预测聚类标签
    y_pred = kmeans.predict(X)

    # 评估聚类效果（使用轮廓系数，需要sklearn.metrics）
    from sklearn.metrics import silhouette_score

    score = silhouette_score(X, y_pred)
    print(f"轮廓系数: {score:.3f}")

    # 打印聚类中心
    print("\n聚类中心坐标:")
    for i, center in enumerate(kmeans.centroids):
        print(f"Cluster {i}: {center}")