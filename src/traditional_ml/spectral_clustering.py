"""
谱聚类（Spectral Clustering）算法实现
基于图论和谱图理论的聚类算法
"""

import numpy as np


class SpectralClustering:
    def __init__(self, n_clusters=3, affinity='rbf', gamma=1.0, n_neighbors=10):
        """
        谱聚类初始化

        参数:
            n_clusters: 聚类数量
            affinity: 相似度度量，'rbf'或'knn'
            gamma: RBF核参数
            n_neighbors: KNN的邻居数量
        """
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.labels = None
        self.affinity_matrix = None

    def fit(self, X):
        """
        训练谱聚类模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
        """
        X = np.array(X)
        n_samples = X.shape[0]

        # 1. 构建相似度矩阵
        self.affinity_matrix = self._build_affinity_matrix(X)

        # 2. 构建拉普拉斯矩阵
        degree_matrix = np.diag(np.sum(self.affinity_matrix, axis=1))
        laplacian = degree_matrix - self.affinity_matrix

        # 3. 计算特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

        # 4. 取前k个特征向量
        self.embedding = eigenvectors[:, :self.n_clusters]

        # 5. 对特征向量进行K-Means聚类
        self.labels = self._kmeans(self.embedding, self.n_clusters)

        print(f"谱聚类训练完成，聚类数: {self.n_clusters}")

    def _build_affinity_matrix(self, X):
        """构建相似度矩阵"""
        n_samples = X.shape[0]
        affinity = np.zeros((n_samples, n_samples))

        if self.affinity == 'rbf':
            # RBF核
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    distance = np.linalg.norm(X[i] - X[j])
                    similarity = np.exp(-self.gamma * distance ** 2)
                    affinity[i, j] = similarity
                    affinity[j, i] = similarity
        else:
            # KNN
            for i in range(n_samples):
                distances = [np.linalg.norm(X[i] - X[j]) for j in range(n_samples)]
                k_nearest = np.argsort(distances)[:self.n_neighbors]
                for j in k_nearest:
                    if i != j:
                        affinity[i, j] = 1
                        affinity[j, i] = 1

        return affinity

    def _kmeans(self, X, k):
        """简单的K-Means聚类"""
        n_samples = X.shape[0]

        # 初始化聚类中心
        random_indices = np.random.choice(n_samples, k, replace=False)
        centroids = X[random_indices]

        # 记录每轮标签（供可视化动画回放）
        self.kmeans_history = []

        # 迭代
        for _ in range(100):
            # 分配簇
            distances = np.zeros((n_samples, k))
            for i in range(k):
                distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
            labels = np.argmin(distances, axis=1)
            self.kmeans_history.append(labels.copy())

            # 更新聚类中心
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        self.kmeans_history.append(labels.copy())
        return labels

    def fit_predict(self, X):
        """
        训练模型并返回聚类标签

        参数:
            X: 训练特征

        返回:
            聚类标签
        """
        self.fit(X)
        return self.labels


if __name__ == '__main__':
    # 示例：使用谱聚类进行聚类
    print("=== 谱聚类算法示例 ===")

    # 生成同心圆数据（传统K-Means无法处理）
    np.random.seed(42)
    theta = np.linspace(0, 2*np.pi, 60)

    # 内圆
    inner = np.column_stack([np.cos(theta) + np.random.randn(60)*0.05,
                             np.sin(theta) + np.random.randn(60)*0.05])

    # 外圆
    outer = np.column_stack([2*np.cos(theta) + np.random.randn(60)*0.05,
                             2*np.sin(theta) + np.random.randn(60)*0.05])

    X_train = np.vstack([inner, outer])

    # 使用不同的相似度度量
    for affinity in ['rbf', 'knn']:
        print(f"\n相似度度量: {affinity}")
        sc = SpectralClustering(n_clusters=2, affinity=affinity, gamma=0.5)
        labels = sc.fit_predict(X_train)

        print(f"聚类标签分布:")
        for cluster_id in range(2):
            count = np.sum(labels == cluster_id)
            print(f"  簇 {cluster_id}: {count}个样本")
