"""
K-Means聚类算法实现
通过迭代更新聚类中心将数据分为K个簇
"""

import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4):
        """
        K-Means初始化

        参数:
            n_clusters: 聚类数量
            max_iters: 最大迭代次数
            tol: 收敛阈值
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.history = []  # 记录每次迭代的中心点，用于动画可视化

    def fit(self, X):
        """
        训练K-Means模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
        """
        X = np.array(X)
        n_samples, n_features = X.shape

        # 随机初始化聚类中心
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        # 记录初始状态
        self.history = [{'centroids': self.centroids.copy()}]

        # 迭代更新
        for i in range(self.max_iters):
            # 分配每个样本到最近的聚类中心
            self.labels = self._assign_clusters(X)

            # 计算新的聚类中心
            new_centroids = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                cluster_points = X[self.labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = cluster_points.mean(axis=0)
                else:
                    new_centroids[k] = self.centroids[k]

            # 检查是否收敛
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            self.history.append({'centroids': new_centroids.copy(),
                                 'labels': self.labels.copy()})
            if centroid_shift < self.tol:
                print(f"K-Means在{i+1}次迭代后收敛")
                break

        print(f"K-Means训练完成，聚类数: {self.n_clusters}")

    def _assign_clusters(self, X):
        """为每个样本分配最近的聚类中心"""
        distances = np.zeros((X.shape[0], self.n_clusters))

        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)

        return np.argmin(distances, axis=1)

    def predict(self, X):
        """
        对新数据进行聚类预测

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            聚类标签，形状为(n_samples,)
        """
        return self._assign_clusters(np.array(X))

    def inertia(self, X):
        """
        计算聚类惯性（样本到其聚类中心的距离平方和）

        参数:
            X: 数据特征

        返回:
            惯性值
        """
        X = np.array(X)
        labels = self._assign_clusters(X)
        inertia = 0

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            inertia += np.sum((cluster_points - self.centroids[k]) ** 2)

        return inertia

    def visualize(self, X, title="K-Means Clustering"):
        """
        可视化聚类结果（仅支持2D数据）

        参数:
            X: 数据特征
            title: 图表标题
        """
        import matplotlib.pyplot as plt  # 仅可视化时需要，避免部署环境强依赖

        X = np.array(X)
        if X.shape[1] != 2:
            print("可视化仅支持2维数据")
            return

        labels = self._assign_clusters(X)

        plt.figure(figsize=(8, 6))
        colors = plt.cm.rainbow(np.linspace(0, 1, self.n_clusters))

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                       c=[colors[k]], label=f'Cluster {k}', alpha=0.6)

        plt.scatter(self.centroids[:, 0], self.centroids[:, 1],
                   c='black', marker='x', s=200, linewidths=3,
                   label='Centroids')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == '__main__':
    # 示例：使用K-Means进行聚类
    print("=== K-Means聚类算法示例 ===")

    # 生成示例数据（3个簇）
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) + np.array([2, 2])
    cluster2 = np.random.randn(30, 2) + np.array([-2, -2])
    cluster3 = np.random.randn(30, 2) + np.array([2, -2])
    X_train = np.vstack([cluster1, cluster2, cluster3])

    # 创建并训练模型
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train)

    # 预测
    labels = kmeans.predict(X_train)
    print(f"聚类标签: {labels[:20]}")

    # 计算惯性
    inertia = kmeans.inertia(X_train)
    print(f"惯性值: {inertia:.4f}")

    # 可视化（如果运行环境支持）
    try:
        kmeans.visualize(X_train)
    except:
        print("可视化功能需要matplotlib支持")