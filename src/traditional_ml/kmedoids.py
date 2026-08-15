"""
K-Medoids（K中心点）聚类算法实现
与K-Means类似，但使用实际数据点作为聚类中心
"""

import numpy as np


class KMedoids:
    def __init__(self, n_clusters=3, max_iters=100):
        """
        K-Medoids初始化

        参数:
            n_clusters: 聚类数量
            max_iters: 最大迭代次数
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.medoids = None
        self.labels = None

    def fit(self, X):
        """
        训练K-Medoids模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
        """
        X = np.array(X)
        n_samples = X.shape[0]

        # 随机初始化中心点
        medoid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.medoids = X[medoid_indices]
        self.medoid_indices = medoid_indices.copy()

        # 计算距离矩阵
        distances = self._compute_distance_matrix(X)

        # 记录训练过程（质心与标签快照，供可视化动画回放）
        self.history = []

        def _record(indices, labels):
            self.history.append({
                'centroids': X[indices].copy(),
                'labels': labels.copy(),
            })

        for iteration in range(self.max_iters):
            # 分配样本到最近的中心点
            medoid_distances = distances[:, self.medoid_indices]
            labels = np.argmin(medoid_distances, axis=1)
            _record(self.medoid_indices, labels)

            # 更新中心点
            new_medoid_indices = self.medoid_indices.copy()

            for k in range(self.n_clusters):
                cluster_points = np.where(labels == k)[0]

                if len(cluster_points) == 0:
                    continue

                # 在簇内寻找最佳中心点
                best_cost = float('inf')
                best_medoid = self.medoid_indices[k]

                for candidate in cluster_points:
                    # 计算候选中心点的总距离
                    cost = np.sum(distances[candidate, cluster_points])
                    if cost < best_cost:
                        best_cost = cost
                        best_medoid = candidate

                new_medoid_indices[k] = best_medoid

            # 检查是否收敛
            if np.array_equal(new_medoid_indices, self.medoid_indices):
                print(f"在第 {iteration+1} 轮收敛")
                break

            self.medoid_indices = new_medoid_indices

        # 最终分配
        medoid_distances = distances[:, self.medoid_indices]
        self.labels = np.argmin(medoid_distances, axis=1)
        self.medoids = X[self.medoid_indices]

        print(f"K-Medoids训练完成，聚类数: {self.n_clusters}")
        print(f"中心点索引: {self.medoid_indices}")

    def _compute_distance_matrix(self, X):
        """计算距离矩阵"""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def predict(self, X):
        """
        对新数据进行聚类预测

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            聚类标签，形状为(n_samples,)
        """
        X = np.array(X)
        distances = np.zeros((X.shape[0], self.n_clusters))

        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - self.medoids[k], axis=1)

        return np.argmin(distances, axis=1)

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

    def inertia(self, X):
        """
        计算聚类惯性（样本到中心点的距离总和）

        参数:
            X: 数据特征

        返回:
            惯性值
        """
        X = np.array(X)
        labels = self.predict(X)
        inertia = 0

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            inertia += np.sum(np.linalg.norm(cluster_points - self.medoids[k], axis=1))

        return inertia


if __name__ == '__main__':
    # 示例：使用K-Medoids进行聚类
    print("=== K-Medoids聚类算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) + np.array([2, 2])
    cluster2 = np.random.randn(30, 2) + np.array([-2, -2])
    cluster3 = np.random.randn(30, 2) + np.array([2, -2])
    X_train = np.vstack([cluster1, cluster2, cluster3])

    # 创建并训练模型
    kmedoids = KMedoids(n_clusters=3)
    labels = kmedoids.fit_predict(X_train)

    print(f"\n聚类标签分布:")
    for cluster_id in range(3):
        count = np.sum(labels == cluster_id)
        print(f"  簇 {cluster_id}: {count}个样本")

    # 计算惯性
    inertia = kmedoids.inertia(X_train)
    print(f"\n惯性值: {inertia:.4f}")
