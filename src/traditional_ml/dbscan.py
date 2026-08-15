"""
DBSCAN（Density-Based Spatial Clustering）聚类算法实现
基于密度的聚类算法，能够发现任意形状的簇
"""

import numpy as np
from collections import deque


class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        """
        DBSCAN初始化

        参数:
            eps: 邻域半径
            min_samples: 形成核心点的最小邻域样本数
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.n_clusters = 0

    def fit(self, X):
        """
        训练DBSCAN模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
        """
        X = np.array(X)
        n_samples = X.shape[0]

        # 初始化所有点为噪声点（标签为-1）
        self.labels = np.full(n_samples, -1)

        # 计算距离矩阵
        distances = self._compute_distance_matrix(X)

        # 找到所有核心点
        core_points = np.sum(distances <= self.eps, axis=1) >= self.min_samples

        # 记录扩展过程（每处理若干个样本记录一次标签快照，供可视化动画回放）
        self.history = []
        self.history.append(self.labels.copy())
        record_every = max(1, n_samples // 60)

        processed = 0
        cluster_id = 0

        # 对每个核心点进行聚类
        for i in range(n_samples):
            if self.labels[i] != -1 or not core_points[i]:
                continue

            # 开始新的簇
            self.labels[i] = cluster_id

            # 使用队列进行区域查询
            queue = deque([i])

            while queue:
                current_point = queue.popleft()

                # 找到当前点的邻域
                neighbors = np.where(distances[current_point] <= self.eps)[0]

                for neighbor in neighbors:
                    if self.labels[neighbor] == -1:
                        self.labels[neighbor] = cluster_id

                        # 如果邻居也是核心点，加入队列
                        if core_points[neighbor]:
                            queue.append(neighbor)

                processed += 1
                if processed % record_every == 0:
                    self.history.append(self.labels.copy())

            cluster_id += 1
            self.history.append(self.labels.copy())

        self.n_clusters = cluster_id
        n_noise = np.sum(self.labels == -1)

        print(f"DBSCAN训练完成")
        print(f"发现簇数量: {self.n_clusters}")
        print(f"噪声点数量: {n_noise}")

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
        print("警告：DBSCAN不支持对新数据进行预测，请使用fit方法")
        return None

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
    # 示例：使用DBSCAN进行聚类
    print("=== DBSCAN聚类算法示例 ===")

    # 生成示例数据（两个圆形簇 + 噪声）
    np.random.seed(42)

    # 生成圆形簇
    theta = np.linspace(0, 2*np.pi, 50)
    cluster1 = np.column_stack([2*np.cos(theta) + np.random.randn(50)*0.1,
                                2*np.sin(theta) + np.random.randn(50)*0.1])
    cluster2 = np.column_stack([6 + 1.5*np.cos(theta) + np.random.randn(50)*0.1,
                                6 + 1.5*np.sin(theta) + np.random.randn(50)*0.1])

    # 添加噪声
    noise = np.random.uniform(-1, 8, (20, 2))
    X_train = np.vstack([cluster1, cluster2, noise])

    # 创建并训练模型
    dbscan = DBSCAN(eps=0.8, min_samples=3)
    labels = dbscan.fit_predict(X_train)

    print(f"\n聚类标签分布:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"噪声点: {count}个")
        else:
            print(f"簇 {label}: {count}个样本")