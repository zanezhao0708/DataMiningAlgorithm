"""
层次聚类（Hierarchical Clustering）算法实现
通过构建层次树结构进行聚类，支持凝聚和分裂方法
"""

import numpy as np


class HierarchicalClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        """
        层次聚类初始化

        参数:
            n_clusters: 目标簇数量
            linkage: 连接方式，'single'（单链接）、'complete'（全链接）或'average'（平均链接）
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels = None
        self.linkage_matrix = []

    def fit(self, X):
        """
        训练层次聚类模型（凝聚方法）

        参数:
            X: 训练特征，形状为(n_samples, n_features)
        """
        X = np.array(X)
        n_samples = X.shape[0]

        # 初始化每个样本为一个簇
        clusters = [{i} for i in range(n_samples)]

        # 计算距离矩阵
        distances = self._compute_distance_matrix(X)

        # 保存合并历史
        self.linkage_matrix = []
        # 记录合并过程（每次合并后所有样本的簇标签快照，供可视化动画回放）
        self.history = []

        def _record():
            snap = np.zeros(n_samples, dtype=int)
            for cid, cluster in enumerate(clusters):
                for sample_id in cluster:
                    snap[sample_id] = cid
            self.history.append(snap)

        _record()
        cluster_sizes = {i: 1 for i in range(n_samples)}
        current_cluster_id = n_samples

        # 迭代合并簇
        while len(clusters) > self.n_clusters:
            # 找到最近的两个簇
            min_dist = float('inf')
            merge_i, merge_j = 0, 1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._cluster_distance(clusters[i], clusters[j], distances, cluster_sizes)
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j

            # 记录合并历史
            cluster_i = list(clusters[merge_i])[0] if len(clusters[merge_i]) == 1 else merge_i
            cluster_j = list(clusters[merge_j])[0] if len(clusters[merge_j]) == 1 else merge_j

            self.linkage_matrix.append([
                cluster_i if cluster_i < n_samples else cluster_i,
                cluster_j if cluster_j < n_samples else cluster_j,
                min_dist,
                len(clusters[merge_i]) + len(clusters[merge_j])
            ])

            # 合并簇
            new_cluster = clusters[merge_i] | clusters[merge_j]
            clusters.pop(merge_j)
            clusters.pop(merge_i)
            clusters.append(new_cluster)
            _record()

            # 更新簇大小
            cluster_sizes[current_cluster_id] = len(new_cluster)
            current_cluster_id += 1

        # 分配标签
        self.labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for sample_id in cluster:
                self.labels[sample_id] = cluster_id

        print(f"层次聚类训练完成")
        print(f"簇数量: {self.n_clusters}")
        print(f"合并次数: {len(self.linkage_matrix)}")

    def _compute_distance_matrix(self, X):
        """计算距离矩阵"""
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def _cluster_distance(self, cluster1, cluster2, distances, cluster_sizes):
        """计算两个簇之间的距离"""
        if self.linkage == 'single':
            # 单链接：最小距离
            min_dist = float('inf')
            for i in cluster1:
                for j in cluster2:
                    if distances[i, j] < min_dist:
                        min_dist = distances[i, j]
            return min_dist

        elif self.linkage == 'complete':
            # 全链接：最大距离
            max_dist = 0
            for i in cluster1:
                for j in cluster2:
                    if distances[i, j] > max_dist:
                        max_dist = distances[i, j]
            return max_dist

        else:  # average
            # 平均链接：平均距离
            total_dist = 0
            count = 0
            for i in cluster1:
                for j in cluster2:
                    total_dist += distances[i, j]
                    count += 1
            return total_dist / count if count > 0 else 0

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
    # 示例：使用层次聚类进行聚类
    print("=== 层次聚类算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    cluster1 = np.random.randn(15, 2) + np.array([0, 0])
    cluster2 = np.random.randn(15, 2) + np.array([5, 5])
    cluster3 = np.random.randn(15, 2) + np.array([5, 0])
    X_train = np.vstack([cluster1, cluster2, cluster3])

    # 使用不同的连接方式
    for linkage in ['single', 'complete', 'average']:
        print(f"\n连接方式: {linkage}")
        hc = HierarchicalClustering(n_clusters=3, linkage=linkage)
        labels = hc.fit_predict(X_train)

        print(f"聚类标签: {labels[:20]}")
        for cluster_id in range(3):
            count = np.sum(labels == cluster_id)
            print(f"簇 {cluster_id}: {count}个样本")