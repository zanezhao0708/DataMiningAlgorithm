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
        训练层次聚类模型（凝聚方法，Lance-Williams 增量更新簇间距离）

        参数:
            X: 训练特征，形状为(n_samples, n_features)
        """
        X = np.array(X)
        n_samples = X.shape[0]
        n_target = max(1, self.n_clusters)

        # 样本级距离矩阵（向量化计算）
        distances = self._compute_distance_matrix(X)

        self.linkage_matrix = []
        # 记录合并过程（每次合并后所有样本的簇标签快照，供可视化动画回放）
        self.history = []

        # 活动簇状态：初始每个样本自成一簇，编号 0..n-1；合并产生的新簇编号依次为 n, n+1, ...
        active = list(range(n_samples))
        sizes = {i: 1 for i in range(n_samples)}
        members = {i: [i] for i in range(n_samples)}

        # 簇间距离矩阵 D[i,j] = 当前第 i、j 个活动簇之间的距离
        D = distances.copy()

        def _record():
            snap = np.zeros(n_samples, dtype=int)
            for cid, cluster_id in enumerate(active):
                for sample_id in members[cluster_id]:
                    snap[sample_id] = cid
            self.history.append(snap)

        _record()
        next_id = n_samples

        while len(active) > n_target:
            k = len(active)
            # 上三角中找最近的一对簇
            iu = np.triu_indices(k, k=1)
            flat = D[iu]
            m = int(np.argmin(flat))
            a, b = int(iu[0][m]), int(iu[1][m])
            merge_dist = float(flat[m])
            ida, idb = active[a], active[b]
            na, nb = sizes[ida], sizes[idb]

            # 其余簇与新簇的距离（Lance-Williams 公式，对三种链接均精确）
            keep = [t for t in range(k) if t != a and t != b]
            if keep:
                da, db = D[a, keep], D[b, keep]
                if self.linkage == 'single':
                    dn = np.minimum(da, db)
                elif self.linkage == 'complete':
                    dn = np.maximum(da, db)
                else:  # average
                    dn = (na * da + nb * db) / (na + nb)

            self.linkage_matrix.append([ida, idb, merge_dist, na + nb])

            # 合并 a、b 为新簇
            members[next_id] = members[ida] + members[idb]
            sizes[next_id] = na + nb
            for t in sorted((a, b), reverse=True):
                active.pop(t)
            D = D[np.ix_(keep, keep)] if keep else np.zeros((0, 0))
            if keep:
                col = dn.reshape(-1, 1)
                D = np.block([[D, col], [col.T, np.zeros((1, 1))]])
            active.append(next_id)
            next_id += 1
            _record()

        # 分配标签
        self.labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(active):
            for sample_id in members[cluster]:
                self.labels[sample_id] = cluster_id

        print(f"层次聚类训练完成")
        print(f"簇数量: {self.n_clusters}")
        print(f"合并次数: {len(self.linkage_matrix)}")

    def _compute_distance_matrix(self, X):
        """计算距离矩阵（向量化）"""
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt((diff ** 2).sum(-1))

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