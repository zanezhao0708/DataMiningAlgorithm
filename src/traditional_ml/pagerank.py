"""
PageRank算法实现
基于链接分析的网页重要性排序算法
"""

import numpy as np


class PageRank:
    def __init__(self, damping_factor=0.85, max_iters=100, tol=1e-6):
        """
        PageRank初始化

        参数:
            damping_factor: 阻尼系数（通常为0.85）
            max_iters: 最大迭代次数
            tol: 收敛阈值
        """
        self.damping_factor = damping_factor
        self.max_iters = max_iters
        self.tol = tol
        self.scores = None

    def fit(self, adjacency_matrix):
        """
        计算PageRank分数

        参数:
            adjacency_matrix: 邻接矩阵，形状为(n_nodes, n_nodes)
        """
        adjacency_matrix = np.array(adjacency_matrix, dtype=float)
        n_nodes = adjacency_matrix.shape[0]

        # 计算出度
        out_degrees = np.sum(adjacency_matrix, axis=1)
        out_degrees[out_degrees == 0] = 1  # 防止除零

        # 构建转移矩阵
        transition_matrix = adjacency_matrix / out_degrees.reshape(-1, 1)

        # 初始化PageRank分数
        scores = np.ones(n_nodes) / n_nodes

        # 迭代计算
        for iteration in range(self.max_iters):
            # PageRank公式: PR = (1-d)/N + d * M^T * PR
            new_scores = (1 - self.damping_factor) / n_nodes + \
                        self.damping_factor * np.dot(transition_matrix.T, scores)

            # 检查收敛
            if np.linalg.norm(new_scores - scores) < self.tol:
                print(f"在第 {iteration+1} 轮收敛")
                break

            scores = new_scores

        self.scores = scores

        print(f"PageRank计算完成，节点数: {n_nodes}")

    def get_ranked_nodes(self, node_names=None):
        """
        获取按PageRank分数排序的节点

        参数:
            node_names: 节点名称列表

        返回:
            排序后的节点列表
        """
        if node_names is None:
            node_names = [f'node_{i}' for i in range(len(self.scores))]

        ranked = sorted(zip(node_names, self.scores), key=lambda x: -x[1])
        return ranked


if __name__ == '__main__':
    # 示例：使用PageRank计算网页重要性
    print("=== PageRank算法示例 ===")

    # 构建简单的网页链接图（5个网页）
    # 1 -> 2, 3
    # 2 -> 3, 4
    # 3 -> 1, 5
    # 4 -> 1, 3
    # 5 -> 2, 4
    adjacency_matrix = np.array([
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0]
    ])

    node_names = ['网页A', '网页B', '网页C', '网页D', '网页E']

    # 创建并训练模型
    pagerank = PageRank()
    pagerank.fit(adjacency_matrix)

    # 获取排序结果
    ranked_nodes = pagerank.get_ranked_nodes(node_names)

    print(f"\n网页重要性排名:")
    for rank, (node, score) in enumerate(ranked_nodes, 1):
        print(f"  {rank}. {node}: {score:.4f}")
