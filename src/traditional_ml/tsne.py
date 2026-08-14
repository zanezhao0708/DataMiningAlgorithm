"""
t-SNE（t-Distributed Stochastic Neighbor Embedding）算法实现
当下最流行的非线性降维与可视化算法之一
通过保持高维空间的局部邻居结构，将数据映射到2D/3D
"""

import numpy as np


class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0,
                 n_iter=1000, early_exaggeration=4.0, momentum=0.8, random_state=None):
        """
        t-SNE初始化

        参数:
            n_components: 降维后的维度（通常为2或3）
            perplexity: 困惑度，可理解为有效邻居数量，通常取5-50
            learning_rate: 学习率
            n_iter: 迭代次数
            early_exaggeration: 早期放大系数，利于簇分离
            momentum: 动量系数
            random_state: 随机种子
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.momentum = momentum
        self.random_state = random_state
        self.embedding_ = None
        self.kl_divergence_ = None

    def _compute_pairwise_distances(self, X):
        """计算成对欧氏距离平方矩阵"""
        sum_X = np.sum(X ** 2, axis=1)
        distances = sum_X[:, None] + sum_X[None, :] - 2 * np.dot(X, X.T)
        np.maximum(distances, 0, out=distances)
        np.fill_diagonal(distances, 0)
        return distances

    def _binary_search_perplexity(self, distances, tol=1e-5, max_iter=50):
        """二分搜索每个点的sigma，使条件分布的熵匹配目标困惑度"""
        n = distances.shape[0]
        target = np.log(self.perplexity)
        P = np.zeros((n, n))
        betas = np.ones(n)

        for i in range(n):
            beta = betas[i]
            beta_min, beta_max = -np.inf, np.inf
            # 排除自身
            dist_i = np.delete(distances[i], i)

            for _ in range(max_iter):
                # 以 beta = 1/(2*sigma^2) 计算条件概率
                probs = np.exp(-beta * dist_i)
                sum_probs = np.sum(probs)
                if sum_probs == 0:
                    probs = np.full(n - 1, 1.0 / (n - 1))
                    sum_probs = 1.0
                else:
                    probs = probs / sum_probs

                # 计算熵
                entropy = -np.sum(probs * np.log(probs + 1e-12))

                diff = entropy - target
                if abs(diff) < tol:
                    break

                if diff > 0:
                    # 熵太大 → 分布太均匀 → 增大beta（减小sigma）
                    beta_min = beta
                    if beta_max != np.inf:
                        beta = (beta + beta_max) / 2
                    elif beta >= 1e6:
                        break
                    else:
                        beta = min(beta * 2, 1e6)
                else:
                    # 熵太小 → 分布太集中 → 减小beta（增大sigma）
                    beta_max = beta
                    beta = (beta_min + beta) / 2 if beta_min != -np.inf else beta / 2

            betas[i] = beta
            # 写回（跳过对角线）
            row = np.exp(-beta * distances[i])
            row[i] = 0
            P[i] = row

        return P, betas

    def _compute_joint_probabilities(self, X):
        """计算高维空间的联合概率分布 P"""
        distances = self._compute_pairwise_distances(X)
        P, _ = self._binary_search_perplexity(distances)

        # 对称化：P_ij = (P_j|i + P_i|j) / 2n
        P = (P + P.T) / (2 * P.shape[0])
        np.fill_diagonal(P, 0)

        # 归一化并做截断，避免极端小值导致数值不稳定
        total = np.sum(P)
        if total == 0:
            # 极端退化情况：退回均匀分布
            P = np.full_like(P, 1.0 / (P.shape[0] * (P.shape[0] - 1)))
            np.fill_diagonal(P, 0)
        else:
            P = np.maximum(P / total, 1e-12)
        return P

    def _compute_q_probs(self, Y):
        """计算低维空间的联合概率分布 Q（学生t分布，自由度1）"""
        distances = self._compute_pairwise_distances(Y)
        # t分布核：1 / (1 + dist)
        inv = 1.0 / (1.0 + distances)
        np.fill_diagonal(inv, 0)
        Q = inv / np.sum(inv)
        Q = np.maximum(Q, 1e-12)
        return Q, inv

    def _compute_gradients(self, P, Q, Y, inv):
        """计算KL散度对Y的梯度"""
        PQ_diff = P - Q
        # 梯度 = 4 * sum_j (P_ij - Q_ij)(y_i - y_j) * inv_ij
        grad = np.zeros_like(Y)
        for i in range(Y.shape[0]):
            diff = Y[i] - Y
            grad[i] = 4 * np.sum((PQ_diff[i] * inv[i])[:, None] * diff, axis=0)
        return grad

    def fit_transform(self, X):
        """
        训练并返回降维结果

        参数:
            X: 输入数据，形状为(n_samples, n_features)

        返回:
            降维后的数据，形状为(n_samples, n_components)
        """
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 限制困惑度不超过样本数
        if self.perplexity >= n_samples:
            self.perplexity = max(5.0, (n_samples - 1) / 3)
            print(f"困惑度过大，已调整为 {self.perplexity:.1f}")

        # 1. 计算高维联合概率
        P = self._compute_joint_probabilities(X)

        # 2. 随机初始化低维嵌入
        Y = np.random.randn(n_samples, self.n_components) * 1e-4
        velocity = np.zeros_like(Y)

        # 3. 梯度下降（带动量）
        exaggerate_until = min(250, self.n_iter // 4)
        for iteration in range(self.n_iter):
            Q, inv = self._compute_q_probs(Y)
            grad = self._compute_gradients(P, Q, Y, inv)

            # 早期放大P，促进簇分离
            if iteration < exaggerate_until:
                grad *= self.early_exaggeration

            # 动量更新
            velocity = self.momentum * velocity - self.learning_rate * grad
            Y += velocity

            # 中心化，防止漂移
            Y -= np.mean(Y, axis=0)

            if iteration % 100 == 0:
                Q, _ = self._compute_q_probs(Y)
                kl = np.sum(P * np.log(P / Q))
                print(f"迭代 {iteration}, KL散度: {kl:.4f}")

        Q, _ = self._compute_q_probs(Y)
        self.kl_divergence_ = np.sum(P * np.log(P / Q))
        self.embedding_ = Y

        print(f"t-SNE降维完成，KL散度: {self.kl_divergence_:.4f}")
        return Y

    def fit(self, X):
        """训练模型"""
        self.fit_transform(X)
        return self


if __name__ == '__main__':
    # 示例：使用t-SNE对多类高斯数据进行降维可视化
    print("=== t-SNE降维算法示例 ===")

    # 生成三类高维高斯数据
    np.random.seed(42)
    class0 = np.random.randn(50, 10)
    class1 = np.random.randn(50, 10) + 3
    class2 = np.random.randn(50, 10) - 3

    X = np.vstack([class0, class1, class2])
    y = np.hstack([np.zeros(50), np.ones(50), np.full(50, 2)])

    print(f"原始数据维度: {X.shape}")

    # 降维到2D
    tsne = TSNE(n_components=2, perplexity=15, n_iter=500, random_state=42)
    X_embedded = tsne.fit_transform(X)

    print(f"降维后维度: {X_embedded.shape}")
    print(f"最终KL散度: {tsne.kl_divergence_:.4f}")

    # 检查各类别在2D上是否分离
    print("\n各类别降维后的均值（检查分离程度）:")
    for c in [0, 1, 2]:
        mean = X_embedded[y == c].mean(axis=0)
        print(f"  类别{c}: [{mean[0]:.2f}, {mean[1]:.2f}]")
