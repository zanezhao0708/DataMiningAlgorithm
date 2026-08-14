"""
高斯混合模型（Gaussian Mixture Model）算法实现
概率聚类模型，假设数据由多个高斯分布混合生成
"""

import numpy as np


class GaussianMixtureModel:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        """
        GMM初始化

        参数:
            n_components: 混合成分数量
            max_iter: 最大迭代次数
            tol: 收敛阈值
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.means = None
        self.covariances = None
        self.labels = None

    def fit(self, X):
        """
        训练GMM模型（使用EM算法）

        参数:
            X: 训练特征，形状为(n_samples, n_features)
        """
        X = np.array(X)
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.ones(self.n_components) / self.n_components
        random_indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[random_indices]
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])

        # EM算法迭代
        prev_log_likelihood = -float('inf')

        for iteration in range(self.max_iter):
            # E步：计算后验概率
            responsibilities = self._expectation(X)

            # M步：更新参数
            self._maximization(X, responsibilities)

            # 计算对数似然
            log_likelihood = self._compute_log_likelihood(X)

            if iteration % 10 == 0:
                print(f"迭代 {iteration}, 对数似然: {log_likelihood:.4f}")

            # 检查收敛
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"在第 {iteration+1} 轮收敛")
                break

            prev_log_likelihood = log_likelihood

        # 分配标签
        self.labels = np.argmax(responsibilities, axis=1)

        print(f"GMM训练完成，成分数量: {self.n_components}")

    def _expectation(self, X):
        """E步：计算后验概率"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self._multivariate_gaussian(
                X, self.means[k], self.covariances[k]
            )

        # 归一化
        responsibilities = responsibilities / (responsibilities.sum(axis=1, keepdims=True) + 1e-10)

        return responsibilities

    def _maximization(self, X, responsibilities):
        """M步：更新参数"""
        n_samples, n_features = X.shape

        for k in range(self.n_components):
            # 更新权重
            self.weights[k] = responsibilities[:, k].sum() / n_samples

            # 更新均值
            self.means[k] = np.sum(responsibilities[:, k].reshape(-1, 1) * X, axis=0) / responsibilities[:, k].sum()

            # 更新协方差矩阵
            diff = X - self.means[k]
            self.covariances[k] = np.dot(
                responsibilities[:, k].reshape(-1, 1) * diff,
                diff.T
            ) / responsibilities[:, k].sum()

            # 添加小的对角线元素防止奇异矩阵
            self.covariances[k] += 1e-6 * np.eye(n_features)

    def _multivariate_gaussian(self, X, mean, covariance):
        """计算多元高斯概率密度"""
        n_features = X.shape[1]
        diff = X - mean

        try:
            # 计算行列式和逆矩阵
            det = np.linalg.det(covariance)
            inv = np.linalg.inv(covariance)

            # 计算概率密度
            exponent = np.sum(diff @ inv * diff, axis=1)
            probability = np.exp(-0.5 * exponent) / np.sqrt((2 * np.pi) ** n_features * det)

        except:
            # 如果协方差矩阵奇异，使用伪逆
            probability = np.ones(X.shape[0]) * 1e-10

        return probability

    def _compute_log_likelihood(self, X):
        """计算对数似然"""
        n_samples = X.shape[0]
        likelihood = np.zeros(n_samples)

        for k in range(self.n_components):
            likelihood += self.weights[k] * self._multivariate_gaussian(
                X, self.means[k], self.covariances[k]
            )

        return np.sum(np.log(likelihood + 1e-10))

    def predict(self, X):
        """
        对新数据进行聚类预测

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            聚类标签，形状为(n_samples,)
        """
        responsibilities = self._expectation(np.array(X))
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        """
        预测每个样本属于各个成分的概率

        参数:
            X: 测试特征

        返回:
            概率矩阵，形状为(n_samples, n_components)
        """
        return self._expectation(np.array(X))

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
    # 示例：使用GMM进行聚类
    print("=== 高斯混合模型算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    cluster1 = np.random.randn(50, 2) * 0.5 + np.array([0, 0])
    cluster2 = np.random.randn(50, 2) * 0.5 + np.array([3, 3])
    cluster3 = np.random.randn(50, 2) * 0.5 + np.array([3, 0])
    X_train = np.vstack([cluster1, cluster2, cluster3])

    # 创建并训练模型
    gmm = GaussianMixtureModel(n_components=3, max_iter=100)
    labels = gmm.fit_predict(X_train)

    print(f"\n聚类标签: {labels[:20]}")

    # 查看每个簇的样本数
    for cluster_id in range(3):
        count = np.sum(labels == cluster_id)
        print(f"簇 {cluster_id}: {count}个样本")

    # 查看成分权重
    print(f"\n各成分权重: {gmm.weights}")