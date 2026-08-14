"""
主成分分析（Principal Component Analysis）降维算法实现
通过线性变换将数据投影到低维空间，保留主要信息
"""

import numpy as np


class PCA:
    def __init__(self, n_components=None):
        """
        PCA初始化

        参数:
            n_components: 保留的主成分数量，如果为None则保留所有成分
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """
        训练PCA模型，计算主成分

        参数:
            X: 训练特征，形状为(n_samples, n_features)
        """
        X = np.array(X)
        n_samples, n_features = X.shape

        # 标准化数据（中心化）
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 计算协方差矩阵
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # 按特征值降序排序
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 保存主成分
        if self.n_components is None:
            self.n_components = n_features

        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = self.explained_variance / np.sum(eigenvalues)

        print(f"PCA训练完成，主成分数量: {self.n_components}")
        print(f"累计方差贡献率: {np.sum(self.explained_variance_ratio):.4f}")

    def transform(self, X):
        """
        将数据投影到主成分空间

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            降维后的数据，形状为(n_samples, n_components)
        """
        X_centered = np.array(X) - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """
        训练模型并转换数据

        参数:
            X: 训练特征，形状为(n_samples, n_features)

        返回:
            降维后的数据，形状为(n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        将降维后的数据还原到原始空间

        参数:
            X_transformed: 降维后的数据，形状为(n_samples, n_components)

        返回:
            还原的数据，形状为(n_samples, n_features)
        """
        return np.dot(X_transformed, self.components.T) + self.mean


if __name__ == '__main__':
    # 示例：使用PCA进行降维
    print("=== PCA降维算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    X = np.random.randn(100, 10)

    # 创建并训练模型
    pca = PCA(n_components=3)
    X_transformed = pca.fit_transform(X)

    print(f"\n原始数据形状: {X.shape}")
    print(f"降维后数据形状: {X_transformed.shape}")
    print(f"各主成分方差贡献率: {pca.explained_variance_ratio}")
    print(f"累计方差贡献率: {np.sum(pca.explained_variance_ratio):.4f}")

    # 还原数据
    X_reconstructed = pca.inverse_transform(X_transformed)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"\n重建误差: {reconstruction_error:.6f}")