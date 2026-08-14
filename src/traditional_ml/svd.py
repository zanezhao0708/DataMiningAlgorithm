"""
奇异值分解（Singular Value Decomposition）算法实现
矩阵分解技术，用于降维、数据压缩和推荐系统
"""

import numpy as np


class SVD:
    def __init__(self, n_components=None):
        """
        SVD初始化

        参数:
            n_components: 保留的奇异值数量，默认为None（保留全部）
        """
        self.n_components = n_components
        self.U = None
        self.s = None
        self.Vt = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """
        对矩阵进行奇异值分解

        参数:
            X: 输入矩阵，形状为(m, n)
        """
        X = np.array(X)

        # 执行奇异值分解
        self.U, s_full, self.Vt = np.linalg.svd(X, full_matrices=False)

        # 使用完整奇异值的平方和作为分母计算方差贡献率
        total_variance = np.sum(s_full ** 2)

        # 保留指定数量的奇异值
        if self.n_components is not None:
            self.U = self.U[:, :self.n_components]
            s = s_full[:self.n_components]
            self.Vt = self.Vt[:self.n_components, :]
        else:
            s = s_full

        self.s = s
        self.explained_variance_ratio = (s ** 2) / total_variance

        print(f"SVD分解完成，奇异值数量: {len(s)}")
        print(f"累计方差贡献率: {np.sum(self.explained_variance_ratio):.4f}")

    def transform(self, X):
        """
        将数据投影到奇异向量空间

        参数:
            X: 输入矩阵，形状为(m, n)

        返回:
            投影后的数据
        """
        return np.dot(np.array(X), self.Vt.T)

    def fit_transform(self, X):
        """
        分解并转换数据

        参数:
            X: 输入矩阵

        返回:
            投影后的数据
        """
        self.fit(X)
        return np.dot(np.array(X), self.Vt.T)

    def inverse_transform(self, X_transformed=None):
        """
        重构原始矩阵

        参数:
            X_transformed: 投影后的数据（如果为None则使用完整分解）

        返回:
            重构的矩阵
        """
        if X_transformed is not None:
            return np.dot(X_transformed, self.Vt)
        else:
            return np.dot(self.U * self.s, self.Vt)

    def reconstruct(self):
        """
        使用分解结果重构原始矩阵

        返回:
            重构的矩阵
        """
        return np.dot(self.U * self.s, self.Vt)


if __name__ == '__main__':
    # 示例：使用SVD进行降维和矩阵压缩
    print("=== 奇异值分解算法示例 ===")

    # 生成示例矩阵
    np.random.seed(42)
    X = np.random.randn(100, 50)

    # 使用SVD降维
    svd = SVD(n_components=10)
    X_transformed = svd.fit_transform(X)

    print(f"\n原始矩阵形状: {X.shape}")
    print(f"降维后形状: {X_transformed.shape}")
    print(f"累计方差贡献率: {np.sum(svd.explained_variance_ratio):.4f}")

    # 重构矩阵
    X_reconstructed = svd.reconstruct()
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"\n重构误差: {reconstruction_error:.6f}")

    # 显示压缩率
    original_size = X.shape[0] * X.shape[1]
    compressed_size = svd.U.shape[0] * svd.U.shape[1] + len(svd.s) + svd.Vt.shape[0] * svd.Vt.shape[1]
    compression_ratio = compressed_size / original_size
    print(f"压缩率: {compression_ratio:.2%}")
