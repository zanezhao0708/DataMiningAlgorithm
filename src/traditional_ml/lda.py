"""
线性判别分析（Linear Discriminant Analysis）算法实现
有监督降维，最大化类间差异同时最小化类内差异
"""

import numpy as np


class LDA:
    def __init__(self, n_components=None):
        """
        LDA初始化

        参数:
            n_components: 降维后的维度，默认为类别数-1
        """
        self.n_components = n_components
        self.scalings = None
        self.explained_variance_ratio = None

    def fit(self, X, y):
        """
        训练LDA模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练标签，形状为(n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)

        if self.n_components is None:
            self.n_components = min(n_classes - 1, n_features)
        # 至少保留1个方向（单类别等退化场景），且不超过特征数
        self.n_components = max(1, min(self.n_components, n_features))

        # 计算总体均值
        overall_mean = np.mean(X, axis=0)

        # 计算类内散度矩阵
        S_w = np.zeros((n_features, n_features))
        # 计算类间散度矩阵
        S_b = np.zeros((n_features, n_features))

        for c in classes:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)

            # 类内散度（直接按离差平方和计算，单样本类不会产生NaN）
            diff_c = X_c - mean_c
            S_w += diff_c.T @ diff_c

            # 类间散度
            n_c = len(X_c)
            mean_diff = (mean_c - overall_mean).reshape(-1, 1)
            S_b += n_c * np.dot(mean_diff, mean_diff.T)

        # 类内散度加微小正则，避免奇异（退化数据下 S_w 可能为全零矩阵）
        S_w += 1e-10 * np.eye(n_features) * max(1.0, np.trace(S_w) / n_features)

        # 求解广义特征值问题（S_w 已正则化，pinv 兜底奇异情形）
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(
                np.linalg.solve(S_w, S_b)
            )
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(
                np.linalg.pinv(S_w).dot(S_b)
            )

        # 按特征值降序排序
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 保存投影矩阵
        self.scalings = eigenvectors[:, :self.n_components]
        total_var = np.sum(np.abs(eigenvalues))
        self.explained_variance_ratio = (np.abs(eigenvalues[:self.n_components])
                                         / total_var if total_var > 0
                                         else np.zeros(self.n_components))

        print(f"LDA训练完成，降维后维度: {self.n_components}")
        print(f"累计判别能力: {np.sum(self.explained_variance_ratio):.4f}")

    def transform(self, X):
        """
        将数据投影到判别空间

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            降维后的数据，形状为(n_samples, n_components)
        """
        return np.dot(np.array(X), self.scalings)

    def fit_transform(self, X, y):
        """
        训练模型并转换数据

        参数:
            X: 训练特征
            y: 训练标签

        返回:
            降维后的数据
        """
        self.fit(X, y)
        return self.transform(X)


if __name__ == '__main__':
    # 示例：使用LDA进行降维
    print("=== 线性判别分析算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    class0 = np.random.randn(30, 5) + np.array([0, 0, 0, 0, 0])
    class1 = np.random.randn(30, 5) + np.array([3, 3, 3, 3, 3])
    class2 = np.random.randn(30, 5) + np.array([-3, -3, -3, -3, -3])

    X_train = np.vstack([class0, class1, class2])
    y_train = np.hstack([np.zeros(30), np.ones(30), np.full(30, 2)])

    # 创建并训练模型
    lda = LDA(n_components=2)
    X_transformed = lda.fit_transform(X_train, y_train)

    print(f"\n原始数据形状: {X_train.shape}")
    print(f"降维后数据形状: {X_transformed.shape}")
    print(f"各判别方向判别能力: {lda.explained_variance_ratio}")

    # 展示降维后各类别的分离情况
    print("\n降维后各类别样本均值:")
    for c in [0, 1, 2]:
        class_mean = np.mean(X_transformed[y_train == c], axis=0)
        print(f"  类别{c}: {class_mean}")
