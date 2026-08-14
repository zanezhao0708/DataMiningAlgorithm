"""
多项式回归（Polynomial Regression）算法实现
通过添加多项式特征拟合非线性关系
"""

import numpy as np


class PolynomialRegression:
    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000):
        """
        多项式回归初始化

        参数:
            degree: 多项式阶数
            learning_rate: 学习率
            n_iterations: 迭代次数
        """
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.X_poly = None

    def _polynomial_features(self, X):
        """
        生成多项式特征

        参数:
            X: 原始特征

        返回:
            多项式特征
        """
        X = np.array(X)
        n_samples, n_features = X.shape

        # 生成多项式特征
        features = [X]

        for d in range(2, self.degree + 1):
            features.append(np.power(X, d))

        return np.hstack(features)

    def fit(self, X, y):
        """
        训练多项式回归模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练目标，形状为(n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        # 生成多项式特征
        self.X_poly = self._polynomial_features(X)
        n_samples, n_poly_features = self.X_poly.shape

        # 标准化多项式特征（高阶特征数值范围大，不标准化会导致梯度下降发散）
        self.poly_mean = self.X_poly.mean(axis=0)
        self.poly_std = self.X_poly.std(axis=0)
        self.poly_std[self.poly_std == 0] = 1
        self.X_poly = (self.X_poly - self.poly_mean) / self.poly_std

        # 初始化参数
        self.weights = np.zeros(n_poly_features)
        self.bias = 0

        # 梯度下降训练
        for i in range(self.n_iterations):
            # 预测
            y_pred = np.dot(self.X_poly, self.weights) + self.bias

            # 计算梯度
            dw = (1 / n_samples) * np.dot(self.X_poly.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                loss = self._compute_loss(y, y_pred)
                print(f"迭代 {i}, 损失: {loss:.4f}")

        print(f"多项式回归训练完成，阶数: {self.degree}")

    def _compute_loss(self, y_true, y_pred):
        """计算均方误差"""
        return np.mean((y_true - y_pred) ** 2)

    def predict(self, X):
        """
        对新数据进行预测

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测结果，形状为(n_samples,)
        """
        X_poly = self._polynomial_features(np.array(X))
        # 使用训练时保存的均值和标准差做同样的标准化
        X_poly = (X_poly - self.poly_mean) / self.poly_std
        return np.dot(X_poly, self.weights) + self.bias

    def score(self, X, y):
        """
        计算R²分数

        参数:
            X: 测试特征
            y: 真实目标值

        返回:
            R²分数
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return r_squared


if __name__ == '__main__':
    # 示例：使用多项式回归拟合非线性数据
    print("=== 多项式回归算法示例 ===")

    # 生成非线性数据（二次函数）
    np.random.seed(42)
    X_train = np.random.uniform(-3, 3, (100, 1))
    y_train = 2 * X_train.ravel() ** 2 - 3 * X_train.ravel() + 1 + np.random.randn(100) * 0.5

    X_test = np.random.uniform(-3, 3, (20, 1))
    y_test = 2 * X_test.ravel() ** 2 - 3 * X_test.ravel() + 1 + np.random.randn(20) * 0.5

    # 尝试不同的多项式阶数
    for degree in [1, 2, 3]:
        print(f"\n多项式阶数: {degree}")
        poly_reg = PolynomialRegression(degree=degree, learning_rate=0.01, n_iterations=1000)
        poly_reg.fit(X_train, y_train)

        # 预测并评估
        r2 = poly_reg.score(X_test, y_test)
        print(f"R²分数: {r2:.4f}")

        # 预测几个样本
        sample_predictions = poly_reg.predict(X_test[:5])
        print(f"前5个样本预测: {sample_predictions}")
        print(f"前5个真实值: {y_test[:5]}")