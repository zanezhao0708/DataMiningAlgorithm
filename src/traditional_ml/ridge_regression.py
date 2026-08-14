"""
岭回归（Ridge Regression）算法实现
带L2正则化的线性回归，防止过拟合
"""

import numpy as np


class RidgeRegression:
    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000, method='gradient_descent'):
        """
        岭回归初始化

        参数:
            alpha: 正则化强度
            learning_rate: 学习率
            n_iterations: 迭代次数
            method: 求解方法，'gradient_descent'或'closed_form'
        """
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        训练岭回归模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练目标，形状为(n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        if self.method == 'gradient_descent':
            self._fit_gradient_descent(X, y)
        else:
            self._fit_closed_form(X, y)

        print(f"岭回归训练完成，正则化参数alpha: {self.alpha}")

    def _fit_gradient_descent(self, X, y):
        """使用梯度下降训练"""
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            # 预测
            y_pred = np.dot(X, self.weights) + self.bias

            # 计算梯度（包含L2正则化项）
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (2 * self.alpha / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 100 == 0:
                loss = self._compute_loss(X, y)
                print(f"迭代 {i}, 损失: {loss:.4f}")

    def _fit_closed_form(self, X, y):
        """使用闭式解训练"""
        n_samples, n_features = X.shape

        # 添加偏置项
        X_b = np.c_[np.ones(n_samples), X]

        # 岭回归的闭式解: θ = (X^T X + αI)^(-1) X^T y
        A = np.eye(n_features + 1)
        A[0, 0] = 0  # 不对偏置项进行正则化

        theta = np.linalg.inv(X_b.T.dot(X_b) + self.alpha * A).dot(X_b.T).dot(y)

        self.bias = theta[0]
        self.weights = theta[1:]

    def _compute_loss(self, X, y):
        """计算损失（MSE + L2正则化）"""
        y_pred = np.dot(X, self.weights) + self.bias
        mse = np.mean((y - y_pred) ** 2)
        l2_penalty = self.alpha * np.sum(self.weights ** 2)
        return mse + l2_penalty

    def predict(self, X):
        """
        对新数据进行预测

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测结果，形状为(n_samples,)
        """
        return np.dot(np.array(X), self.weights) + self.bias

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
    # 示例：使用岭回归进行预测
    print("=== 岭回归算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    X_train = 3 * np.random.rand(100, 3)
    y_train = 2 + np.dot(X_train, [1.5, -2.0, 1.0]) + np.random.randn(100) * 0.5

    X_test = 3 * np.random.rand(20, 3)
    y_test = 2 + np.dot(X_test, [1.5, -2.0, 1.0]) + np.random.randn(20) * 0.5

    # 方法1: 梯度下降
    print("\n方法1: 梯度下降")
    ridge_gd = RidgeRegression(alpha=1.0, learning_rate=0.01, n_iterations=1000)
    ridge_gd.fit(X_train, y_train)
    r2_gd = ridge_gd.score(X_test, y_test)
    print(f"权重: {ridge_gd.weights}")
    print(f"偏置: {ridge_gd.bias:.4f}")
    print(f"R²分数: {r2_gd:.4f}")

    # 方法2: 闭式解
    print("\n方法2: 闭式解")
    ridge_cf = RidgeRegression(alpha=1.0, method='closed_form')
    ridge_cf.fit(X_train, y_train)
    r2_cf = ridge_cf.score(X_test, y_test)
    print(f"权重: {ridge_cf.weights}")
    print(f"偏置: {ridge_cf.bias:.4f}")
    print(f"R²分数: {r2_cf:.4f}")