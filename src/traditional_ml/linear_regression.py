"""
线性回归（Linear Regression）算法实现
通过最小二乘法拟合线性模型
"""

import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, method='gradient_descent'):
        """
        线性回归初始化

        参数:
            learning_rate: 学习率（用于梯度下降）
            n_iterations: 迭代次数（用于梯度下降）
            method: 求解方法，'gradient_descent'或'normal_equation'
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        训练线性回归模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练目标，形状为(n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        if self.method == 'gradient_descent':
            self._fit_gradient_descent(X, y)
        else:
            self._fit_normal_equation(X, y)

        print(f"线性回归训练完成，方法: {self.method}")

    def _fit_gradient_descent(self, X, y):
        """使用梯度下降训练"""
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降
        for i in range(self.n_iterations):
            # 预测
            y_pred = np.dot(X, self.weights) + self.bias

            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 打印损失
            if i % 100 == 0:
                loss = self._mse(y_pred, y)
                print(f"迭代 {i}, 损失: {loss:.4f}")

    def _fit_normal_equation(self, X, y):
        """使用正规方程训练"""
        # 添加偏置项（x0 = 1）
        X_b = np.c_[np.ones(X.shape[0]), X]

        # 使用正规方程求解: θ = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

        self.bias = theta[0]
        self.weights = theta[1:]

    def _mse(self, y_true, y_pred):
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
    # 示例：使用线性回归进行预测
    print("=== 线性回归算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    X_train = 2 * np.random.rand(100, 1)
    y_train = 4 + 3 * X_train.ravel() + np.random.randn(100)

    X_test = 2 * np.random.rand(20, 1)
    y_test = 4 + 3 * X_test.ravel() + np.random.randn(20)

    # 方法1: 梯度下降
    print("\n方法1: 梯度下降")
    lr_gd = LinearRegression(learning_rate=0.1, n_iterations=1000, method='gradient_descent')
    lr_gd.fit(X_train, y_train)
    predictions_gd = lr_gd.predict(X_test)
    r2_gd = lr_gd.score(X_test, y_test)
    print(f"权重: {lr_gd.weights}, 偏置: {lr_gd.bias:.4f}")
    print(f"R²分数: {r2_gd:.4f}")

    # 方法2: 正规方程
    print("\n方法2: 正规方程")
    lr_ne = LinearRegression(method='normal_equation')
    lr_ne.fit(X_train, y_train)
    predictions_ne = lr_ne.predict(X_test)
    r2_ne = lr_ne.score(X_test, y_test)
    print(f"权重: {lr_ne.weights}, 偏置: {lr_ne.bias:.4f}")
    print(f"R²分数: {r2_ne:.4f}")