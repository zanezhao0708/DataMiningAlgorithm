"""
Lasso回归（Least Absolute Shrinkage and Selection Operator）算法实现
带L1正则化的线性回归，可以实现特征选择
"""

import numpy as np


class LassoRegression:
    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000):
        """
        Lasso回归初始化

        参数:
            alpha: 正则化强度
            learning_rate: 学习率
            n_iterations: 迭代次数
        """
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        训练Lasso回归模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练目标，形状为(n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 记录训练过程（权重快照，供可视化动画回放）
        self.history = []
        step = max(1, self.n_iterations // 40)

        # 梯度下降训练
        for i in range(self.n_iterations):
            # 预测
            y_pred = np.dot(X, self.weights) + self.bias

            # 计算梯度（包含L1正则化项）
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # L1正则化使用软阈值函数
            dw += (self.alpha / n_samples) * np.sign(self.weights)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % step == 0 or i == self.n_iterations - 1:
                self.history.append({'iter': i,
                                     'weights': self.weights.copy(),
                                     'bias': float(self.bias),
                                     'loss': float(self._compute_loss(X, y))})

            if i % 100 == 0:
                loss = self._compute_loss(X, y)
                n_nonzero = np.sum(np.abs(self.weights) > 1e-4)
                print(f"迭代 {i}, 损失: {loss:.4f}, 非零权重数: {n_nonzero}/{n_features}")

        print(f"Lasso回归训练完成，正则化参数alpha: {self.alpha}")
        print(f"非零权重特征数: {np.sum(np.abs(self.weights) > 1e-4)}/{n_features}")

    def _compute_loss(self, X, y):
        """计算损失（MSE + L1正则化）"""
        y_pred = np.dot(X, self.weights) + self.bias
        mse = np.mean((y - y_pred) ** 2)
        l1_penalty = self.alpha * np.sum(np.abs(self.weights))
        return mse + l1_penalty

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

    def get_feature_importance(self, feature_names=None):
        """
        获取特征重要性（基于权重绝对值）

        参数:
            feature_names: 特征名称列表

        返回:
            特征重要性字典
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.weights))]

        importance = dict(zip(feature_names, np.abs(self.weights)))
        return {k: v for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)}


if __name__ == '__main__':
    # 示例：使用Lasso回归进行预测和特征选择
    print("=== Lasso回归算法示例 ===")

    # 生成示例数据（只有部分特征有用）
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    # 只有前3个特征影响目标值
    y_train = 1 + 1.5 * X_train[:, 0] - 2.0 * X_train[:, 1] + 0.5 * X_train[:, 2] + np.random.randn(100) * 0.5

    X_test = np.random.randn(20, 10)
    y_test = 1 + 1.5 * X_test[:, 0] - 2.0 * X_test[:, 1] + 0.5 * X_test[:, 2] + np.random.randn(20) * 0.5

    # 创建并训练模型
    lasso = LassoRegression(alpha=0.1, learning_rate=0.01, n_iterations=2000)
    lasso.fit(X_train, y_train)

    # 预测并评估
    r2 = lasso.score(X_test, y_test)
    print(f"\nR²分数: {r2:.4f}")

    # 查看特征重要性
    feature_names = [f'feature_{i}' for i in range(10)]
    importance = lasso.get_feature_importance(feature_names)

    print("\n特征重要性（按权重绝对值排序）:")
    for feature, weight in importance.items():
        print(f"  {feature}: {weight:.4f}")