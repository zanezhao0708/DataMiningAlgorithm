"""
感知机（Perceptron）算法实现
最简单的神经网络模型，用于二分类问题
"""

import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        感知机初始化

        参数:
            learning_rate: 学习率
            n_iterations: 迭代次数
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        训练感知机模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练标签，形状为(n_samples,)
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

        # 训练
        for i in range(self.n_iterations):
            n_errors = 0

            for idx, (x_i, y_i) in enumerate(zip(X, y)):
                # 预测
                prediction = self.predict_single(x_i)

                # 更新权重（如果预测错误）
                if y_i != prediction:
                    self.weights += self.learning_rate * y_i * x_i
                    self.bias += self.learning_rate * y_i
                    n_errors += 1

            if i % step == 0 or i == self.n_iterations - 1 or n_errors == 0:
                self.history.append({'iter': i,
                                     'weights': self.weights.copy(),
                                     'bias': float(self.bias),
                                     'errors': int(n_errors)})

            if i % 100 == 0:
                accuracy = self.score(X, y)
                print(f"迭代 {i}, 错误数: {n_errors}, 准确率: {accuracy:.4f}")

            # 如果没有错误，提前停止
            if n_errors == 0:
                print(f"在第 {i+1} 轮收敛")
                break

        print(f"感知机训练完成")

    def predict_single(self, x):
        """对单个样本进行预测"""
        linear_output = np.dot(x, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)

    def predict(self, X):
        """
        对多个样本进行预测

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测结果，形状为(n_samples,)
        """
        linear_output = np.dot(np.array(X), self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)

    def score(self, X, y):
        """
        计算准确率

        参数:
            X: 测试特征
            y: 真实标签

        返回:
            准确率
        """
        predictions = self.predict(X)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy


if __name__ == '__main__':
    # 示例：使用感知机进行分类
    print("=== 感知机算法示例 ===")

    # 生成示例数据（线性可分）
    np.random.seed(42)
    class_pos = np.random.randn(50, 2) + np.array([2, 2])
    class_neg = np.random.randn(50, 2) + np.array([-2, -2])
    X_train = np.vstack([class_pos, class_neg])
    y_train = np.hstack([np.ones(50), -np.ones(50)])

    X_test = np.random.randn(20, 2) * 2
    y_test = np.where(X_test[:, 0] + X_test[:, 1] > 0, 1, -1)

    # 创建并训练模型
    perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
    perceptron.fit(X_train, y_train)

    # 预测并评估
    predictions = perceptron.predict(X_test)
    accuracy = perceptron.score(X_test, y_test)

    print(f"\n预测结果: {predictions[:10]}")
    print(f"真实标签: {y_test[:10]}")
    print(f"准确率: {accuracy:.4f}")
    print(f"权重: {perceptron.weights}, 偏置: {perceptron.bias:.4f}")