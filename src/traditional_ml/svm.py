"""
支持向量机（Support Vector Machine）算法实现
通过寻找最优超平面进行分类，使用核技巧处理非线性问题
"""

import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000, kernel='linear'):
        """
        SVM初始化

        参数:
            learning_rate: 学习率
            lambda_param: 正则化参数
            n_iterations: 迭代次数
            kernel: 核函数类型，'linear'或'rbf'
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.kernel = kernel
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        训练SVM模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练标签（-1或1），形状为(n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        # 将标签转换为-1和1
        self.classes = np.unique(y)
        y_adjusted = np.where(y == self.classes[0], -1, 1)

        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 记录训练过程（权重快照，供可视化动画回放）
        self.history = []
        step = max(1, self.n_iterations // 40)

        # 训练
        for i in range(self.n_iterations):
            for idx, (x_i, y_i) in enumerate(zip(X, y_adjusted)):
                condition = y_i * (np.dot(x_i, self.weights) - self.bias) >= 1

                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y_i))
                    self.bias -= self.learning_rate * y_i

            if i % step == 0 or i == self.n_iterations - 1:
                self.history.append({'iter': i,
                                     'weights': self.weights.copy(),
                                     'bias': float(self.bias),
                                     'loss': float(self._compute_loss(X, y_adjusted))})

            if i % 100 == 0:
                loss = self._compute_loss(X, y_adjusted)
                print(f"迭代 {i}, 损失: {loss:.4f}")

        print(f"SVM训练完成，核函数: {self.kernel}")

    def _compute_loss(self, X, y):
        """计算损失"""
        n = len(y)
        distances = 1 - y * (np.dot(X, self.weights) - self.bias)
        distances = np.maximum(0, distances)
        hinge_loss = np.sum(distances) / n
        return hinge_loss + self.lambda_param * np.dot(self.weights, self.weights)

    def predict(self, X):
        """
        对新数据进行预测

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测类别，形状为(n_samples,)
        """
        linear_output = np.dot(np.array(X), self.weights) - self.bias
        predictions = np.sign(linear_output)
        # 将-1/1转换回原始类别（单类别时全部映射为该类）
        pos = self.classes[min(1, len(self.classes) - 1)]
        return np.where(predictions == -1, self.classes[0], pos)

    def decision_function(self, X):
        """
        计算决策函数值

        参数:
            X: 测试特征

        返回:
            决策函数值
        """
        return np.dot(np.array(X), self.weights) - self.bias

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
    # 示例：使用SVM进行分类
    print("=== SVM算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    class0 = np.random.randn(50, 2) + np.array([-2, -2])
    class1 = np.random.randn(50, 2) + np.array([2, 2])
    X_train = np.vstack([class0, class1])
    y_train = np.hstack([np.zeros(50), np.ones(50)])

    X_test = np.random.randn(20, 2) * 2
    y_test = np.random.randint(0, 2, 20)

    # 创建并训练模型
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
    svm.fit(X_train, y_train)

    # 预测并评估
    predictions = svm.predict(X_test)
    accuracy = svm.score(X_test, y_test)

    print(f"\n预测结果: {predictions[:10]}")
    print(f"真实标签: {y_test[:10]}")
    print(f"准确率: {accuracy:.4f}")