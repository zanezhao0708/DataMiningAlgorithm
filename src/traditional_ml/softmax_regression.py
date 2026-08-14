"""
Softmax回归（Softmax Regression）算法实现
多分类逻辑回归，使用softmax函数输出类别概率
"""

import numpy as np


class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, reg_lambda=0.01):
        """
        Softmax回归初始化

        参数:
            learning_rate: 学习率
            n_iterations: 迭代次数
            reg_lambda: 正则化强度
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        训练Softmax回归模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练标签（类别从0开始），形状为(n_samples,)
        """
        X = np.array(X)
        y = np.array(y).astype(int)

        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))

        # 初始化参数
        self.weights = np.random.randn(n_features, self.n_classes) * 0.01
        self.bias = np.zeros(self.n_classes)

        # 将标签转换为one-hot编码
        y_one_hot = np.eye(self.n_classes)[y]

        # 记录训练过程（权重快照，供可视化动画回放）
        self.history = []
        step = max(1, self.n_iterations // 40)

        # 梯度下降
        for i in range(self.n_iterations):
            # 计算得分
            scores = np.dot(X, self.weights) + self.bias

            # 计算概率
            probabilities = self._softmax(scores)

            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (probabilities - y_one_hot)) + \
                 (self.reg_lambda / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(probabilities - y_one_hot, axis=0)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % step == 0 or i == self.n_iterations - 1:
                self.history.append({'iter': i,
                                     'weights': self.weights.copy(),
                                     'bias': self.bias.copy(),
                                     'loss': float(self._compute_loss(y_one_hot, probabilities))})

            if i % 100 == 0:
                loss = self._compute_loss(y_one_hot, probabilities)
                print(f"迭代 {i}, 损失: {loss:.4f}")

        print(f"Softmax回归训练完成，类别数: {self.n_classes}")

    def _softmax(self, scores):
        """Softmax函数"""
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def _compute_loss(self, y_one_hot, probabilities):
        """计算交叉熵损失"""
        eps = 1e-10
        loss = -np.sum(y_one_hot * np.log(probabilities + eps)) / len(y_one_hot)
        loss += (self.reg_lambda / (2 * len(y_one_hot))) * np.sum(self.weights ** 2)
        return loss

    def predict_proba(self, X):
        """
        预测概率

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测概率，形状为(n_samples, n_classes)
        """
        scores = np.dot(np.array(X), self.weights) + self.bias
        return self._softmax(scores)

    def predict(self, X):
        """
        预测类别

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测类别，形状为(n_samples,)
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

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
    # 示例：使用Softmax回归进行多分类
    print("=== Softmax回归算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    class0 = np.random.randn(40, 3) + np.array([0, 0, 0])
    class1 = np.random.randn(40, 3) + np.array([3, 3, 3])
    class2 = np.random.randn(40, 3) + np.array([-3, -3, -3])

    X_train = np.vstack([class0, class1, class2])
    y_train = np.hstack([np.zeros(40), np.ones(40), np.full(40, 2)])

    X_test = np.random.randn(30, 3) * 2
    y_test = np.random.randint(0, 3, 30)

    # 创建并训练模型
    softmax = SoftmaxRegression(learning_rate=0.1, n_iterations=1000)
    softmax.fit(X_train, y_train)

    # 预测并评估
    predictions = softmax.predict(X_test)
    probabilities = softmax.predict_proba(X_test)
    accuracy = softmax.score(X_test, y_test)

    print(f"\n预测结果: {predictions[:10]}")
    print(f"真实标签: {y_test[:10]}")
    print(f"准确率: {accuracy:.4f}")
    print(f"\n预测概率（前3个样本）:")
    print(probabilities[:3])
