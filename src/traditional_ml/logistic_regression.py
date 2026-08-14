"""
逻辑回归（Logistic Regression）分类算法实现
使用sigmoid函数将线性组合映射到[0,1]区间进行二分类
"""

import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        逻辑回归初始化

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
        训练逻辑回归模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练标签（0或1），形状为(n_samples,)
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

        # 梯度下降
        for i in range(self.n_iterations):
            # 前向传播
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % step == 0 or i == self.n_iterations - 1:
                self.history.append({'iter': i,
                                     'weights': self.weights.copy(),
                                     'bias': float(self.bias),
                                     'loss': float(self._compute_loss(y, y_pred))})

            # 打印损失
            if i % 100 == 0:
                loss = self._compute_loss(y, y_pred)
                print(f"迭代 {i}, 损失: {loss:.4f}")

        print(f"逻辑回归训练完成")

    def _sigmoid(self, z):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def _compute_loss(self, y_true, y_pred):
        """计算二元交叉熵损失"""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def predict_proba(self, X):
        """
        预测概率

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测概率，形状为(n_samples,)
        """
        linear_model = np.dot(np.array(X), self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        预测类别

        参数:
            X: 测试特征，形状为(n_samples, n_features)
            threshold: 分类阈值

        返回:
            预测类别，形状为(n_samples,)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

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

    def confusion_matrix(self, X, y):
        """
        计算混淆矩阵

        参数:
            X: 测试特征
            y: 真实标签

        返回:
            混淆矩阵 (TP, TN, FP, FN)
        """
        predictions = self.predict(X)
        tp = np.sum((predictions == 1) & (y == 1))
        tn = np.sum((predictions == 0) & (y == 0))
        fp = np.sum((predictions == 1) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))
        return tp, tn, fp, fn

    def precision_recall_f1(self, X, y):
        """
        计算精确率、召回率和F1分数

        参数:
            X: 测试特征
            y: 真实标签

        返回:
            (precision, recall, f1)
        """
        tp, tn, fp, fn = self.confusion_matrix(X, y)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1


if __name__ == '__main__':
    # 示例：使用逻辑回归进行分类
    print("=== 逻辑回归算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    # 生成两个簇的数据
    class0 = np.random.randn(50, 2) + np.array([-2, -2])
    class1 = np.random.randn(50, 2) + np.array([2, 2])
    X_train = np.vstack([class0, class1])
    y_train = np.hstack([np.zeros(50), np.ones(50)])

    X_test = np.random.randn(20, 2) * 2
    y_test = np.random.randint(0, 2, 20)

    # 创建并训练模型
    lr = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    lr.fit(X_train, y_train)

    # 预测并评估
    predictions = lr.predict(X_test)
    probabilities = lr.predict_proba(X_test)
    accuracy = lr.score(X_test, y_test)
    precision, recall, f1 = lr.precision_recall_f1(X_test, y_test)

    print(f"\n预测结果: {predictions[:10]}")
    print(f"预测概率: {probabilities[:10]}")
    print(f"真实标签: {y_test[:10]}")
    print(f"\n准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")