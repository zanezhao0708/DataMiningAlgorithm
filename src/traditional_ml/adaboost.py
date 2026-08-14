"""
AdaBoost（Adaptive Boosting）集成学习算法实现
通过迭代训练弱分类器并组合成强分类器
"""

import numpy as np


class AdaBoost:
    def __init__(self, n_estimators=50):
        """
        AdaBoost初始化

        参数:
            n_estimators: 弱分类器数量
        """
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []

    def fit(self, X, y):
        """
        训练AdaBoost模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练标签（-1或1），形状为(n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        # 将标签转换为-1和1
        self.classes = np.unique(y)
        y_adjusted = np.where(y == self.classes[0], -1, 1)

        n_samples = X.shape[0]

        # 初始化样本权重
        weights = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            # 训练弱分类器（决策树桩）
            estimator = self._train_stump(X, y_adjusted, weights)

            # 计算分类误差
            predictions = self._stump_predict(X, estimator)
            incorrect = predictions != y_adjusted
            error = np.sum(weights * incorrect)

            # 避免误差为0或1
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # 计算分类器权重
            estimator_weight = 0.5 * np.log((1 - error) / error)

            # 更新样本权重
            weights = weights * np.exp(-estimator_weight * y_adjusted * predictions)
            weights = weights / np.sum(weights)  # 归一化

            # 保存分类器和权重
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)

            if i % 10 == 0:
                print(f"已训练 {i+1}/{self.n_estimators} 个弱分类器，误差: {error:.4f}")

        print(f"AdaBoost训练完成，弱分类器数量: {self.n_estimators}")

    def _train_stump(self, X, y, weights):
        """训练决策树桩"""
        n_samples, n_features = X.shape

        best_stump = {
            'feature_idx': 0,
            'threshold': 0,
            'direction': 1
        }
        min_error = float('inf')

        # 对每个特征寻找最佳分裂点
        for feature_idx in range(n_features):
            feature_values = np.sort(np.unique(X[:, feature_idx]))

            for threshold in feature_values:
                # 尝试两种方向
                for direction in [1, -1]:
                    predictions = direction * np.where(X[:, feature_idx] <= threshold, 1, -1)
                    error = np.sum(weights * (predictions != y))

                    if error < min_error:
                        min_error = error
                        best_stump = {
                            'feature_idx': feature_idx,
                            'threshold': threshold,
                            'direction': direction
                        }

        return best_stump

    def _stump_predict(self, X, stump):
        """决策树桩预测"""
        feature_idx = stump['feature_idx']
        threshold = stump['threshold']
        direction = stump['direction']

        return direction * np.where(X[:, feature_idx] <= threshold, 1, -1)

    def predict(self, X):
        """
        对新数据进行预测

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测类别，形状为(n_samples,)
        """
        X = np.array(X)

        # 加权投票
        predictions = np.zeros(X.shape[0])
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            predictions += weight * self._stump_predict(X, estimator)

        # 转换为类别标签
        predictions = np.sign(predictions)
        return np.where(predictions == -1, self.classes[0], self.classes[1])

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
    # 示例：使用AdaBoost进行分类
    print("=== AdaBoost算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.where(X_train[:, 0] + X_train[:, 1] > 0, 1, 0)

    X_test = np.random.randn(20, 5)
    y_test = np.where(X_test[:, 0] + X_test[:, 1] > 0, 1, 0)

    # 创建并训练模型
    ada = AdaBoost(n_estimators=50)
    ada.fit(X_train, y_train)

    # 预测并评估
    predictions = ada.predict(X_test)
    accuracy = ada.score(X_test, y_test)

    print(f"\n预测结果: {predictions[:10]}")
    print(f"真实标签: {y_test[:10]}")
    print(f"准确率: {accuracy:.4f}")