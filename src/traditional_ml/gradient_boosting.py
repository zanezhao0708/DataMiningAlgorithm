"""
梯度提升决策树（Gradient Boosting Decision Tree）算法实现
通过迭代训练回归树，拟合前一轮的残差
"""

import numpy as np


class _RegressionTree:
    """用于梯度提升的回归树（基于均方误差分裂）"""

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _mse(self, y):
        """计算均方误差"""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def _best_split(self, X, y):
        """寻找最佳分裂点（基于方差减少）"""
        best_gain = 0
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape
        parent_mse = self._mse(y)

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_idx = X[:, feature_idx] <= threshold
                right_idx = ~left_idx

                if np.sum(left_idx) < self.min_samples_split or np.sum(right_idx) < self.min_samples_split:
                    continue

                # 加权MSE
                n_left, n_right = np.sum(left_idx), np.sum(right_idx)
                weighted_mse = (n_left / n_samples) * self._mse(y[left_idx]) + \
                              (n_right / n_samples) * self._mse(y[right_idx])

                gain = parent_mse - weighted_mse

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow(self, X, y, depth):
        """递归构建回归树"""
        n_samples = X.shape[0]

        # 停止条件
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return {'leaf': True, 'value': np.mean(y)}

        feature_idx, threshold = self._best_split(X, y)

        if feature_idx is None:
            return {'leaf': True, 'value': np.mean(y)}

        left_idx = X[:, feature_idx] <= threshold
        right_idx = ~left_idx

        return {
            'leaf': False,
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': self._grow(X[left_idx], y[left_idx], depth + 1),
            'right': self._grow(X[right_idx], y[right_idx], depth + 1)
        }

    def fit(self, X, y):
        """训练回归树"""
        self.tree = self._grow(np.array(X), np.array(y), 0)

    def _predict_single(self, x, node):
        """对单个样本预测"""
        if node['leaf']:
            return node['value']

        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

    def predict(self, X):
        """对多个样本预测"""
        return np.array([self._predict_single(x, self.tree) for x in np.array(X)])


class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        梯度提升初始化

        参数:
            n_estimators: 基学习器数量
            learning_rate: 学习率
            max_depth: 每棵树的最大深度
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        """
        训练梯度提升模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练目标，形状为(n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        # 初始化预测值
        self.initial_prediction = np.mean(y)
        y_pred = np.full(len(y), self.initial_prediction)

        # 迭代训练
        for i in range(self.n_estimators):
            # 计算残差（负梯度）
            residuals = y - y_pred

            # 训练回归树拟合残差
            tree = _RegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # 更新预测值
            y_pred += self.learning_rate * tree.predict(X)

            if i % 20 == 0:
                loss = np.mean((y - y_pred) ** 2)
                print(f"迭代 {i}, 损失: {loss:.4f}")

        print(f"梯度提升训练完成，基学习器数量: {self.n_estimators}")

    def predict(self, X):
        """
        对新数据进行预测

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测结果，形状为(n_samples,)
        """
        X = np.array(X)
        y_pred = np.full(X.shape[0], self.initial_prediction)

        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

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
    # 示例：使用梯度提升进行回归
    print("=== 梯度提升决策树算法示例 ===")

    # 生成非线性数据
    np.random.seed(42)
    X_train = np.random.rand(100, 2)
    y_train = np.sin(X_train[:, 0] * 3) + X_train[:, 1] ** 2 + np.random.randn(100) * 0.1

    X_test = np.random.rand(20, 2)
    y_test = np.sin(X_test[:, 0] * 3) + X_test[:, 1] ** 2 + np.random.randn(20) * 0.1

    # 创建并训练模型
    gb = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)
    gb.fit(X_train, y_train)

    # 预测并评估
    r2 = gb.score(X_test, y_test)
    print(f"\nR²分数: {r2:.4f}")

    # 预测几个样本
    sample_predictions = gb.predict(X_test[:5])
    print(f"前5个样本预测: {sample_predictions}")
    print(f"前5个真实值: {y_test[:5]}")
