"""
决策树（Decision Tree）算法实现
通过递归地选择最优特征进行分裂来构建树结构
"""

import numpy as np
from collections import Counter


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        """
        决策树初始化

        参数:
            max_depth: 树的最大深度
            min_samples_split: 分裂节点所需的最小样本数
            min_samples_leaf: 叶节点的最小样本数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        """
        训练决策树模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练标签，形状为(n_samples,)
        """
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)
        print(f"决策树训练完成，最大深度: {self.max_depth}")

    def _gini(self, y):
        """计算基尼系数"""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _information_gain(self, y, left_idx, right_idx):
        """计算信息增益（使用基尼系数）"""
        n = len(y)
        n_left, n_right = len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0:
            return 0

        # 计算分裂后的加权基尼系数
        gini_left = self._gini(y[left_idx])
        gini_right = self._gini(y[right_idx])
        weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right

        # 信息增益 = 分裂前的基尼系数 - 分裂后的加权基尼系数
        return self._gini(y) - weighted_gini

    def _best_split(self, X, y):
        """寻找最佳分裂点"""
        best_gain = -1
        best_feature_idx = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                # 分裂数据
                left_idx = np.where(X[:, feature_idx] <= threshold)[0]
                right_idx = np.where(X[:, feature_idx] > threshold)[0]

                # 检查是否满足最小样本数要求
                if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                    continue

                # 计算信息增益
                gain = self._information_gain(y, left_idx, right_idx)

                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _grow_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples = X.shape[0]

        # 停止条件
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}

        # 寻找最佳分裂
        feature_idx, threshold = self._best_split(X, y)

        if feature_idx is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}

        # 分裂数据
        left_idx = np.where(X[:, feature_idx] <= threshold)[0]
        right_idx = np.where(X[:, feature_idx] > threshold)[0]

        # 递归构建左右子树
        left_subtree = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return {
            'leaf': False,
            'feature_idx': feature_idx,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def _predict_single(self, x, node):
        """对单个样本进行预测"""
        if node['leaf']:
            return node['value']

        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

    def predict(self, X):
        """
        对多个样本进行预测

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测结果，形状为(n_samples,)
        """
        return np.array([self._predict_single(x, self.tree) for x in X])

    def score(self, X, y):
        """
        计算模型准确率

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
    # 示例：使用决策树进行分类
    print("=== 决策树算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    X_train = np.random.randn(100, 4)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    X_test = np.random.randn(20, 4)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

    # 创建并训练模型
    dt = DecisionTree(max_depth=5)
    dt.fit(X_train, y_train)

    # 预测并评估
    predictions = dt.predict(X_test)
    accuracy = dt.score(X_test, y_test)

    print(f"预测结果: {predictions[:10]}")
    print(f"真实标签: {y_test[:10]}")
    print(f"准确率: {accuracy:.4f}")