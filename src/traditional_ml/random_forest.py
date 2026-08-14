"""
随机森林（Random Forest）算法实现
集成多个决策树，通过投票机制进行分类
"""

import numpy as np
from .decision_tree import DecisionTree
from collections import Counter


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        """
        随机森林初始化

        参数:
            n_trees: 决策树数量
            max_depth: 每棵树的最大深度
            min_samples_split: 分裂节点所需的最小样本数
            n_features: 每棵树使用的特征数量（默认使用全部特征）
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        """
        训练随机森林模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练标签，形状为(n_samples,)
        """
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        if self.n_features is None:
            self.n_features = int(np.sqrt(n_features))

        # 训练多棵决策树
        for i in range(self.n_trees):
            # Bootstrap采样
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_subset = X[indices]
            y_subset = y[indices]

            # 随机选择特征子集
            feature_indices = np.random.choice(n_features, self.n_features, replace=False)
            X_subset = X_subset[:, feature_indices]

            # 训练决策树
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_subset, y_subset)

            self.trees.append({
                'tree': tree,
                'feature_indices': feature_indices
            })

            if i % 5 == 0:
                print(f"已训练 {i+1}/{self.n_trees} 棵树")

        print(f"随机森林训练完成，树数量: {self.n_trees}")

    def predict(self, X):
        """
        对新数据进行预测

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测结果，形状为(n_samples,)
        """
        X = np.array(X)
        tree_predictions = []

        # 收集每棵树的预测结果
        for tree_info in self.trees:
            tree = tree_info['tree']
            feature_indices = tree_info['feature_indices']
            X_subset = X[:, feature_indices]
            predictions = tree.predict(X_subset)
            tree_predictions.append(predictions)

        # 投票决定最终结果
        tree_predictions = np.array(tree_predictions)
        final_predictions = []

        for i in range(X.shape[0]):
            sample_predictions = tree_predictions[:, i]
            most_common = Counter(sample_predictions).most_common(1)[0][0]
            final_predictions.append(most_common)

        return np.array(final_predictions)

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
    # 示例：使用随机森林进行分类
    print("=== 随机森林算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.choice([0, 1, 2], size=100)

    X_test = np.random.randn(20, 10)
    y_test = np.random.choice([0, 1, 2], size=20)

    # 创建并训练模型
    rf = RandomForest(n_trees=10, max_depth=8, n_features=5)
    rf.fit(X_train, y_train)

    # 预测并评估
    predictions = rf.predict(X_test)
    accuracy = rf.score(X_test, y_test)

    print(f"\n预测结果: {predictions[:10]}")
    print(f"真实标签: {y_test[:10]}")
    print(f"准确率: {accuracy:.4f}")