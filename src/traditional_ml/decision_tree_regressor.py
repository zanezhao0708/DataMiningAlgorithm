"""
决策树回归算法实现
用MSE作为分裂准则拟合非线性阶梯状曲线，逐深度记录快照用于可视化动画
"""

import numpy as np


class DecisionTreeRegressor:
    def __init__(self, max_depth=6, min_samples_split=2):
        """
        决策树回归初始化

        参数:
            max_depth: 最大深度
            min_samples_split: 内部节点再划分所需最小样本数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.history = []  # 按深度记录树快照，展示曲线从平线到阶梯的演化

    def fit(self, X, y):
        """
        训练决策树回归模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练目标，形状为(n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).ravel()
        self.tree = None
        self.history = []

        # 按深度增量生长：每加深一层记录一次快照，
        # 展示拟合曲线从"一条平线"逐步变成"非线性阶梯"的过程
        for depth in range(1, self.max_depth + 1):
            tree = self._grow(X, y, 0, depth)
            pred = self._predict_tree(X, tree)
            mse = float(np.mean((y - pred) ** 2))
            self.history.append({'depth': depth, 'tree': tree, 'loss': mse})

        self.tree = self.history[-1]['tree']
        print(f"决策树回归训练完成，深度: {self.max_depth}")

    def _grow(self, X, y, depth, max_depth):
        """限制深度的递归建树"""
        # 叶节点：达到深度上限 / 样本过少 / 目标一致
        if (depth >= max_depth or len(y) < self.min_samples_split
                or np.allclose(y, y[0])):
            return {'leaf': True, 'value': float(y.mean())}

        best = self._best_split(X, y)
        if best is None:
            return {'leaf': True, 'value': float(y.mean())}

        feature, threshold = best
        left_mask = X[:, feature] <= threshold
        return {
            'leaf': False,
            'feature': feature,
            'threshold': float(threshold),
            'left': self._grow(X[left_mask], y[left_mask], depth + 1, max_depth),
            'right': self._grow(X[~left_mask], y[~left_mask], depth + 1, max_depth),
        }

    def _best_split(self, X, y):
        """MSE下降最大的分裂点（向量化扫描全部候选阈值）"""
        n_samples, n_features = X.shape
        best_gain, best_split = 0.0, None
        parent_sq = np.sum((y - y.mean()) ** 2)

        for f in range(n_features):
            order = np.argsort(X[:, f])
            xs, ys = X[order, f], y[order]
            # 候选阈值取相邻不同值的中点
            diff_pos = np.nonzero(np.diff(xs) > 1e-12)[0]
            for pos in diff_pos:
                threshold = (xs[pos] + xs[pos + 1]) / 2
                y_left, y_right = ys[:pos + 1], ys[pos + 1:]
                if len(y_left) < 1 or len(y_right) < 1:
                    continue
                gain = parent_sq - (
                    np.sum((y_left - y_left.mean()) ** 2)
                    + np.sum((y_right - y_right.mean()) ** 2)
                )
                if gain > best_gain + 1e-12:
                    best_gain, best_split = gain, (f, threshold)

        return best_split

    def _predict_tree(self, X, tree):
        """按给定树结构预测"""
        X = np.array(X, dtype=float)
        return np.array([self._predict_single(x, tree) for x in X])

    def _predict_single(self, x, node):
        while not node['leaf']:
            node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
        return node['value']

    def predict(self, X):
        """
        对新数据进行预测

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测结果，形状为(n_samples,)
        """
        return self._predict_tree(X, self.tree)

    def score(self, X, y):
        """计算R²分数"""
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=float).ravel()
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / (ss_tot + 1e-12)
