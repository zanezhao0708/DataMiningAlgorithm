"""
K近邻（K-Nearest Neighbors）算法实现
通过计算距离找到最近的K个邻居进行分类或回归
"""

import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3, task_type='classification'):
        """
        KNN算法初始化

        参数:
            k: 邻居数量，默认为3
            task_type: 任务类型，'classification'或'regression'
        """
        self.k = k
        self.task_type = task_type
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        训练模型（实际上只是存储训练数据）

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练标签，形状为(n_samples,)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        print(f"KNN模型训练完成，训练样本数: {len(self.X_train)}")

    def _euclidean_distance(self, x1, x2):
        """计算欧氏距离"""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_single(self, x):
        """对单个样本进行预测"""
        # 计算测试样本与所有训练样本的距离
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # 找到距离最近的k个样本的索引
        k_indices = np.argsort(distances)[:self.k]

        # 获取k个最近邻居的标签
        k_labels = self.y_train[k_indices]

        # 根据任务类型进行预测
        if self.task_type == 'classification':
            # 分类任务：投票决定类别
            most_common = Counter(k_labels).most_common(1)
            return most_common[0][0]
        else:
            # 回归任务：取平均值
            return np.mean(k_labels)

    def predict(self, X):
        """
        对多个样本进行预测（向量化实现，批量计算距离矩阵）

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测结果，形状为(n_samples,)
        """
        X = np.array(X)
        if self.task_type == 'regression':
            # 回归仍逐样本取均值（简单场景）
            return np.array([self._predict_single(x) for x in X])

        # 分类：向量化距离矩阵 (n_test, n_train)
        diff = X[:, None, :] - self.X_train[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        k_indices = np.argpartition(dist_sq, self.k, axis=1)[:, :self.k]
        k_labels = self.y_train[k_indices]

        # 多数投票：对每个测试样本统计邻居标签
        predictions = np.empty(X.shape[0], dtype=self.y_train.dtype)
        for i in range(X.shape[0]):
            labels, counts = np.unique(k_labels[i], return_counts=True)
            predictions[i] = labels[np.argmax(counts)]
        return predictions

    def score(self, X, y):
        """
        计算模型准确率（分类）或R²分数（回归）

        参数:
            X: 测试特征
            y: 真实标签

        返回:
            得分
        """
        predictions = self.predict(X)

        if self.task_type == 'classification':
            accuracy = np.sum(predictions == y) / len(y)
            return accuracy
        else:
            # R² 分数
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            return r_squared


if __name__ == '__main__':
    # 示例：使用KNN进行分类
    print("=== KNN算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    X_train = np.random.randn(100, 2)
    y_train = np.random.choice([0, 1, 2], size=100)

    X_test = np.random.randn(20, 2)
    y_test = np.random.choice([0, 1, 2], size=20)

    # 创建并训练模型
    knn = KNN(k=5, task_type='classification')
    knn.fit(X_train, y_train)

    # 预测并评估
    predictions = knn.predict(X_test)
    accuracy = knn.score(X_test, y_test)

    print(f"预测结果: {predictions[:10]}")
    print(f"真实标签: {y_test[:10]}")
    print(f"准确率: {accuracy:.4f}")