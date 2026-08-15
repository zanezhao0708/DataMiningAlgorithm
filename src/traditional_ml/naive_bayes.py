"""
朴素贝叶斯（Naive Bayes）分类器实现
基于贝叶斯定理和特征条件独立假设
"""

import numpy as np


class NaiveBayes:
    def __init__(self):
        """朴素贝叶斯分类器初始化"""
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        """
        训练朴素贝叶斯模型

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练标签，形状为(n_samples,)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        # 对每个类别计算均值、方差和先验概率
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0)
            self.priors[c] = X_c.shape[0] / n_samples

        print(f"朴素贝叶斯模型训练完成，类别数: {len(self.classes)}")

    def _gaussian_probability(self, x, mean, var):
        """计算高斯概率密度"""
        eps = 1e-4  # 防止方差为0
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var + eps))
        # 概率下限截断，避免 log(0) 产生 -inf
        return np.maximum(coeff * exponent, 1e-12)

    def _predict_single(self, x):
        """对单个样本进行预测"""
        posteriors = []

        for c in self.classes:
            # 计算先验概率的对数
            prior = np.log(self.priors[c])

            # 计算条件概率的对数
            likelihood = np.sum(np.log(
                self._gaussian_probability(x, self.mean[c], self.var[c])
            ))

            # 后验概率 = 先验概率 × 条件概率
            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """
        对多个样本进行预测（向量化实现，公式与 _predict_single 一致）

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测结果，形状为(n_samples,)
        """
        X = np.array(X)
        eps = 1e-4
        log_floor = np.log(1e-12)  # 与 _gaussian_probability 的截断下限一致
        best = np.full(len(X), self.classes[0])
        best_score = np.full(len(X), -np.inf)
        for c in self.classes:
            var = self.var[c]
            # log N(x; mean, var) 逐特征计算后截断下限再求和（与 _predict_single 一致）
            log_pdf = np.maximum(-0.5 * np.log(2.0 * np.pi * var + eps)
                                 - (X - self.mean[c]) ** 2 / (2 * var + eps),
                                 log_floor).sum(axis=1)
            score = np.log(self.priors[c]) + log_pdf
            mask = score > best_score
            best[mask] = c
            best_score[mask] = score[mask]
        return best

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
    # 示例：使用朴素贝叶斯进行分类
    print("=== 朴素贝叶斯算法示例 ===")

    # 生成示例数据
    np.random.seed(42)
    X_train = np.random.randn(100, 3)
    y_train = np.random.choice([0, 1, 2], size=100)

    X_test = np.random.randn(20, 3)
    y_test = np.random.choice([0, 1, 2], size=20)

    # 创建并训练模型
    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    # 预测并评估
    predictions = nb.predict(X_test)
    accuracy = nb.score(X_test, y_test)

    print(f"预测结果: {predictions[:10]}")
    print(f"真实标签: {y_test[:10]}")
    print(f"准确率: {accuracy:.4f}")