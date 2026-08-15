"""
MLP 回归算法实现
用神经网络拟合任意非线性曲线，记录逐epoch权重快照用于可视化动画
"""

import numpy as np


class MLPRegressor:
    def __init__(self, hidden_layers=(16,), learning_rate=0.05, n_epochs=300,
                 activation='tanh', random_state=None):
        """
        MLP回归初始化

        参数:
            hidden_layers: 隐藏层宽度元组
            learning_rate: 学习率
            n_epochs: 训练轮数
            activation: 隐藏层激活函数 'tanh' 或 'relu'
            random_state: 随机种子
        """
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.activation = activation
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.history = []  # 每 epoch 的权重快照与损失，供动画回放

    def _init_params(self, n_features):
        """Xavier初始化权重（回归输出为单值连续量）"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        layer_sizes = [n_features] + list(self.hidden_layers) + [1]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(1.0 / layer_sizes[i])
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale)
            self.biases.append(np.zeros(layer_sizes[i + 1]))

    def _activate(self, z):
        if self.activation == 'tanh':
            return np.tanh(z)
        return np.maximum(0, z)

    def _activate_grad(self, z):
        if self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        return (z > 0).astype(float)

    def _forward(self, X, weights=None, biases=None):
        """前向传播（可传入快照权重），返回输出与各层缓存"""
        Ws = weights if weights is not None else self.weights
        bs = biases if biases is not None else self.biases
        activations = [X]
        zs = []
        a = X
        for i, (W, b) in enumerate(zip(Ws, bs)):
            z = a @ W + b
            zs.append(z)
            if i == len(Ws) - 1:
                a = z  # 回归输出层为线性
            else:
                a = self._activate(z)
            activations.append(a)
        return activations, zs

    def fit(self, X, y):
        """
        训练MLP回归模型（全批量梯度下降）

        参数:
            X: 训练特征，形状为(n_samples, n_features)
            y: 训练目标，形状为(n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).ravel()
        n_samples, n_features = X.shape

        # 目标标准化，加速收敛且便于统一学习率
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        y_norm = (y - self.y_mean_) / self.y_std_

        self._init_params(n_features)
        self.history = []

        # 快照采样：最多40帧，均匀覆盖整个训练过程
        step = max(1, self.n_epochs // 40)
        snapshot_idx = set(range(0, self.n_epochs, step))
        snapshot_idx.add(self.n_epochs - 1)

        for epoch in range(self.n_epochs):
            activations, zs = self._forward(X)
            y_pred = activations[-1].ravel()
            loss = float(np.mean((y_pred - y_norm) ** 2))

            if epoch in snapshot_idx:
                self.history.append({
                    'epoch': epoch,
                    'weights': [w.copy() for w in self.weights],
                    'biases': [b.copy() for b in self.biases],
                    'loss': loss,
                })

            # 反向传播：输出层为线性，delta = (pred - y) * 1
            delta = (y_pred - y_norm).reshape(-1, 1)
            deltas = [None] * len(self.weights)
            deltas[-1] = delta
            for i in range(len(self.weights) - 2, -1, -1):
                delta = deltas[i + 1] @ self.weights[i + 1].T * self._activate_grad(zs[i])
                deltas[i] = delta

            # 参数更新
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * (activations[i].T @ deltas[i]) / n_samples
                self.biases[i] -= self.learning_rate * deltas[i].mean(axis=0)

        final_loss = self.history[-1]['loss'] if self.history else float('nan')
        print(f"MLP回归训练完成，最终损失: {final_loss:.4f}")

    def _predict_normalized(self, X, weights=None, biases=None):
        """返回标准化空间的预测值"""
        X = np.array(X, dtype=float)
        activations, _ = self._forward(X, weights, biases)
        return activations[-1].ravel()

    def predict(self, X):
        """
        对新数据进行预测

        参数:
            X: 测试特征，形状为(n_samples, n_features)

        返回:
            预测结果，形状为(n_samples,)
        """
        return self._predict_normalized(X) * self.y_std_ + self.y_mean_

    def score(self, X, y):
        """计算R²分数"""
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=float).ravel()
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / (ss_tot + 1e-12)
