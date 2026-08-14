"""
多层感知机（MLP）神经网络实现
支持任意层数与宽度，记录逐epoch训练历史用于可视化动画
"""

import numpy as np


class MLP:
    def __init__(self, hidden_layers=(8,), learning_rate=0.05, n_epochs=200,
                 activation='relu', random_state=None):
        """
        MLP初始化

        参数:
            hidden_layers: 隐藏层宽度元组，如(8,)、(6,6)
            learning_rate: 学习率
            n_epochs: 训练轮数
            activation: 激活函数 'relu' 或 'tanh'
            random_state: 随机种子
        """
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.activation = activation
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.history = []  # 每个epoch的{loss, accuracy}

    def _init_params(self, n_features, n_classes):
        """He初始化权重"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        layer_sizes = [n_features] + list(self.hidden_layers) + [n_classes]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
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

    def _forward(self, X):
        """前向传播，缓存每层输入与激活"""
        activations = [X]
        zs = []
        a = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            zs.append(z)
            if i == len(self.weights) - 1:
                # 输出层softmax
                z = z - z.max(axis=1, keepdims=True)
                e = np.exp(z)
                a = e / e.sum(axis=1, keepdims=True)
            else:
                a = self._activate(z)
            activations.append(a)
        return activations, zs

    def _cross_entropy(self, probs, y_onehot):
        return -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))

    def fit(self, X, y):
        """
        训练MLP，记录逐epoch历史

        参数:
            X: 训练特征 (n_samples, n_features)
            y: 训练标签 (n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y).astype(int)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        y_onehot = np.eye(n_classes)[y]

        self._init_params(n_features, n_classes)
        self.history = []

        for epoch in range(self.n_epochs):
            # 前向
            activations, zs = self._forward(X)
            probs = activations[-1]
            loss = self._cross_entropy(probs, y_onehot)
            acc = float(np.mean(np.argmax(probs, axis=1) == y))
            self.history.append({'loss': float(loss), 'accuracy': acc})

            # 反向传播
            delta = (probs - y_onehot) / n_samples
            for i in range(len(self.weights) - 1, -1, -1):
                dW = activations[i].T @ delta
                db = delta.sum(axis=0)
                if i > 0:
                    delta = (delta @ self.weights[i].T) * self._activate_grad(zs[i - 1])
                self.weights[i] -= self.learning_rate * dW
                self.biases[i] -= self.learning_rate * db

        print(f"MLP训练完成，最终损失: {self.history[-1]['loss']:.4f}")

    def predict_proba(self, X):
        """预测类别概率"""
        activations, _ = self._forward(np.array(X, dtype=float))
        return activations[-1]

    def predict(self, X):
        """预测类别"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        """计算准确率"""
        return float(np.mean(self.predict(X) == np.array(y).astype(int)))


if __name__ == '__main__':
    print("=== MLP神经网络示例 ===")
    np.random.seed(42)
    # 螺旋数据
    n = 100
    X, y = [], []
    for c in range(2):
        r = np.linspace(0.2, 1.5, n)
        t = 2.5 * np.pi * r + c * np.pi
        X.append(np.column_stack([r * np.sin(t), r * np.cos(t)]))
        y.extend([c] * n)
    X, y = np.vstack(X), np.array(y)

    mlp = MLP(hidden_layers=(8, 8), learning_rate=0.05, n_epochs=300, random_state=42)
    mlp.fit(X, y)
    print(f"训练准确率: {mlp.score(X, y):.4f}")
    print(f"损失曲线: 起始{mlp.history[0]['loss']:.3f} → 结束{mlp.history[-1]['loss']:.3f}")
