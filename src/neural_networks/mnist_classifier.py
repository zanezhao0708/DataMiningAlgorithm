import numpy as np
import pickle

"""
MLP整体结构 - 手写数字识别 (MNIST)
input: 28*28=784
hidden1: 1000 (ReLU)
hidden2: 400 (ReLU)
hidden3: 50 (ReLU)
output: 10 (Softmax)
"""

print("正在加载 MNIST 数据集...")
with open('../../data/data_NN.pkl', 'rb') as f:
    dataset = pickle.load(f)

X_train = dataset['x_train']  # 训练集特征 (N, 784)
t_train = dataset['t_train']  # 训练集真实标签 (N,)

# 将标签转为 One-Hot 独热编码
Y_train = np.eye(10)[t_train]  # MNIST 有10个类别
print(f"数据加载成功！训练样本数: {X_train.shape[0]}")

# 超参数设置
epochs = 100
batch_size = 100
learning_rate = 0.01
input_dim = 784

# 权重和偏置初始化
# *0.01 防止进入激活函数饱和区
np.random.seed(42)
w1 = np.random.randn(input_dim, 1000) * 0.01
b1 = np.zeros((1, 1000))

w2 = np.random.randn(1000, 400) * 0.01
b2 = np.zeros((1, 400))

w3 = np.random.randn(400, 50) * 0.01
b3 = np.zeros((1, 50))

w4 = np.random.randn(50, 10) * 0.01
b4 = np.zeros((1, 10))


# 激活函数
def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)


def relu_derivative(x):
    """ReLU 导数"""
    return (x > 0).astype(float)


def softmax(x):
    """Softmax 激活函数"""
    x = x - np.max(x, axis=1, keepdims=True)  # 防止数值溢出
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(pred, true):
    """交叉熵损失函数"""
    batch_size = pred.shape[0]
    prob = np.sum(pred * true, axis=1)
    return -np.sum(np.log(prob + 1e-7)) / batch_size


def predict(X):
    """前向传播预测"""
    layer1 = relu(np.dot(X, w1) + b1)
    layer2 = relu(np.dot(layer1, w2) + b2)
    layer3 = relu(np.dot(layer2, w3) + b3)
    output = softmax(np.dot(layer3, w4) + b4)
    return output, layer1, layer2, layer3


def accuracy(X, Y):
    """计算准确率"""
    pred, _, _, _ = predict(X)
    predictions = np.argmax(pred, axis=1)
    labels = np.argmax(Y, axis=1)
    return np.mean(predictions == labels)


# 训练循环
print("开始训练...")
for epoch in range(epochs):
    # 随机抽取小批量数据
    batch_indices = np.random.choice(len(X_train), batch_size, replace=False)
    input_batch = X_train[batch_indices]  # (100, 784)
    label_batch = Y_train[batch_indices]  # (100, 10)

    # 前向传播
    layer1_z = np.dot(input_batch, w1) + b1
    layer1 = relu(layer1_z)

    layer2_z = np.dot(layer1, w2) + b2
    layer2 = relu(layer2_z)

    layer3_z = np.dot(layer2, w3) + b3
    layer3 = relu(layer3_z)

    output_z = np.dot(layer3, w4) + b4
    output = softmax(output_z)

    # 反向传播
    # 输出层误差
    Doutput = output - label_batch

    # 第四层梯度
    dW4 = np.dot(layer3.T, Doutput) / batch_size
    db4 = np.sum(Doutput, axis=0, keepdims=True) / batch_size

    # 第三层误差传播
    Dlayer3 = np.dot(Doutput, w4.T) * relu_derivative(layer3_z)
    dW3 = np.dot(layer2.T, Dlayer3) / batch_size
    db3 = np.sum(Dlayer3, axis=0, keepdims=True) / batch_size

    # 第二层误差传播
    Dlayer2 = np.dot(Dlayer3, w3.T) * relu_derivative(layer2_z)
    dW2 = np.dot(layer1.T, Dlayer2) / batch_size
    db2 = np.sum(Dlayer2, axis=0, keepdims=True) / batch_size

    # 第一层误差传播
    Dlayer1 = np.dot(Dlayer2, w2.T) * relu_derivative(layer1_z)
    dW1 = np.dot(input_batch.T, Dlayer1) / batch_size
    db1 = np.sum(Dlayer1, axis=0, keepdims=True) / batch_size

    # 更新权重和偏置
    w4 -= learning_rate * dW4
    b4 -= learning_rate * db4
    w3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    w2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    w1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # 每10轮打印一次损失和准确率
    if epoch % 10 == 0:
        loss_val = cross_entropy_loss(output, label_batch)
        acc = accuracy(input_batch, label_batch)
        print(f"第 {epoch:>3} 轮, 损失值: {loss_val:.4f}, 批次准确率: {acc:.2%}")

# 最终评估
final_loss = cross_entropy_loss(output, label_batch)
final_acc = accuracy(X_train[:1000], Y_train[:1000])  # 在前1000个样本上测试
print(f"\n训练完成！")
print(f"最终损失值: {final_loss:.4f}")
print(f"验证集准确率: {final_acc:.2%}")