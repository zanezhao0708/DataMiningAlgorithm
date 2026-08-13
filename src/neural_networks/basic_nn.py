#实现一个三层前馈神经网络，完成3分类任务。要求如下：
#（1）网络结构：第一隐藏层30个神经元，激活函数为sigmoid，
# 第二隐藏层50个神经元，激活函数为sigmoid，
# 输出层3个神经元，使用softmax处理，损失函数为交叉熵损失；
#（2）训练配置：优化算法使用SGD或Adam，学习超参数自定，采用小批量训练方法，每轮从训练集随机选择100个样本训练，总轮次数为1000轮；

import pickle
import numpy as np

print("正在加载数据集...")
with open('../../data/data_NN.pkl', 'rb') as f:
    dataset = pickle.load(f)

X_train = dataset['x_train'] # 训练集特征 (N, 784)
t_train = dataset['t_train'] # 训练集真实标签 (N,)

# 将真实的标签转为 One-Hot 独热编码，尺寸变为 (N, 3)
Y_train = np.eye(3)[t_train] 
print(f"数据加载成功！训练样本数: {X_train.shape[0]}")


epochs = 1000
batch_size = 100
learning_rate = 0.5
input_dim = X_train.shape[1] # 784

#假设输入为input_dim,对于单个神经元，第一层30个神经元，所以单个样本的尺寸为30，而由于我们要求小批量学习
#对于第一层100样本为一批次则输入尺寸为（100，input_dim），输出尺寸为（100，30）
#得权重矩阵W为（input_dim，30）
#每一个神经元只需要一个固定的偏置值,所以偏置值b的维度为（1，30）
w1 = np.random.randn(input_dim, 30) * 0.01
b1 = np.zeros((1, 30))

w2 = np.random.randn(30, 50) * 0.01
b2 = np.zeros((1, 50))

w3 = np.random.randn(50, 3) * 0.01
b3 = np.zeros((1, 3))

def sigom(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    # 减去最大值是为了防止计算 exp 时数值溢出报错，不影响数学结果
    x = x - np.max(x, axis=1, keepdims=True) 
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def loss(output, label):
    return -np.sum(np.log(output + 1e-7) * label) / label.shape[0]
    
#我们对损失函数对output求偏导得到output - label
#根据链式求导法则，我们的dW 和 db应该是对output*output对W和b求偏导
for epoch in range(epochs):
    # (a) 随机抽取小批量数据
    batch_indices = np.random.choice(len(X_train), batch_size, replace=False)
    input_batch = X_train[batch_indices] # 尺寸 (100, 784)
    label_batch = Y_train[batch_indices] # 尺寸 (100, 3)

    layer1_z = np.dot(input_batch, w1) + b1
    layer1 = sigom(layer1_z)

    layer2_z = np.dot(layer1, w2) + b2
    layer2 = sigom(layer2_z)

    layer3_z = np.dot(layer2, w3) + b3
    output = softmax(layer3_z)

    Doutput = output - label_batch
    
    dW3 = np.dot(layer2.T, Doutput) / batch_size
    db3 = np.sum(Doutput, axis=0, keepdims=True) / batch_size

    Dlayer2_error = np.dot(Doutput, w3.T) * (layer2 * (1 - layer2))
    dW2 = np.dot(layer1.T, Dlayer2_error) / batch_size
    db2 = np.sum(Dlayer2_error, axis=0, keepdims=True) / batch_size

    Dlayer1_error = np.dot(Dlayer2_error, w2.T) * (layer1 * (1 - layer1))
    dW1 = np.dot(input_batch.T, Dlayer1_error) / batch_size
    db1 = np.sum(Dlayer1_error, axis=0, keepdims=True) / batch_size

    w3 = w3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3
    
    w2 = w2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    w1 = w1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    if epoch % 100 == 0:
        loss_val = loss(output, label_batch)
        print(f"第 {epoch:>4} 轮, 当前 Loss 损失值: {loss_val:.4f}")

print(f"第 1000 轮, 最终 Loss 损失值: {loss(output, label_batch):.4f}")