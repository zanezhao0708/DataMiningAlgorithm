
#实现一个三层前馈神经网络，完成3分类任务。要求如下：
#（1）网络结构：第一隐藏层30个神经元，激活函数为sigmoid，
# 第二隐藏层50个神经元，激活函数为sigmoid，
# 输出层3个神经元，使用softmax处理，损失函数为交叉熵损失；
#（2）训练配置：优化算法使用SGD或Adam，学习超参数自定，采用小批量训练方法，每轮从训练集随机选择100个样本训练，总轮次数为1000轮；

import torch
import numpy as np

learning_rate = 0.1

#假设输入为input_dim,对于单个神经元，第一层30个神经元，所以单个样本的尺寸为30，而由于我们要求小批量学习
#对于第一层100样本为一批次则输入尺寸为（100，input_dim），输出尺寸为（100，30）
#得权重矩阵W为（input_dim，30）
#每一个神经元只需要一个固定的偏置值,所以偏置值b的维度为（1，30）


#对于第二层权重W的尺寸为（30，50），偏置值b的尺寸为（1，50）
#对于第三层权重W的尺寸为（50，3），偏置值b的尺寸为（1，3）

def sigom(x):
    return 1/1+np.exp(-x)
def softmax(x):
    x = np.exp(x)
    sum = np.sum(x,axis=1,keepdims=True)
    return x / sum

def loss(output,label):
    return -np.sum(np.log(output+ 1e-7)  * label)/label.shape[0]
    



layer1 = np.dot(input , w1) + b1
layer1 = sigom(layer1)

layer2 = np.dot(layer1 , w2) + b2
layer2 = sigom(layer2)

layer3 = np.dot(layer2 , w3) + b3
output = softmax(layer3)


#新的权重 = 老的权重 - 学习率 * 梯度
w3 = w3 - learning_rate * dw3