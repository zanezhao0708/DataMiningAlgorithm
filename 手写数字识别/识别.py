import numpy as np
import PIL
import matplotlib
import os
"""
MLP整体结构
input:28*28=784
hiden1: 1000
hiden2: 400
hiden3: 50
output: 10
"""

#根据计算可知进行初始化： 
#w1(784,1000),b1(1,1000)
#w2(1000,400),b2(1,400)
#w3(400,50),b3(1,50)
#w4(50,10),b4(1,10)


#*0.01防止进入激活函数因为输入过大而进入饱和区
w1 = np.random.randn(784,1000)*0.01
b1 = np.zeros((1,1000))

w2 = np.random.randn(1000,400)*0.01
b2 = np.zeros((1,400))

w3 = np.random.randn(400,50)*0.01
b3 = np.zeros((1,50))

w4 = np.random.randn(50,10)*0.01
b4 = np.zeros((1,10))

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    x = x - np.max(x, axis=1 , keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=1 , keepdims =True)

def loss(pred,true):
    batchsize = pred.shape[0]
    prob = np.sum(pred * true , axis=1)
    key = -np.log(prob + 1e-7)
    
    return np.sum(key)/batchsize