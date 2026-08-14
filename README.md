# 数据挖掘算法实现集合

本项目包含多种数据挖掘和机器学习算法的 Python 实现，涵盖神经网络、传统机器学习算法等。

## 项目结构

```
.
├── data/                           # 数据文件目录
│   ├── data_NN.pkl                 # 训练数据集
│   └── my_dataset.csv              # CSV格式数据
├── src/                            # 源代码目录
│   ├── neural_networks/            # 神经网络实现
│   │   ├── basic_nn.py             # 基础三层神经网络
│   │   └── mnist_classifier.py     # MNIST手写数字识别
│   ├── traditional_ml/             # 传统机器学习算法
│   │   ├── knn.py                  # K近邻算法
│   │   ├── decision_tree.py        # 决策树
│   │   ├── naive_bayes.py          # 朴素贝叶斯
│   │   ├── kmeans.py               # K-Means聚类
│   │   ├── linear_regression.py    # 线性回归
│   │   └── logistic_regression.py  # 逻辑回归
│   └── utils/                      # 工具函数
│       └── data_loader.py          # 数据加载与处理工具
├── requirements.txt                # Python依赖列表
└── README.md                       # 项目说明文档
```

## 功能模块

### 一、神经网络模块 (neural_networks/)

#### 1. 基础神经网络 (basic_nn.py)

实现一个三层前馈神经网络，完成3分类任务。

**网络结构：**
- 输入层：784个神经元 (28×28图像)
- 隐藏层1：30个神经元，Sigmoid激活函数
- 隐藏层2：50个神经元，Sigmoid激活函数
- 输出层：3个神经元，Softmax激活函数

**训练配置：**
- 优化算法：SGD (随机梯度下降)
- 学习率：0.5
- 批量大小：100
- 训练轮数：1000
- 损失函数：交叉熵损失

#### 2. MNIST手写数字识别 (mnist_classifier.py)

实现一个四层神经网络用于手写数字识别。

**网络结构：**
- 输入层：784个神经元 (28×28图像)
- 隐藏层1：1000个神经元，ReLU激活函数
- 隐藏层2：400个神经元，ReLU激活函数
- 隐藏层3：50个神经元，ReLU激活函数
- 输出层：10个神经元 (0-9数字分类)，Softmax激活函数

**训练配置：**
- 优化算法：SGD
- 学习率：0.01
- 批量大小：100
- 训练轮数：100
- 损失函数：交叉熵损失

**特性：**
- 完整的反向传播实现
- 训练过程可视化
- 准确率评估功能

### 二、传统机器学习模块 (traditional_ml/)

#### 1. K近邻算法 (knn.py)
基于实例的学习算法，通过计算距离找到最近的K个邻居进行分类或回归。

**算法特点：**
- 距离度量：欧氏距离
- 支持分类和回归任务
- 参数：邻居数量k

**使用示例：**
```python
from src.traditional_ml import KNN

model = KNN(k=5, task_type='classification')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 2. 决策树 (decision_tree.py)
基于树结构的分类算法，通过递归地选择最优特征进行分裂。

**算法特点：**
- 分裂准则：基尼系数
- 支持最大深度、最小样本数等参数
- 易于理解和可视化

**使用示例：**
```python
from src.traditional_ml import DecisionTree

model = DecisionTree(max_depth=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 3. 朴素贝叶斯 (naive_bayes.py)
基于贝叶斯定理的概率分类器，假设特征条件独立。

**算法特点：**
- 使用高斯分布建模
- 计算速度快，适合小数据集
- 输出概率估计

**使用示例：**
```python
from src.traditional_ml import NaiveBayes

model = NaiveBayes()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 4. K-Means聚类 (kmeans.py)
无监督学习算法，通过迭代更新聚类中心将数据分为K个簇。

**算法特点：**
- 支持自定义聚类数量
- 提供收敛判断
- 支持聚类可视化（2D数据）

**使用示例：**
```python
from src.traditional_ml import KMeans

model = KMeans(n_clusters=3)
model.fit(X_train)
labels = model.predict(X_test)
```

#### 5. 线性回归 (linear_regression.py)
回归算法，拟合特征与目标值之间的线性关系。

**算法特点：**
- 支持梯度下降和正规方程两种求解方法
- 输出权重系数和截距
- 提供R²评分

**使用示例：**
```python
from src.traditional_ml import LinearRegression

model = LinearRegression(method='normal_equation')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 6. 逻辑回归 (logistic_regression.py)
二分类算法，使用sigmoid函数将线性组合映射到[0,1]区间。

**算法特点：**
- 输出概率估计
- 提供精确率、召回率、F1分数等评估指标
- 支持自定义分类阈值

**使用示例：**
```python
from src.traditional_ml import LogisticRegression

model = LogisticRegression(learning_rate=0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 三、数据加载工具 (utils/)

#### 数据加载工具 (data_loader.py)

提供数据加载和转换功能：
- 读取 PKL 格式数据文件
- 数据格式转换 (PKL → CSV)
- 数据集信息展示

## 安装与运行

### 环境要求

- Python 3.6+
- NumPy
- Pandas (可选，用于数据处理)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

运行基础神经网络：
```bash
cd src/neural_networks
python basic_nn.py
```

运行MNIST分类器：
```bash
cd src/neural_networks
python mnist_classifier.py
```

运行传统机器学习算法：
```bash
# K近邻算法
cd src/traditional_ml
python knn.py

# 决策树
cd src/traditional_ml
python decision_tree.py

# 朴素贝叶斯
cd src/traditional_ml
python naive_bayes.py

# K-Means聚类
cd src/traditional_ml
python kmeans.py

# 线性回归
cd src/traditional_ml
python linear_regression.py

# 逻辑回归
cd src/traditional_ml
python logistic_regression.py
```

运行数据加载工具：
```bash
cd src/utils
python data_loader.py
```

## 技术特点

1. **纯 NumPy 实现**：所有神经网络模型均使用 NumPy 手动实现，便于理解底层原理
2. **模块化设计**：清晰的代码结构，便于扩展和维护
3. **详细注释**：代码包含详细的中文注释，易于学习理解
4. **完整训练流程**：包含前向传播、反向传播、梯度更新等完整训练过程

## 数据格式

训练数据采用 PKL 格式存储，包含以下字段：
- `x_train`：训练集特征，形状为 (N, 784)
- `t_train`：训练集标签，形状为 (N,)

## 许可证

本项目仅供学习和研究使用。

## 作者

zanezhao0708

## 更新日志

- **2024-08-13**: 重构项目结构，补齐MNIST分类器完整训练代码
- **初始版本**: 实现基础三层神经网络