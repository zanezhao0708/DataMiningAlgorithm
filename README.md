# 数据挖掘算法实现集合

本项目包含多种数据挖掘和机器学习算法的 Python 实现，涵盖神经网络、传统机器学习算法，并附带一个交互式的 [ML算法可视化实验室](#ml算法可视化实验室)。

## 项目结构

```
.
├── data/                           # 数据文件目录
│   ├── data_NN.pkl                 # 训练数据集
│   └── my_dataset.csv              # CSV格式数据
├── visualization_lab/              # ML算法可视化实验室（Web应用）
│   ├── main.py                     # FastAPI后端
│   ├── services/                   # 数据集生成与算法运行服务
│   └── static/                     # 前端页面（Canvas可视化/决策树SVG）
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
│   │   ├── logistic_regression.py  # 逻辑回归
│   │   ├── svm.py                  # 支持向量机
│   │   ├── random_forest.py        # 随机森林
│   │   ├── pca.py                  # 主成分分析
│   │   ├── dbscan.py               # DBSCAN聚类
│   │   ├── adaboost.py             # AdaBoost集成学习
│   │   ├── perceptron.py           # 感知机
│   │   ├── ridge_regression.py     # 岭回归
│   │   ├── lasso_regression.py     # Lasso回归
│   │   ├── hierarchical_clustering.py  # 层次聚类
│   │   ├── gmm.py                  # 高斯混合模型
│   │   ├── apriori.py              # Apriori关联规则
│   │   ├── polynomial_regression.py    # 多项式回归
│   │   ├── kmedoids.py             # K-Medoids聚类
│   │   ├── hmm.py                  # 隐马尔可夫模型
│   │   ├── pagerank.py             # PageRank算法
│   │   ├── svd.py                  # 奇异值分解
│   │   ├── fpgrowth.py             # FP-Growth关联规则
│   │   ├── gradient_boosting.py    # 梯度提升决策树
│   │   ├── lda.py                  # 线性判别分析
│   │   ├── softmax_regression.py   # Softmax回归
│   │   ├── collaborative_filtering.py  # 协同过滤推荐
│   │   ├── spectral_clustering.py  # 谱聚类
│   │   ├── tsne.py                 # t-SNE降维
│   │   └── mlp.py                  # 多层感知机神经网络
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

#### 7. 支持向量机 (svm.py)
寻找最优超平面进行分类，适用于高维数据。

**算法特点：**
- 支持线性核函数
- 通过正则化参数控制模型复杂度
- 提供决策函数值

**使用示例：**
```python
from src.traditional_ml import SVM

model = SVM(learning_rate=0.001, lambda_param=0.01)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 8. 随机森林 (random_forest.py)
集成多个决策树，通过投票机制进行分类。

**算法特点：**
- 使用Bootstrap采样和特征随机选择
- 支持多棵树集成
- 提供准确率评估

**使用示例：**
```python
from src.traditional_ml import RandomForest

model = RandomForest(n_trees=10, max_depth=8)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 9. 主成分分析 (pca.py)
降维算法，通过线性变换保留主要信息。

**算法特点：**
- 支持自定义主成分数量
- 输出方差贡献率
- 支持数据还原

**使用示例：**
```python
from src.traditional_ml import PCA

model = PCA(n_components=3)
X_transformed = model.fit_transform(X_train)
print(f"累计方差贡献率: {np.sum(model.explained_variance_ratio):.4f}")
```

#### 10. DBSCAN聚类 (dbscan.py)
基于密度的聚类算法，能够发现任意形状的簇。

**算法特点：**
- 不需要指定簇数量
- 自动识别噪声点
- 支持任意形状的簇

**使用示例：**
```python
from src.traditional_ml import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
labels = model.fit_predict(X_train)
```

#### 11. AdaBoost集成学习 (adaboost.py)
通过迭代训练弱分类器并组合成强分类器。

**算法特点：**
- 使用决策树桩作为弱分类器
- 自动调整样本权重
- 提供分类器权重

**使用示例：**
```python
from src.traditional_ml import AdaBoost

model = AdaBoost(n_estimators=50)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 12. 感知机 (perceptron.py)
最简单的神经网络模型，用于二分类问题。

**算法特点：**
- 在线学习算法
- 收敛性保证（线性可分数据）
- 计算简单快速

**使用示例：**
```python
from src.traditional_ml import Perceptron

model = Perceptron(learning_rate=0.01)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 13. 岭回归 (ridge_regression.py)
带L2正则化的线性回归，防止过拟合。

**算法特点：**
- 支持梯度下降和闭式解两种方法
- L2正则化防止过拟合
- 输出权重系数和截距

**使用示例：**
```python
from src.traditional_ml import RidgeRegression

model = RidgeRegression(alpha=1.0, method='closed_form')
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 14. Lasso回归 (lasso_regression.py)
带L1正则化的线性回归，可以实现特征选择。

**算法特点：**
- L1正则化实现特征选择
- 输出特征重要性
- 稀疏解

**使用示例：**
```python
from src.traditional_ml import LassoRegression

model = LassoRegression(alpha=0.1)
model.fit(X_train, y_train)
importance = model.get_feature_importance(feature_names)
```

#### 15. 层次聚类 (hierarchical_clustering.py)
通过构建层次树结构进行聚类。

**算法特点：**
- 支持单链接、全链接和平均链接
- 不需要预设簇数量（可使用树状图）
- 提供合并历史

**使用示例：**
```python
from src.traditional_ml import HierarchicalClustering

model = HierarchicalClustering(n_clusters=3, linkage='average')
labels = model.fit_predict(X_train)
```

#### 16. 高斯混合模型 (gmm.py)
概率聚类模型，假设数据由多个高斯分布混合生成。

**算法特点：**
- 使用EM算法训练
- 输出概率估计
- 软聚类（样本可以属于多个簇）

**使用示例：**
```python
from src.traditional_ml import GaussianMixtureModel

model = GaussianMixtureModel(n_components=3)
labels = model.fit_predict(X_train)
probabilities = model.predict_proba(X_test)
```

#### 17. Apriori关联规则 (apriori.py)
通过频繁项集挖掘发现数据项之间的关联关系。

**算法特点：**
- 挖掘频繁项集
- 生成关联规则
- 输出支持度和置信度

**使用示例：**
```python
from src.traditional_ml import Apriori

model = Apriori(min_support=0.3, min_confidence=0.6)
model.fit(transactions)
top_rules = model.get_top_rules(n=10)
```

#### 18. 多项式回归 (polynomial_regression.py)
通过添加多项式特征拟合非线性关系。

**算法特点：**
- 支持自定义多项式阶数
- 自动生成多项式特征
- 适合非线性数据

**使用示例：**
```python
from src.traditional_ml import PolynomialRegression

model = PolynomialRegression(degree=2)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 19. K-Medoids聚类 (kmedoids.py)
与K-Means类似的聚类算法，但使用实际数据点作为聚类中心，对离群点更鲁棒。

**算法特点：**
- 使用实际数据点作为中心（medoid）
- 对离群点和噪声更鲁棒
- 支持任意距离度量

**使用示例：**
```python
from src.traditional_ml import KMedoids

model = KMedoids(n_clusters=3)
labels = model.fit_predict(X_train)
```

#### 20. 隐马尔可夫模型 (hmm.py)
用于处理序列数据的统计模型，包含隐藏状态和观测状态。

**算法特点：**
- 使用Baum-Welch算法训练
- 使用Viterbi算法解码
- 适用于时序数据建模

**使用示例：**
```python
from src.traditional_ml import HMM

model = HMM(n_states=2, n_observations=3)
model.fit(observations, n_iterations=50)
hidden_states = model.predict(observations)
```

#### 21. PageRank (pagerank.py)
基于链接分析的网页重要性排序算法，Google搜索引擎的核心算法。

**算法特点：**
- 基于链接分析
- 支持阻尼系数调整
- 输出节点重要性排名

**使用示例：**
```python
from src.traditional_ml import PageRank

model = PageRank(damping_factor=0.85)
model.fit(adjacency_matrix)
ranked = model.get_ranked_nodes(node_names)
```

#### 22. 奇异值分解 (svd.py)
矩阵分解技术，用于降维、数据压缩和推荐系统。

**算法特点：**
- 矩阵分解
- 支持降维和数据压缩
- 提供方差贡献率

**使用示例：**
```python
from src.traditional_ml import SVD

model = SVD(n_components=10)
X_transformed = model.fit_transform(X)
X_reconstructed = model.reconstruct()
```

#### 23. FP-Growth关联规则 (fpgrowth.py)
比Apriori更高效的频繁项集挖掘算法，使用FP树结构。

**算法特点：**
- 无需生成候选项集
- 使用FP树压缩数据
- 比Apriori效率更高

**使用示例：**
```python
from src.traditional_ml import FPGrowth

model = FPGrowth(min_support=0.3, min_confidence=0.6)
model.fit(transactions)
rules = model.get_top_rules(n=10)
```

#### 24. 梯度提升决策树 (gradient_boosting.py)
通过迭代训练回归树拟合残差的集成学习方法。

**算法特点：**
- 迭代拟合残差
- 支持学习率调整
- 强大的回归能力

**使用示例：**
```python
from src.traditional_ml import GradientBoosting

model = GradientBoosting(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 25. 线性判别分析 (lda.py)
有监督降维算法，最大化类间差异同时最小化类内差异。

**算法特点：**
- 有监督降维
- 最大化类别可分性
- 适用于分类问题

**使用示例：**
```python
from src.traditional_ml import LDA

model = LDA(n_components=2)
X_transformed = model.fit_transform(X_train, y_train)
```

#### 26. Softmax回归 (softmax_regression.py)
多分类逻辑回归，使用softmax函数输出类别概率。

**算法特点：**
- 支持多分类
- 输出概率分布
- 支持正则化

**使用示例：**
```python
from src.traditional_ml import SoftmaxRegression

model = SoftmaxRegression(learning_rate=0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

#### 27. 协同过滤 (collaborative_filtering.py)
基于用户或物品相似度的推荐系统算法。

**算法特点：**
- 支持基于用户和基于物品
- 余弦相似度度量
- 生成个性化推荐

**使用示例：**
```python
from src.traditional_ml import CollaborativeFiltering

model = CollaborativeFiltering(method='user_based', k=5)
model.fit(ratings_matrix)
recommendations = model.recommend(user_id=0)
```

#### 28. 谱聚类 (spectral_clustering.py)
基于图论和谱图理论的聚类算法，能处理任意形状的簇。

**算法特点：**
- 基于图论
- 支持RBF和KNN相似度
- 能发现任意形状的簇

**使用示例：**
```python
from src.traditional_ml import SpectralClustering

model = SpectralClustering(n_clusters=2, affinity='rbf')
labels = model.fit_predict(X_train)
```

#### 29. t-SNE降维 (tsne.py)
当下最流行的非线性降维可视化算法，通过保持高维邻居结构将数据映射到2D/3D。

**算法特点：**
- 二分搜索困惑度匹配的邻域宽度
- t分布核 + KL散度梯度下降（带动量与早期放大）
- 适合高维数据的可视化探索

**使用示例：**
```python
from src.traditional_ml import TSNE

model = TSNE(n_components=2, perplexity=20, n_iter=500, random_state=42)
X_embedded = model.fit_transform(X)
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

# 支持向量机
cd src/traditional_ml
python svm.py

# 随机森林
cd src/traditional_ml
python random_forest.py

# 主成分分析
cd src/traditional_ml
python pca.py

# DBSCAN聚类
cd src/traditional_ml
python dbscan.py

# AdaBoost集成学习
cd src/traditional_ml
python adaboost.py

# 感知机
cd src/traditional_ml
python perceptron.py

# 岭回归
cd src/traditional_ml
python ridge_regression.py

# Lasso回归
cd src/traditional_ml
python lasso_regression.py

# 层次聚类
cd src/traditional_ml
python hierarchical_clustering.py

# 高斯混合模型
cd src/traditional_ml
python gmm.py

# Apriori关联规则
cd src/traditional_ml
python apriori.py

# 多项式回归
cd src/traditional_ml
python polynomial_regression.py

# K-Medoids聚类
cd src/traditional_ml
python kmedoids.py

# 隐马尔可夫模型
cd src/traditional_ml
python hmm.py

# PageRank算法
cd src/traditional_ml
python pagerank.py

# 奇异值分解
cd src/traditional_ml
python svd.py

# FP-Growth关联规则
cd src/traditional_ml
python fpgrowth.py

# 梯度提升决策树
cd src/traditional_ml
python gradient_boosting.py

# 线性判别分析
cd src/traditional_ml
python lda.py

# Softmax回归
cd src/traditional_ml
python softmax_regression.py

# 协同过滤推荐
cd src/traditional_ml
python collaborative_filtering.py

# 谱聚类
cd src/traditional_ml
python spectral_clustering.py
```

运行数据加载工具：
```bash
cd src/utils
python data_loader.py
```

## 技术特点

1. **纯 NumPy 实现**：所有算法均使用 NumPy 手动实现，便于理解底层原理
2. **模块化设计**：清晰的代码结构，便于扩展和维护
3. **详细注释**：代码包含详细的中文注释，易于学习理解
4. **完整训练流程**：包含模型训练、预测、评估等完整流程
5. **算法多样性**：涵盖监督学习、无监督学习、降维、关联规则等多种类型

## 数据格式

训练数据采用 PKL 格式存储，包含以下字段：
- `x_train`：训练集特征，形状为 (N, 784)
- `t_train`：训练集标签，形状为 (N,)

## ML算法可视化实验室

交互式算法可视化 Web 应用，将本仓库 23 个算法变成"左边调参数、右边看效果"的实验台。

```bash
pip install numpy fastapi uvicorn
cd visualization_lab && python main.py
# 浏览器打开 http://localhost:8000
```

**功能一览：**

| 任务 | 能力 |
|------|------|
| 分类 | 10 种分类器实时绘制决策边界，树模型附带树结构 SVG，MLP 支持逐 epoch 训练动画与损失曲线 |
| 聚类 | 6 种聚类算法，K-Means 支持迭代动画回放（质心轨迹） |
| 回归 | 4 种回归算法拟合曲线对比，实时 R² |
| 降维 | PCA / LDA / SVD / t-SNE 二维投影 |

其他特性：训练/测试集划分（空心点标记测试样本）、数据集缩略图一键切换、TensorFlow Playground 风格 UI。

详见 [visualization_lab/README.md](visualization_lab/README.md)。

## 许可证

本项目仅供学习和研究使用。

## 作者

zanezhao0708

## 更新日志

- **2026-08-14(三)**: 可视化实验室大升级——新增 MLP 神经网络（逐epoch训练动画+损失曲线）、训练/测试集划分、数据集缩略图、TF Playground 风格 UI；KNN 预测向量化提速 33 倍
- **2026-08-14(二)**: 新增 t-SNE 降维算法与 ML算法可视化实验室（FastAPI Web 应用，23 个算法交互式可视化）；修复 GMM 协方差更新、多项式回归数值发散、SVD 方差贡献率等 bug
- **2026-08-14**: 新增10个算法（K-Medoids、HMM、PageRank、SVD、FP-Growth、梯度提升、LDA、Softmax回归、协同过滤、谱聚类）
- **2024-08-13**: 重构项目结构，补齐MNIST分类器完整训练代码
- **初始版本**: 实现基础三层神经网络