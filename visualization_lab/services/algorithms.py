"""
算法注册与运行服务
将 src/traditional_ml 中的算法包装为统一的可视化任务接口
"""

import sys
import os
import numpy as np

# 让服务能导入项目算法库（src/traditional_ml）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from traditional_ml.knn import KNN
from traditional_ml.decision_tree import DecisionTree
from traditional_ml.logistic_regression import LogisticRegression
from traditional_ml.svm import SVM
from traditional_ml.naive_bayes import NaiveBayes
from traditional_ml.perceptron import Perceptron
from traditional_ml.adaboost import AdaBoost
from traditional_ml.random_forest import RandomForest
from traditional_ml.softmax_regression import SoftmaxRegression
from traditional_ml.kmeans import KMeans
from traditional_ml.kmedoids import KMedoids
from traditional_ml.dbscan import DBSCAN
from traditional_ml.gmm import GaussianMixtureModel
from traditional_ml.hierarchical_clustering import HierarchicalClustering
from traditional_ml.spectral_clustering import SpectralClustering
from traditional_ml.linear_regression import LinearRegression
from traditional_ml.polynomial_regression import PolynomialRegression
from traditional_ml.ridge_regression import RidgeRegression
from traditional_ml.lasso_regression import LassoRegression
from traditional_ml.pca import PCA
from traditional_ml.lda import LDA
from traditional_ml.svd import SVD
from traditional_ml.tsne import TSNE
from traditional_ml.mlp import MLP

GRID_SIZE = 50  # 决策边界网格分辨率


def _slider(key, label, lo, hi, default, step=1):
    """构造前端滑块参数的schema"""
    return {'key': key, 'label': label, 'min': lo, 'max': hi,
            'step': step, 'default': default}


# ============ 分类算法 ============
CLASSIFIERS = {
    'knn': {
        'name': 'K近邻 KNN',
        'params': [_slider('k', '邻居数 K', 1, 25, 5)],
        'build': lambda p: KNN(k=p['k']),
    },
    'decision_tree': {
        'name': '决策树',
        'tree': True,
        'params': [_slider('max_depth', '最大深度', 1, 12, 4)],
        'build': lambda p: DecisionTree(max_depth=p['max_depth']),
    },
    'logistic_regression': {
        'name': '逻辑回归',
        'params': [_slider('learning_rate', '学习率', 0.01, 2.0, 0.5, 0.01),
                   _slider('n_iterations', '迭代次数', 50, 3000, 800, 50)],
        'build': lambda p: LogisticRegression(learning_rate=p['learning_rate'],
                                              n_iterations=p['n_iterations']),
    },
    'svm': {
        'name': '支持向量机 SVM',
        'params': [_slider('lambda_param', '正则强度 λ', 0.001, 0.5, 0.01, 0.001),
                   _slider('n_iterations', '迭代次数', 50, 3000, 1000, 50)],
        'build': lambda p: SVM(lambda_param=p['lambda_param'],
                               n_iterations=p['n_iterations']),
    },
    'naive_bayes': {
        'name': '朴素贝叶斯',
        'params': [],
        'build': lambda p: NaiveBayes(),
    },
    'perceptron': {
        'name': '感知机',
        'params': [_slider('learning_rate', '学习率', 0.01, 1.0, 0.1, 0.01),
                   _slider('n_iterations', '迭代次数', 10, 1000, 100, 10)],
        'build': lambda p: Perceptron(learning_rate=p['learning_rate'],
                                      n_iterations=p['n_iterations']),
    },
    'adaboost': {
        'name': 'AdaBoost',
        'params': [_slider('n_estimators', '弱分类器数', 5, 60, 20)],
        'build': lambda p: AdaBoost(n_estimators=p['n_estimators']),
    },
    'random_forest': {
        'name': '随机森林',
        'tree': 'forest',
        'params': [_slider('n_trees', '树数量', 3, 30, 10),
                   _slider('max_depth', '最大深度', 2, 12, 6)],
        'build': lambda p: RandomForest(n_trees=p['n_trees'], max_depth=p['max_depth']),
    },
    'softmax_regression': {
        'name': 'Softmax回归',
        'params': [_slider('learning_rate', '学习率', 0.01, 1.0, 0.1, 0.01),
                   _slider('n_iterations', '迭代次数', 50, 3000, 1000, 50)],
        'build': lambda p: SoftmaxRegression(learning_rate=p['learning_rate'],
                                             n_iterations=p['n_iterations']),
    },
    'mlp': {
        'name': '神经网络 MLP',
        'animated': True,
        'params': [{'key': 'arch', 'label': '网络结构', 'type': 'select',
                    'options': ['单层', '双层'], 'default': '双层'},
                   _slider('hidden', '隐藏层宽度', 2, 16, 8),
                   _slider('learning_rate', '学习率', 0.01, 0.5, 0.05, 0.01),
                   _slider('n_epochs', '训练轮数', 50, 500, 300, 50)],
        'build': lambda p: MLP(
            hidden_layers=(p['hidden'], p['hidden']) if p.get('arch') == '双层' else (p['hidden'],),
            learning_rate=p['learning_rate'],
            n_epochs=p['n_epochs'], random_state=42),
    },
}

# 这些算法要求标签为 ±1（内部按符号判断）
PM1_ALGOS = {'svm', 'perceptron', 'adaboost'}


def _train_test_split(X, y, test_ratio, seed=42):
    """分层随机划分训练集与测试集"""
    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = max(1, int(n * test_ratio))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], test_idx


def run_classification(algorithm, params, X, y, test_ratio=0.2):
    """训练分类器并生成决策边界网格，支持训练/测试划分与MLP动画"""
    spec = CLASSIFIERS[algorithm]
    model = spec['build'](params)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y).tolist()

    # 训练/测试划分
    X_tr, X_te, y_tr, y_te, test_idx = _train_test_split(X, y, test_ratio)

    # ±1 算法：转换标签并在预测后映射回来
    if algorithm in PM1_ALGOS:
        y_fit = np.where(y_tr == classes[0], -1, 1)
    else:
        y_fit = y_tr.astype(int)

    model.fit(X_tr, y_fit)

    # 决策边界网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xs = np.linspace(x_min, x_max, GRID_SIZE)
    ys = np.linspace(y_min, y_max, GRID_SIZE)
    mesh = np.array([[a, b] for b in ys for a in xs])

    def _map(pred):
        pred = pred.ravel().astype(int)
        if algorithm in PM1_ALGOS:
            pred = np.where(pred == -1, 0, 1)
        return pred

    pred = _map(model.predict(mesh))
    train_pred = _map(model.predict(X_tr))
    test_pred = _map(model.predict(X_te))

    result = {
        'grid': pred.tolist(),
        'grid_size': GRID_SIZE,
        'x_range': [float(x_min), float(x_max)],
        'y_range': [float(y_min), float(y_max)],
        'accuracy': float(np.mean(train_pred == y_tr)),
        'test_accuracy': float(np.mean(test_pred == y_te)),
        'test_indices': test_idx.tolist(),
        'n_classes': len(classes),
    }

    # MLP：生成逐epoch训练动画帧（决策边界演化 + 损失曲线）
    if algorithm == 'mlp':
        frames = _mlp_frames(model, mesh, X_tr, y_tr, GRID_SIZE)
        result['frames'] = frames
        result['history'] = model.history

    # 树模型附带树结构（随机森林展示第一棵树）
    if spec.get('tree') == 'forest':
        tree = model.trees[0]['tree'].tree
        result['tree'] = serialize_tree(tree, feature_names=['x₁', 'x₂'])
    elif spec.get('tree'):
        result['tree'] = serialize_tree(model.tree, feature_names=['x₁', 'x₂'])

    return result


def _mlp_frames(model, mesh, X_tr, y_tr, grid_size, max_frames=40):
    """重放MLP训练过程，采样决策边界帧用于动画"""
    # 重新训练并记录中间状态（用相同随机种子保证可复现）
    n_epochs = model.n_epochs
    step = max(1, n_epochs // max_frames)
    frames = []

    # 保存当前权重，从头重放
    np.random.seed(model.random_state)
    n_features = X_tr.shape[1]
    n_classes = len(np.unique(y_tr))
    y_onehot = np.eye(n_classes)[y_tr.astype(int)]
    model._init_params(n_features, n_classes)

    for epoch in range(n_epochs):
        activations, zs = model._forward(X_tr)
        probs = activations[-1]
        loss = model._cross_entropy(probs, y_onehot)
        acc = float(np.mean(np.argmax(probs, axis=1) == y_tr))

        if epoch % step == 0 or epoch == n_epochs - 1:
            grid_pred = np.argmax(model.predict_proba(mesh), axis=1)
            frames.append({
                'epoch': epoch,
                'grid': grid_pred.tolist(),
                'loss': float(loss),
                'accuracy': acc,
            })

        # 一步梯度更新
        delta = (probs - y_onehot) / len(y_tr)
        for i in range(len(model.weights) - 1, -1, -1):
            dW = activations[i].T @ delta
            db = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ model.weights[i].T) * model._activate_grad(zs[i - 1])
            model.weights[i] -= model.learning_rate * dW
            model.biases[i] -= model.learning_rate * db

    return frames


def serialize_tree(node, feature_names):
    """把决策树嵌套dict序列化为前端可渲染的JSON"""
    if node.get('leaf'):
        return {'leaf': True, 'value': int(node['value'])}
    return {
        'leaf': False,
        'feature': feature_names[node['feature_idx']],
        'threshold': round(float(node['threshold']), 3),
        'left': serialize_tree(node['left'], feature_names),
        'right': serialize_tree(node['right'], feature_names),
    }


# ============ 聚类算法 ============
CLUSTERERS = {
    'kmeans': {
        'name': 'K-Means',
        'animated': True,
        'params': [_slider('n_clusters', '聚类数 K', 2, 6, 3)],
        'build': lambda p: KMeans(n_clusters=p['n_clusters'], max_iters=50),
    },
    'kmedoids': {
        'name': 'K-Medoids',
        'params': [_slider('n_clusters', '聚类数 K', 2, 6, 3)],
        'build': lambda p: KMedoids(n_clusters=p['n_clusters'], max_iters=50),
    },
    'dbscan': {
        'name': 'DBSCAN',
        'params': [_slider('eps', '邻域半径 ε', 0.05, 1.2, 0.3, 0.05),
                   _slider('min_samples', '最小样本数', 3, 15, 5)],
        'build': lambda p: DBSCAN(eps=p['eps'], min_samples=p['min_samples']),
    },
    'gmm': {
        'name': '高斯混合 GMM',
        'params': [_slider('n_components', '成分数 K', 2, 6, 3)],
        'build': lambda p: GaussianMixtureModel(n_components=p['n_components'], max_iter=100),
    },
    'hierarchical': {
        'name': '层次聚类',
        'params': [_slider('n_clusters', '聚类数 K', 2, 6, 3),
                   {'key': 'linkage', 'label': '链接方式', 'type': 'select',
                    'options': ['single', 'complete', 'average'], 'default': 'complete'}],
        'build': lambda p: HierarchicalClustering(n_clusters=p['n_clusters'],
                                                  linkage=p['linkage']),
    },
    'spectral': {
        'name': '谱聚类',
        'params': [_slider('n_clusters', '聚类数 K', 2, 6, 2),
                   _slider('gamma', 'RBF γ', 1, 60, 20)],
        'build': lambda p: SpectralClustering(n_clusters=p['n_clusters'],
                                              affinity='rbf', gamma=p['gamma']),
    },
}


def run_clustering(algorithm, params, X, y=None):
    """运行聚类算法，返回标签、中心与K-Means动画帧"""
    spec = CLUSTERERS[algorithm]
    model = spec['build'](params)
    X = np.asarray(X)

    model.fit(X)
    labels = np.asarray(model.labels).astype(int).tolist()

    result = {
        'labels': labels,
        'n_found': len(set(labels)),
        'silhouette': float(silhouette(X, np.asarray(labels))),
    }

    # 中心点
    if hasattr(model, 'centroids') and getattr(model, 'centroids', None) is not None:
        result['centers'] = np.asarray(model.centroids).tolist()
    elif hasattr(model, 'medoids') and getattr(model, 'medoids', None) is not None:
        result['centers'] = np.asarray(model.medoids).tolist()

    # DBSCAN 噪声标记（-1）
    if algorithm == 'dbscan':
        result['noise'] = (np.asarray(labels) == -1).tolist()

    # K-Means 迭代动画帧
    if spec.get('animated') and getattr(model, 'history', None):
        frames = []
        for i, frame in enumerate(model.history):
            f = {'step': i, 'centroids': frame['centroids'].tolist()}
            if 'labels' in frame:
                f['labels'] = frame['labels'].astype(int).tolist()
            frames.append(f)
        result['frames'] = frames

    return result


def silhouette(X, labels):
    """轮廓系数（简化实现），衡量聚类质量，越大越好"""
    if len(set(labels.tolist())) < 2:
        return 0.0
    n = len(labels)
    if n > 400:  # 大数据集抽样以加速
        idx = np.random.choice(n, 400, replace=False)
        X, labels = X[idx], labels[idx]
        n = 400

    dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    unique = set(labels.tolist())
    if len(unique) < 2 or -1 in unique and len(unique) < 3:
        return 0.0

    scores = []
    for i in range(n):
        own = labels == labels[i]
        own[i] = False
        if not own.any():
            scores.append(0)
            continue
        a = dist[i, own].mean()
        b = min(dist[i, labels == c].mean() for c in unique if c != labels[i] and c != -1)
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0)
    return float(np.mean(scores))


# ============ 回归算法 ============
REGRESSORS = {
    'linear_regression': {
        'name': '线性回归',
        'params': [_slider('n_iterations', '迭代次数', 100, 5000, 2000, 100)],
        'build': lambda p: LinearRegression(n_iterations=p['n_iterations'],
                                            method='normal_equation'),
    },
    'polynomial_regression': {
        'name': '多项式回归',
        'params': [_slider('degree', '多项式阶数', 1, 9, 3)],
        'build': lambda p: PolynomialRegression(degree=p['degree']),
    },
    'ridge_regression': {
        'name': '岭回归 (L2)',
        'params': [_slider('alpha', '正则强度 α', 0.0, 10.0, 1.0, 0.1)],
        'build': lambda p: RidgeRegression(alpha=p['alpha'], method='normal_equation'),
    },
    'lasso_regression': {
        'name': 'Lasso回归 (L1)',
        'params': [_slider('alpha', '正则强度 α', 0.01, 5.0, 0.5, 0.01),
                   _slider('n_iterations', '迭代次数', 100, 3000, 1000, 100)],
        'build': lambda p: LassoRegression(alpha=p['alpha'],
                                           n_iterations=p['n_iterations']),
    },
}


def run_regression(algorithm, params, X, y):
    """拟合一维回归并在区间上采样拟合曲线"""
    spec = REGRESSORS[algorithm]
    model = spec['build'](params)
    model.fit(np.asarray(X), np.asarray(y))

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    curve_x = np.linspace(x_min - 0.3, x_max + 0.3, 120).reshape(-1, 1)
    curve_y = model.predict(curve_x)

    pred = model.predict(np.asarray(X)).ravel()
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    return {
        'curve': [[float(a), float(b)] for a, b in zip(curve_x.ravel(), curve_y)],
        'r2': 1 - ss_res / (ss_tot + 1e-12),
    }


# ============ 降维算法 ============
REDUCERS = {
    'pca': {
        'name': '主成分分析 PCA',
        'params': [{'key': 'n_components', 'label': '保留主成分数', 'type': 'select',
                    'options': [2], 'default': 2}],
        'desc': '线性降维，最大化方差保留',
        'build': lambda p: PCA(n_components=2),
    },
    'lda': {
        'name': '线性判别分析 LDA',
        'params': [],
        'desc': '有监督降维，最大化类间分离',
        'build': lambda p: LDA(n_components=2),
    },
    'svd': {
        'name': '奇异值分解 SVD',
        'params': [],
        'desc': '矩阵分解降维',
        'build': lambda p: SVD(n_components=2),
    },
    'tsne': {
        'name': 't-SNE',
        'params': [_slider('perplexity', '困惑度', 5, 50, 20),
                   _slider('n_iter', '迭代次数', 100, 1000, 400, 50)],
        'desc': '非线性降维，保持局部邻居结构',
        'build': lambda p: TSNE(n_components=2, perplexity=p['perplexity'],
                                n_iter=p['n_iter'], random_state=42),
    },
}


def run_dim_reduction(algorithm, params, X, y):
    """降维到2D并返回投影坐标"""
    spec = REDUCERS[algorithm]
    model = spec['build'](params)
    X = np.asarray(X)

    if algorithm == 'lda':
        emb = model.fit_transform(X, np.asarray(y, dtype=int))
    elif algorithm == 'svd':
        emb = model.fit_transform(X)
    else:
        emb = model.fit_transform(X)

    result = {'embedding': np.asarray(emb).tolist()}

    if hasattr(model, 'explained_variance_ratio') and model.explained_variance_ratio is not None:
        result['explained'] = [round(float(v), 4)
                               for v in np.asarray(model.explained_variance_ratio).ravel()]
        result['explained_total'] = float(np.sum(result['explained']))
    if algorithm == 'tsne':
        result['kl'] = float(model.kl_divergence_)

    return result


# 统一的任务分发
RUNNERS = {
    'classification': (CLASSIFIERS, run_classification),
    'clustering': (CLUSTERERS, run_clustering),
    'regression': (REGRESSORS, run_regression),
    'dim_reduction': (REDUCERS, run_dim_reduction),
}


def get_catalog():
    """返回全部算法目录（供前端渲染）"""
    catalog = {}
    for task, (registry, _) in RUNNERS.items():
        catalog[task] = [
            {
                'id': aid,
                'name': spec['name'],
                'params': spec['params'],
                'animated': bool(spec.get('animated')),
                'has_tree': bool(spec.get('tree')),
                'desc': spec.get('desc', ''),
            }
            for aid, spec in registry.items()
        ]
    return catalog


def run(task, algorithm, params, X, y, test_ratio=0.2):
    """统一执行入口"""
    if task not in RUNNERS:
        raise ValueError(f"未知任务: {task}")
    registry, runner = RUNNERS[task]
    if algorithm not in registry:
        raise ValueError(f"未知算法: {algorithm}")
    if task == 'classification':
        return runner(algorithm, params, np.asarray(X, dtype=float),
                      np.asarray(y, dtype=float), test_ratio=test_ratio)
    return runner(algorithm, params, np.asarray(X, dtype=float),
                  np.asarray(y, dtype=float))
