"""
数据集生成服务
为可视化实验室提供经典的2D玩具数据集和高维降维数据集
"""

import numpy as np


def make_moons(n_samples=300, noise=0.15, seed=None):
    """双月牙数据集：非线性二分类的经典数据"""
    rng = np.random.RandomState(seed)
    n = n_samples // 2

    outer_x = np.cos(np.linspace(0, np.pi, n))
    outer_y = np.sin(np.linspace(0, np.pi, n))
    inner_x = 1 - np.cos(np.linspace(0, np.pi, n))
    inner_y = 1 - np.sin(np.linspace(0, np.pi, n)) - 0.5

    X = np.vstack([
        np.column_stack([outer_x, outer_y]),
        np.column_stack([inner_x, inner_y])
    ])
    X += noise * rng.randn(X.shape[0], 2)
    y = np.array([0] * n + [1] * n)
    return X, y


def make_circles(n_samples=300, noise=0.15, seed=None):
    """同心圆数据集：测试非线性分类与谱聚类"""
    rng = np.random.RandomState(seed)
    n = n_samples // 2

    outer = np.linspace(0, 2 * np.pi, n)
    inner = np.linspace(0, 2 * np.pi, n)

    X = np.vstack([
        np.column_stack([np.cos(outer), np.sin(outer)]),
        np.column_stack([np.cos(inner) * 0.4, np.sin(inner) * 0.4])
    ])
    X += noise * rng.randn(X.shape[0], 2)
    y = np.array([0] * n + [1] * n)
    return X, y


def make_blobs(n_samples=300, n_classes=3, noise=0.5, seed=None):
    """高斯团块数据集：聚类与多分类的基础数据"""
    rng = np.random.RandomState(seed)
    n = n_samples // n_classes
    centers = rng.uniform(-3, 3, size=(n_classes, 2))
    X = np.vstack([
        center + noise * rng.randn(n, 2) for center in centers
    ])
    y = np.array([i for i in range(n_classes) for _ in range(n)])
    return X, y


def make_xor(n_samples=300, noise=0.15, seed=None):
    """异或（四象限）数据集：线性分类器的天然克星"""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1.5, 1.5, size=(n_samples, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    mask = rng.rand(n_samples) > (noise * 0.5)
    # noise 参数体现为拉近象限间距离的扰动
    X += noise * 0.3 * rng.randn(n_samples, 2)
    return X[mask], y[mask]


def make_spiral(n_samples=300, noise=0.15, seed=None):
    """双螺旋数据集：最难的2D分类任务之一"""
    rng = np.random.RandomState(seed)
    n = n_samples // 2
    X = []
    y = []

    for class_id in range(2):
        r = np.linspace(0.2, 1.5, n)
        t = 2.5 * np.pi * r + class_id * np.pi
        x1 = r * np.sin(t) + noise * rng.randn(n)
        x2 = r * np.cos(t) + noise * rng.randn(n)
        X.append(np.column_stack([x1, x2]))
        y.extend([class_id] * n)

    return np.vstack(X), np.array(y)


def make_regression(kind='sin', n_samples=120, noise=0.2, seed=None):
    """一维回归数据集：线性或正弦波形"""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3, 3, size=(n_samples, 1))
    if kind == 'linear':
        y = 2.5 * X[:, 0] - 1.0
    else:
        y = np.sin(1.5 * X[:, 0]) + 0.5 * X[:, 0]
    y += noise * rng.randn(n_samples)
    return X, y


def make_highdim(n_samples=300, n_features=10, n_classes=3, seed=None):
    """高维高斯数据集：用于降维算法演示"""
    rng = np.random.RandomState(seed)
    n = n_samples // n_classes
    centers = rng.uniform(-4, 4, size=(n_classes, n_features))
    X = np.vstack([
        center + rng.randn(n, n_features) for center in centers
    ])
    y = np.array([i for i in range(n_classes) for _ in range(n)])
    return X, y


# 数据集注册表：供前端下拉选择
CLASSIFICATION_DATASETS = {
    'moons': {'name': '双月牙', 'generator': make_moons},
    'circles': {'name': '同心圆', 'generator': make_circles},
    'xor': {'name': '异或象限', 'generator': make_xor},
    'spiral': {'name': '双螺旋', 'generator': make_spiral},
    'blobs2': {'name': '双高斯团', 'generator': make_blobs},
}

CLUSTERING_DATASETS = {
    'blobs': {'name': '高斯团块', 'generator': make_blobs},
    'moons': {'name': '双月牙', 'generator': make_moons},
    'circles': {'name': '同心圆', 'generator': make_circles},
}


def previews(task, n_samples=150, seed=7):
    """批量生成某任务下所有数据集的预览数据（供前端缩略图）"""
    registry = {
        'classification': CLASSIFICATION_DATASETS,
        'clustering': CLUSTERING_DATASETS,
        'regression': {'sin': None, 'linear': None},
        'dim_reduction': {'highdim': None},
    }.get(task, {})

    out = {}
    for ds_id in registry:
        try:
            X, y, _, _ = generate(ds_id, task, n_samples, 0.15, 3, seed)
            out[ds_id] = {'X': X, 'y': y}
        except ValueError:
            continue
    return out


def generate(dataset_id, task, n_samples, noise, n_classes=3, seed=None):
    """统一的数据集生成入口"""
    if task == 'classification':
        spec = CLASSIFICATION_DATASETS.get(dataset_id)
        if spec is None:
            raise ValueError(f"未知数据集: {dataset_id}")
        X, y = spec['generator'](n_samples=n_samples, noise=noise, seed=seed)
    elif task == 'clustering':
        spec = CLUSTERING_DATASETS.get(dataset_id)
        if spec is None:
            raise ValueError(f"未知数据集: {dataset_id}")
        if dataset_id == 'blobs':
            X, y = make_blobs(n_samples=n_samples, n_classes=n_classes,
                              noise=noise, seed=seed)
        else:
            X, y = spec['generator'](n_samples=n_samples, noise=noise, seed=seed)
    elif task == 'regression':
        X, y = make_regression(kind=dataset_id, n_samples=n_samples,
                               noise=noise, seed=seed)
    elif task == 'dim_reduction':
        X, y = make_highdim(n_samples=n_samples, n_features=10,
                            n_classes=n_classes, seed=seed)
    else:
        raise ValueError(f"未知任务: {task}")

    return X.tolist(), y.tolist(), X, y
