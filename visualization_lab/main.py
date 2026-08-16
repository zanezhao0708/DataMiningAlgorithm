"""
ML算法可视化实验室 - FastAPI后端
提供算法目录、数据集生成、训练与预测API
"""

import io
import os
import sys
import contextlib

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from services.datasets import generate, previews
from services.algorithms import get_catalog, run
from services.theory import THEORY

app = FastAPI(title="ML算法可视化实验室")


@app.middleware("http")
async def no_cache(request, call_next):
    """静态资源与API禁用缓存，保证部署更新后用户立即拿到新版本"""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


class DatasetRequest(BaseModel):
    task: str
    dataset: str = 'moons'
    n_samples: int = 300
    noise: float = 0.15
    n_classes: int = 3
    seed: Optional[int] = 42


class RunRequest(BaseModel):
    task: str
    algorithm: str
    params: dict = {}
    X: list
    y: Optional[list] = None
    test_ratio: float = 0.2


class CompareItem(BaseModel):
    id: str
    params: dict = {}


class CompareRequest(BaseModel):
    task: str = 'classification'
    algorithms: list  # [{id, params?}]
    X: list
    y: Optional[list] = None
    test_ratio: float = 0.2


@app.get("/api/algorithms")
def api_catalog():
    """返回全部算法目录"""
    return get_catalog()


@app.get("/api/previews/{task}")
def api_previews(task: str):
    """返回某任务下所有数据集的预览数据（缩略图用）"""
    try:
        return previews(task)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/dataset")
def api_dataset(req: DatasetRequest):
    """生成数据集"""
    try:
        X_json, y_json, X, y = generate(req.dataset, req.task, req.n_samples,
                                        req.noise, req.n_classes, req.seed)
        return {'X': X_json, 'y': y_json}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/source/{algorithm}")
def api_source(algorithm: str):
    """返回算法实现的Python源码（供前端源码面板展示）"""
    import inspect
    from services.algorithms import RUNNERS
    for registry, _ in RUNNERS.values():
        if algorithm in registry:
            spec = registry[algorithm]
            defaults = {p['key']: p['default'] for p in spec['params']}
            cls = type(spec['build'](defaults))
            path = inspect.getfile(cls)
            # 相对仓库根目录（visualization_lab 的上一级），便于拼接 GitHub 链接
            rel = os.path.relpath(path, os.path.join(os.path.dirname(__file__), '..'))
            return {'path': rel.replace(os.sep, '/'), 'source': inspect.getsource(cls)}
    raise HTTPException(status_code=404, detail=f"未知算法: {algorithm}")


@app.get("/api/theory/{algorithm}")
def api_theory(algorithm: str):
    """返回算法的理论说明（原理 + 为什么 work）"""
    md = THEORY.get(algorithm)
    if md is None:
        raise HTTPException(status_code=404, detail=f"未知算法: {algorithm}")
    return {'algorithm': algorithm, 'markdown': md}


@app.post("/api/run")
def api_run(req: RunRequest):
    """训练算法并返回可视化数据"""
    try:
        # 吞掉算法内部print，避免污染日志
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(req.task, req.algorithm, req.params, req.X, req.y,
                         test_ratio=req.test_ratio)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/api/compare")
def api_compare(req: CompareRequest):
    """同一数据集上并排运行多个算法（对比模式，只取最终边界，不生成动画帧）"""
    import time
    results = []
    for item in req.algorithms[:4]:
        aid = item.get('id') if isinstance(item, dict) else item
        params = item.get('params') or {} if isinstance(item, dict) else {}
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r = run(req.task, aid, params, req.X, req.y,
                        test_ratio=req.test_ratio, frames=False)
            r['id'] = aid
            r['elapsed_ms'] = round((time.perf_counter() - t0) * 1000)
            results.append(r)
        except Exception as e:
            results.append({'id': aid, 'error': f"{type(e).__name__}: {e}"})
    return {'results': results}


# 静态前端（挂载在最后，避免覆盖API路由）
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == '__main__':
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # 云平台（Render等）通过 PORT 注入端口
    print("ML算法可视化实验室启动中...")
    print(f"访问 http://localhost:{port} 打开实验室")
    uvicorn.run(app, host="0.0.0.0", port=port)
