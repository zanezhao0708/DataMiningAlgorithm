"""
ML算法可视化实验室 - FastAPI后端
提供算法目录、数据集生成、训练与预测API
"""

import io
import sys
import contextlib

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from services.datasets import generate
from services.algorithms import get_catalog, run

app = FastAPI(title="ML算法可视化实验室")


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


@app.get("/api/algorithms")
def api_catalog():
    """返回全部算法目录"""
    return get_catalog()


@app.post("/api/dataset")
def api_dataset(req: DatasetRequest):
    """生成数据集"""
    try:
        X_json, y_json, X, y = generate(req.dataset, req.task, req.n_samples,
                                        req.noise, req.n_classes, req.seed)
        return {'X': X_json, 'y': y_json}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/run")
def api_run(req: RunRequest):
    """训练算法并返回可视化数据"""
    try:
        # 吞掉算法内部print，避免污染日志
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(req.task, req.algorithm, req.params, req.X, req.y)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


# 静态前端（挂载在最后，避免覆盖API路由）
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == '__main__':
    import uvicorn
    print("ML算法可视化实验室启动中...")
    print("访问 http://localhost:8000 打开实验室")
    uvicorn.run(app, host="0.0.0.0", port=8000)
