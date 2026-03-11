import random
import threading
import time
from typing import Dict, List
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
from contextlib import asynccontextmanager
app = FastAPI(
    title="Task Manager API",
    description="Task distribution and statistics service",
    version="1.0.0"
)
# 线程锁（FastAPI 异步环境下仍建议使用锁保护共享状态）
lock = threading.Lock()
# 全局状态
task_stats: Dict[str, Dict] = {}
initialized = False
# 日志文件路径（建议后续改为使用 pathlib 或环境变量）
LOG_FILE = "/HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL/ROLL/roll/pipeline/agentic/env/android/TaskManager.json"
def load_task_stats():
    """从文件加载 task_stats"""
    global initialized
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        for task, data in loaded.items():
            task_stats[task] = {
                'assigned': 0,
                'total_attempts': data['complete_attempts'],
                'complete_attempts': data['complete_attempts'],
                'success_count': data['success_count'],
                'failure_count': data['failure_count'],
                'average_steps': data['average_steps'],
                'average_time': data['average_time'],
                'average_success_rate': data['average_success_rate']
            }
        initialized = bool(task_stats)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Failed to load task_stats: {e}")

def save_task_stats():
    """将 task_stats 序列化为 JSON 并覆盖写入文件"""
    try:
        to_save = {
            task: {
                'complete_attempts': stats['complete_attempts'],
                'success_count': stats['success_count'],
                'failure_count': stats['failure_count'],
                'average_steps': stats['average_steps'],
                'average_time': stats['average_time'],
                'average_success_rate': stats['average_success_rate']
            } for task, stats in task_stats.items()
        }
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save task_stats: {e}")
# ---------------------------
# 请求/响应模型（推荐使用 Pydantic）
# ---------------------------
class InitializeRequest(BaseModel):
    task_list: List[str]
class CompleteTaskRequest(BaseModel):
    task: str
    success: bool
    steps: int
    time: float # 建议使用 float 表示秒数，更精确
class TaskResponse(BaseModel):
    task: str
class HealthResponse(BaseModel):
    status: str
    initialized: bool
    task_count: int
    timestamp: float
# ---------------------------
# 路由
# ---------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查接口"""
    with lock:
        return HealthResponse(
            status="healthy",
            initialized=initialized,
            task_count=len(task_stats),
            timestamp=time.time()
        )
@app.post("/initialize")
async def initialize(req: InitializeRequest):
    """初始化任务列表（幂等）"""
    global initialized
    with lock:
        if initialized:
            return {"message": "Already initialized"}
        # 使用 set 去重
        for task in set(req.task_list):
            if task not in task_stats:
                task_stats[task] = {
                    'assigned': 0,
                    'total_attempts': 0,
                    'complete_attempts': 0,
                    'success_count': 0,
                    'failure_count': 0,
                    'average_steps': 0,
                    'average_time': 0.0,
                    'average_success_rate': 0.0
                }
        initialized = True
        save_task_stats()
    return {"message": "Initialized"}
@app.get("/get_task", response_model=TaskResponse)
async def get_task():
    """获取一个待执行的任务（负载最低优先 + 随机）"""
    with lock:
        if not task_stats:
            raise HTTPException(status_code=400, detail="TaskManager not initialized")
        # 按照 total_attempts 最少的优先
        sums = {task: stats['total_attempts'] for task, stats in task_stats.items()}
        min_sum = min(sums.values())
        candidates = [task for task, s in sums.items() if s == min_sum]
        selected = random.choice(candidates)
        task_stats[selected]['total_attempts'] += 1
        task_stats[selected]['assigned'] += 1
        return TaskResponse(task=selected)
    
@app.post("/complete_task")
async def complete_task(req: CompleteTaskRequest):
    """任务完成上报"""
    global initialized
    with lock:
        if req.task not in task_stats:
            raise HTTPException(status_code=404, detail=f"Task {req.task} not found")
        stats = task_stats[req.task]
        stats['assigned'] -= 1
        old_complete = stats['complete_attempts']
        stats['complete_attempts'] += 1
        if req.success:
            stats['success_count'] += 1
        else:
            stats['failure_count'] += 1
        stats['average_steps'] = (stats['average_steps'] * old_complete + req.steps) / stats['complete_attempts'] if stats['complete_attempts'] > 0 else 0
        stats['average_time'] = (stats['average_time'] * old_complete + req.time) / stats['complete_attempts'] if stats['complete_attempts'] > 0 else 0.0
        stats['average_success_rate'] = stats['success_count'] / stats['complete_attempts'] if stats['complete_attempts'] > 0 else 0.0
        # 每次完成都持久化（频率高时可考虑改为定时持久化或批量）
        save_task_stats()
        initialized = True
    return {"message": "Completed"}

# 可选：添加一个查看当前状态的调试接口
@app.get("/stats")
async def get_stats():
    with lock:
        tasks = {
            task: {
                'complete_attempts': stats['complete_attempts'],
                'success_count': stats['success_count'],
                'failure_count': stats['failure_count'],
                'average_steps': stats['average_steps'],
                'average_time': stats['average_time'],
                'average_success_rate': stats['average_success_rate']
            } for task, stats in task_stats.items()
        }
        attempted_tasks = 0
        sum_success_rates = 0.0
        total_complete = 0
        for stats in task_stats.values():
            if stats['complete_attempts'] > 0:
                attempted_tasks += 1
                sum_success_rates += stats['average_success_rate']
                total_complete += stats['complete_attempts']
        global_stats = {
            'average_success_rate': sum_success_rates / attempted_tasks if attempted_tasks > 0 else 0.0,
            'total_complete_times': total_complete,
            'total_attempted_task_types': attempted_tasks,
            'average_completions_per_task': total_complete / attempted_tasks if attempted_tasks > 0 else 0.0
        }
        return {'tasks': tasks, 'global': global_stats}
if __name__ == "__main__":
    import uvicorn
    # load_task_stats()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5001,
        # reload=True, # 开发时开启热重载
        workers=1, # 生产建议根据 CPU 核数调整
    )