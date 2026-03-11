import random
import threading
import time
from typing import Dict, List
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
from contextlib import asynccontextmanager
from datetime import datetime
import asyncio

# 全局变量
last_save_time = 0.0
SAVE_INTERVAL_SECONDS = 60.0   # 每60秒保存一次，可调整
N_TASK = 2
# 线程锁（FastAPI 异步环境下仍建议使用锁保护共享状态）
lock = threading.Lock()
random.seed(42)
# 全局状态
task_stats: Dict[str, Dict] = {}
initialized = False
# 日志文件路径（建议后续改为使用 pathlib 或环境变量）
LOG_FILE = "/HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL/ROLL/roll/pipeline/agentic/env/android/TaskEvalManager.json"

# 新的后台定时保存任务
async def periodic_save():
    while True:
        await asyncio.sleep(SAVE_INTERVAL_SECONDS)
        with lock:
            if time.time() - last_save_time >= SAVE_INTERVAL_SECONDS:
                save_task_stats()

# lifespan 管理（FastAPI 推荐方式）
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动后台定时保存
    asyncio.create_task(periodic_save())
    
    yield
    
    # 程序结束时最后保存一次
    with lock:
        save_task_stats()

# 修改 app 创建方式
app = FastAPI(
    title="Task Manager API",
    description="Task distribution and statistics service",
    version="1.0.0",
    lifespan=lifespan
)


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
    global last_save_time
    try:
        to_save = {
            task: {
                'complete_attempts': stats['complete_attempts'],
                'success_count': stats['success_count'],
                'failure_count': stats['failure_count'],
                'average_steps': stats['average_steps'],
                'average_time': stats['average_time'],
                'average_success_rate': stats['average_success_rate']
            }
            for task, stats in sorted(task_stats.items())   # 按任务名字典序排序
        }

        global_stats = compute_global_stats()

        final_data = {
            "last_saved": datetime.utcnow().isoformat() + "Z",  # UTC ISO格式带Z
            "tasks": to_save ,
            "global_stats": global_stats
        }

        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        

                    
        last_save_time = time.time()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stats saved to disk")
        
    except Exception as e:
        print(f"Failed to save task_stats: {e}")
        
        
def compute_global_stats():
    attempted_tasks = 0
    sum_success_rates = 0.0
    total_success = 0
    total_complete = 0
    
    for stats in task_stats.values():
        if stats['complete_attempts'] > 0:
            attempted_tasks += 1
            sum_success_rates += stats['average_success_rate']
            total_success += stats['success_count']
            total_complete += stats['complete_attempts']
    
    return {
        "total_success_count": total_success,
        "total_complete_count": total_complete,
        "total_task_types": len(task_stats),
        "attempted_task_types": attempted_tasks,
        "global_success_rate": (total_success / total_complete) if total_complete > 0 else 0.0,
        "average_success_rate": (sum_success_rates / attempted_tasks) if attempted_tasks > 0 else 0.0,
        "average_completions_per_task": (total_complete / attempted_tasks) if attempted_tasks > 0 else 0.0
    }
        

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
            print("Already initialized")
            return {"message": "Already initialized"}
        
        # task_stats = {}
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
        # save_task_stats()
    return {"message": "Initialized"}


@app.get("/get_task", response_model=TaskResponse)
async def get_task():
#任务分配逻辑：优先分配尝试次数少的任务；如果尝试次数相同，则分配assigned最少的任务；如果仍然相同，则随机选择一个任务。
    with lock:
        if not task_stats:
            raise HTTPException(status_code=400, detail="TaskEvalManager not initialized")

        # 所有任务都参与排序，不再过滤 assigned <= 1
        sorted_tasks = sorted(
            task_stats.keys(),
            key=lambda t: (task_stats[t]['total_attempts'], task_stats[t]['assigned'])
        )

        best_task = sorted_tasks[0]
        min_attempts = task_stats[best_task]['total_attempts']

        if min_attempts >= N_TASK:
            print("get task: finish")
            return TaskResponse(task="finish")

        # 找到所有 total_attempts == min_attempts 且 assigned 最小的任务
        min_assigned = task_stats[best_task]['assigned']
        candidates = [
            t for t in sorted_tasks
            if task_stats[t]['total_attempts'] == min_attempts
            and task_stats[t]['assigned'] == min_assigned
        ]

        selected = random.choice(candidates)

        task_stats[selected]['total_attempts'] += 1
        task_stats[selected]['assigned'] += 1

        print(f"get task: {selected}")
        return TaskResponse(task=selected)
    
@app.post("/complete_task")
async def complete_task(req: CompleteTaskRequest):
    """任务完成上报"""
    global initialized
    with lock:
        initialized = False # 下次执行初始化时进行重置
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
        # print("save task stats!")
        # save_task_stats()
        
        
    return {"message": "Completed"}


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
            }
            for task, stats in sorted(task_stats.items())   # 按任务名字典序
        }
        
        global_stats = compute_global_stats()
        
        return {
            'tasks': tasks,
            'global': global_stats,
            'last_updated': time.time()
        }
    

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