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
SAVE_INTERVAL_SECONDS = 60.0
N_TASK = 2
GROUP_SIZE = 1  # 每次连续分配 GROUP_SIZE 个同类任务，需与训练端 group_size 一致
lock = threading.Lock()
random.seed(42)
task_stats: Dict[str, Dict] = {}
initialized = False
LOG_FILE = "/HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL/ROLL/roll/pipeline/agentic/env/android/GuiTaskEvalManager.json"

# 批次分配状态
current_batch_task: str = None
current_batch_remaining: int = 0


async def periodic_save():
    while True:
        await asyncio.sleep(SAVE_INTERVAL_SECONDS)
        with lock:
            if time.time() - last_save_time >= SAVE_INTERVAL_SECONDS:
                save_task_stats()


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(periodic_save())
    yield
    with lock:
        save_task_stats()


app = FastAPI(
    title="Task Manager API",
    description="Task distribution and statistics service",
    version="1.0.0",
    lifespan=lifespan
)


def load_task_stats():
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
            for task, stats in sorted(task_stats.items())
        }
        global_stats = compute_global_stats()
        final_data = {
            "last_saved": datetime.utcnow().isoformat() + "Z",
            "tasks": to_save,
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


class InitializeRequest(BaseModel):
    task_list: List[str]
    group_size: int = GROUP_SIZE  # 可在初始化时传入 group_size


class CompleteTaskRequest(BaseModel):
    task: str
    success: bool
    steps: int
    time: float


class TaskResponse(BaseModel):
    task: str


class HealthResponse(BaseModel):
    status: str
    initialized: bool
    task_count: int
    timestamp: float


@app.get("/health", response_model=HealthResponse)
async def health():
    with lock:
        return HealthResponse(
            status="healthy",
            initialized=initialized,
            task_count=len(task_stats),
            timestamp=time.time()
        )


@app.post("/initialize")
async def initialize(req: InitializeRequest):
    # global initialized, GROUP_SIZE
    # with lock:
    #     if initialized:
    #         print("Already initialized")
    #         return {"message": "Already initialized"}
    #     GROUP_SIZE = req.group_size
    #     for task in set(req.task_list):
    #         if task not in task_stats:
    #             task_stats[task] = {
    #                 'assigned': 0,
    #                 'total_attempts': 0,
    #                 'complete_attempts': 0,
    #                 'success_count': 0,
    #                 'failure_count': 0,
    #                 'average_steps': 0,
    #                 'average_time': 0.0,
    #                 'average_success_rate': 0.0
    #             }
    #     initialized = True
    
    global initialized, GROUP_SIZE, current_batch_task, current_batch_remaining

    empty_stats = {
        'assigned': 0,
        'total_attempts': 0,
        'complete_attempts': 0,
        'success_count': 0,
        'failure_count': 0,
        'average_steps': 0,
        'average_time': 0.0,
        'average_success_rate': 0.0
    }

    with lock:
        if initialized:
            print("Already initialized")
            return {"message": "Already initialized"}

        GROUP_SIZE = req.group_size

        # 可选：只保留本次 task_list
        task_stats.clear()

        for task in set(req.task_list):
            task_stats[task] = dict(empty_stats)

        current_batch_task = None
        current_batch_remaining = 0
        initialized = True
        
    return {"message": "Initialized", "group_size": GROUP_SIZE}


@app.get("/get_task", response_model=TaskResponse)
async def get_task():
    """
    任务分配逻辑：每次连续分配 GROUP_SIZE 个同类任务。
    当一个批次分配完毕后，根据调度算法选取下一个任务开启新批次。
    """
    global current_batch_task, current_batch_remaining

    with lock:
        if not task_stats:
            raise HTTPException(status_code=400, detail="TaskEvalManager not initialized")

        # 如果当前批次仍有剩余，继续分配同一任务
        if current_batch_remaining > 0 and current_batch_task is not None:
            # 验证该任务仍然在 task_stats 中
            if current_batch_task in task_stats:
                current_batch_remaining -= 1
                task_stats[current_batch_task]['total_attempts'] += 1
                task_stats[current_batch_task]['assigned'] += 1
                print(f"get task (batch continue): {current_batch_task} (remaining: {current_batch_remaining})")
                return TaskResponse(task=current_batch_task)
            else:
                # 任务已被移除，重置批次
                current_batch_task = None
                current_batch_remaining = 0

        # 批次已用完或无批次，选择新任务开启新批次
        sorted_tasks = sorted(
            task_stats.keys(),
            key=lambda t: (task_stats[t]['total_attempts'], task_stats[t]['assigned'])
        )

        best_task = sorted_tasks[0]
        min_attempts = task_stats[best_task]['total_attempts']

        if min_attempts >= N_TASK * GROUP_SIZE:
            print("get task: finish")
            current_batch_task = None
            current_batch_remaining = 0
            return TaskResponse(task="finish")

        min_assigned = task_stats[best_task]['assigned']
        candidates = [
            t for t in sorted_tasks
            if task_stats[t]['total_attempts'] == min_attempts
            and task_stats[t]['assigned'] == min_assigned
        ]

        selected = random.choice(candidates)

        # 开启新批次：当前请求算第一个，剩余 GROUP_SIZE - 1 个
        current_batch_task = selected
        current_batch_remaining = GROUP_SIZE - 1

        task_stats[selected]['total_attempts'] += 1
        task_stats[selected]['assigned'] += 1

        print(f"get task (new batch): {selected} (remaining: {current_batch_remaining})")
        return TaskResponse(task=selected)


@app.post("/complete_task")
async def complete_task(req: CompleteTaskRequest):
    global initialized
    with lock:
        initialized = False
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
            for task, stats in sorted(task_stats.items())
        }
        global_stats = compute_global_stats()
        return {
            'tasks': tasks,
            'global': global_stats,
            'last_updated': time.time()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5001,
        workers=1,
    )