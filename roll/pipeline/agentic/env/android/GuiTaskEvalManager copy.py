import random
import threading
import time
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
from contextlib import asynccontextmanager
from datetime import datetime
import asyncio
import hashlib
import os

# 全局变量
last_save_time = 0.0
SAVE_INTERVAL_SECONDS = 30.0
N_TASK = 5 
GROUP_SIZE = 1  # 每次连续分配 GROUP_SIZE 个同类任务，需与训练端 group_size 一致
lock = threading.Lock()
task_stats: Dict[str, Dict] = {}
initialized = False
LOG_FILE = None
# 批次分配状态
current_batch_task: str = None
current_batch_remaining: int = 0
GLOBAL_SEED = 42
global_step = 0
timestamp = time.strftime("%Y%m%d_%H%M%S")


def deterministic_choice(candidates: List[str], seed: int, step: int) -> str:
    key = f"{seed}-{step}-{'|'.join(sorted(candidates))}"
    h = hashlib.md5(key.encode()).hexdigest()
    idx = int(h, 16) % len(candidates)
    return sorted(candidates)[idx]


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

        tasks_data = loaded.get("tasks", loaded) if isinstance(loaded, dict) else {}
        for task, data in tasks_data.items():
            task_stats[task] = {
                'assigned': 0,
                'total_attempts': data.get('total_attempts', data.get('complete_attempts', 0)),
                'complete_attempts': data.get('complete_attempts', 0),
                'success_count': data.get('success_count', 0),
                'failure_count': data.get('failure_count', 0),
                'average_steps': data.get('average_steps', 0),
                'average_time': data.get('average_time', 0.0),
                'average_success_rate': data.get('average_success_rate', 0.0),
                'returned_count': data.get('returned_count', 0),
                'last_return_reason': data.get('last_return_reason'),
                'last_return_time': data.get('last_return_time'),
            }
        initialized = bool(task_stats)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"Failed to load task_stats: {e}")
        
def save_task_stats():
    global last_save_time
    try:
        if not LOG_FILE:
            return

        to_save = {
            task: {
                'total_attempts': stats['total_attempts'],
                'complete_attempts': stats['complete_attempts'],
                'success_count': stats['success_count'],
                'failure_count': stats['failure_count'],
                'average_steps': stats['average_steps'],
                'average_time': stats['average_time'],
                'average_success_rate': stats['average_success_rate'],
                'returned_count': stats.get('returned_count', 0),
                'last_return_reason': stats.get('last_return_reason'),
                'last_return_time': stats.get('last_return_time'),
            }
            for task, stats in sorted(task_stats.items())
            if stats['complete_attempts'] >= 1 or stats.get('returned_count', 0) >= 1
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
    seed: int = 42
    n_task: int = N_TASK  # 可在初始化时传入 n_task

class CompleteTaskRequest(BaseModel):
    task: str
    success: bool
    steps: int
    time: float

class ReturnTaskRequest(BaseModel):
    task: str
    reason: str = "env_failed"
    rollback_total_attempts: bool = True

class TaskResponse(BaseModel):
    task: str


@app.post("/initialize")
async def initialize(req: InitializeRequest):
    global initialized, GROUP_SIZE, N_TASK ,current_batch_task, current_batch_remaining, timestamp , GLOBAL_SEED , global_step , LOG_FILE

    empty_stats = {
    'assigned': 0,
    'total_attempts': 0,
    'complete_attempts': 0,
    'success_count': 0,
    'failure_count': 0,
    'average_steps': 0,
    'average_time': 0.0,
    'average_success_rate': 0.0,
    'returned_count': 0,
    'last_return_reason': None,
    'last_return_time': None,
}

    with lock:
        if initialized:
            print("Already initialized")
            return {"message": "Already initialized" , "group_size": GROUP_SIZE ,"timestamp": timestamp}


        GROUP_SIZE = req.group_size
        N_TASK = req.n_task
        GLOBAL_SEED = req.seed
        global_step = 0
        # 可选：只保留本次 task_list
        task_stats.clear()

        # for task in set(req.task_list):
        #     task_stats[task] = dict(empty_stats)
        unique_tasks = list(dict.fromkeys(req.task_list)) # 防止同名任务
        for task in unique_tasks:
            task_stats[task] = dict(empty_stats)

        current_batch_task = None
        current_batch_remaining = 0
        initialized = True
        timestamp = time.strftime("%Y-%m-%d_%H%M%S")
        
        LOG_FILE = f"/HOME/hitsz_xdeng/hitsz_xdeng_2/HDD_POOL/ROLL/trajectories/{timestamp}/{timestamp}.json"
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        print(f"Initialized with {len(unique_tasks)} tasks, group_size: {GROUP_SIZE}, seed: {GLOBAL_SEED}, n_task: {N_TASK}")
        
    return {"message": "Initialized", "group_size": GROUP_SIZE , "timestamp": timestamp}


@app.get("/get_task", response_model=TaskResponse)
async def get_task():
    """
    任务分配逻辑：每次连续分配 GROUP_SIZE 个同类任务。
    当一个批次分配完毕后，根据调度算法选取下一个任务开启新批次。
    """
    global current_batch_task, current_batch_remaining , GLOBAL_SEED , global_step

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

        selected = deterministic_choice(candidates, GLOBAL_SEED, global_step)
        global_step += 1

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
        if initialized:
            initialized = False
        if req.task not in task_stats:
            raise HTTPException(status_code=404, detail=f"Task {req.task} not found")
        
        stats = task_stats[req.task]
        stats["assigned"] = max(stats["assigned"] - 1, 0)
        old_complete = stats["complete_attempts"]
        stats["complete_attempts"] += 1
        if req.success:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
        stats["average_steps"] = (stats["average_steps"] * old_complete + req.steps) / stats["complete_attempts"] if stats["complete_attempts"] > 0 else 0
        stats["average_time"] = (stats["average_time"] * old_complete + req.time) / stats["complete_attempts"] if stats["complete_attempts"] > 0 else 0.0
        stats["average_success_rate"] = stats["success_count"] / stats["complete_attempts"] if stats["complete_attempts"] > 0 else 0.0
    print(f"complete task: {req.task}, success: {req.success}, steps: {req.steps}, time: {req.time:.2f}s")
    return {"message": "Completed"}

@app.post("/return_task")
async def return_task(req: ReturnTaskRequest):
    with lock:
        if req.task not in task_stats:
            raise HTTPException(status_code=404, detail=f"Task {req.task} not found")

        stats = task_stats[req.task]
        stats['assigned'] = max(stats['assigned'] - 1, 0)
        if req.rollback_total_attempts:
            stats['total_attempts'] = max(stats['total_attempts'] - 1, 0)

        stats['returned_count'] = stats.get('returned_count', 0) + 1
        stats['last_return_reason'] = req.reason
        stats['last_return_time'] = time.time()

    print(f"return task: {req.task}, reason: {req.reason}, rollback_total_attempts={req.rollback_total_attempts}")
    return {"message": "Returned"}

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