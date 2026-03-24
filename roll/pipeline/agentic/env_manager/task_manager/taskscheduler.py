
import hashlib

def deterministic_choice(candidates: List[str], seed: int, step: int) -> str:
    key = f"{seed}-{step}-{'|'.join(sorted(candidates))}"
    h = hashlib.md5(key.encode()).hexdigest()
    idx = int(h, 16) % len(candidates)
    return sorted(candidates)[idx]



# ------------------------ 调度抽象接口 ------------------------
class TaskScheduler:
    def select_new_batch_task(
        self,
        task_stats: Dict[str, Dict],
        n_task: int,
        group_size: int,
        seed: int,
        step: int
    ) -> Tuple[str, int]:
        """
        返回 (selected_task, next_step)。
        selected_task 可为 "finish" 表示结束。
        """
        raise NotImplementedError


class DefaultTaskScheduler(TaskScheduler):
    def select_new_batch_task(
        self,
        task_stats: Dict[str, Dict],
        n_task: int,
        group_size: int,
        seed: int,
        step: int
    ) -> Tuple[str, int]:
        sorted_tasks = sorted(
            task_stats.keys(),
            key=lambda t: (task_stats[t]['total_attempts'], task_stats[t]['assigned'])
        )

        best_task = sorted_tasks[0]
        min_attempts = task_stats[best_task]['total_attempts']
        if min_attempts >= n_task * group_size:
            return "finish", step

        min_assigned = task_stats[best_task]['assigned']
        candidates = [
            t for t in sorted_tasks
            if task_stats[t]['total_attempts'] == min_attempts
            and task_stats[t]['assigned'] == min_assigned
        ]

        selected = deterministic_choice(candidates, seed, step)
        return selected, step + 1

class _BaseScheduler(TaskScheduler):
    def _check_finish(self, task_stats: Dict[str, Dict], n_task: int, group_size: int) -> bool:
        min_attempts = min(task_stats[t]['total_attempts'] for t in task_stats)
        return min_attempts >= n_task * group_size

    def _pick_with_tie_break(self, candidates: List[str], seed: int, step: int) -> str:
        return deterministic_choice(candidates, seed, step)


class ModerateSuccessRateScheduler(_BaseScheduler):
    """
    成功率适中优先：优先 success_rate 接近 target 的任务。
    同时保证冷启动：样本不足的任务优先探索。
    """
    def __init__(self, target: float = 0.5, min_samples: int = 3):
        self.target = target
        self.min_samples = min_samples

    def select_new_batch_task(self, task_stats, n_task, group_size, seed, step):
        if self._check_finish(task_stats, n_task, group_size):
            return "finish", step

        def key(t: str):
            s = task_stats[t]
            complete = s['complete_attempts']
            warmup_done = complete >= self.min_samples
            distance = abs(s['average_success_rate'] - self.target) if complete > 0 else 0.0
            return (warmup_done, distance, s['total_attempts'], s['assigned'])

        best = min(key(t) for t in task_stats)
        candidates = [t for t in task_stats if key(t) == best]
        selected = self._pick_with_tie_break(candidates, seed, step)
        return selected, step + 1


class StaleFirstScheduler(_BaseScheduler):
    """
    太久没被调度优先：记录每个任务最近一次被选为新批次的 step。
    """
    def __init__(self):
        self.last_selected_step: Dict[str, int] = {}

    def select_new_batch_task(self, task_stats, n_task, group_size, seed, step):
        if self._check_finish(task_stats, n_task, group_size):
            return "finish", step

        min_attempts = min(task_stats[t]['total_attempts'] for t in task_stats)
        pool = [t for t in task_stats if task_stats[t]['total_attempts'] == min_attempts]

        def stale_score(t: str):
            return step - self.last_selected_step.get(t, -10**9)

        max_stale = max(stale_score(t) for t in pool)
        candidates = [t for t in pool if stale_score(t) == max_stale]

        selected = self._pick_with_tie_break(candidates, seed, step)
        self.last_selected_step[selected] = step
        return selected, step + 1


class HybridScheduler(_BaseScheduler):
    """
    混合策略：先保证公平(最小 total_attempts)，再优先久未调度，再偏向成功率接近 target。
    """
    def __init__(self, target: float = 0.5, min_samples: int = 2):
        self.target = target
        self.min_samples = min_samples
        self.last_selected_step: Dict[str, int] = {}

    def select_new_batch_task(self, task_stats, n_task, group_size, seed, step):
        if self._check_finish(task_stats, n_task, group_size):
            return "finish", step

        min_attempts = min(task_stats[t]['total_attempts'] for t in task_stats)
        pool = [t for t in task_stats if task_stats[t]['total_attempts'] == min_attempts]

        def key(t: str):
            s = task_stats[t]
            complete = s['complete_attempts']
            warmup_done = complete >= self.min_samples
            stale = step - self.last_selected_step.get(t, -10**9)
            distance = abs(s['average_success_rate'] - self.target) if complete > 0 else 0.0
            return (warmup_done, -stale, distance, s['assigned'])

        best = min(key(t) for t in pool)
        candidates = [t for t in pool if key(t) == best]

        selected = self._pick_with_tie_break(candidates, seed, step)
        self.last_selected_step[selected] = step
        return selected, step + 1