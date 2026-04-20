import requests
from datetime import datetime
from typing import Any, Dict

from roll.utils.logging import get_logger


logger = get_logger()


class TaskManagerUtilsMixin:
    def _http_get_json(self, url: str, timeout: int = 20) -> Dict[str, Any]:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _http_post_json(self, url: str, payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _resolve_active_manager(self) -> str:
        return self.task_manager_train_url if self.scheduler_mode == "train" else self.task_manager_eval_url

    def _resolve_peer_manager(self) -> str:
        return self.task_manager_eval_url if self.scheduler_mode == "train" else self.task_manager_train_url

    def _sync_timestamp_only(self, shared_timestamp: str | None) -> str:
        """
        只做 timestamp 协调，不初始化对端 manager。
        优先级:
        1) 显式 shared_timestamp
        2) 任一已初始化 manager 的 timestamp
        3) 新生成 timestamp
        """
        if shared_timestamp:
            return shared_timestamp

        active_url = self._resolve_active_manager()
        peer_url = self._resolve_peer_manager()

        ts_candidates = []
        for url in (active_url, peer_url):
            try:
                info = self._http_get_json(f"{url}/info")
                if info.get("initialized") and info.get("timestamp"):
                    ts_candidates.append(info["timestamp"])
            except Exception:
                pass

        ts_candidates = list(dict.fromkeys(ts_candidates))
        if len(ts_candidates) > 1:
            raise RuntimeError(f"manager timestamp conflict: {ts_candidates}")
        if len(ts_candidates) == 1:
            return ts_candidates[0]
        return datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def _initialize_active_task_manager(
        self,
        task_list: list[str],
        group_size: int,
        n_task: int,
        seed: int,
        shared_timestamp: str | None,
    ) -> str:
        active_url = self._resolve_active_manager()
        final_timestamp = self._sync_timestamp_only(shared_timestamp)

        payload = {
            "task_list": task_list,
            "group_size": group_size,
            "n_task": n_task,
            "seed": seed,
            "timestamp": final_timestamp,
            "mode": self.scheduler_mode,
        }

        try:
            self._http_post_json(f"{active_url}/initialize", payload)
            info = self._http_get_json(f"{active_url}/info")
        except requests.exceptions.HTTPError as e:
            # 并发初始化常见为 409，优先回收 manager 当前 timestamp
            resp = getattr(e, "response", None)
            if resp is not None and resp.status_code == 409:
                try:
                    body = resp.json()
                    detail = body.get("detail", {}) if isinstance(body, dict) else {}
                    current = detail.get("current", {}) if isinstance(detail, dict) else {}
                    ts = current.get("timestamp") if isinstance(current, dict) else None
                    if ts:
                        logger.warning(
                            f"initialize conflict, use manager timestamp={ts}, local={final_timestamp}"
                        )
                        return ts
                except Exception:
                    pass

            # 兜底：再读一次 /info，拿 manager 真值
            try:
                info = self._http_get_json(f"{active_url}/info")
                ts = info.get("timestamp")
                if info.get("initialized") and ts:
                    logger.warning(
                        f"initialize failed but manager initialized, use timestamp={ts}, local={final_timestamp}"
                    )
                    return ts
            except Exception:
                pass

            # 最后保留原行为
            return final_timestamp
        except Exception:
            try:
                info = self._http_get_json(f"{active_url}/info")
                ts = info.get("timestamp")
                if info.get("initialized") and ts:
                    logger.warning(
                        f"initialize exception fallback to manager timestamp={ts}, local={final_timestamp}"
                    )
                    return ts
            except Exception:
                pass
            return final_timestamp

        manager_ts = info.get("timestamp")
        if manager_ts and manager_ts != final_timestamp:
            logger.warning(
                f"manager timestamp differs, use manager timestamp={manager_ts}, local={final_timestamp}"
            )
            return manager_ts

        return final_timestamp

    def _return_task_once(self, reason: str, rollback_total_attempts: bool = True):
        if self.assigned_task is None or self._task_returned_for_current_episode:
            return
        try:
            payload = {
                "task": self.assigned_task,
                "reason": reason,
                "rollback_total_attempts": rollback_total_attempts,
            }
            requests.post(f"{self.task_manager_url}/return_task", json=payload, timeout=20)
        except Exception as e:
            logger.warning(f"return_task failed: {e}")
        finally:
            self._task_returned_for_current_episode = True
