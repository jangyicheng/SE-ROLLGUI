from __future__ import annotations

from typing import Any, Dict, Mapping


def _size_in_bytes(value: Any) -> int:
    if value is None:
        return 0

    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes

    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)

    if isinstance(value, str):
        return len(value.encode("utf-8", errors="ignore"))

    if isinstance(value, list):
        if not value:
            return 0
        if isinstance(value[0], (int, float, bool)):
            return len(value) * 8

    element_size = getattr(value, "element_size", None)
    numel = getattr(value, "numel", None)
    if callable(element_size) and callable(numel):
        try:
            return int(element_size() * numel())
        except Exception:
            return 0

    return 0


def _estimate_messages_chars(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0

    total = 0
    for message in messages[:8]:
        if not isinstance(message, Mapping):
            continue

        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
            continue

        if isinstance(content, list):
            for item in content[:8]:
                if not isinstance(item, Mapping):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    total += len(text)

    return total


def collect_gui_traj_memory_snapshot(env_managers: Mapping[int, Any], max_env_details: int = 8) -> Dict[str, Any]:
    gui_env_count = 0
    active_envs = 0
    history_total = 0
    frames_total = 0

    observation_bytes_total = 0
    messages_bytes_total = 0
    non_tensor_bytes_total = 0
    frames_bytes_total = 0

    env_details = []

    for env_id, env_manager in env_managers.items():
        if env_manager.__class__.__name__ != "GuiTrajEnvManager" and "GuiTrajEnvManager" not in {
            base.__name__ for base in env_manager.__class__.__mro__
        }:
            continue

        gui_env_count += 1
        cache = getattr(env_manager, "rollout_cache", None)
        if cache is None:
            continue

        history = getattr(cache, "history", None) or []
        frames = getattr(cache, "frames", None) or []
        history_len = len(history)
        frames_len = len(frames)

        history_total += history_len
        frames_total += frames_len

        if history_len > 0 or frames_len > 0:
            active_envs += 1

        observation_sample = 0
        messages_sample = 0
        non_tensor_sample = 0

        if history and isinstance(history[-1], Mapping):
            last_item = history[-1]
            observation_sample = _size_in_bytes(last_item.get("observation"))
            messages_sample = _estimate_messages_chars(last_item.get("messages"))

            non_tensor_batch = last_item.get("non_tensor_batch")
            if isinstance(non_tensor_batch, Mapping):
                for value in list(non_tensor_batch.values())[:8]:
                    non_tensor_sample += _size_in_bytes(value)

        frames_sample = _size_in_bytes(frames[-1]) if frames_len > 0 else 0

        observation_bytes = observation_sample * history_len
        messages_bytes = messages_sample * history_len
        non_tensor_bytes = non_tensor_sample * history_len
        frame_bytes = frames_sample * frames_len

        observation_bytes_total += observation_bytes
        messages_bytes_total += messages_bytes
        non_tensor_bytes_total += non_tensor_bytes
        frames_bytes_total += frame_bytes

        env_total_bytes = observation_bytes + messages_bytes + non_tensor_bytes + frame_bytes
        env_details.append(
            (
                env_total_bytes,
                f"env={env_id},cache_mb={env_total_bytes / (1024 * 1024):.2f},history={history_len},frames={frames_len}",
            )
        )

    env_details.sort(key=lambda x: x[0], reverse=True)
    detail_lines = [item[1] for item in env_details[:max_env_details]]
    if len(env_details) > max_env_details:
        detail_lines.append(f"...(+{len(env_details) - max_env_details} envs)")

    cache_total_bytes = observation_bytes_total + messages_bytes_total + non_tensor_bytes_total + frames_bytes_total

    return {
        "memory/gui/env_count": gui_env_count,
        "memory/gui/active_envs": active_envs,
        "memory/gui/history_total": history_total,
        "memory/gui/frames_total": frames_total,
        "memory/gui/cache_observation_mb": observation_bytes_total / (1024 * 1024),
        "memory/gui/cache_messages_mb": messages_bytes_total / (1024 * 1024),
        "memory/gui/cache_non_tensor_mb": non_tensor_bytes_total / (1024 * 1024),
        "memory/gui/cache_frames_mb": frames_bytes_total / (1024 * 1024),
        "memory/gui/cache_estimated_mb": cache_total_bytes / (1024 * 1024),
        "memory/gui/top_envs": " | ".join(detail_lines),
    }


def _estimate_dataproto_bytes(rollout: Any) -> int:
    if rollout is None:
        return 0

    total = 0
    batch = getattr(rollout, "batch", None)
    if batch is not None:
        try:
            for _, tensor in batch.items():
                total += _size_in_bytes(tensor)
        except Exception:
            pass

    non_tensor_batch = getattr(rollout, "non_tensor_batch", None)
    if isinstance(non_tensor_batch, Mapping):
        for value in list(non_tensor_batch.values())[:8]:
            total += _size_in_bytes(value)

    return total


def collect_group_queue_memory_snapshot(group_queue_manager: Any) -> Dict[str, Any]:
    group_queues = getattr(group_queue_manager, "group_queues", None)
    task_to_group_id = getattr(group_queue_manager, "task_to_group_id", None)

    if not isinstance(group_queues, Mapping):
        return {
            "memory/group_queue/group_queues_count": 0,
            "memory/group_queue/task_to_group_id_count": 0,
            "memory/group_queue/pending_groups_total": 0,
            "memory/group_queue/buffered_rollouts_total": 0,
            "memory/group_queue/max_rollouts_in_group": 0,
            "memory/group_queue/rollout_payload_estimated_mb": 0.0,
        }

    pending_groups_total = 0
    buffered_rollouts_total = 0
    max_rollouts_in_group = 0
    sample_rollout = None

    for queue in group_queues.values():
        groups = getattr(queue, "groups", None)
        if not isinstance(groups, Mapping):
            continue

        pending_groups_total += len(groups)

        for group_data in groups.values():
            rollouts = getattr(group_data, "rollouts", None) or []
            rollout_count = len(rollouts)
            buffered_rollouts_total += rollout_count
            max_rollouts_in_group = max(max_rollouts_in_group, rollout_count)

            if sample_rollout is None:
                for rollout in rollouts:
                    if rollout is not None:
                        sample_rollout = rollout
                        break

    sample_rollout_bytes = _estimate_dataproto_bytes(sample_rollout)
    rollout_payload_estimated_mb = (sample_rollout_bytes * buffered_rollouts_total) / (1024 * 1024)

    return {
        "memory/group_queue/group_queues_count": len(group_queues),
        "memory/group_queue/task_to_group_id_count": len(task_to_group_id) if isinstance(task_to_group_id, Mapping) else 0,
        "memory/group_queue/pending_groups_total": pending_groups_total,
        "memory/group_queue/buffered_rollouts_total": buffered_rollouts_total,
        "memory/group_queue/max_rollouts_in_group": max_rollouts_in_group,
        "memory/group_queue/rollout_payload_estimated_mb": rollout_payload_estimated_mb,
    }
