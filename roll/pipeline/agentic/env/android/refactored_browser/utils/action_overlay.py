from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw


def rescale_coordinates(point, width, height):
    return [round(point[0] / 999 * width), round(point[1] / 999 * height)]


def parse_args(raw_args):
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def pick_point(args, keys):
    for key in keys:
        if key not in args:
            continue
        value = args[key]
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return [float(value[0]), float(value[1])]
            except Exception:
                continue
        if isinstance(value, dict) and "x" in value and "y" in value:
            try:
                return [float(value["x"]), float(value["y"])]
            except Exception:
                continue
    return None


def get_image_path_for_step(attempt_dir: Path, step_num: int) -> Path:
    idx = max(int(step_num) - 1, 0)
    candidate = attempt_dir / f"step_{idx:03d}.png"
    if not candidate.exists():
        candidate = attempt_dir / f"step_{int(step_num):03d}.png"
    return candidate


def extract_action_and_args(step_data: dict):
    raw_action = step_data.get("action", "")
    raw_args = parse_args(step_data.get("args"))
    if isinstance(raw_action, str) and raw_action.lower() in {"click", "tap", "long_press", "swipe"}:
        return raw_action.lower(), raw_args

    text = str(raw_action)
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.S)
    if not match:
        return "", {}

    try:
        payload = json.loads(match.group(1))
    except Exception:
        return "", {}

    arguments = payload.get("arguments", {}) if isinstance(payload, dict) else {}
    action = str(arguments.get("action", "")).lower()
    return action, arguments if isinstance(arguments, dict) else {}


def draw_action_overlay(img_path: Path, step_data: dict):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    action, args = extract_action_and_args(step_data)
    p1 = pick_point(args, ["coordinate", "coord", "position", "start", "from"])
    p2 = pick_point(args, ["coordinate2", "coord2", "end", "to"])

    if action in ["click", "tap"] and p1 is not None:
        x, y = rescale_coordinates(p1, width, height)
        radius = 10
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red")
    elif action == "long_press" and p1 is not None:
        x, y = rescale_coordinates(p1, width, height)
        radius = 15
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline="yellow", width=4)
    elif action == "swipe" and p1 is not None and p2 is not None:
        x1, y1 = rescale_coordinates(p1, width, height)
        x2, y2 = rescale_coordinates(p2, width, height)
        draw.line((x1, y1, x2, y2), fill="blue", width=5)
        draw.ellipse((x2 - 6, y2 - 6, x2 + 6, y2 + 6), fill="blue")

    return img


def _overlay_cache_dir(img_path: Path) -> Path:
    cache_dir = img_path.parent / ".overlay_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_overlay_image_path(img_path: Path, step_data: dict) -> Path:
    cache_dir = _overlay_cache_dir(img_path)

    try:
        step_payload = json.dumps(step_data, ensure_ascii=False, sort_keys=True)
    except Exception:
        step_payload = str(step_data)

    src_sig = f"{img_path.resolve()}|{img_path.stat().st_mtime_ns}|{img_path.stat().st_size}"
    digest = hashlib.sha1(f"{src_sig}|{step_payload}".encode("utf-8")).hexdigest()[:16]
    cached_path = cache_dir / f"overlay_{digest}.png"

    if cached_path.exists():
        return cached_path

    overlay = draw_action_overlay(img_path, step_data)
    overlay.save(cached_path, format="PNG")
    return cached_path
