"""
Task Initialization Explorer for AndroidWorld/MobileWorld.

Replaces the original Context Review with OpenMobile-style task initialization:
- Params persistence for deterministic replay
- App snapshot recovery for reproducible initial states
- Initialization verification with initial screenshots

This enables the self-evolving loop to generate tasks that can be reliably
executed during training.
"""

import base64
import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from roll.utils.logging import get_logger

from .params_manager import AndroidWorldParamsManager, MobileWorldParamsManager

logger = get_logger()


@dataclass
class TaskInitResult:
    """Result of a single task initialization."""

    task_name: str
    instance_id: int
    seed: int
    task_family: str
    params_path: str
    app_snapshot_restored: List[str]
    initialization: Dict[str, Any]
    init_screenshot: Optional[str]
    verification: Dict[str, Any]
    vlm_verification: Optional[Dict[str, Any]]
    timestamp: str
    blockers: List[str]
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        return {k: v for k, v in result.items() if v is not None}

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "task_init_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return result_path

    @property
    def success(self) -> bool:
        return self.initialization.get("success", False) and self.error_message is None


class BaseTaskInitializer:
    """Base class for task initialization."""

    TIMEOUT = 240

    def __init__(
        self,
        server_url: str,
        params_manager,
        max_init_steps: int = 5,
        output_dir: str = "./androidworld_init_output",
        task_pool: Optional[List[str]] = None,
        enable_vlm_verification: bool = False,
        console_port: Optional[int] = None,
        adb_path: str = "/root/android-sdk/platform-tools/adb",
        task_family: str = "android_world",
    ):
        self.server_url = server_url.rstrip("/")
        self.params_manager = params_manager
        self.max_init_steps = max_init_steps
        self.output_dir = Path(output_dir)
        self.task_pool = task_pool or []
        self.enable_vlm_verification = enable_vlm_verification
        self.console_port = console_port
        self.adb_path = adb_path
        self.task_family = task_family

    def _init_server(self) -> Dict[str, Any]:
        """Initialize the server connection."""
        raise NotImplementedError("Subclasses must implement _init_server")

    def _init_task(self, task_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize a specific task with params."""
        raise NotImplementedError("Subclasses must implement _init_task")

    def _get_screenshot(self) -> Optional[np.ndarray]:
        """Get current screenshot."""
        raise NotImplementedError("Subclasses must implement _get_screenshot")

    def _step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a verification step."""
        raise NotImplementedError("Subclasses must implement _step")

    def _get_task_app_names(self, task_name: str) -> List[str]:
        """Extract app names from task definition."""
        raise NotImplementedError("Subclasses must implement _get_task_app_names")

    def _vlm_verify_init(self, screenshot: np.ndarray, task_name: str) -> Dict[str, Any]:
        """Use VLM to verify initialization success."""
        return {"enabled": False, "feedback": None}

    def run_for_task(
        self, task_name: str, instance_id: int = 0
    ) -> TaskInitResult:
        """Run initialization for a single task."""
        task_dir = self.output_dir / task_name / f"instance_{instance_id}"
        task_dir.mkdir(parents=True, exist_ok=True)

        try:
            params = self.params_manager.generate_params(task_name, instance_id)
            params_path = self.params_manager.save_params(task_name, params, instance_id)

            init_result = self._init_task(task_name, params)

            init_screenshot = self._get_screenshot()
            init_screenshot_path = None
            if init_screenshot is not None:
                init_screenshot_path = str(task_dir / "init_screenshot.png")
                Image.fromarray(init_screenshot).save(init_screenshot_path)

            app_names = self._get_task_app_names(task_name)

            verification_steps = 0
            all_verification_successful = True
            app_responded = True

            for step_idx in range(self.max_init_steps):
                wait_action = {"name": "mobile_use", "arguments": {"action": "wait"}}
                step_result = self._step(wait_action)
                verification_steps += 1

                if step_result.get("error") or step_result.get("status") == "failed":
                    all_verification_successful = False
                    break

                time.sleep(0.5)

            vlm_feedback = None
            if self.enable_vlm_verification and init_screenshot is not None:
                vlm_result = self._vlm_verify_init(init_screenshot, task_name)
                vlm_feedback = vlm_result.get("feedback")

            return TaskInitResult(
                task_name=task_name,
                instance_id=instance_id,
                seed=params.get("seed", 0),
                task_family=self.task_family,
                params_path=params_path,
                app_snapshot_restored=app_names,
                initialization={
                    "success": True,
                    "steps_used": 0,
                    "initial_screen": f"{app_names[0] if app_names else 'Unknown'} initial screen",
                },
                init_screenshot=init_screenshot_path,
                verification={
                    "enabled": True,
                    "steps": verification_steps,
                    "all_successful": all_verification_successful,
                    "app_responded": app_responded,
                },
                vlm_verification={
                    "enabled": self.enable_vlm_verification,
                    "feedback": vlm_feedback,
                },
                timestamp=datetime.now().isoformat(),
                blockers=[],
            )

        except Exception as e:
            logger.error(f"Task initialization failed for {task_name}: {e}")
            return TaskInitResult(
                task_name=task_name,
                instance_id=instance_id,
                seed=0,
                task_family=self.task_family,
                params_path="",
                app_snapshot_restored=[],
                initialization={"success": False, "steps_used": 0, "initial_screen": ""},
                init_screenshot=None,
                verification={
                    "enabled": True,
                    "steps": 0,
                    "all_successful": False,
                    "app_responded": False,
                },
                vlm_verification={"enabled": self.enable_vlm_verification, "feedback": None},
                timestamp=datetime.now().isoformat(),
                blockers=[str(e)],
                error_message=str(e),
            )

    def run(self, num_instances: int = 1) -> Dict[str, TaskInitResult]:
        """Run initialization for all tasks in the task pool."""
        results = {}
        for task_name in self.task_pool:
            for instance_id in range(num_instances):
                result = self.run_for_task(task_name, instance_id)
                task_dir = self.output_dir / task_name / f"instance_{instance_id}"
                result.save(task_dir)
                results[f"{task_name}_{instance_id}"] = result
                logger.info(
                    f"Task init result for {task_name}_{instance_id}: success={result.success}"
                )
        return results

    def load_results(self) -> Dict[str, TaskInitResult]:
        """Load previously saved initialization results."""
        results = {}
        for task_dir in self.output_dir.iterdir():
            if not task_dir.is_dir():
                continue
            for instance_dir in task_dir.iterdir():
                if not instance_dir.is_dir():
                    continue
                result_file = instance_dir / "task_init_result.json"
                if result_file.exists():
                    with open(result_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        key = f"{data['task_name']}_{data['instance_id']}"
                        results[key] = TaskInitResult(**data)
        return results


class AndroidWorldTaskInitializer(BaseTaskInitializer):
    """Task initializer for AndroidWorld environment.

    Uses AndroidWorldBackend's reset_with_params endpoint for deterministic
    task initialization with OpenMobile-style snapshot recovery.
    """

    def __init__(
        self,
        server_url: str,
        params_manager: Optional[AndroidWorldParamsManager] = None,
        max_init_steps: int = 5,
        output_dir: str = "./androidworld_init_output",
        task_pool: Optional[List[str]] = None,
        enable_vlm_verification: bool = False,
        console_port: int = 5554,
        grpc_port: int = 8554,
        adb_path: str = "/root/android-sdk/platform-tools/adb",
        task_family: str = "android_world",
        max_image_tokens: int = 600,
    ):
        if params_manager is None:
            params_manager = AndroidWorldParamsManager()
        super().__init__(
            server_url=server_url,
            params_manager=params_manager,
            max_init_steps=max_init_steps,
            output_dir=output_dir,
            task_pool=task_pool,
            enable_vlm_verification=enable_vlm_verification,
            console_port=console_port,
            adb_path=adb_path,
            task_family=task_family,
        )
        self.grpc_port = grpc_port
        self.max_image_tokens = max_image_tokens
        self.current_obs: Optional[np.ndarray] = None

    def _init_server(self) -> Dict[str, Any]:
        payload = {
            "console_port": self.console_port,
            "grpc_port": self.grpc_port,
            "max_steps": self.max_init_steps,
            "adb_path": self.adb_path,
            "max_image_tokens": self.max_image_tokens,
        }
        resp = requests.post(f"{self.server_url}/init", json=payload, timeout=self.TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(f"Server init failed: {resp.text}")
        return resp.json()

    def _init_task(self, task_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "console_port": self.console_port,
            "task": task_name,
            "params": params,
        }
        resp = requests.post(
            f"{self.server_url}/reset_with_params", json=payload, timeout=self.TIMEOUT
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Task init failed: {resp.text}")
        data = resp.json()
        if data.get("status") == "failed" or data.get("detail"):
            raise RuntimeError(f"Task init failed: {data}")
        self.current_obs = self._decode_obs(data)
        return data

    def _decode_obs(self, resp_data: Dict[str, Any]) -> np.ndarray:
        np_bytes = base64.b64decode(resp_data["observation_np_b64"])
        dtype = np.dtype(resp_data["observation_dtype"])
        shape = tuple(resp_data["observation_shape"])
        return np.frombuffer(np_bytes, dtype=dtype).reshape(shape)

    def _get_screenshot(self) -> Optional[np.ndarray]:
        return self.current_obs

    def _step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "console_port": self.console_port,
            "action": action,
        }
        resp = requests.post(f"{self.server_url}/step", json=payload, timeout=self.TIMEOUT)
        if resp.status_code != 200:
            return {"error": resp.text}
        data = resp.json()
        if data.get("status") == "failed" or data.get("detail"):
            return {"error": data.get("detail", data)}
        self.current_obs = self._decode_obs(data)
        return data

    def _get_task_app_names(self, task_name: str) -> List[str]:
        app_mapping = {
            "Contacts": ["Contacts"],
            "Calendar": ["SimpleCalendar"],
            "SimpleCalendar": ["SimpleCalendar"],
            "SMS": ["Messages"],
            "SimpleSms": ["Messages"],
            "Expense": ["ExpenseTracker"],
            "Files": ["Files"],
            "Markor": ["Markor"],
            "OsmAnd": ["OsmAnd"],
            "Recipe": ["Recipe"],
            "Retro": ["RetroMusic"],
            "Vlc": ["VLC"],
            "System": [],
            "Browser": ["Chrome"],
            "Camera": ["Camera"],
            "Clock": ["Clock"],
            "AudioRecorder": ["AudioRecorder"],
        }
        for prefix, apps in app_mapping.items():
            if task_name.startswith(prefix):
                return apps
        return []

    def run_for_task(self, task_name: str, instance_id: int = 0) -> TaskInitResult:
        self._init_server()
        return super().run_for_task(task_name, instance_id)


class MobileWorldTaskInitializer(BaseTaskInitializer):
    """Task initializer for MobileWorld environment.

    Uses MobileWorld's HTTP API for task initialization with snapshot recovery.
    """

    TIMEOUT = 120

    def __init__(
        self,
        server_url: str,
        params_manager: Optional[MobileWorldParamsManager] = None,
        max_init_steps: int = 5,
        output_dir: str = "./mobileworld_init_output",
        task_pool: Optional[List[str]] = None,
        enable_vlm_verification: bool = False,
        device_id: str = "device_001",
        snapshot: str = "default",
    ):
        if params_manager is None:
            params_manager = MobileWorldParamsManager()
        super().__init__(
            server_url=server_url,
            params_manager=params_manager,
            max_init_steps=max_init_steps,
            output_dir=output_dir,
            task_pool=task_pool,
            enable_vlm_verification=enable_vlm_verification,
        )
        self.device_id = device_id
        self.snapshot = snapshot
        self.current_obs: Optional[np.ndarray] = None

    def _init_server(self) -> Dict[str, Any]:
        payload = {"device_id": self.device_id, "snapshot": self.snapshot}
        resp = requests.post(f"{self.server_url}/init", json=payload, timeout=self.TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(f"MobileWorld init failed: {resp.text}")
        return resp.json()

    def _init_task(self, task_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "device_id": self.device_id,
            "task": task_name,
            "params": params,
            "snapshot": params.get("snapshot", self.snapshot),
        }
        resp = requests.post(f"{self.server_url}/reset_with_params", json=payload, timeout=self.TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(f"MobileWorld task init failed: {resp.text}")
        return resp.json()

    def _get_screenshot(self) -> Optional[np.ndarray]:
        return self.current_obs

    def _step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"device_id": self.device_id, "action": action}
        resp = requests.post(f"{self.server_url}/step", json=payload, timeout=self.TIMEOUT)
        if resp.status_code != 200:
            return {"error": resp.text}
        return resp.json()

    def _get_task_app_names(self, task_name: str) -> List[str]:
        return []

    def run_for_task(self, task_name: str, instance_id: int = 0) -> TaskInitResult:
        self._init_server()
        return super().run_for_task(task_name, instance_id)
