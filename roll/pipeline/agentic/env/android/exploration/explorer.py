"""
AndroidWorld and MobileWorld Free Exploration Module.

Performs unconstrained free exploration of mobile GUI environments,
recording action trajectories and discovered apps for curriculum generation.
"""

import base64
import hashlib
import json
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
from PIL import Image

from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class ExplorationResult:
    """Result of a single exploration episode."""

    exploration_id: str
    timestamp: str
    environment: str
    model: str
    max_steps: int
    actual_steps: int
    discovered_apps: List[str]
    discovered_action_types: List[str]
    trajectory_file: str
    trajectory_dir: str
    screenshots_dir: str
    init_screenshot: Optional[str] = None
    final_screenshot: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / f"{self.exploration_id}_result.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return result_path


@dataclass
class ExplorationStep:
    """Single step in an exploration trajectory."""

    step: int
    action: Union[str, Dict]
    action_type: str
    observation_b64: str
    screenshot_path: Optional[str] = None
    current_app: Optional[str] = None
    ui_elements: Optional[List[str]] = None
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class BaseExplorer:
    """Base class for mobile environment exploration."""

    DEFAULT_EXPLORATION_INSTRUCTION = (
        "You have full access to this Android phone. Your goal is to explore the most "
        "common and essential features of the current app and the entire system as "
        "thoroughly as possible. Explore different screens, try various UI elements "
        "(buttons, menus, inputs, toggles), and discover what actions are available. "
        "Screenshots will be captured at each step. Do NOT say 'done' or 'finish' "
        "until you have explored at least 30 steps. Try to cover: app launch, settings "
        "navigation, content creation, search, and file/media operations if available."
    )

    def __init__(
        self,
        server_url: str,
        model_client: Any = None,
        model_type: str = "gui_owl",
        max_steps: int = 50,
        output_dir: str = "./androidworld_exploration_output",
        exploration_id: Optional[str] = None,
        log_trajectory: bool = True,
        save_screenshots: bool = True,
        console_port: Optional[int] = None,
        grpc_port: Optional[int] = None,
        adb_path: str = "/root/android-sdk/platform-tools/adb",
        task_family: str = "android_world",
        seed: int = 42,
        instruction: Optional[str] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.model_client = model_client
        self.model_type = model_type
        self.max_steps = max_steps
        self.output_dir = Path(output_dir)
        self.exploration_id = exploration_id or f"exp_{self.environment_name}_{uuid.uuid4().hex[:8]}"
        self.log_trajectory = log_trajectory
        self.save_screenshots = save_screenshots
        self.console_port = console_port
        self.grpc_port = grpc_port
        self.adb_path = adb_path
        self.task_family = task_family
        self.seed = seed
        self.instruction = instruction or self.DEFAULT_EXPLORATION_INSTRUCTION

        self.exploration_dir = self.output_dir / self.exploration_id
        self.screenshots_dir = self.exploration_dir / "screenshots"
        self.trajectory_dir = self.exploration_dir / "trajectory"

        if self.save_screenshots:
            self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        if self.log_trajectory:
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)

        self.trajectory: List[ExplorationStep] = []
        self.discovered_apps: List[str] = []
        self.discovered_action_types: List[str] = []

    @property
    def environment_name(self) -> str:
        raise NotImplementedError("Subclasses must implement environment_name")

    def _init_server(self) -> Dict[str, Any]:
        """Initialize the server/environment."""
        raise NotImplementedError("Subclasses must implement _init_server")

    def _reset(self, go_home: bool = True) -> Dict[str, Any]:
        """Reset the environment for exploration."""
        raise NotImplementedError("Subclasses must implement _reset")

    def _step(self, action: Union[str, Dict]) -> Dict[str, Any]:
        """Execute a single step in the environment."""
        raise NotImplementedError("Subclasses must implement _step")

    def _get_screenshot(self) -> np.ndarray:
        """Get current screenshot as numpy array."""
        raise NotImplementedError("Subclasses must implement _get_screenshot")

    def _get_current_app(self) -> Optional[str]:
        """Get the currently focused app name."""
        raise NotImplementedError("Subclasses must implement _get_current_app")

    def _parse_action(self, action: Any) -> str:
        """Parse action to determine action type."""
        if isinstance(action, str):
            if "<tool_call>" in action:
                return "xml_action"
            return "text"
        elif isinstance(action, dict):
            args = action.get("arguments", {})
            action_name = args.get("action", "unknown")
            return action_name
        return "unknown"

    def _query_model(self, observation: Any, instruction: str) -> Union[str, Dict]:
        """Query the VLM model for next action."""
        if self.model_client is None:
            return self._random_action()
        try:
            response = self.model_client.generate(observation, instruction)
            return response
        except Exception as e:
            logger.warning(f"Model query failed, using random action: {e}")
            return self._random_action()

    def _random_action(self) -> Dict[str, Any]:
        """Generate a random exploration action."""
        actions = [
            {"name": "mobile_use", "arguments": {"action": "click", "coordinate": [300, 400]}},
            {"name": "mobile_use", "arguments": {"action": "swipe", "coordinate": [300, 600], "coordinate2": [300, 200]}},
            {"name": "mobile_use", "arguments": {"action": "long_press", "coordinate": [300, 400]}},
            {"name": "mobile_use", "arguments": {"action": "type", "text": "test"}},
            {"name": "mobile_use", "arguments": {"action": "wait"}},
            {"name": "mobile_use", "arguments": {"action": "navigate_back"}},
        ]
        return random.choice(actions)

    def _format_action_for_model(self, action: Union[str, Dict]) -> str:
        """Format action for model input."""
        if isinstance(action, dict):
            return json.dumps(action, ensure_ascii=False)
        return action

    def _extract_action_type(self, action: Union[str, Dict]) -> str:
        """Extract the type of action performed."""
        if isinstance(action, dict):
            args = action.get("arguments", {})
            return args.get("action", "unknown")
        return "text"

    def run(self) -> ExplorationResult:
        """Run the full exploration episode."""
        start_time = time.time()
        try:
            self._init_server()
            init_obs = self._reset(go_home=True)
            init_screenshot_path = None
            if self.save_screenshots:
                init_screenshot = self._get_screenshot()
                init_screenshot_path = str(self.screenshots_dir / "step_000_init.png")
                Image.fromarray(init_screenshot).save(init_screenshot_path)

            current_app = self._get_current_app()
            if current_app and current_app not in self.discovered_apps:
                self.discovered_apps.append(current_app)

            for step_idx in range(self.max_steps):
                obs = self._get_screenshot()
                action = self._query_model(obs, self.instruction)
                formatted_action = self._format_action_for_model(action)
                action_type = self._extract_action_type(formatted_action)

                if action_type not in self.discovered_action_types:
                    self.discovered_action_types.append(action_type)

                step_result = self._step(formatted_action)

                screenshot_path = None
                if self.save_screenshots:
                    screenshot = self._get_screenshot()
                    screenshot_path = str(self.screenshots_dir / f"step_{step_idx:03d}.png")
                    Image.fromarray(screenshot).save(screenshot_path)

                current_app = self._get_current_app()
                if current_app and current_app not in self.discovered_apps:
                    self.discovered_apps.append(current_app)

                step_data = ExplorationStep(
                    step=step_idx,
                    action=formatted_action,
                    action_type=action_type,
                    observation_b64=base64.b64encode(obs.tobytes()).decode() if isinstance(obs, np.ndarray) else "",
                    screenshot_path=screenshot_path,
                    current_app=current_app,
                    timestamp=datetime.now().isoformat(),
                )
                self.trajectory.append(step_data)

                if step_result.get("terminate", False):
                    logger.info(f"Exploration terminated at step {step_idx}")
                    break

            final_screenshot_path = None
            if self.save_screenshots and self.trajectory:
                try:
                    final_obs = self._get_screenshot()
                    final_screenshot_path = str(self.screenshots_dir / "final.png")
                    Image.fromarray(final_obs).save(final_screenshot_path)
                except Exception:
                    pass

            return ExplorationResult(
                exploration_id=self.exploration_id,
                timestamp=datetime.now().isoformat(),
                environment=self.environment_name,
                model=getattr(self.model_client, "model_name", "unknown") if self.model_client else "random",
                max_steps=self.max_steps,
                actual_steps=len(self.trajectory),
                discovered_apps=self.discovered_apps,
                discovered_action_types=self.discovered_action_types,
                trajectory_file=str(self.trajectory_dir / "trajectory.jsonl"),
                trajectory_dir=str(self.trajectory_dir),
                screenshots_dir=str(self.screenshots_dir),
                init_screenshot=init_screenshot_path,
                final_screenshot=final_screenshot_path,
                success=True,
            )

        except Exception as e:
            logger.error(f"Exploration failed: {e}")
            return ExplorationResult(
                exploration_id=self.exploration_id,
                timestamp=datetime.now().isoformat(),
                environment=self.environment_name,
                model=getattr(self.model_client, "model_name", "unknown") if self.model_client else "random",
                max_steps=self.max_steps,
                actual_steps=len(self.trajectory),
                discovered_apps=self.discovered_apps,
                discovered_action_types=self.discovered_action_types,
                trajectory_file=str(self.trajectory_dir / "trajectory.jsonl"),
                trajectory_dir=str(self.trajectory_dir),
                screenshots_dir=str(self.screenshots_dir),
                success=False,
                error_message=str(e),
            )

    def save_trajectory(self) -> Path:
        """Save the exploration trajectory to disk."""
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = self.trajectory_dir / "trajectory.jsonl"
        with open(trajectory_path, "w", encoding="utf-8") as f:
            for step in self.trajectory:
                f.write(json.dumps(step.to_dict(), ensure_ascii=False) + "\n")
        return trajectory_path

    def save_result(self) -> Path:
        """Save the exploration result to disk."""
        result = ExplorationResult(
            exploration_id=self.exploration_id,
            timestamp=datetime.now().isoformat(),
            environment=self.environment_name,
            model=getattr(self.model_client, "model_name", "unknown") if self.model_client else "random",
            max_steps=self.max_steps,
            actual_steps=len(self.trajectory),
            discovered_apps=self.discovered_apps,
            discovered_action_types=self.discovered_action_types,
            trajectory_file=str(self.trajectory_dir / "trajectory.jsonl"),
            trajectory_dir=str(self.trajectory_dir),
            screenshots_dir=str(self.screenshots_dir),
            success=len(self.trajectory) > 0,
        )
        return result.save(self.exploration_dir)


class AndroidWorldExplorer(BaseExplorer):
    """AndroidWorld environment free exploration explorer.

    Reuses the HTTP API from remote_android.py but with exploration-specific
    configurations: longer max_steps, no reward judge, trajectory logging.
    """

    TIMEOUT = 240

    def __init__(
        self,
        server_url: str,
        model_client: Any = None,
        model_type: str = "gui_owl",
        max_steps: int = 50,
        output_dir: str = "./androidworld_exploration_output",
        exploration_id: Optional[str] = None,
        log_trajectory: bool = True,
        save_screenshots: bool = True,
        console_port: int = 5554,
        grpc_port: int = 8554,
        adb_path: str = "/root/android-sdk/platform-tools/adb",
        task_family: str = "android_world",
        seed: int = 42,
        instruction: Optional[str] = None,
        max_image_tokens: int = 600,
    ):
        super().__init__(
            server_url=server_url,
            model_client=model_client,
            model_type=model_type,
            max_steps=max_steps,
            output_dir=output_dir,
            exploration_id=exploration_id,
            log_trajectory=log_trajectory,
            save_screenshots=save_screenshots,
            console_port=console_port,
            grpc_port=grpc_port,
            adb_path=adb_path,
            task_family=task_family,
            seed=seed,
            instruction=instruction,
        )
        self.max_image_tokens = max_image_tokens
        self.current_obs: Optional[np.ndarray] = None

    @property
    def environment_name(self) -> str:
        return "AndroidWorld"

    def _init_server(self) -> Dict[str, Any]:
        payload = {
            "console_port": self.console_port,
            "grpc_port": self.grpc_port,
            "max_steps": self.max_steps,
            "adb_path": self.adb_path,
            "max_image_tokens": self.max_image_tokens,
        }
        resp = requests.post(f"{self.server_url}/init", json=payload, timeout=self.TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(f"Server init failed: {resp.text}")
        return resp.json()

    def _reset(self, go_home: bool = True) -> Dict[str, Any]:
        payload = {
            "console_port": self.console_port,
            "go_home": go_home,
            "task": "android_world_task",
            "task_family": self.task_family,
            "seed": self.seed,
        }
        resp = requests.post(f"{self.server_url}/reset", json=payload, timeout=self.TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(f"Reset failed: {resp.text}")
        data = resp.json()
        if data.get("status") == "failed" or data.get("detail"):
            raise RuntimeError(f"Reset failed: {data}")
        return data

    def _step(self, action: Union[str, Dict]) -> Dict[str, Any]:
        payload = {
            "console_port": self.console_port,
            "action": action,
        }
        resp = requests.post(f"{self.server_url}/step", json=payload, timeout=self.TIMEOUT)
        if resp.status_code != 200:
            return {"terminate": False, "error": resp.text}
        data = resp.json()
        if data.get("status") == "failed" or data.get("detail"):
            return {"terminate": True, "error": data.get("detail", data)}
        return data

    def _decode_obs(self, resp_data: Dict[str, Any]) -> np.ndarray:
        np_bytes = base64.b64decode(resp_data["observation_np_b64"])
        dtype = np.dtype(resp_data["observation_dtype"])
        shape = tuple(resp_data["observation_shape"])
        return np.frombuffer(np_bytes, dtype=dtype).reshape(shape)

    def _get_screenshot(self) -> np.ndarray:
        if self.current_obs is not None:
            return self.current_obs
        payload = {"console_port": self.console_port, "action": {"name": "mobile_use", "arguments": {"action": "wait"}}}
        resp = requests.post(f"{self.server_url}/step", json=payload, timeout=self.TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            self.current_obs = self._decode_obs(data)
            return self.current_obs
        raise RuntimeError("Failed to get screenshot")

    def _get_current_app(self) -> Optional[str]:
        if not self.trajectory:
            return None
        last_step = self.trajectory[-1]
        return last_step.current_app

    def _query_model(self, observation: Any, instruction: str) -> Union[str, Dict]:
        if self.model_client is None:
            return self._random_action()
        try:
            response = self.model_client.generate(observation, instruction)
            if isinstance(response, dict):
                return response
            return {"name": "mobile_use", "arguments": {"action": "wait"}}
        except Exception as e:
            logger.warning(f"Model query failed: {e}")
            return self._random_action()


class MobileWorldExplorer(BaseExplorer):
    """MobileWorld environment free exploration explorer.

    Uses MobileWorld's HTTP API for exploration.
    """

    TIMEOUT = 120

    def __init__(
        self,
        server_url: str,
        model_client: Any = None,
        model_type: str = "gui_owl",
        max_steps: int = 50,
        output_dir: str = "./mobileworld_exploration_output",
        exploration_id: Optional[str] = None,
        log_trajectory: bool = True,
        save_screenshots: bool = True,
        console_port: int = 5554,
        grpc_port: int = 8554,
        instruction: Optional[str] = None,
        snapshot: str = "default",
    ):
        super().__init__(
            server_url=server_url,
            model_client=model_client,
            model_type=model_type,
            max_steps=max_steps,
            output_dir=output_dir,
            exploration_id=exploration_id,
            log_trajectory=log_trajectory,
            save_screenshots=save_screenshots,
            console_port=console_port,
            grpc_port=grpc_port,
            adb_path="",
            task_family="mobile_world",
            seed=42,
            instruction=instruction,
        )
        self.snapshot = snapshot
        self.current_obs: Optional[np.ndarray] = None
        self.current_screenshot_b64: Optional[str] = None

    @property
    def environment_name(self) -> str:
        return "MobileWorld"

    def _init_server(self) -> Dict[str, Any]:
        payload = {
            "console_port": self.console_port,
            "grpc_port": self.grpc_port,
            "snapshot": self.snapshot,
        }
        resp = requests.post(f"{self.server_url}/init", json=payload, timeout=self.TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(f"MobileWorld init failed: {resp.text}")
        return resp.json()

    def _reset(self, go_home: bool = True) -> Dict[str, Any]:
        payload = {
            "console_port": self.console_port,
            "grpc_port": self.grpc_port,
            "snapshot": self.snapshot,
        }
        resp = requests.post(f"{self.server_url}/reset", json=payload, timeout=self.TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(f"MobileWorld reset failed: {resp.text}")
        return resp.json()

    def _step(self, action: Union[str, Dict]) -> Dict[str, Any]:
        payload = {
            "console_port": self.console_port,
            "grpc_port": self.grpc_port,
            "action": action,
        }
        resp = requests.post(f"{self.server_url}/step", json=payload, timeout=self.TIMEOUT)
        if resp.status_code != 200:
            return {"terminate": False, "error": resp.text}
        return resp.json()

    def _get_screenshot(self) -> np.ndarray:
        if self.current_obs is not None:
            return self.current_obs
        payload = {"console_port": self.console_port, "grpc_port": self.grpc_port, "action": {"name": "wait"}}
        resp = requests.post(f"{self.server_url}/step", json=payload, timeout=self.TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            if "observation_b64" in data:
                img_bytes = base64.b64decode(data["observation_b64"])
                img = Image.open(io.BytesIO(img_bytes))
                self.current_obs = np.array(img)
                return self.current_obs
        raise RuntimeError("Failed to get MobileWorld screenshot")

    def _get_current_app(self) -> Optional[str]:
        if not self.trajectory:
            return None
        return self.trajectory[-1].current_app


import io
