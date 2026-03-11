import base64
import json
import subprocess
import time
from io import BytesIO

import numpy as np
from android_world import registry, suite_utils
from android_world.env import env_launcher, json_action
from gem import Env
from PIL import Image
from qwen_vl_utils import smart_resize
from rich import print

from roll.utils.logging import get_logger


logger = get_logger()


def rescale_coordinates(point, width, height):
    point = [round(point[0] / 999 * width), round(point[1] / 999 * height)]
    return point


def process_image(image: np.ndarray, max_image_tokens: int = 1000) -> tuple[str, np.ndarray]:
    """
    Process an image for Qwen VL models (thinking variant).
    Uses a tighter resize cap consistent with the thinking DUN agent.
    """
    image_2 = Image.fromarray(image)
    width, height = image_2.size

    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        factor=32,
        max_pixels=max_image_tokens * 32 * 32,  # image_tokens = h_bar * w_bar / (32*32) + 2
    )

    image_2 = image_2.resize((resized_width, resized_height))

    buffer = BytesIO()
    image_2.save(buffer, format="PNG")
    processed_bytes = buffer.getvalue()

    return base64.b64encode(processed_bytes).decode("utf-8"), np.array(image_2)


def convert_action_space(action: dict):
    """convert qwen3-vl action space to android_world action space."""
    converted_action = json_action.JSONAction(action_type=json_action.UNKNOWN)
    if action["arguments"]["action"] == "click":
        converted_action.action_type = json_action.CLICK
        converted_action.x = action["arguments"]["coordinate"][0]
        converted_action.y = action["arguments"]["coordinate"][1]
    elif action["arguments"]["action"] == "long_press":
        converted_action.action_type = json_action.LONG_PRESS
        converted_action.x = action["arguments"]["coordinate"][0]
        converted_action.y = action["arguments"]["coordinate"][1]
    elif action["arguments"]["action"] == "swipe":
        converted_action.action_type = json_action.SWIPE
        converted_action.x = action["arguments"]["coordinate"][0]
        converted_action.y = action["arguments"]["coordinate"][1]
        converted_action.x2 = action["arguments"]["coordinate2"][0]
        converted_action.y2 = action["arguments"]["coordinate2"][1]
        converted_action.direction = None
    elif action["arguments"]["action"] == "type":
        converted_action.action_type = json_action.INPUT_TEXT
        converted_action.text = action["arguments"]["text"]
    elif action["arguments"]["action"] == "answer":
        converted_action.action_type = json_action.ANSWER
        converted_action.text = action["arguments"]["text"]
    elif action["arguments"]["action"] == "open":
        converted_action.action_type = json_action.OPEN_APP
        converted_action.app_name = action["arguments"]["text"]
    elif action["arguments"]["action"] == "system_button":
        button = action["arguments"]["button"]
        if button == "Back":
            converted_action.action_type = json_action.NAVIGATE_BACK
        elif button == "Home":
            converted_action.action_type = json_action.NAVIGATE_HOME
        elif button == "Enter":
            converted_action.action_type = json_action.KEYBOARD_ENTER
    elif action["arguments"]["action"] == "wait":
        converted_action.action_type = json_action.WAIT
    elif action["arguments"]["action"] == "terminate":
        converted_action.action_type = json_action.STATUS
        if action["arguments"]["status"] == "success":
            converted_action.goal_status = "success"
        else:
            converted_action.goal_status = "infeasible"
    return converted_action


class AndroidEnv(Env):
    def __init__(
        self,
        adb_path: str = "/root/android-sdk/platform-tools/adb",
        console_ports: list[int] = [],
        grpc_ports: list[int] = [],
        task: str | None = None,
        task_family: str = "android_world",
        max_steps: int = 10,
        group_seed: int = 0,
        max_image_tokens: int = 600,
        **kwargs,
    ):
        self.env_id = kwargs.get("android_env_id", 0)
        self.adb_path = adb_path
        self.console_port = eval(console_ports)[self.env_id]
        self.grpc_port = eval(grpc_ports)[self.env_id]
        self.max_steps = max_steps
        self.group_seed = group_seed
        self.group_id = kwargs.get("android_group_id", 0)
        self.max_image_tokens = max_image_tokens

        self.env = env_launcher.load_and_setup_env(
            console_port=self.console_port,
            emulator_setup=False,
            adb_path=self.adb_path,
            grpc_port=self.grpc_port,
        )
        self.name = f"emulator-{self.console_port}"

        self.logical_screen_size = self.env.logical_screen_size
        self.orientation = self.env.orientation
        self.physical_frame_boundary = self.env.physical_frame_boundary
        self.task_registry = registry.TaskRegistry().get_registry(family=task_family)
        self.current_step = 0
        self.task = None
        self.first_render = True
        if task:
            task: list[str] = task.split(",")
            task: str = task[self.env_id % len(task)]
            logger.info(f"Group-{self.group_id}[Env-{self.env_id}]({self.name}): {task}")
            for name, task_type in self.task_registry.items():
                if name == task:
                    self.task_type = task_type
                    self.task = suite_utils._instantiate_task(task_type, seed=42, env=self.env)
                    break

    def step(self, action: str | json_action.JSONAction):
        ensure_emulator_running(self.console_port, self.grpc_port)
        self.current_step += 1
        extra_reward = 0.0
        try:
            if isinstance(action, str):
                # for qwen3-vl action space
                try:
                    action = json.loads(action.split("<tool_call>\n")[1].split("\n</tool_call>")[0])
                except Exception:
                    action = {"arguments": {"action": "wait"}}
                if action["arguments"]["action"] in ["click", "long_press"]:
                    action["arguments"]["coordinate"] = rescale_coordinates(
                        action["arguments"]["coordinate"],
                        self.raw_width,
                        self.raw_height,
                    )
                elif action["arguments"]["action"] == "swipe":
                    action["arguments"]["coordinate"] = rescale_coordinates(
                        action["arguments"]["coordinate"],
                        self.raw_width,
                        self.raw_height,
                    )
                    action["arguments"]["coordinate2"] = rescale_coordinates(
                        action["arguments"]["coordinate2"],
                        self.raw_width,
                        self.raw_height,
                    )
                action = convert_action_space(action)
            self.env.execute_action(action)
        except Exception:
            extra_reward = -0.1
            pass
        obs = self.render()
        is_success = False
        if self.task:
            is_success = self.task.is_successful(self.env)
        if action.action_type == json_action.STATUS or is_success > 0.5:
            terminate = True
        else:
            terminate = self.current_step >= self.max_steps
        return obs, 1.0 + extra_reward if is_success > 0.5 else 0.0 + extra_reward, terminate, None, {}

    def reset(self, *, go_home: bool = True, seed: int | None = None) -> tuple[np.ndarray, dict]:
        ensure_emulator_running(self.console_port, self.grpc_port)

        super().reset(seed=seed)
        self.current_step = 0
        self.first_render = True
        self.env.reset(go_home=go_home)
        self.env.hide_automation_ui()
        if self.task:
            self.task.tear_down(self.env)
            self.task = suite_utils._instantiate_task(self.task_type, seed=42, env=self.env)
            self.task.initialize_task(self.env)
            logger.info(f"Task: {self.task.goal}")
        obs = self.render()
        return obs, {}

    def render(self, mode="rgb_array") -> np.ndarray:
        ensure_emulator_running(self.console_port, self.grpc_port)

        state = self.env.get_state(wait_to_stabilize=True)
        raw = Image.fromarray(state.pixels)
        b64, obs = process_image(state.pixels.copy(), max_image_tokens=self.max_image_tokens)
        if self.first_render:
            processed_img = Image.fromarray(obs)
            self.raw_width, self.raw_height = raw.size
            self.processed_width, self.processed_height = processed_img.size
        return obs

    def close(self):
        if self.task:
            self.task.tear_down(self.env)
        self.env.close()


def ensure_emulator_running(console_port, grpc_port, timeout=1200):
    device_serial = f"emulator-{console_port}"

    # --- 1. 检查是否存活 ---
    is_alive = False
    try:
        output = subprocess.run(["adb", "devices"], capture_output=True, text=True).stdout
        if f"{device_serial}\tdevice" in output:
            is_alive = True
    except Exception:
        pass

    if is_alive:
        return True

    # --- 2. 启动模拟器 ---
    print(f"Starting {device_serial} in background...")
    cmd = [
        "emulator",
        "-avd",
        "pixel_6_api33_AndroidWorldAvd_emulator",
        "-no-window",
        "-no-audio",
        "-no-snapshot-save",
        "-read-only",
        "-grpc",
        str(grpc_port),
        "-port",
        str(console_port),
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # --- 3. 等待 Boot 完成 (·Polling) ---
    print(f"Waiting for {device_serial} to fully boot (Timeout: {timeout}s)...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            # -s 指定设备，检查 sys.boot_completed 属性是否为 1
            res = subprocess.run(
                ["adb", "-s", device_serial, "shell", "getprop", "sys.boot_completed"], capture_output=True, text=True
            )
            if res.stdout.strip() == "1":
                print(f"🎉 {device_serial} is ready!")
                return True
        except subprocess.SubprocessError:
            pass  # adb 可能还没连上，忽略错误继续重试

        time.sleep(2)  # 每 2 秒检查一次

    print(f"Timeout: Device did not boot within {timeout} seconds.")
    return False


if __name__ == "__main__":
    env = AndroidEnv(console_port=5554, grpc_port=8554)
    env.reset()
    state = env.render()
    env.close()
    pass
