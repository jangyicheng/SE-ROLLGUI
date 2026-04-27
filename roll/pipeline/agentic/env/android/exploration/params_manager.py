"""
Params Persistence Management for AndroidWorld/MobileWorld Exploration.

Implements deterministic params generation and persistence using the same
approach as OpenMobile: sha256-based seed derivation and pickle storage.

This enables reproducible task initialization for the self-evolving loop.
"""

import hashlib
import json
import pickle
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class TaskParams:
    """Container for task parameters."""

    task_name: str
    instance_id: int
    seed: int
    params: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "instance_id": self.instance_id,
            "seed": self.seed,
            "params": self.params,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: Path) -> "TaskParams":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(**data)


class BaseParamsManager:
    """Base class for params management."""

    DEFAULT_TASK_RANDOM_SEED = 42

    def __init__(
        self,
        params_dir: str = "./exploration_output/params",
        task_random_seed: int = DEFAULT_TASK_RANDOM_SEED,
    ):
        self.params_dir = Path(params_dir)
        self.params_dir.mkdir(parents=True, exist_ok=True)
        self.task_random_seed = task_random_seed

    def _derive_instance_seed(self, task_name: str, instance_id: int) -> int:
        """Derive deterministic seed from task_name and instance_id using sha256."""
        unique_seed_str = f"{self.task_random_seed}_{task_name}_{instance_id}"
        seed = int(hashlib.sha256(unique_seed_str.encode()).hexdigest(), 16) % (2**32)
        return seed

    def generate_params(self, task_name: str, instance_id: int = 0) -> Dict[str, Any]:
        """Generate deterministic params for a task instance."""
        raise NotImplementedError("Subclasses must implement generate_params")

    def save_params(
        self, task_name: str, params: Dict[str, Any], instance_id: int = 0
    ) -> str:
        """Save params to pickle file."""
        seed = params.get("seed", self._derive_instance_seed(task_name, instance_id))
        params_path = self.params_dir / f"{task_name}_{instance_id}_params.pkl"
        task_params = TaskParams(
            task_name=task_name, instance_id=instance_id, seed=seed, params=params
        )
        task_params.save(params_path)
        logger.info(f"Saved params for {task_name}_{instance_id} to {params_path}")
        return str(params_path)

    def load_params(self, task_name: str, instance_id: int = 0) -> Dict[str, Any]:
        """Load saved params from pickle file."""
        params_path = self.params_dir / f"{task_name}_{instance_id}_params.pkl"
        if not params_path.exists():
            raise FileNotFoundError(f"Params file not found: {params_path}")
        task_params = TaskParams.load(params_path)
        return task_params.to_dict()

    def params_exists(self, task_name: str, instance_id: int = 0) -> bool:
        """Check if params file exists."""
        params_path = self.params_dir / f"{task_name}_{instance_id}_params.pkl"
        return params_path.exists()

    def build_params_index(self) -> Dict[str, Dict[str, Any]]:
        """Build an index of all saved params files."""
        index = {}
        for pkl_file in self.params_dir.glob("*_params.pkl"):
            stem = pkl_file.stem
            parts = stem.rsplit("_params", 1)
            if len(parts) < 2:
                continue
            task_instance = parts[0]
            try:
                task_name, instance_id_str = task_instance.rsplit("_", 1)
                instance_id = int(instance_id_str)
            except ValueError:
                continue
            try:
                params = self.load_params(task_name, instance_id)
                index[f"{task_name}_{instance_id}"] = {
                    "params_path": str(pkl_file),
                    "seed": params.get("seed"),
                    "instance_id": instance_id,
                    "task_name": task_name,
                    "params": params.get("params", {}),
                }
            except Exception as e:
                logger.warning(f"Failed to load params from {pkl_file}: {e}")
        return index

    def generate_and_save(
        self, task_name: str, instance_id: int = 0
    ) -> Tuple[Dict[str, Any], str]:
        """Generate params and save to file in one call."""
        params = self.generate_params(task_name, instance_id)
        params_path = self.save_params(task_name, params, instance_id)
        return params, params_path


class AndroidWorldParamsManager(BaseParamsManager):
    """Params manager for AndroidWorld tasks.

    AndroidWorld tasks use deterministic seed derivation and task-specific
    param generation following OpenMobile's approach.
    """

    # Maps task prefixes to their expected param fields
    TASK_PARAM_SCHEMAS = {
        "Contacts": ["name", "phone"],
        "Calendar": ["title", "date", "time", "duration"],
        "SMS": ["recipient", "message"],
        "Email": ["recipient", "subject", "body"],
        "Expense": ["amount", "category", "description"],
        "Files": ["filename", "content"],
        "Markor": ["note_title", "note_content"],
        "OsmAnd": ["location", "name"],
        "Recipe": ["recipe_name", "ingredients"],
        "Retro": ["playlist_name", "songs"],
        "Vlc": ["video_name", "playlist_name"],
        "SimpleSms": ["recipient", "message"],
        "SimpleCalendar": ["event_name", "date", "time"],
        "System": ["setting", "value"],
        "Browser": ["search_query"],
        "Camera": ["mode", "flash"],
        "Clock": ["alarm_time", "timer_duration"],
        "AudioRecorder": ["filename", "duration"],
    }

    def __init__(
        self,
        params_dir: str = "./exploration_output/params/androidworld",
        task_random_seed: int = DEFAULT_TASK_RANDOM_SEED,
    ):
        super().__init__(params_dir=params_dir, task_random_seed=task_random_seed)

    def _get_task_prefix(self, task_name: str) -> str:
        """Extract the app prefix from task name."""
        for prefix in self.TASK_PARAM_SCHEMAS.keys():
            if task_name.startswith(prefix):
                return prefix
        return "Generic"

    def _generate_generic_params(self, seed: int) -> Dict[str, Any]:
        """Generate generic params for unknown task types."""
        random.seed(seed)
        return {
            "random_int": random.randint(0, 1000),
            "random_float": random.uniform(0, 100),
            "random_string": f"test_{random.randint(1, 1000)}",
        }

    def generate_contact_params(self, seed: int) -> Dict[str, Any]:
        """Generate params for Contacts tasks."""
        random.seed(seed)
        first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis"]
        return {
            "name": f"{random.choice(first_names)} {random.choice(last_names)}",
            "phone": f"+1{random.randint(1000000000, 9999999999)}",
            "email": f"user{random.randint(1, 100)}@example.com",
        }

    def generate_calendar_params(self, seed: int) -> Dict[str, Any]:
        """Generate params for Calendar tasks."""
        random.seed(seed)
        titles = ["Team Meeting", "Doctor Appointment", "Gym Session", "Birthday Party", "Conference Call"]
        return {
            "title": random.choice(titles),
            "date": f"2026-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            "time": f"{random.randint(8, 18):02d}:{random.choice([0, 15, 30, 45]):02d}",
            "duration": random.choice([30, 60, 90, 120]),
            "location": f"Room {random.randint(1, 10)}",
        }

    def generate_sms_params(self, seed: int) -> Dict[str, Any]:
        """Generate params for SMS tasks."""
        random.seed(seed)
        messages = [
            "Hello, how are you?",
            "Meeting at 3pm",
            "Can you call me?",
            "See you tomorrow!",
            "Thanks for your help!",
        ]
        return {
            "recipient": f"+1{random.randint(1000000000, 9999999999)}",
            "message": random.choice(messages),
        }

    def generate_expense_params(self, seed: int) -> Dict[str, Any]:
        """Generate params for Expense tasks."""
        random.seed(seed)
        categories = ["Food", "Transport", "Shopping", "Entertainment", "Utilities", "Healthcare"]
        return {
            "amount": round(random.uniform(5.0, 500.0), 2),
            "category": random.choice(categories),
            "description": f"Expense item {random.randint(1, 100)}",
            "currency": "USD",
        }

    def generate_files_params(self, seed: int) -> Dict[str, Any]:
        """Generate params for Files tasks."""
        random.seed(seed)
        extensions = ["txt", "pdf", "doc", "jpg", "png", "mp3"]
        return {
            "filename": f"test_file_{random.randint(1, 1000)}.{random.choice(extensions)}",
            "content": f"Test content {random.randint(1, 1000)}",
            "directory": f"/sdcard/Download/test_{random.randint(1, 10)}",
        }

    def generate_markor_params(self, seed: int) -> Dict[str, Any]:
        """Generate params for Markor/Notes tasks."""
        random.seed(seed)
        return {
            "note_title": f"Note {random.randint(1, 100)}",
            "note_content": f"This is test content {random.randint(1, 1000)}",
            "folder": random.choice(["Notes", "Todo", "Ideas"]),
        }

    def generate_osmand_params(self, seed: int) -> Dict[str, Any]:
        """Generate params for OsmAnd tasks."""
        random.seed(seed)
        locations = ["Central Park", "Times Square", "Airport", "Hospital", "School"]
        return {
            "location": random.choice(locations),
            "name": f"Marker {random.randint(1, 100)}",
            "latitude": random.uniform(40.0, 41.0),
            "longitude": random.uniform(-74.0, -73.0),
        }

    def generate_recipe_params(self, seed: int) -> Dict[str, Any]:
        """Generate params for Recipe tasks."""
        random.seed(seed)
        recipes = ["Pasta Carbonara", "Chicken Curry", "Caesar Salad", "Beef Stew", "Vegetable Soup"]
        return {
            "recipe_name": random.choice(recipes),
            "ingredients": [f"Ingredient {i}" for i in range(random.randint(3, 8))],
            "servings": random.randint(2, 6),
        }

    def generate_system_params(self, seed: int) -> Dict[str, Any]:
        """Generate params for System settings tasks."""
        random.seed(seed)
        settings = [
            ("wifi", "on"),
            ("wifi", "off"),
            ("bluetooth", "on"),
            ("bluetooth", "off"),
            ("brightness", "max"),
            ("brightness", "min"),
        ]
        setting = random.choice(settings)
        return {
            "setting": setting[0],
            "value": setting[1],
        }

    def generate_browser_params(self, seed: int) -> Dict[str, Any]:
        """Generate params for Browser tasks."""
        random.seed(seed)
        queries = [
            "weather today",
            "news headlines",
            "how to cook pasta",
            "nearby restaurants",
            "latest movies",
        ]
        return {
            "search_query": random.choice(queries),
        }

    def generate_generic_app_params(self, seed: int, prefix: str) -> Dict[str, Any]:
        """Generate params based on task prefix."""
        generators = {
            "Contacts": self.generate_contact_params,
            "Calendar": self.generate_calendar_params,
            "SimpleCalendar": self.generate_calendar_params,
            "SMS": self.generate_sms_params,
            "SimpleSms": self.generate_sms_params,
            "Expense": self.generate_expense_params,
            "Files": self.generate_files_params,
            "Markor": self.generate_markor_params,
            "OsmAnd": self.generate_osmand_params,
            "Recipe": self.generate_recipe_params,
            "System": self.generate_system_params,
            "Browser": self.generate_browser_params,
        }
        generator = generators.get(prefix, self._generate_generic_params)
        return generator(seed)

    def generate_params(self, task_name: str, instance_id: int = 0) -> Dict[str, Any]:
        """Generate deterministic params for an AndroidWorld task."""
        seed = self._derive_instance_seed(task_name, instance_id)
        prefix = self._get_task_prefix(task_name)
        params = self.generate_generic_app_params(seed, prefix)
        params["seed"] = seed
        params["instance_id"] = instance_id
        params["task_name"] = task_name
        logger.info(f"Generated params for {task_name}_{instance_id}: seed={seed}")
        return params


class MobileWorldParamsManager(BaseParamsManager):
    """Params manager for MobileWorld tasks.

    MobileWorld uses a different task definition format with snapshot strings
    and arbitrary param fields.
    """

    def __init__(
        self,
        params_dir: str = "./exploration_output/params/mobileworld",
        task_random_seed: int = DEFAULT_TASK_RANDOM_SEED,
    ):
        super().__init__(params_dir=params_dir, task_random_seed=task_random_seed)

    def generate_params(self, task_name: str, instance_id: int = 0) -> Dict[str, Any]:
        """Generate deterministic params for a MobileWorld task."""
        seed = self._derive_instance_seed(task_name, instance_id)
        random.seed(seed)
        return {
            "seed": seed,
            "instance_id": instance_id,
            "task_name": task_name,
            "snapshot": "default",
            "random_int": random.randint(0, 1000),
            "random_string": f"param_{random.randint(1, 1000)}",
        }
