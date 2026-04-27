"""
Model client factory for exploration and curriculum generation.

Provides a unified interface for:
- OpenAI API (cloud)
- vLLM / OpenAI-compatible endpoints (local)
- HuggingFace transformers (local, CPU/GPU)

The base `BaseExplorer._query_model()` expects `model_client.generate(observation, instruction)`,
where `observation` is a numpy screenshot array and `instruction` is a string.
`ExplorerModelWrapper` bridges this to the chat-message-based OpenAI/vLLM API.
"""

from __future__ import annotations

import base64
import io
from typing import Any, List, Literal, Optional, Union

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# System prompt for the exploration agent
# ---------------------------------------------------------------------------

EXPLORATION_SYSTEM_PROMPT = """You are an expert at operating an Android phone through screenshots.

You must respond with a JSON object representing the next action to take.

Available action types:
- `click`: Tap at coordinate [x, y]
- `long_press`: Long press at coordinate [x, y] for specified duration
- `swipe`: Swipe from [x1, y1] to [x2, y2]
- `type`: Input text into the currently focused text field
- `wait`: Wait N seconds for UI to settle
- `navigate_back`: Press the back button
- `terminate`: End the exploration early (use only when fully explored)

Screen resolution: 999x999 pixels. Coordinates start at [0, 0] (top-left).

Always output a valid JSON object like:
{"action": "click", "coordinate": [300, 400]}
{"action": "type", "text": "hello"}
{"action": "terminate"}
"""


# ---------------------------------------------------------------------------
# ExplorerModelWrapper: bridges BaseExplorer to OpenAI/vLLM chat APIs
# ---------------------------------------------------------------------------

class ExplorerModelWrapper:
    """Wraps an OpenAI/vLLM chat client to match BaseExplorer's `generate` interface.

    BaseExplorer calls: `model_client.generate(observation, instruction)`
    This wrapper: converts screenshot + instruction → chat messages → API call → action JSON
    """

    def __init__(
        self,
        chat_client: Any,
        model_name: str = "unknown",
        temperature: float = 1.0,
        max_tokens: int = 256,
    ):
        self._client = chat_client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def model_name_attr(self) -> str:
        return self.model_name

    def generate(self, observation: Any, instruction: str) -> Any:
        """Compatible with BaseExplorer._query_model signature."""
        image_b64 = self._numpy_to_base64(observation)
        messages = [
            {"role": "system", "content": EXPLORATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        try:
            result = self._client.generate(
                messages=messages,
                temperature=self.temperature,
                max_new_tokens=self.max_tokens,
            )
            # _client.generate() returns List[str] by convention
            if isinstance(result, list):
                text = result[0]
            else:
                text = str(result)
            return self._parse_json_action(text)
        except Exception as e:
            # Fallback: return a safe wait action on error
            import json as _json

            try:
                return _json.loads(
                    '{"name": "mobile_use", "arguments": {"action": "wait", "time": 2}}'
                )
            except Exception:
                return {"name": "mobile_use", "arguments": {"action": "wait", "time": 2}}

    @staticmethod
    def _numpy_to_base64(observation: Any) -> str:
        if isinstance(observation, np.ndarray):
            img = Image.fromarray(observation)
        elif isinstance(observation, Image.Image):
            img = observation
        elif isinstance(observation, bytes):
            return base64.b64encode(observation).decode()
        else:
            img = Image.new("RGB", (999, 999), color=(128, 128, 128))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _parse_json_action(text: str) -> dict:
        import json as _json
        import re as _re

        # Strip markdown code fences
        text = text.strip()
        for marker in ["```json", "```"]:
            if text.startswith(marker):
                text = text[len(marker):]
        for marker in ["```", "`"]:
            if text.endswith(marker):
                text = text[: -len(marker)]
        text = text.strip()

        # Try direct JSON parse first
        try:
            parsed = _json.loads(text)
            if isinstance(parsed, dict):
                return _json_to_mobile_action(parsed)
        except Exception:
            pass

        # Fallback: extract first JSON object with regex
        match = _re.search(r"\{[^}]+\}", text, _re.DOTALL)
        if match:
            try:
                parsed = _json.loads(match.group())
                return _json_to_mobile_action(parsed)
            except Exception:
                pass

        # Last resort: return wait
        return {"name": "mobile_use", "arguments": {"action": "wait", "time": 2}}


def _json_to_mobile_action(parsed: dict) -> dict:
    """Normalize various JSON action formats to the mobile_use format."""
    if "name" in parsed and "arguments" in parsed:
        return parsed

    action = parsed.get("action", parsed.get("Action", ""))
    if action == "terminate":
        return {"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}

    args = {}
    if "coordinate" in parsed:
        args["coordinate"] = parsed["coordinate"]
    if "coordinates" in parsed:
        coords = parsed["coordinates"]
        if isinstance(coords, list) and len(coords) >= 2:
            args["coordinate"] = [coords[0], coords[1]]
            if len(coords) >= 4:
                args["coordinate2"] = [coords[2], coords[3]]
    if "text" in parsed:
        args["text"] = parsed["text"]
    if "time" in parsed:
        args["time"] = parsed["time"]
    if "button" in parsed:
        args["button"] = parsed["button"]

    return {"name": "mobile_use", "arguments": {"action": action, **args}}


# ---------------------------------------------------------------------------
# VLMModelFactory: create model clients from config
# ---------------------------------------------------------------------------

BACKENDS: List[str] = ["openai", "vllm", "huggingface"]


class VLMModelFactory:
    """Factory for creating VLM model clients.

    Supports three backends:
    - openai  : Cloud OpenAI API (or any OpenAI-compatible cloud endpoint)
    - vllm     : Local vLLM server with OpenAI-compatible API
    - huggingface: HuggingFace transformers with vLLM or TGI backend
    """

    @staticmethod
    def create(
        backend: Literal["openai", "vllm", "huggingface"],
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
        chat_client_extra_kwargs: Optional[dict] = None,
    ) -> ExplorerModelWrapper:
        """Create an ExplorerModelWrapper with the specified backend.

        Args:
            backend: One of "openai", "vllm", "huggingface"
            model_name: Model name (e.g. "gpt-4o", "Qwen/Qwen2.5-VL-7B-Instruct")
            base_url: Base URL for the API endpoint.
                       - openai: "https://api.openai.com/v1" (default)
                       - vllm:   e.g. "http://localhost:8000/v1"
                       - huggingface: e.g. "http://localhost:8080/v1"
            api_key: API key. If None, reads from OPENAI_API_KEY env var.
                     For vLLM/HuggingFace local servers, set to "EMPTY" or "no-key".
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            chat_client_extra_kwargs: Extra kwargs passed to the underlying chat client

        Returns:
            ExplorerModelWrapper ready to use with BaseExplorer
        """
        import os as _os

        chat_client_extra_kwargs = chat_client_extra_kwargs or {}

        if backend == "vllm":
            chat_client = _create_vllm_client(model_name, base_url, api_key, **chat_client_extra_kwargs)
        elif backend == "openai":
            chat_client = _create_openai_client(model_name, base_url, api_key, **chat_client_extra_kwargs)
        elif backend == "huggingface":
            chat_client = _create_huggingface_client(model_name, base_url, api_key, **chat_client_extra_kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose from: {BACKENDS}")

        return ExplorerModelWrapper(
            chat_client=chat_client,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @staticmethod
    def from_url(
        base_url: str,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 256,
    ) -> ExplorerModelWrapper:
        """Infer backend from URL pattern and create a client.

        Auto-detection rules:
        - localhost:8000 → vLLM
        - api.openai.com → OpenAI
        - localhost:8080 → HuggingFace TGI
        - Otherwise → OpenAI (default)
        """
        import os as _os

        if model_name is None:
            model_name = "auto"

        url_lower = base_url.lower()
        if "localhost:8000" in url_lower or "localhost:8080" in url_lower:
            backend = "vllm"
        elif "api.openai.com" in url_lower:
            backend = "openai"
        else:
            backend = "openai"

        if api_key is None:
            api_key = _os.getenv("OPENAI_API_KEY", "EMPTY")

        return VLMModelFactory.create(
            backend=backend,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ---------------------------------------------------------------------------
# Backend-specific client factories
# ---------------------------------------------------------------------------

def _create_vllm_client(
    model_name: str,
    base_url: Optional[str],
    api_key: Optional[str],
    **kwargs,
) -> Any:
    """Create a vLLM (OpenAI-compatible) chat client."""
    from roll.pipeline.agentic.env.android.mobile.utils import vllm_OpenaiEngine

    if base_url is None:
        base_url = "http://localhost:8000/v1"
    if api_key is None:
        api_key = "EMPTY"

    return vllm_OpenaiEngine(model=model_name, base_url=base_url)


def _create_openai_client(
    model_name: str,
    base_url: Optional[str],
    api_key: Optional[str],
    **kwargs,
) -> Any:
    """Create an OpenAI cloud API chat client."""
    import os as _os
    from roll.pipeline.agentic.env.android.mobile.utils import OpenaiEngine

    if api_key is None:
        api_key = _os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it explicitly or pass api_key='your-key'."
            )

    if base_url:
        # Custom OpenAI-compatible endpoint
        from openai import OpenAI as _OpenAI

        return _OpenAI(api_key=api_key, base_url=base_url)
    else:
        return OpenaiEngine(model=model_name)


def _create_huggingface_client(
    model_name: str,
    base_url: Optional[str],
    api_key: Optional[str],
    **kwargs,
) -> Any:
    """Create a HuggingFace TGI/vLLM chat client.

    Uses the same vllm_OpenaiEngine path since TGI exposes OpenAI-compatible API.
    """
    import os as _os

    if base_url is None:
        base_url = "http://localhost:8080/v1"
    if api_key is None:
        api_key = _os.getenv("HF_TOKEN", "EMPTY")

    from roll.pipeline.agentic.env.android.mobile.utils import vllm_OpenaiEngine

    return vllm_OpenaiEngine(model=model_name, base_url=base_url)
