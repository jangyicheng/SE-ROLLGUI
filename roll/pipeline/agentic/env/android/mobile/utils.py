from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
    OpenAI
)
import os
import backoff

class vllm_OpenaiEngine():
    def __init__(self, model, base_url):
        self.model = model
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)
        self.last_usage = {}

    @staticmethod
    def _extract_usage(response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }


    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
        max_tries=3,
        on_backoff=lambda details: print(f"Retrying in {details['wait']:0.1f} seconds due to {details['exception']}")
    )
    def generate(
        self,
        messages,
        max_new_tokens=4096,
        temperature=0,
        return_metadata=False,
        **kwargs,
    ):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        contents = [choice.message.content for choice in response.choices]
        usage = self._extract_usage(response)
        self.last_usage = usage

        if return_metadata:
            return {
                "contents": contents,
                "usage": usage,
                "model": self.model,
            }
        return contents
        

class OpenaiEngine():
    def __init__(
        self,
        rate_limit=-1,
        model=None,
        **kwargs,
    ) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.client = OpenAI(api_key=self.api_key)

        self.model = model  # Save the model parameter
        self.reasoning_models = ["o4-mini","gpt-5","gpt-5-mini"]
        self.last_usage = {}
        
    def log_error(self, details):
        print(f"Retrying in {details['wait']:0.1f} seconds due to {details['exception']}")

    @staticmethod
    def _extract_usage(response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
        max_tries=3,
        on_backoff=lambda details: print(f"Retrying in {details['wait']:0.1f} seconds due to {details['exception']}")
    )
    def generate(
        self,
        messages,
        max_new_tokens=4096,
        temperature=0,
        return_metadata=False,
        **kwargs,
    ):
        if self.model in self.reasoning_models:
            response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_new_tokens,
            **kwargs,
        )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        contents = [choice.message.content for choice in response.choices]
        usage = self._extract_usage(response)
        self.last_usage = usage

        if return_metadata:
            return {
                "contents": contents,
                "usage": usage,
                "model": self.model,
            }
        return contents
