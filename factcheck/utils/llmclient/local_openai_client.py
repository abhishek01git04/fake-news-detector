import time
import openai
from openai import OpenAI
from .base import BaseClient


class LocalOpenAIClient(BaseClient):
    """Support Local host LLM chatbot with OpenAI API.
    see https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md for example usage.
    """

    def __init__(
        self,
        model: str = "",
        api_config: dict = None,
        max_requests_per_minute=200,
        request_window=60,
    ):
        super().__init__(model, api_config, max_requests_per_minute, request_window)

        openai.api_key = api_config["LOCAL_API_KEY"]
        openai.base_url = api_config["LOCAL_API_URL"]

    def _call(self, messages: str, **kwargs):
        seed = kwargs.get("seed", 42)  # default seed is 42
        assert type(seed) is int, "Seed must be an integer."

        # Free-tier providers (Gemini, Groq) rate-limit aggressively: back off and retry on HTTP 429 / 503.
        max_attempts = 6
        for attempt in range(max_attempts):
            extra = {"seed": seed} if getattr(self, "_send_seed", True) else {}
            try:
                response = openai.chat.completions.create(
                    response_format={"type": "json_object"},
                    model=self.model,
                    messages=messages,
                    **extra,
                )
                break
            except openai.BadRequestError:
                # Some OpenAI-compatible servers (e.g. Gemini) reject `seed`; retry once without it.
                if extra:
                    self._send_seed = False
                    continue
                raise
            except (openai.RateLimitError, openai.InternalServerError) as e:
                if attempt == max_attempts - 1:
                    raise
                wait = min(60, 5 * 2**attempt)
                print(f"Provider busy ({type(e).__name__}), retrying in {wait}s...")
                time.sleep(wait)
            except openai.APIConnectionError as e:
                raise ConnectionError(
                    f"Cannot reach LLM server at {openai.base_url}. If using Ollama, make sure it is running."
                ) from e

        if response.usage is not None:
            self.usage.prompt_tokens += response.usage.prompt_tokens or 0
            self.usage.completion_tokens += response.usage.completion_tokens or 0
        r = response.choices[0].message.content
        return r

    def get_request_length(self, messages):
        # TODO: check if we should return the len(menages) instead
        return 1

    def construct_message_list(
        self,
        prompt_list: list[str],
        system_role: str = "You are a helpful assistant designed to output JSON.",
    ):
        messages_list = list()
        for prompt in prompt_list:
            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt},
            ]
            messages_list.append(messages)
        return messages_list
