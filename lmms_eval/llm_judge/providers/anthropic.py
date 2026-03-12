import os
import time
from typing import Dict, List, Optional

from loguru import logger as eval_logger

from ..base import ServerInterface
from ..protocol import Request, Response, ServerConfig


class AnthropicProvider(ServerInterface):
    """Anthropic API implementation of the Judge interface"""

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.base_url = os.getenv("ANTHROPIC_BASE_URL")

        try:
            from anthropic import Anthropic

            kwargs: Dict = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self.client = Anthropic(**kwargs)
            self.use_client = True
        except ImportError:
            eval_logger.warning(
                "Anthropic client not available. Install with: uv add anthropic"
            )
            self.use_client = False

    def is_available(self) -> bool:
        return bool(self.api_key) and self.use_client

    def evaluate(self, request: Request) -> Response:
        """Evaluate using Anthropic API"""
        if not self.is_available():
            raise ValueError(
                "Anthropic API not configured. "
                "Set ANTHROPIC_API_KEY and install anthropic."
            )

        config = request.config or self.config
        messages = self.prepare_messages(request)

        # Anthropic uses a separate system parameter instead of a
        # system message in the messages list.
        system: Optional[str] = None
        user_messages: List[Dict] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_messages.append(m)

        kwargs: Dict = {
            "model": config.model_name,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": user_messages,
        }
        if system:
            kwargs["system"] = system
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p

        for attempt in range(config.num_retries):
            try:
                response = self.client.messages.create(**kwargs)
                content = response.content[0].text
                model_used = response.model
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }

                return Response(
                    content=content.strip(),
                    model_used=model_used,
                    usage=usage,
                    raw_response=response,
                )

            except Exception as e:
                eval_logger.warning(
                    f"Attempt {attempt + 1}/{config.num_retries} failed: {e!s}"
                )
                if attempt < config.num_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    eval_logger.error(f"All {config.num_retries} attempts failed")
                    raise
