"""
providers/anthropic_provider.py

Implementación del proveedor Anthropic usando el SDK oficial `anthropic`
y la Messages API en modo streaming (client.messages.stream).

Requiere la variable de entorno ANTHROPIC_API_KEY (ver .env.example).
"""

from typing import Iterator, TYPE_CHECKING

from anthropic import Anthropic, AnthropicError

from .base import BaseProvider, ProviderError

if TYPE_CHECKING:
    from core.conversation import Conversation


class AnthropicProvider(BaseProvider):
    name = "Anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-0",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def stream_reply(self, conversation: "Conversation") -> Iterator[str]:
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=conversation.system_prompt,
                messages=conversation.to_anthropic_format(),
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except AnthropicError as exc:
            raise ProviderError(self.name, exc) from exc
        except Exception as exc:  # red, timeouts, etc.
            raise ProviderError(self.name, exc) from exc
