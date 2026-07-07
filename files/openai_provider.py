"""
providers/openai_provider.py

Implementación del proveedor OpenAI usando el SDK oficial `openai`
y la Responses API (client.responses.create) en modo streaming.

Requiere la variable de entorno OPENAI_API_KEY (ver .env.example).
"""

from typing import Iterator, TYPE_CHECKING

from openai import OpenAI, OpenAIError

from .base import BaseProvider, ProviderError

if TYPE_CHECKING:
    from core.conversation import Conversation


class OpenAIProvider(BaseProvider):
    name = "OpenAI"

    def __init__(self, model: str = "gpt-4.1", api_key: str | None = None) -> None:
        # Si no se pasa api_key, el SDK toma automáticamente OPENAI_API_KEY
        # del entorno (por eso conviene cargar el .env antes de instanciar esto).
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def stream_reply(self, conversation: "Conversation") -> Iterator[str]:
        try:
            stream = self.client.responses.create(
                model=self.model,
                input=conversation.to_openai_format(),
                stream=True,
            )
            for event in stream:
                # El evento relevante para texto incremental en la
                # Responses API es "response.output_text.delta".
                if event.type == "response.output_text.delta":
                    yield event.delta
                elif event.type == "response.error":
                    raise RuntimeError(getattr(event, "error", "Unknown OpenAI stream error"))
        except OpenAIError as exc:
            raise ProviderError(self.name, exc) from exc
        except Exception as exc:  # errores de red, timeouts, etc.
            raise ProviderError(self.name, exc) from exc
