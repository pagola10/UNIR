"""
providers/gemini_provider.py

Implementación del proveedor Google Gemini usando el SDK oficial
`google-genai` (paquete: google-genai) en modo streaming
(client.models.generate_content_stream).

Requiere la variable de entorno GEMINI_API_KEY (ver .env.example).
"""

from typing import Iterator, TYPE_CHECKING

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from .base import BaseProvider, ProviderError

if TYPE_CHECKING:
    from core.conversation import Conversation


class GeminiProvider(BaseProvider):
    name = "Google Gemini"

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def stream_reply(self, conversation: "Conversation") -> Iterator[str]:
        try:
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=conversation.to_gemini_format(),
                config=types.GenerateContentConfig(
                    system_instruction=conversation.system_prompt,
                ),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except genai_errors.APIError as exc:
            raise ProviderError(self.name, exc) from exc
        except Exception as exc:  # red, timeouts, etc.
            raise ProviderError(self.name, exc) from exc
