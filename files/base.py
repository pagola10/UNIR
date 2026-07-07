"""
providers/base.py

Define la interfaz común que deben cumplir todos los proveedores,
y la excepción unificada que usa el sistema de fallback para saber
cuándo debe saltar al siguiente proveedor de la cascada.
"""

from abc import ABC, abstractmethod
from typing import Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from core.conversation import Conversation


class ProviderError(Exception):
    """
    Excepción unificada que lanzan los providers cuando falla la llamada
    a su API (errores de conectividad, autenticación, rate limit, etc).

    El chatbot solo necesita capturar este único tipo de excepción para
    decidir si debe hacer fallback al siguiente proveedor, sin tener que
    conocer las excepciones específicas de cada SDK (openai.APIError,
    anthropic.APIError, google.genai.errors...).
    """

    def __init__(self, provider_name: str, original_error: Exception):
        self.provider_name = provider_name
        self.original_error = original_error
        super().__init__(f"[{provider_name}] {type(original_error).__name__}: {original_error}")


class BaseProvider(ABC):
    """Interfaz mínima que debe implementar cada proveedor."""

    name: str = "base"

    @abstractmethod
    def stream_reply(self, conversation: "Conversation") -> Iterator[str]:
        """
        Envía el historial de la conversación al proveedor y devuelve
        un generador que va emitiendo fragmentos de texto (streaming)
        a medida que el modelo los genera.

        Debe lanzar ProviderError si algo falla (red, auth, rate limit...).
        """
        raise NotImplementedError
