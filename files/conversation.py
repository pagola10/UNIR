"""
core/conversation.py

Gestiona el historial de una conversación en un formato neutral
(independiente de proveedor). Cada proveedor se encarga de traducir
este historial a su propio formato (roles, estructura de mensajes, etc).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class Conversation:
    """
    Mantiene el historial completo de una conversación usuario-asistente.

    El historial se guarda con roles neutrales ("user" / "assistant").
    Cada provider (OpenAI, Anthropic, Gemini) es responsable de convertir
    este historial a su propio esquema antes de llamar a su API.
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt or (
            "Eres un asistente conversacional útil, claro y conciso."
        )
        self._messages: list[Message] = []

    def add_user_message(self, content: str) -> None:
        self._messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        self._messages.append(Message(role="assistant", content=content))

    @property
    def messages(self) -> list[Message]:
        """Devuelve el historial completo (solo lectura desde fuera)."""
        return list(self._messages)

    def to_openai_format(self) -> list[dict]:
        """
        Convierte el historial al formato esperado por la Responses API
        de OpenAI: lista de dicts con 'role' y 'content'.
        """
        history = [{"role": "system", "content": self.system_prompt}]
        for m in self._messages:
            history.append({"role": m.role, "content": m.content})
        return history

    def to_anthropic_format(self) -> list[dict]:
        """
        Convierte el historial al formato esperado por la Messages API
        de Anthropic. El system prompt NO va en la lista de mensajes;
        se pasa aparte como parámetro 'system'.
        """
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def to_gemini_format(self) -> list[dict]:
        """
        Convierte el historial al formato esperado por la API de Gemini.
        Gemini usa los roles 'user' y 'model' (no 'assistant'), y el
        system prompt se pasa aparte vía 'system_instruction'.
        """
        role_map = {"user": "user", "assistant": "model"}
        return [
            {"role": role_map[m.role], "parts": [{"text": m.content}]}
            for m in self._messages
        ]

    def __len__(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        self._messages.clear()
