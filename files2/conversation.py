"""
core/conversation.py

Gestiona el estado de la conversación: qué experto está activo y el
historial de mensajes intercambiados con cada uno.

Diseño clave: cada experto mantiene su PROPIO historial de forma
independiente. Así, si el usuario cambia de "Programación" a
"Marketing" y luego vuelve a "Programación", puede optar por seguir
la conversación de programación donde la dejó, sin que los mensajes de
marketing hayan contaminado su contexto (y viceversa).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from experts.expert_prompts import EXPERTS, DEFAULT_EXPERT_KEY, Expert

Role = Literal["user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class Conversation:
    """
    Mantiene el estado completo de la sesión de chat:
      - qué experto está activo en cada momento.
      - el historial de mensajes de CADA experto por separado.
    """

    def __init__(self, initial_expert_key: str = DEFAULT_EXPERT_KEY) -> None:
        if initial_expert_key not in EXPERTS:
            raise ValueError(f"Experto desconocido: {initial_expert_key}")

        self._current_expert_key: str = initial_expert_key
        # Un historial independiente por cada experto disponible.
        self._histories: dict[str, list[Message]] = {key: [] for key in EXPERTS}

    # ---------------------------------------------------------------
    # Propiedades de conveniencia
    # ---------------------------------------------------------------

    @property
    def current_expert(self) -> Expert:
        return EXPERTS[self._current_expert_key]

    @property
    def current_expert_key(self) -> str:
        return self._current_expert_key

    @property
    def messages(self) -> list[Message]:
        """Historial del experto actualmente activo (solo lectura)."""
        return list(self._histories[self._current_expert_key])

    def __len__(self) -> int:
        return len(self._histories[self._current_expert_key])

    # ---------------------------------------------------------------
    # Gestión de mensajes
    # ---------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self._histories[self._current_expert_key].append(
            Message(role="user", content=content)
        )

    def add_assistant_message(self, content: str) -> None:
        self._histories[self._current_expert_key].append(
            Message(role="assistant", content=content)
        )

    # ---------------------------------------------------------------
    # Cambio de experto / reinicio
    # ---------------------------------------------------------------

    def switch_expert(self, new_expert_key: str, reset_history: bool = False) -> Expert:
        """
        Cambia el experto activo.

        Por defecto (reset_history=False) se conserva el historial que
        ese experto ya tuviera de intercambios anteriores en la misma
        sesión, permitiendo retomar la conversación donde se dejó.

        Si reset_history=True, se vacía el historial del experto de
        destino antes de activarlo (útil si el usuario quiere una
        consulta fresca con ese experto).
        """
        if new_expert_key not in EXPERTS:
            raise ValueError(f"Experto desconocido: {new_expert_key}")

        if reset_history:
            self._histories[new_expert_key].clear()

        self._current_expert_key = new_expert_key
        return self.current_expert

    def reset_current(self) -> None:
        """Borra el historial del experto actualmente activo."""
        self._histories[self._current_expert_key].clear()

    def reset_all(self) -> None:
        """Borra el historial de TODOS los expertos."""
        for history in self._histories.values():
            history.clear()

    # ---------------------------------------------------------------
    # Conversión al formato esperado por el SDK de Ollama
    # ---------------------------------------------------------------

    def to_ollama_format(self) -> list[dict]:
        """
        Construye la lista de mensajes en el formato que espera
        ollama.chat(): una lista de dicts con 'role' y 'content',
        empezando por el mensaje de sistema del experto activo,
        seguido del historial de esa conversación.
        """
        messages = [
            {"role": "system", "content": self.current_expert.system_prompt}
        ]
        for m in self._histories[self._current_expert_key]:
            messages.append({"role": m.role, "content": m.content})
        return messages
