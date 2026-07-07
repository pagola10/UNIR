"""
core/chatbot.py

Orquesta la interacción entre la entrada del usuario y el generador
de posts basado en OpenAI. Mantiene un pequeño historial de las ideas
consultadas en la sesión (útil para referencia, no se usa como
contexto conversacional en las llamadas, ya que cada post se genera
de forma independiente a partir de la idea puesta por el usuario).
"""

from dataclasses import dataclass, field
from datetime import datetime

from core.api_client import LinkedinPostGenerator, OpenAIClientError
from models.linkedin_post import LinkedinPost


@dataclass
class HistoryEntry:
    idea: str
    post: LinkedinPost
    timestamp: datetime = field(default_factory=datetime.now)


class LinkedinPostChatbot:
    """
    Clase principal del chatbot: recibe ideas de post en texto libre
    y devuelve objetos `LinkedinPost` ya generados y validados,
    delegando en `LinkedinPostGenerator` la comunicación con OpenAI.
    """

    def __init__(self, generator: LinkedinPostGenerator | None = None) -> None:
        self.generator = generator or LinkedinPostGenerator()
        self.history: list[HistoryEntry] = []

    def generate(self, idea: str) -> LinkedinPost:
        """
        Genera un post a partir de la idea del usuario.

        Puede propagar cualquiera de las excepciones definidas en
        `core.api_client` (OpenAIClientError y sus subclases); se deja
        a propósito sin capturar aquí para que la interfaz de usuario
        (main.py) decida cómo mostrar cada tipo de error al usuario.
        """
        idea = idea.strip()
        if not idea:
            raise ValueError("La idea del post no puede estar vacía.")

        post = self.generator.generate_post(idea)
        self.history.append(HistoryEntry(idea=idea, post=post))
        return post

    def last_posts(self, n: int = 5) -> list[HistoryEntry]:
        """Devuelve los últimos `n` posts generados en esta sesión."""
        return self.history[-n:]
