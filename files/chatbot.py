"""
core/chatbot.py

Contiene la clase Chatbot, responsable de:
  - Mantener la conversación (delegado en Conversation).
  - Implementar la cascada de fallback entre proveedores.
  - Notificar al usuario cuando cambia de proveedor o cuando todos fallan.
"""

from typing import Callable, Iterable

from providers.base import BaseProvider, ProviderError
from core.conversation import Conversation


class AllProvidersFailedError(Exception):
    """Se lanza cuando ningún proveedor de la cascada pudo responder."""

    def __init__(self, errors: list[ProviderError]):
        self.errors = errors
        detail = " | ".join(str(e) for e in errors)
        super().__init__(f"Todos los proveedores fallaron: {detail}")


class Chatbot:
    """
    Orquesta la conversación y el sistema de fallback en cascada.

    providers: lista ordenada por prioridad, p.ej.
        [OpenAIProvider(), AnthropicProvider(), GeminiProvider()]
    El primero que responda con éxito "gana" ese turno; si falla,
    se prueba automáticamente con el siguiente.
    """

    def __init__(
        self,
        providers: list[BaseProvider],
        system_prompt: str | None = None,
        on_provider_switch: Callable[[str, Exception], None] | None = None,
    ) -> None:
        if not providers:
            raise ValueError("Debes pasar al menos un proveedor.")
        self.providers = providers
        self.conversation = Conversation(system_prompt=system_prompt)
        # Callback opcional para notificar a la UI cuando se cambia de
        # proveedor por un fallo (nombre del proveedor que falló, error).
        self.on_provider_switch = on_provider_switch or (lambda name, err: None)

    def send_message(self, user_input: str) -> Iterable[str]:
        """
        Añade el mensaje del usuario al historial e intenta obtener una
        respuesta probando cada proveedor en orden de prioridad.

        Devuelve un generador de fragmentos de texto (streaming).
        Si TODOS los proveedores fallan, lanza AllProvidersFailedError
        y el mensaje del usuario permanece en el historial (para que el
        usuario pueda reintentar sin perder contexto).
        """
        self.conversation.add_user_message(user_input)

        errors: list[ProviderError] = []
        for provider in self.providers:
            try:
                full_reply = yield from self._stream_and_collect(provider)
                self.conversation.add_assistant_message(full_reply)
                return
            except ProviderError as exc:
                errors.append(exc)
                self.on_provider_switch(provider.name, exc)
                continue

        # Si llegamos aquí, todos los proveedores fallaron.
        raise AllProvidersFailedError(errors)

    def _stream_and_collect(self, provider: BaseProvider) -> Iterable[str]:
        """
        Hace streaming de un proveedor concreto, reemitiendo cada
        fragmento hacia quien llamó a send_message() y, al mismo tiempo,
        acumulando el texto completo para guardarlo en el historial.

        Se usa "yield from" + un valor de retorno (vía StopIteration.value)
        para poder tanto emitir fragmentos como devolver el texto completo
        al finalizar sin gastar memoria en dos pasadas separadas.
        """
        chunks: list[str] = []
        for fragment in provider.stream_reply(self.conversation):
            chunks.append(fragment)
            yield fragment
        return "".join(chunks)
