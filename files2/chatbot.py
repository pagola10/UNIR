"""
core/chatbot.py

Contiene la clase ExpertChatbot, responsable de:
  - Verificar que Ollama está accesible y que el modelo gemma3:1b
    está descargado localmente.
  - Enviar el historial de la conversación (con el system prompt del
    experto activo) al modelo, en modo streaming.
  - Traducir los errores específicos del SDK de `ollama` a excepciones
    propias, claras y fáciles de manejar desde la interfaz de consola.

Todo el funcionamiento es 100% local/offline: no se realiza ninguna
llamada a servicios en la nube, únicamente al servidor local de Ollama
(por defecto en http://localhost:11434).
"""

from typing import Iterator

import ollama
from ollama import ResponseError

from core.conversation import Conversation

MODEL_NAME = "gemma3:1b"


class OllamaNotAvailableError(Exception):
    """
    Se lanza cuando no se puede establecer conexión con el servicio
    local de Ollama (por ejemplo, si `ollama serve` no está corriendo).
    """


class ModelNotAvailableError(Exception):
    """
    Se lanza cuando Ollama está accesible pero el modelo requerido
    (gemma3:1b) no está descargado localmente.
    """


class ExpertChatbot:
    """
    Orquesta la comunicación con Ollama para el modelo `gemma3:1b`,
    usando el historial y el system prompt del experto activo en cada
    momento (gestionados por `Conversation`).
    """

    def __init__(self, conversation: Conversation, model: str = MODEL_NAME) -> None:
        self.conversation = conversation
        self.model = model

    # ---------------------------------------------------------------
    # Verificación de disponibilidad (llamar al arrancar la app)
    # ---------------------------------------------------------------

    def verify_availability(self) -> None:
        """
        Comprueba que:
          1. El servicio de Ollama está corriendo y es alcanzable.
          2. El modelo `gemma3:1b` está entre los modelos descargados.

        Lanza OllamaNotAvailableError o ModelNotAvailableError según
        corresponda. Se recomienda llamar a este método una vez, al
        arrancar la aplicación, para dar un mensaje de error claro
        antes de empezar a chatear.
        """
        try:
            local_models_response = ollama.list()
        except ConnectionError as exc:
            raise OllamaNotAvailableError(
                "No se pudo conectar con Ollama. Asegúrate de que el "
                "servicio está corriendo (ejecuta 'ollama serve' o abre "
                "la aplicación de Ollama) y vuelve a intentarlo."
            ) from exc
        except Exception as exc:  # cualquier otro fallo de red/servicio
            raise OllamaNotAvailableError(
                f"No se pudo conectar con Ollama: {exc}"
            ) from exc

        downloaded_names = {m.model for m in local_models_response.models}
        # Ollama normaliza los nombres añadiendo ':latest' si no se
        # especifica tag; comprobamos ambas variantes por seguridad.
        model_is_present = self.model in downloaded_names or any(
            name.split(":")[0] == self.model.split(":")[0] for name in downloaded_names
        )

        if not model_is_present:
            raise ModelNotAvailableError(
                f"El modelo '{self.model}' no está descargado localmente. "
                f"Descárgalo primero ejecutando en tu terminal:\n"
                f"    ollama pull {self.model}"
            )

    # ---------------------------------------------------------------
    # Envío de mensajes (streaming)
    # ---------------------------------------------------------------

    def send_message(self, user_input: str) -> Iterator[str]:
        """
        Añade el mensaje del usuario al historial del experto activo,
        y devuelve un generador que emite la respuesta del modelo en
        streaming, fragmento a fragmento.

        Al finalizar el streaming, la respuesta completa se guarda
        también en el historial como mensaje del asistente, de forma
        que el contexto se conserva para el siguiente turno.
        """
        self.conversation.add_user_message(user_input)

        try:
            stream = ollama.chat(
                model=self.model,
                messages=self.conversation.to_ollama_format(),
                stream=True,
            )
        except ConnectionError as exc:
            raise OllamaNotAvailableError(
                "Se perdió la conexión con Ollama. Comprueba que el "
                "servicio sigue corriendo e inténtalo de nuevo."
            ) from exc
        except ResponseError as exc:
            if exc.status_code == 404:
                raise ModelNotAvailableError(
                    f"El modelo '{self.model}' no está disponible en "
                    f"Ollama. Ejecuta 'ollama pull {self.model}' y "
                    f"vuelve a intentarlo."
                ) from exc
            raise

        collected: list[str] = []
        try:
            for chunk in stream:
                fragment = chunk.message.content or ""
                if fragment:
                    collected.append(fragment)
                    yield fragment
        except ConnectionError as exc:
            raise OllamaNotAvailableError(
                "Se perdió la conexión con Ollama durante la generación "
                "de la respuesta."
            ) from exc
        finally:
            # Guardamos en el historial lo que se haya generado hasta
            # el momento, incluso si el streaming se interrumpió, para
            # no perder contexto de lo que el modelo alcanzó a decir.
            if collected:
                self.conversation.add_assistant_message("".join(collected))
