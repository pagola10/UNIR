"""
core/api_client.py

Encapsula toda la comunicación con la API de OpenAI. Usa
`client.responses.parse()` junto con `text_format=LinkedinPost` para
garantizar que la respuesta del modelo sea siempre un objeto
`LinkedinPost` válido (Structured Outputs), en lugar de confiar en
que el modelo "entienda" el formato solo por instrucciones de texto.

Traduce los distintos fallos posibles (errores de red/autenticación,
rechazos de contenido, respuestas incompletas, fallos de validación)
a excepciones propias, claras y fáciles de manejar desde capas
superiores (`core/chatbot.py`).
"""

import openai
from openai import OpenAI
from pydantic import ValidationError

from models.linkedin_post import LinkedinPost

DEFAULT_MODEL = "gpt-4o-2024-08-06"

SYSTEM_PROMPT = (
    "Eres un redactor experto en contenido para LinkedIn, especializado en "
    "escribir posts profesionales que generan buen engagement (me gusta, "
    "comentarios, compartidos).\n\n"
    "A partir de la idea que te dé el usuario, genera un post de LinkedIn "
    "completo. Ten en cuenta:\n"
    "- El título debe ser un gancho corto y atractivo.\n"
    "- El contenido debe tener una estructura clara: una apertura que "
    "capte la atención, un desarrollo con valor real (consejos, "
    "reflexiones o datos), y un cierre que invite a la interacción "
    "(una pregunta, una reflexión, una llamada a la acción).\n"
    "- Usa un tono profesional pero cercano y humano, evitando sonar "
    "genérico o excesivamente corporativo.\n"
    "- Los hashtags deben ser relevantes y estar en español salvo que la "
    "idea del usuario sea claramente técnica y los hashtags en inglés "
    "sean más habituales en ese sector.\n"
    "- Elige la categoría que mejor represente el tema del post."
)


class OpenAIClientError(Exception):
    """Error base para cualquier fallo relacionado con la API de OpenAI."""


class APIConnectionOrAuthError(OpenAIClientError):
    """
    Fallo de conectividad con la API de OpenAI, o de autenticación
    (API key inválida/ausente).
    """


class RateLimitOrQuotaError(OpenAIClientError):
    """Se ha excedido el límite de peticiones o la cuota disponible."""


class ContentRefusedError(OpenAIClientError):
    """
    El modelo ha rechazado generar el contenido solicitado (por
    ejemplo, por motivos de política de contenido).
    """

    def __init__(self, refusal_message: str):
        self.refusal_message = refusal_message
        super().__init__(f"El modelo rechazó la solicitud: {refusal_message}")


class IncompleteResponseError(OpenAIClientError):
    """
    La respuesta del modelo quedó incompleta (por ejemplo, se alcanzó
    el límite de tokens antes de terminar de generar el JSON).
    """


class InvalidStructuredOutputError(OpenAIClientError):
    """
    La respuesta no pudo interpretarse como un LinkedinPost válido
    (fallo de validación de Pydantic), o el modelo no devolvió ningún
    contenido utilizable.
    """


class LinkedinPostGenerator:
    """
    Cliente responsable de pedir a OpenAI que genere un `LinkedinPost`
    estructurado a partir de la idea proporcionada por el usuario.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        # El SDK toma automáticamente OPENAI_API_KEY del entorno si no
        # se pasa api_key explícitamente (por eso se carga el .env
        # antes de instanciar esta clase, ver main.py).
        self.client = OpenAI()
        self.model = model

    def generate_post(self, idea: str) -> LinkedinPost:
        """
        Genera un LinkedinPost estructurado a partir de la idea del
        usuario. Lanza una de las excepciones definidas en este
        módulo si algo falla, para que la capa superior pueda mostrar
        un mensaje claro al usuario.
        """
        try:
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": idea},
                ],
                text_format=LinkedinPost,
            )
        except openai.AuthenticationError as exc:
            raise APIConnectionOrAuthError(
                "La API key de OpenAI no es válida o no está configurada. "
                "Revisa la variable OPENAI_API_KEY en tu archivo .env."
            ) from exc
        except openai.APIConnectionError as exc:
            raise APIConnectionOrAuthError(
                "No se pudo conectar con la API de OpenAI. Comprueba tu "
                "conexión a internet e inténtalo de nuevo."
            ) from exc
        except openai.RateLimitError as exc:
            raise RateLimitOrQuotaError(
                "Se ha alcanzado el límite de peticiones o la cuota "
                "disponible en tu cuenta de OpenAI. Espera un momento "
                "antes de volver a intentarlo."
            ) from exc
        except openai.BadRequestError as exc:
            raise OpenAIClientError(
                f"La API rechazó la petición por ser inválida: {exc}"
            ) from exc
        except openai.APIStatusError as exc:
            raise OpenAIClientError(
                f"La API de OpenAI devolvió un error: {exc}"
            ) from exc

        # Respuesta incompleta (p.ej. se agotó max_output_tokens antes
        # de terminar de generar el JSON estructurado).
        if getattr(response, "status", None) == "incomplete":
            reason = getattr(response.incomplete_details, "reason", "desconocido")
            raise IncompleteResponseError(
                f"La respuesta del modelo quedó incompleta (motivo: {reason})."
            )

        # Comprobar si el modelo rechazó la solicitud (refusal) antes
        # de intentar acceder al contenido parseado.
        refusal_text = self._extract_refusal(response)
        if refusal_text:
            raise ContentRefusedError(refusal_text)

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise InvalidStructuredOutputError(
                "El modelo no devolvió contenido estructurado utilizable."
            )

        try:
            # Revalidación explícita: aunque el SDK ya construye el
            # objeto Pydantic, esto garantiza que cualquier violación
            # de las reglas del modelo (p.ej. si en el futuro se
            # añaden validadores personalizados) se detecte aquí.
            return LinkedinPost.model_validate(parsed.model_dump())
        except ValidationError as exc:
            raise InvalidStructuredOutputError(
                f"La respuesta del modelo no cumple el schema esperado: {exc}"
            ) from exc

    @staticmethod
    def _extract_refusal(response) -> str | None:
        """
        Busca en la respuesta un bloque de tipo 'refusal' (el modelo
        se negó a generar el contenido) y devuelve su texto, o None
        si no hubo ningún rechazo.
        """
        output_items = getattr(response, "output", None) or []
        for item in output_items:
            if getattr(item, "type", None) != "message":
                continue
            for content_block in getattr(item, "content", None) or []:
                if getattr(content_block, "type", None) == "refusal":
                    return getattr(content_block, "refusal", "Contenido rechazado.")
        return None
