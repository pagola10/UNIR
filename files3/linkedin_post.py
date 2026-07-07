"""
models/linkedin_post.py

Define el schema de salida estructurada para los posts de LinkedIn
generados por el modelo. Esta clase Pydantic se pasa directamente al
parámetro `text_format` de `client.responses.parse()`, y OpenAI la usa
para forzar (structured output, modo strict) que la respuesta del
modelo cumpla exactamente esta estructura.

Nota de diseño: el modo "strict" de Structured Outputs de OpenAI NO
soporta restricciones de validación como `min_length`, `max_length` o
`pattern` a nivel de schema (se ignoran silenciosamente a la hora de
restringir la generación). Por eso:
  - Usamos `Literal` para `category`, ya que los enumerados sí están
    soportados y sí restringen realmente la generación del modelo.
  - Para `hashtags` y `content` documentamos las expectativas de
    longitud en la propia `description` del campo (que el modelo SÍ
    lee como instrucción), y opcionalmente podemos revalidar esas
    expectativas nosotros mismos tras recibir la respuesta si se
    necesitara una garantía dura.
"""

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "Tecnología",
    "Negocios",
    "Carrera Profesional",
    "Liderazgo",
    "Marketing",
    "Desarrollo Personal",
]


class LinkedinPost(BaseModel):
    """Estructura de un post de LinkedIn generado por el modelo."""

    title: str = Field(
        description=(
            "Un título/gancho corto y atractivo para el post, de máximo "
            "unas 12 palabras, pensado para captar la atención en el feed."
        )
    )
    content: str = Field(
        description=(
            "El cuerpo del post de LinkedIn, entre 3 y 8 párrafos cortos, "
            "en un tono profesional pero cercano, con saltos de línea "
            "entre párrafos para facilitar la lectura."
        )
    )
    hashtags: list[str] = Field(
        description=(
            "Entre 3 y 6 hashtags relevantes para el post, cada uno "
            "escrito sin espacios y comenzando por '#', por ejemplo "
            "'#InteligenciaArtificial'."
        )
    )
    category: Category = Field(
        description="La categoría temática que mejor describe el post."
    )
