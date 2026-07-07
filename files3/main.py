"""
main.py

Punto de entrada de la aplicación. CLI interactiva que pide al
usuario la idea de un post de LinkedIn y usa la API de OpenAI con
Structured Outputs (Pydantic) para generar un `LinkedinPost` validado
y mostrarlo de forma organizada por terminal.

Uso:
    python main.py

Comandos disponibles:
    /salir   -> termina la aplicación
"""

import sys

from dotenv import load_dotenv

from core.api_client import (
    APIConnectionOrAuthError,
    ContentRefusedError,
    IncompleteResponseError,
    InvalidStructuredOutputError,
    OpenAIClientError,
    RateLimitOrQuotaError,
)
from core.chatbot import LinkedinPostChatbot
from models.linkedin_post import LinkedinPost

DIVIDER = "=" * 62


def print_header() -> None:
    print(DIVIDER)
    print("  ✍️  Generador de Posts de LinkedIn (OpenAI · Structured Outputs)")
    print(DIVIDER)
    print(
        "\nDescribe la idea de tu post y generaré un título, contenido, "
        "hashtags y categoría listos para publicar.\n"
        "Escribe /salir en cualquier momento para terminar.\n"
    )


def print_post(post: LinkedinPost) -> None:
    print("\n" + "-" * 62)
    print(f"📌 TÍTULO:\n{post.title}\n")
    print(f"📝 CONTENIDO:\n{post.content}\n")
    print(f"🏷️  HASHTAGS:\n{' '.join(post.hashtags)}\n")
    print(f"📂 CATEGORÍA:\n{post.category}")
    print("-" * 62 + "\n")


def run_cli() -> None:
    load_dotenv()  # carga OPENAI_API_KEY desde el archivo .env

    print_header()

    try:
        chatbot = LinkedinPostChatbot()
    except Exception as exc:
        print(f"[Error] No se pudo inicializar el cliente de OpenAI: {exc}")
        sys.exit(1)

    while True:
        try:
            idea = input("💡 Idea para tu post (o /salir): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSesión interrumpida. ¡Hasta pronto!")
            break

        if not idea:
            continue

        if idea.lower() == "/salir":
            print("¡Hasta pronto! 👋")
            break

        print("\nGenerando post...\n")

        try:
            post = chatbot.generate(idea)
            print_post(post)

        except ContentRefusedError as exc:
            print(
                f"\n[Solicitud rechazada] El modelo no pudo generar este "
                f"post: {exc.refusal_message}\n"
                "Prueba a reformular la idea con un enfoque distinto.\n"
            )
        except IncompleteResponseError as exc:
            print(
                f"\n[Respuesta incompleta] {exc}\n"
                "Prueba con una idea más breve o inténtalo de nuevo.\n"
            )
        except InvalidStructuredOutputError as exc:
            print(
                f"\n[Error de validación] La respuesta del modelo no tuvo "
                f"el formato esperado: {exc}\n"
                "Puedes intentarlo de nuevo.\n"
            )
        except RateLimitOrQuotaError as exc:
            print(f"\n[Límite alcanzado] {exc}\n")
        except APIConnectionOrAuthError as exc:
            print(f"\n[Error de conexión] {exc}\n")
            # Un fallo de autenticación no se va a resolver reintentando
            # con la misma configuración; salimos con un código de error.
            sys.exit(1)
        except OpenAIClientError as exc:
            print(f"\n[Error de la API] {exc}\n")
        except ValueError as exc:
            print(f"\n[Entrada no válida] {exc}\n")


if __name__ == "__main__":
    try:
        run_cli()
    except Exception as exc:  # último recurso: no dejar morir la app sin explicar por qué
        print(f"\n[Error fatal] {exc}", file=sys.stderr)
        sys.exit(1)
