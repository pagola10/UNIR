"""
main.py

Punto de entrada de la aplicación. Levanta una CLI interactiva que
conversa con el usuario usando un sistema de fallback automático
entre OpenAI -> Anthropic -> Google Gemini.

Uso:
    python main.py

Comandos especiales dentro del chat:
    /salir   -> termina la conversación de forma limpia
    /reset   -> borra el historial y empieza una conversación nueva
"""

import sys

from dotenv import load_dotenv

from core.chatbot import AllProvidersFailedError, Chatbot
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.gemini_provider import GeminiProvider


SYSTEM_PROMPT = "Eres un asistente conversacional útil, claro y conciso."

BANNER = """
==================================================
  Chatbot con fallback automático (OpenAI -> Anthropic -> Gemini)
  Escribe tu mensaje y pulsa Enter.
  Comandos: /salir para terminar, /reset para limpiar el historial.
==================================================
"""


def notify_provider_switch(failed_provider_name: str, error: Exception) -> None:
    """
    Callback que se ejecuta cada vez que un proveedor falla y el
    chatbot va a intentar con el siguiente de la cascada.
    """
    print(
        f"\n[!] {failed_provider_name} no está disponible ahora mismo "
        f"({error}). Probando con el siguiente proveedor...\n"
    )


def build_chatbot() -> Chatbot:
    """
    Instancia los tres proveedores en orden de prioridad y crea el
    Chatbot. Si algún SDK no encuentra su API key en el entorno, la
    instanciación del cliente no falla inmediatamente (los SDKs suelen
    validar la key en la primera llamada), así que el fallback en
    tiempo de ejecución cubre igualmente ese caso.
    """
    providers = [
        OpenAIProvider(model="gpt-4.1"),
        AnthropicProvider(model="claude-sonnet-4-0"),
        GeminiProvider(model="gemini-2.5-flash"),
    ]
    return Chatbot(
        providers=providers,
        system_prompt=SYSTEM_PROMPT,
        on_provider_switch=notify_provider_switch,
    )


def run_cli() -> None:
    load_dotenv()  # carga OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY desde .env

    chatbot = build_chatbot()
    print(BANNER)

    while True:
        try:
            user_input = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSesión interrumpida. ¡Hasta pronto!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/salir":
            print("Cerrando el chatbot. ¡Hasta pronto!")
            break

        if user_input.lower() == "/reset":
            chatbot.conversation.clear()
            print("[Historial de conversación reiniciado]\n")
            continue

        print("Asistente: ", end="", flush=True)
        try:
            for fragment in chatbot.send_message(user_input):
                print(fragment, end="", flush=True)
            print("\n")
        except AllProvidersFailedError as exc:
            print(
                "\n[Error] Ningún proveedor pudo responder en este momento.\n"
                f"Detalle: {exc}\n"
                "Tu mensaje se ha conservado en el historial; puedes "
                "reintentar en unos segundos.\n"
            )


if __name__ == "__main__":
    try:
        run_cli()
    except Exception as exc:  # último recurso: no dejar morir la app sin explicar por qué
        print(f"\n[Error fatal] {exc}", file=sys.stderr)
        sys.exit(1)
