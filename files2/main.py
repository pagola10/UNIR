"""
main.py

Punto de entrada de la aplicación. CLI interactiva que permite
conversar con tres expertos temáticos (Programación, Marketing y
Jurídico-Legal) usando el modelo local `gemma3:1b` a través de Ollama,
de forma completamente offline.

Uso:
    python main.py

Comandos disponibles dentro del chat:
    /expertos   -> muestra el menú para cambiar de experto
    /reiniciar  -> borra el historial del experto activo
    /salir      -> termina la aplicación
"""

import sys

from core.chatbot import (
    ExpertChatbot,
    ModelNotAvailableError,
    OllamaNotAvailableError,
)
from core.conversation import Conversation
from experts.expert_prompts import EXPERTS, DEFAULT_EXPERT_KEY

DIVIDER = "=" * 60


def print_header() -> None:
    print(DIVIDER)
    print("  🤖 Chatbot de Expertos Temáticos (Ollama · gemma3:1b · offline)")
    print(DIVIDER)


def print_experts_menu() -> None:
    print("\nExpertos disponibles:")
    for i, expert in enumerate(EXPERTS.values(), start=1):
        print(f"  {i}. {expert.emoji}  {expert.display_name}")
        print(f"       {expert.short_description}")
    print()


def choose_expert_interactively(current_key: str | None = None) -> tuple[str, bool]:
    """
    Muestra el menú de expertos y pide al usuario que elija uno.
    Devuelve (clave_del_experto, reset_history) donde reset_history
    indica si el usuario pidió explícitamente empezar de cero con ese
    experto.
    """
    expert_keys = list(EXPERTS.keys())
    print_experts_menu()

    while True:
        choice = input(
            "Elige un experto (número), o Enter para cancelar: "
        ).strip()
        if choice == "":
            # Cancelar: se mantiene el experto actual (o el de por defecto)
            return (current_key or DEFAULT_EXPERT_KEY), False

        if choice.isdigit() and 1 <= int(choice) <= len(expert_keys):
            selected_key = expert_keys[int(choice) - 1]
            if selected_key == current_key:
                # Ya estaba en ese experto: preguntamos si quiere reiniciar
                confirm = input(
                    "Ya estás hablando con este experto. "
                    "¿Quieres reiniciar la conversación? (s/N): "
                ).strip().lower()
                return selected_key, confirm == "s"
            return selected_key, False

        print("Opción no válida. Introduce el número de un experto de la lista.\n")


def print_active_expert_banner(conversation: Conversation) -> None:
    expert = conversation.current_expert
    turns = len(conversation)
    print(f"\n--- {expert.emoji} Experto activo: {expert.display_name} "
          f"(intercambios en esta sesión: {turns}) ---")
    print("Comandos: /expertos  ·  /reiniciar  ·  /salir\n")


def run_cli() -> None:
    print_header()
    print("\nComprobando disponibilidad de Ollama y del modelo gemma3:1b...")

    conversation = Conversation(initial_expert_key=DEFAULT_EXPERT_KEY)
    chatbot = ExpertChatbot(conversation=conversation)

    try:
        chatbot.verify_availability()
    except OllamaNotAvailableError as exc:
        print(f"\n[Error] {exc}")
        sys.exit(1)
    except ModelNotAvailableError as exc:
        print(f"\n[Error] {exc}")
        sys.exit(1)

    print("✔ Ollama está disponible y el modelo gemma3:1b está listo.\n")

    # Selección inicial del experto
    initial_key, _ = choose_expert_interactively(current_key=None)
    conversation.switch_expert(initial_key)

    print_active_expert_banner(conversation)

    while True:
        try:
            user_input = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSesión interrumpida. ¡Hasta pronto!")
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command == "/salir":
            print("Cerrando el chatbot. ¡Hasta pronto!")
            break

        if command == "/reiniciar":
            conversation.reset_current()
            print("[Historial del experto actual reiniciado]\n")
            print_active_expert_banner(conversation)
            continue

        if command == "/expertos":
            new_key, reset = choose_expert_interactively(
                current_key=conversation.current_expert_key
            )
            conversation.switch_expert(new_key, reset_history=reset)
            print_active_expert_banner(conversation)
            continue

        # Mensaje normal: se envía al experto activo
        expert = conversation.current_expert
        print(f"{expert.emoji} {expert.display_name}: ", end="", flush=True)
        try:
            for fragment in chatbot.send_message(user_input):
                print(fragment, end="", flush=True)
            print("\n")
        except ModelNotAvailableError as exc:
            print(f"\n[Error] {exc}\n")
        except OllamaNotAvailableError as exc:
            print(f"\n[Error] {exc}\n")


if __name__ == "__main__":
    try:
        run_cli()
    except Exception as exc:  # último recurso: no dejar morir la app sin explicar por qué
        print(f"\n[Error fatal] {exc}", file=sys.stderr)
        sys.exit(1)
