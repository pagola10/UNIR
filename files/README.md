# Chatbot con fallback automático entre proveedores de IA

Chatbot conversacional en Python que mantiene el historial completo de la
conversación y encadena automáticamente tres proveedores de IA
(**OpenAI → Anthropic Claude → Google Gemini**) para garantizar continuidad
del servicio si alguno falla.

## Estructura del proyecto

```
├── main.py                     # Punto de entrada (CLI interactiva)
├── providers/
│   ├── __init__.py
│   ├── base.py                 # Interfaz común (BaseProvider) y ProviderError
│   ├── openai_provider.py      # Responses API de OpenAI, streaming
│   ├── anthropic_provider.py   # Messages API de Anthropic, streaming
│   └── gemini_provider.py      # google-genai, streaming
├── core/
│   ├── __init__.py
│   ├── chatbot.py              # Cascada de fallback
│   └── conversation.py         # Historial de conversación
├── requirements.txt
├── .env                        # Plantilla de variables de entorno (sin claves reales)
└── README.md
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuración

Edita el archivo `.env` en la raíz del proyecto y añade tus claves reales:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
```

> El archivo `.env` está incluido en `.gitignore`: nunca lo subas a un
> repositorio público con tus claves reales.

No es obligatorio tener las tres claves configuradas: si un proveedor no
tiene clave válida, el sistema de fallback automáticamente saltará al
siguiente en la cascada.

## Uso

```bash
python main.py
```

Dentro del chat:

- Escribe con normalidad para conversar con el asistente.
- `/reset` — borra el historial y empieza una conversación nueva.
- `/salir` — termina la aplicación de forma limpia.

## Cómo funciona el fallback

1. `Chatbot.send_message()` añade el mensaje del usuario al historial
   (`Conversation`) y recorre la lista de proveedores en orden de prioridad:
   `OpenAI → Anthropic → Gemini`.
2. Para cada proveedor, intenta obtener la respuesta en **streaming**. Si
   la llamada tiene éxito, el fragmento de texto se va mostrando en tiempo
   real y, al terminar, la respuesta completa se guarda en el historial.
3. Si un proveedor lanza `ProviderError` (fallo de red, autenticación,
   rate limit, etc.), se captura, se notifica al usuario mediante un
   callback (`on_provider_switch`) y se prueba automáticamente con el
   siguiente proveedor de la cascada, **sin perder el historial previo**.
4. Si los tres proveedores fallan, se lanza `AllProvidersFailedError` y el
   mensaje del usuario permanece en el historial para poder reintentar.

Cada proveedor traduce el historial neutral (`Conversation`) a su propio
formato:

| Proveedor | Roles | System prompt |
|---|---|---|
| OpenAI | `user` / `assistant` | Como mensaje con rol `system` dentro del array |
| Anthropic | `user` / `assistant` | Parámetro `system` aparte |
| Gemini | `user` / `model` | Parámetro `system_instruction` aparte |

## Notas de diseño

- **`ProviderError`**: cada proveedor traduce las excepciones específicas
  de su SDK (`OpenAIError`, `AnthropicError`, `google.genai.errors.APIError`,
  además de errores de red genéricos) a una única excepción común, para que
  la lógica de fallback en `core/chatbot.py` no necesite conocer los detalles
  internos de cada SDK.
- **Streaming end-to-end**: los tres proveedores devuelven generadores de
  texto, y `main.py` los imprime con `print(fragment, end="", flush=True)`
  para que el usuario vea la respuesta aparecer en tiempo real.
- **Historial neutral**: `Conversation` no conoce ningún detalle de ningún
  proveedor; solo guarda mensajes con rol `user`/`assistant` y expone
  métodos de conversión (`to_openai_format`, `to_anthropic_format`,
  `to_gemini_format`).
