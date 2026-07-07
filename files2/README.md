# Chatbot de Expertos Temáticos (Ollama · gemma3:1b · 100% offline)

Chatbot de consola que permite conversar con tres expertos temáticos
diferenciados —**Programación**, **Marketing** y **Jurídico-Legal**—
usando el modelo local `gemma3:1b` a través del SDK oficial de
**Ollama**. Todo el procesamiento ocurre en tu máquina: no se realiza
ninguna llamada a servicios en la nube.

## Estructura del proyecto

```
├── main.py                    # Punto de entrada (CLI interactiva)
├── experts/
│   ├── __init__.py
│   └── expert_prompts.py      # System prompts de cada experto
├── core/
│   ├── __init__.py
│   ├── chatbot.py             # Conexión con Ollama + manejo de errores
│   └── conversation.py        # Historial de conversación por experto
├── requirements.txt
└── README.md
```

## Requisitos previos

1. Tener [Ollama](https://ollama.com) instalado y corriendo en tu máquina
   (`ollama serve`, o simplemente abrir la aplicación de Ollama).
2. Haber descargado el modelo `gemma3:1b`:

   ```bash
   ollama pull gemma3:1b
   ```

## Instalación

```bash
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
python main.py
```

Al arrancar, la aplicación:

1. Comprueba que Ollama está accesible y que `gemma3:1b` está descargado.
   Si algo falta, muestra un mensaje de error claro con instrucciones
   (por ejemplo, el comando `ollama pull gemma3:1b` a ejecutar).
2. Te pide elegir con qué experto quieres empezar a hablar.
3. Entra en un bucle de conversación normal.

Comandos disponibles durante el chat:

| Comando       | Acción                                                          |
|---------------|------------------------------------------------------------------|
| `/expertos`   | Abre el menú para cambiar de experto (conservando o reiniciando su historial) |
| `/reiniciar`  | Borra el historial del experto actualmente activo               |
| `/salir`      | Termina la aplicación                                            |

## Cómo funciona

### Los tres expertos

Cada experto (`experts/expert_prompts.py`) es, en esencia, un **system
prompt** distinto que se envía al modelo junto con el historial de la
conversación. Ese prompt define:

- El rol y la personalidad del experto (ingeniero de software senior,
  consultor de marketing, asesor jurídico generalista).
- El vocabulario y los frameworks que debe usar.
- Los límites de su ámbito (por ejemplo, el experto jurídico siempre
  aclara que su respuesta no sustituye asesoría legal profesional, y
  el experto de marketing no entra en detalles técnicos de código).

### Historial por experto

`core/conversation.py` mantiene un historial **independiente por cada
experto**. Esto significa que si cambias de "Programación" a
"Marketing" y luego regresas a "Programación", puedes elegir seguir la
conversación de programación exactamente donde la dejaste, sin que los
mensajes de marketing se hayan mezclado en su contexto.

Al cambiar de experto (`/expertos`) puedes elegir:
- **Conservar** el historial de ese experto (para retomar una
  conversación anterior con él en la misma sesión).
- **Reiniciar** su historial (para empezar una consulta nueva).

### Conexión con Ollama y manejo de errores

`core/chatbot.py` usa la librería oficial `ollama` para Python:

```python
import ollama

stream = ollama.chat(
    model="gemma3:1b",
    messages=conversation.to_ollama_format(),  # incluye el system prompt del experto activo
    stream=True,
)
for chunk in stream:
    print(chunk.message.content, end="", flush=True)
```

Antes de empezar a chatear, `verify_availability()` comprueba con
`ollama.list()` que el servicio está accesible y que `gemma3:1b`
aparece entre los modelos descargados. Los errores se traducen a dos
excepciones propias y fáciles de manejar desde la CLI:

- **`OllamaNotAvailableError`**: no se pudo conectar con el servicio
  de Ollama (por ejemplo, si no está corriendo). Se captura el
  `ConnectionError` que lanza el SDK de `ollama` en este caso.
- **`ModelNotAvailableError`**: Ollama está accesible, pero
  `gemma3:1b` no está descargado (se detecta tanto en la comprobación
  inicial como capturando un `ResponseError` con `status_code == 404`
  durante el chat).

Estos mismos errores se vuelven a comprobar durante el envío de cada
mensaje, no solo al arrancar, por si Ollama se detiene a mitad de la
sesión.

## Notas de diseño

- El streaming se implementa como en el resto de proveedores vistos
  en el curso: `stream=True` devuelve un iterador de fragmentos, que
  se van imprimiendo con `print(fragment, end="", flush=True)` para
  dar sensación de respuesta en tiempo real.
- Aunque el streaming se interrumpa por un error de conexión, el
  texto generado hasta ese punto se guarda igualmente en el
  historial, para no perder contexto de lo que el modelo alcanzó a
  responder.
