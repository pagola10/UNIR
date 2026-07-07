# Generador de Posts de LinkedIn (OpenAI · Structured Outputs · Pydantic)

Chatbot de consola que genera posts de LinkedIn a partir de una idea
escrita por el usuario, usando la API de OpenAI con **Structured
Outputs** para garantizar que la respuesta del modelo siempre tenga
una estructura válida y predecible: título, contenido, hashtags y
categoría.

## Estructura del proyecto

```
├── main.py                # Punto de entrada (CLI interactiva)
├── models/
│   ├── __init__.py
│   └── linkedin_post.py   # Modelo Pydantic (schema de salida)
├── core/
│   ├── __init__.py
│   ├── chatbot.py         # Lógica principal del chatbot
│   └── api_client.py      # Cliente de OpenAI + manejo de errores
├── requirements.txt
├── .env                    # Plantilla de variables de entorno (sin claves reales)
└── README.md
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuración

Edita el archivo `.env` en la raíz del proyecto y añade tu clave real:

```
OPENAI_API_KEY=sk-...
```

> El archivo `.env` está incluido en `.gitignore`: nunca lo subas a un
> repositorio público con tu clave real.

## Uso

```bash
python main.py
```

Escribe la idea de tu post (por ejemplo: *"Reflexión sobre cómo la IA
está cambiando el trabajo de los desarrolladores"*) y el chatbot
generará y mostrará:

- **Título**: un gancho corto y atractivo.
- **Contenido**: el cuerpo del post, con estructura de apertura,
  desarrollo y cierre.
- **Hashtags**: entre 3 y 6 hashtags relevantes.
- **Categoría**: una de las categorías predefinidas (Tecnología,
  Negocios, Carrera Profesional, Liderazgo, Marketing, Desarrollo
  Personal).

Escribe `/salir` en cualquier momento para terminar.

## Cómo funciona

### El modelo Pydantic (`models/linkedin_post.py`)

```python
class LinkedinPost(BaseModel):
    title: str
    content: str
    hashtags: list[str]
    category: Literal["Tecnología", "Negocios", ...]
```

Esta clase se pasa directamente como `text_format` a
`client.responses.parse()`. OpenAI usa su schema (generado
automáticamente a partir de las anotaciones de tipo) para **restringir
la generación del modelo a nivel de token**, garantizando que la
salida sea siempre JSON válido con exactamente estos cuatro campos.

> **Nota técnica**: el modo *strict* de Structured Outputs de OpenAI
> no aplica restricciones tipo `min_length`/`max_length`/`pattern` a
> nivel de schema (se ignoran para la generación). Por eso `category`
> usa un `Literal` (los enumerados sí se aplican de verdad), y las
> expectativas de longitud de `hashtags`/`content` se indican en la
> `description` de cada campo, que el modelo sí lee como instrucción.

### El cliente de OpenAI (`core/api_client.py`)

```python
response = client.responses.parse(
    model="gpt-4o-2024-08-06",
    input=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": idea},
    ],
    text_format=LinkedinPost,
)
post = response.output_parsed  # instancia de LinkedinPost ya validada
```

`LinkedinPostGenerator.generate_post()` comprueba, en este orden:

1. **Respuesta incompleta** (`response.status == "incomplete"`, por
   ejemplo si se alcanza el límite de tokens antes de terminar de
   generar el JSON) → `IncompleteResponseError`.
2. **Rechazo del modelo** (`refusal`, cuando el modelo se niega a
   generar el contenido por política de seguridad) → `ContentRefusedError`.
3. **Ausencia de contenido parseado** o **fallo de validación de
   Pydantic** → `InvalidStructuredOutputError`.

Además, traduce los errores del SDK de OpenAI (autenticación, límites
de peticiones, problemas de red, peticiones inválidas) a excepciones
propias y descriptivas.

### El chatbot (`core/chatbot.py`)

`LinkedinPostChatbot` es una capa fina sobre `LinkedinPostGenerator`
que valida la entrada del usuario y mantiene un pequeño historial de
los posts generados en la sesión (útil para referencia).

## Manejo de errores

| Situación | Excepción | Qué ve el usuario |
|---|---|---|
| API key inválida o ausente | `APIConnectionOrAuthError` | Mensaje claro + la app termina (no tiene sentido reintentar) |
| Sin conexión a internet | `APIConnectionOrAuthError` | Mensaje claro, se puede reintentar |
| Límite de peticiones/cuota excedido | `RateLimitOrQuotaError` | Mensaje claro, se puede reintentar más tarde |
| El modelo rechaza la idea (política de contenido) | `ContentRefusedError` | Se muestra el motivo del rechazo |
| Respuesta cortada por límite de tokens | `IncompleteResponseError` | Se sugiere reformular con una idea más breve |
| JSON no cumple el schema esperado | `InvalidStructuredOutputError` | Se muestra el detalle de validación |
| Idea vacía | `ValueError` | Se pide al usuario que escriba algo |

En todos los casos (salvo el fallo de autenticación, que detiene la
app), el bucle principal en `main.py` continúa, permitiendo que el
usuario intente generar otro post sin tener que reiniciar la
aplicación.
