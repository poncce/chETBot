# Chatbot Escolar

Asistente virtual para trámites académicos y administrativos de una escuela técnica. Responde consultas sobre **prácticas profesionalizantes**, **seguridad en talleres**, **mesas de examen**, **documentación** y **contacto**.

Está compuesto por una red neuronal (MLP) que clasifica la intención de cada pregunta y un frontend de chat que consume la API.

## Stack

| Componente | Tecnología |
|---|---|
| Backend | Python + Flask |
| Modelo | TensorFlow / Keras (MLP de 2 capas ocultas) |
| Datos | `intents.json` (patrones y respuestas) |
| Frontend | HTML + CSS + JavaScript (sin dependencias) |
| Deploy | Render (blueprint `render.yaml`) |

## Estructura del proyecto

```
.
├── app.py            # API Flask: POST /chat, GET /, GET /health
├── train.py          # Entrena la red neuronal y guarda el modelo
├── utils.py          # Preprocesamiento compartido (normalización, stemming, bag-of-words)
├── intents.json      # Conocimiento: intenciones, patrones y respuestas
├── chatbot_model.h5  # Modelo entrenado
├── words.pkl         # Vocabulario (palabras que conoce el modelo)
├── classes.pkl       # Lista de intenciones
├── index.html        # Frontend del chat (servido por Flask)
├── requirements.txt  # Dependencias
├── runtime.txt       # Versión de Python para Render
└── render.yaml       # Blueprint de deploy en Render
```

Los archivos del modelo (`chatbot_model.h5`, `words.pkl`, `classes.pkl`) ya están entrenados e incluidos en el repositorio, por lo que **no es obligatorio reentrenar** para poner el bot en marcha.

## Requisitos

- Python **3.12** (el proyecto está probado con la 3.12.10, ver `runtime.txt`)
- `pip` y `venv` disponibles
- ~8 GB de disco libres: TensorFlow instala paquetes pesados

## Ejecutar localmente

### 1. Crear el entorno virtual e instalar dependencias

En Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

En Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Si estuviste dentro del entorno y querés salir: `deactivate`.

### 2. (Opcional) Entrenar el modelo

Solo es necesario si modificaste `intents.json` (agregaste/quitaste intenciones o respuestas) o si los archivos del modelo no existen.

```bash
python train.py
```

El script lee `intents.json`, entrena la red durante 200 épocas y regenera `chatbot_model.h5`, `words.pkl` y `classes.pkl`.

### 3. Levantar el servidor

```bash
python app.py
```

El servidor queda escuchando en **http://localhost:5000** y el frontend del chat se sirve en la raíz:

```
http://localhost:5000/
```

### 4. Usar el chat

- **Desde el navegador:** entrá a `http://localhost:5000/` y probá preguntas como *"¿cuándo son las mesas de previas?"* o *"¿cómo saco la constancia de alumno regular?"*.
- **Desde la terminal:** el endpoint responde en `POST /chat`:

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"cuales son las normas de seguridad en el taller\"}"
```

Respuesta esperada:

```json
{"response": "En talleres y laboratorios es obligatorio: ..."}
```

- **Endpoints disponibles:**

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Sirve el frontend (`index.html`) |
| POST | `/chat` | Recibe `{"message": "..."}`, devuelve `{"response": "..."}` |
| GET | `/health` | Healthcheck: `{"status": "ok"}` |

> El frontend también puede abrirse como archivo local (`index.html` en el navegador); en ese caso se conecta a `http://localhost:5000` automáticamente.

## Cómo funciona

1. El usuario envía una pregunta a `POST /chat` → `app.py`.
2. `utils.normalize_text()` normaliza el texto y lo tokeniza (minúsculas, sin acentos, sin puntuación, con stemming en español).
3. `utils.bag_of_words()` convierte los tokens en un vector booleano contra el vocabulario.
4. La red neuronal predice la probabilidad de cada intención.
5. Si la confianza es menor al **60%** o más de la mitad de las palabras del usuario son desconocidas, responde un mensaje de fallback y deriva a preceptoría (RF-04).

## Configuración

Parámetros ajustables en `app.py`:

| Parámetro | Ubicación | Valor por defecto | Descripción |
|---|---|---|---|
| `CONFIDENCE_THRESHOLD` | `app.py:24` | `0.60` | Confianza mínima para aceptar una predicción |
| `MIN_KNOWN_RATIO` | `app.py:29` | `0.5` | Fracción mínima de palabras del usuario presentes en el vocabulario |
| `FALLBACK_RESPONSE` | `app.py:33` | — | Mensaje que se devuelve cuando no hay match |

El puerto de escucha por defecto es `5000` (`app.py:131`).

## Agregar nuevas respuestas

1. Editá `intents.json`: agregá patrones a intentos existentes o creá un nuevo `tag` con `patterns` y `responses`.
2. Reentrená el modelo: `python train.py`.
3. Reiniciá el servidor para recargar el modelo (`app.py` carga el modelo al arrancar).