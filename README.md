# Chatbot Escolar con IA — PSR-TP03-C2

Asistente virtual que resuelve dudas recurrentes sobre trámites académicos y
administrativos de la escuela técnica, usando una **red neuronal (MLP)** en
Python expuesta como **API REST**.

Corresponde al Trabajo Práctico de *Programación sobre Redes*.

## Requerimientos funcionales cubiertos

- **RF-01 — PLN**: entiende la intención aunque haya errores ortográficos,
  abreviaturas o sinónimos. Ej.: "constancia de alumno", "certificado regular",
  "papel para la obra social" → misma respuesta.
- **RF-02 — Ejes temáticos** (mínimo 4):
  1. Prácticas Profesionalizantes / Pasantías (requisitos y fechas).
  2. Protocolo de seguridad y uso de herramientas en Talleres y Laboratorios.
  3. Mesas de examen (previas y equivalencias).
  4. Emisión de documentación (constancia de alumno regular, analíticos).
- **RF-03 — API REST**: `POST /chat`, recibe `{"message": "..."}` y devuelve
  `{"response": "..."}`.
- **RF-04 — Fallback**: si la red no supera ~60% de confianza (o la consulta
  está fuera de vocabulario) responde un mensaje genérico y deriva a
  preceptoría.

## Estructura

```
chatbot-escolar/
├── venv/                 # entorno virtual (no se versiona)
├── intents.json          # base de conocimiento (tags, patterns, responses)
├── utils.py              # preprocesado compartido (normalización + stemmer + bow)
├── train.py              # entrena el MLP y guarda modelo + vocabulario
├── app.py                # API REST Flask (POST /chat) con CORS y fallback
├── words.pkl             # vocabulario (generado por train.py)
├── classes.pkl           # tags de intenciones (generado por train.py)
├── chatbot_model.h5      # pesos de la red (generado por train.py)
├── index.html            # frontend conectado al endpoint real
├── requirements.txt
└── README.md
```

## Instalación

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

> Requiere Python 3.10+ (probado con 3.12 y TensorFlow 2.21).

## Ejecución

1. **Entrenar el modelo** (una sola vez, genera `words.pkl`, `classes.pkl`
   y `chatbot_model.h5`):

   ```bash
   python train.py
   ```

2. **Levantar la API**:

   ```bash
   python app.py
   ```

   El servidor queda en `http://localhost:5000`.

3. **Abrir el chat**: entrá a `http://localhost:5000/` (o abrí `index.html`
   como archivo local; CORS está habilitado para ambos casos).

## Uso de la API

```bash
curl -X POST http://localhost:5000/chat \
     -H "Content-Type: application/json" \
     -d '{"message":"¿cómo consigo una pasantía?"}'
```

Respuesta:

```json
{"response": "Para las pasantías necesitas: 1) ser alumno de 6to o 7mo..." }
```

Endpoints:

| Método | Ruta     | Descripción                                  |
|--------|----------|----------------------------------------------|
| GET    | `/`      | Sirve el frontend `index.html`               |
| GET    | `/health`| Verifica que el servidor está vivo            |
| POST   | `/chat`  | Envía `{"message": "..."}` → `{"response"}`  |

## Implementación

- **Preprocesado** (`utils.py`): minúsculas, sin acentos, sin puntuación,
  *stemmer* español simple y *bag-of-words*. Las mismas funciones se usan en
  entrenamiento e inferencia para que el preprocesado del input sea idéntico.
- **Red neuronal** (`train.py`): MLP secuencial con 2 capas ocultas (128 y 64
  neuronas, ReLU), dropout 0.5 y softmax; `adam` + `categorical_crossentropy`.
- **API** (`app.py`): carga el modelo y el vocabulario al iniciar, aplica el
  umbral de confianza de 60 % (RF-04) más un chequeo de cobertura del
  vocabulario para filtrar preguntas fuera de dominio, nunca crashea con
  inputs vacíos o malformados (devuelve HTTP 400 con el fallback) y habilita
  CORS para que el frontend pueda consumirla desde cualquier origen.

## Desacople backend–frontend

El backend (`app.py`) expone la API de forma independiente. El frontend
(`index.html`) consume `POST /chat` mediante `fetch` y funciona tanto servido
desde Flask (`/`) como abierto como archivo local.

## Deployment en Render

Este repo está preparado para deployar gratis en [Render](https://render.com)
(archivo `render.yaml` + `runtime.txt`). El modelo ya viene entrenado y
versionado, por lo que **no se entrena en el servidor**: solo se instalan
dependencias (`tensorflow-cpu` para un build liviano) y se levanta con
Gunicorn.

Pasos:

1. Subí el proyecto a un repo en GitHub:
   ```bash
   git remote add origin https://github.com/TU_USUARIO/chatbot-escolar.git
   git push -u origin main
   ```
2. En [dashboard.render.com](https://dashboard.render.com) → **New → Blueprint**,
   conectá el repo. Render lee `render.yaml` y crea el web service automáticamente.
   (Alternativa manual: **New → Web Service**, conectá el repo y usá los comandos
   de `render.yaml` como Build/Start.)
3. Al terminar el build, te da una URL tipo `https://chatbot-escolar.onrender.com`.
   El frontend queda en la raíz (`/`) y la API en `POST /chat`.

Notas:
- Se usa un único worker porque el modelo se carga en memoria al arrancar
  (instancia free con 512 MB de RAM).
- El primer deploy tarda unos minutos (descarga de TensorFlow). El artefacto
  `.h5` ya está en el repo, así que no hay etapa de entrenamiento.