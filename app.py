# app.py
# Backend API REST del chatbot. Expone POST /chat.
# Recibe {"message": "..."} y devuelve {"response": "..."}.
# Si la red neuronal no supera el umbral de confianza (RF-04),
# responde con un mensaje generico y deriva a preceptoria.

import os
import pickle
import random

import json
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from utils import normalize_text, bag_of_words

# Umbral de confianza RF-04: por debajo de 60% -> fallback.
CONFIDENCE_THRESHOLD = 0.60

# Fracion minima de palabras del usuario que deben pertenecer al
# vocabulario de entrenamiento. Complementa el umbral: una pregunta
# fuera de dominio suele tener casi todas sus palabras "desconocidas"
# y no debe pasar el filtro aunque la confianza supere el 60%.
MIN_KNOWN_RATIO = 0.5

# Mensaje de fallback para preguntas fuera de dominio (RF-04).
FALLBACK_RESPONSE = (
    "No estoy seguro de haber entendido tu consulta. Prob\u00e1 reformular "
    "la pregunta o escrib\u00ed a preceptor\u00eda: "
    "preceptoria@campus.edu.ar para una respuesta precisa."
)

# Directorio base del proyecto (donde viven modelo y frontend).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_model_files():
    """Carga el modelo y los metadatos guardados por train.py.

    Devuelve (model, words, classes) o lanza RuntimeError si falta algo.
    """
    from tensorflow.keras.models import load_model

    if not all(os.path.exists(os.path.join(BASE_DIR, f)) for f in
               ("chatbot_model.h5", "words.pkl", "classes.pkl")):
        raise RuntimeError(
            "Faltan archivos del modelo. Ejecuta primero: python train.py"
        )

    model = load_model(os.path.join(BASE_DIR, "chatbot_model.h5"))
    with open(os.path.join(BASE_DIR, "words.pkl"), "rb") as f:
        words = pickle.load(f)
    with open(os.path.join(BASE_DIR, "classes.pkl"), "rb") as f:
        classes = pickle.load(f)
    return model, words, classes


# --- Carga inicial del modelo (al arrancar el servidor) ---
model, words, classes = load_model_files()
print(f"Modelo cargado. {len(classes)} intenciones, {len(words)} palabras en vocabulario.")

# Tambien cargamos las respuestas una sola vez al iniciar.
with open(os.path.join(BASE_DIR, "intents.json"), "r", encoding="utf-8") as f:
    intents = json.load(f)["intents"]

app = Flask(__name__, static_folder=None)
# CORS habilitado para que el frontend pueda consumir la API
# incluso si se abre index.html como archivo local (file://).
CORS(app)


@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint principal: procesa la pregunta del usuario."""
    # Manejo de errores: input no JSON o sin 'message'.
    if not request.is_json:
        return jsonify({"response": FALLBACK_RESPONSE}), 400

    data = request.get_json()
    message = (data or {}).get("message", "")

    # Input vacio o no textual.
    if not message or not isinstance(message, str) or not message.strip():
        return jsonify({"response": FALLBACK_RESPONSE}), 400

    try:
        # 1) Preprocesamos igual que en train.py y construimos el bag.
        tokens = normalize_text(message)
        if not tokens:  # mensaje sin palabras reconocibles (solo simbolos)
            return jsonify({"response": FALLBACK_RESPONSE})

        bow = bag_of_words(tokens, words)

        # 2) Predecimos probabilidades por intencion.
        prediction = model.predict(np.array([bow]), verbose=0)[0]

        # 3) Umbral de confianza RF-04 + cobertura de vocabulario.
        max_index = int(np.argmax(prediction))
        confidence = float(prediction[max_index])
        known_ratio = sum(1 for t in tokens if t in words) / len(tokens)
        if confidence < CONFIDENCE_THRESHOLD or known_ratio < MIN_KNOWN_RATIO:
            return jsonify({"response": FALLBACK_RESPONSE})

        # 4) Respuesta aleatoria del intent elegido.
        tag = classes[max_index]
        for intent in intents:
            if intent["tag"] == tag:
                response = random.choice(intent["responses"])
                return jsonify({"response": response})

    except Exception:  # nunca crashear con inputs raros
        return jsonify({"response": FALLBACK_RESPONSE})

    return jsonify({"response": FALLBACK_RESPONSE})


@app.route("/health", methods=["GET"])
def health():
    """Endpoint util para verificar que el servidor esta vivo."""
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    """Sirve el frontend (index.html) desde el servidor por comodidad."""
    return send_from_directory(BASE_DIR, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)