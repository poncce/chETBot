# train.py
# Entrena una red neuronal (MLP) para clasificar intenciones del chatbot
# y guarda el modelo + vocabulario en disco.

import json
import pickle
import random

import numpy as np
import tensorflow as tf

# Mismo preprocesado que usara app.py en el momento de predecir.
from utils import normalize_text, bag_of_words


def load_intents(path="intents.json"):
    """Lee el archivo de conocimiento y devuelve la lista de intents."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["intents"]


def build_training_data(intents):
    """Construye el vocabulario y los conjuntos de entrenamiento.

    - words:  lista de palabras unicas de todos los patterns (stemizadas)
    - classes: lista de tags (intenciones)
    - training: lista de (bag_of_words, one_hot_tag)
    """
    patterns = []
    tags = []

    for intent in intents:
        for pattern in intent["patterns"]:
            patterns.append(normalize_text(pattern))
            tags.append(intent["tag"])

    # Vocabulario global de palabras unicas.
    words = sorted(set(sum(patterns, [])))
    classes = sorted(set(tags))
    print(f"{len(patterns)} patterns en {len(classes)} intenciones.")

    training = []
    for i, pattern in enumerate(patterns):
        bag = bag_of_words(pattern, words)

        # One-hot encoding del tag correspondiente.
        output_row = [0] * len(classes)
        output_row[classes.index(tags[i])] = 1
        training.append([bag, output_row])

    random.shuffle(training)  # evita que la red aprenda en orden fijo

    # Separamos en matrices X (features) e y (target).
    training = np.array(training, dtype=object)
    X = np.array(training[:, 0].tolist(), dtype=np.float32)
    y = np.array(training[:, 1].tolist(), dtype=np.float32)
    print(f"Vocabulario: {len(words)} palabras.")

    return words, classes, X, y


def build_model(input_size, output_size):
    """Crea la red neuronal secuencial (MLP de 2 capas ocultas)."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_shape=(input_size,), activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(output_size, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    intents = load_intents()
    words, classes, X, y = build_training_data(intents)

    model = build_model(len(words), len(classes))
    model.fit(X, y, epochs=200, batch_size=8, verbose=1)

    # Guardamos modelo y metadatos en disco para app.py.
    model.save("chatbot_model.h5")
    with open("words.pkl", "wb") as f:
        pickle.dump(words, f)
    with open("classes.pkl", "wb") as f:
        pickle.dump(classes, f)

    print("Modelo entrenado y guardado: chatbot_model.h5, words.pkl, classes.pkl")


if __name__ == "__main__":
    main()