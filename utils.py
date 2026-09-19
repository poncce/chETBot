# utils.py
# Funciones compartidas de preprocesamiento.
# Se usan TANTO en train.py como en app.py para que el preprocesado
# del input del usuario sea identico al utilizado en el entrenamiento.

import re


# Mapa para quitar acentos sin romper la codificacion.
ACCENT_MAP = {
    "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
    "ü": "u", "ñ": "n",
}


def normalize_text(text):
    """Normaliza un texto: minusculas, sin acentos, sin puntuacion.

    Devuelve una lista de tokens (palabras) ya 'stemizadas'.
    Ejemplo: "Muchas GRACIAS!!!" -> ['much', 'graci']
    """
    text = text.lower()

    # Reemplazamos letras con acentos/ene por su version simple.
    for accented, plain in ACCENT_MAP.items():
        text = text.replace(accented, plain)

    # Todo lo que no sea letra (a-z) se convierte en espacio.
    text = re.sub(r"[^a-z]", " ", text)

    # Tokenizamos por espacios y aplicamos el stemmer.
    tokens = [stem(word) for word in text.split()]
    return tokens


def stem(word):
    """Stemmer espanol muy sencillo (suficiente para el TP).

    Reduce plurales y algunas terminaciones para que "practicas",
    "practica" o "practicando" terminen en la misma raiz.
    """
    w = word

    # Terminaciones largas de sustantivos/verbos -> raiz.
    for suffix in ("amientos", "imientos", "aciones", "esiones",
                   "amientos", "encias", "mientos", "antes", "antes"):
        if len(w) > len(suffix) and w.endswith(suffix):
            return w[: -len(suffix)]

    # Plurales simples ('s', 'es', 'as', 'os').
    if len(w) > 3 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("as"):
        return w[:-1]
    if len(w) > 3 and w.endswith("os"):
        return w[:-1]
    if len(w) > 3 and w.endswith("s"):
        return w[:-1]

    return w


def bag_of_words(tokens, words):
    """Convierte una lista de tokens en un vector bag-of-words.

    Cada posicion del vector vale 1 si la palabra (stemizada) aparece
    en la lista global de palabras del vocabulario, 0 en caso contrario.
    """
    bag = [0] * len(words)
    for token in tokens:
        for i, word in enumerate(words):
            if word == token:
                bag[i] = 1
    return bag