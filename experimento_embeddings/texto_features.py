"""Features de texto por participante, a partir de las transcripciones de Whisper
(features/transcripciones/NNN.json, solo la voz del participante; el 66 incluye al
entrevistador).

- emb: sentence-transformers `paraphrase-multilingual-mpnet-base-v2`, media de los
       fragmentos de la transcripción -> 768 dimensiones.
- lex: features interpretables fijadas A PRIORI (listas abajo, antes de ver resultados):
       tasas por cada 100 palabras de primera persona singular, emociones negativas,
       emociones positivas, palabras absolutistas y negaciones, más número de palabras y
       palabras por minuto. Motivación: marcadores lingüísticos de depresión reportados en
       la literatura (foco en uno mismo, afecto negativo, pensamiento absolutista).

Salida: features/texto_features.npz (pids, emb, lex, lex_nombres)  -> gitignored.
Uso:    python experimento_embeddings/texto_features.py
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

REPO = Path(__file__).resolve().parent.parent
TRANS = REPO / "features" / "transcripciones"
OUT = REPO / "features" / "texto_features.npz"
MODELO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

LEXICOS = {
    "primera_persona": {"yo", "me", "mi", "mis", "mio", "mia", "mios", "mias", "conmigo"},
    "emocion_negativa": {
        "triste", "tristeza", "llorar", "lloro", "llore", "lloraba", "soledad", "cansado", "cansada",
        "cansancio", "miedo", "temor", "preocupa", "preocupado", "preocupada", "preocupacion", "estres",
        "estresado", "estresada", "ansiedad", "ansioso", "ansiosa", "nervios", "nervioso", "nerviosa",
        "depresion", "deprimido", "deprimida", "mal", "dolor", "dificil", "problema", "problemas", "sufrir",
        "sufro", "culpa", "rabia", "enojo", "aburrido", "aburrida", "desanimado", "desanimada", "vacio",
        "muerte", "murio"},
    "emocion_positiva": {
        "feliz", "felicidad", "alegre", "alegria", "bien", "bueno", "buena", "tranquilo", "tranquila",
        "contento", "contenta", "disfruto", "disfrutar", "gusta", "encanta", "amor", "divertido",
        "divertida", "emocionado", "emocionada"},
    "absolutista": {"siempre", "nunca", "nada", "nadie", "todo", "todos", "todas", "completamente",
                    "totalmente", "jamas", "absolutamente", "ninguno", "ninguna"},
    "negacion": {"no", "ni", "tampoco"},
}


def palabras(texto):
    t = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z]+", t)


def main():
    archivos = sorted(TRANS.glob("*.json"))
    st = SentenceTransformer(MODELO, device="cuda")
    pids, emb, lex = [], [], []
    for f in archivos:
        d = json.loads(f.read_text(encoding="utf-8"))
        frag = [x["texto"] for x in d.get("fragmentos", []) if x["texto"]]
        texto = " ".join(frag)
        w = palabras(texto)
        n = max(len(w), 1)
        fila = [100 * sum(p in L for p in w) / n for L in LEXICOS.values()]
        fila += [len(w), len(w) / max(d["duracion_s"], 1) * 60]
        vec = st.encode(frag or [texto], batch_size=32, normalize_embeddings=True).mean(axis=0)
        pids.append(d["pid"]); emb.append(vec); lex.append(fila)
    np.savez_compressed(OUT, pids=np.array(pids), emb=np.array(emb, dtype=np.float32),
                        lex=np.array(lex, dtype=np.float32),
                        lex_nombres=np.array(list(LEXICOS) + ["n_palabras", "palabras_por_min"]))
    print(f"OK -> {OUT}: {len(pids)} participantes, emb {np.array(emb).shape}, lex {np.array(lex).shape}")


if __name__ == "__main__":
    main()
