"""Micro-ventanas DENSAS de voz (ronda 4, exploratoria dirigida).

La réplica R (la mejor voz en depresión) toma UN trozo de 0.19 s cada 2.5 s: usa solo el
7.6% del audio. Aquí se cubre TODO el audio editado con trozos contiguos de d segundos,
d en {0.1, 0.19, 0.5, 1.0}. Cada trozo pasa por wav2vec2-large-robust; se toma la última
capa y se promedian sus frames. Por participante se guardan la media y la desviación
sobre trozos, para cada duración.
La duración NO se elige a ojo: es un hiperparámetro que se elige en la CV interna
(buscar_mejor_modelo.py, rama RD).

Mismo preprocesamiento que R / Psiquiatria: 16 kHz, normalización por pico, y la
normalización por trozo del feature extractor.
Entrada: features/audio_editado_denoised/NNN_recortado_denoised.wav (79, validados)
Salida:  features/emb_microdensas/NNN.npz  (gitignored). Reanudable por participante.

Uso:  python experimento_embeddings/extraer_microventanas_densas.py
"""
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

REPO = Path(__file__).resolve().parent.parent
AUDIO = REPO / "features" / "audio_editado_denoised"
OUT = REPO / "features" / "emb_microdensas"
MODEL_ID = "facebook/wav2vec2-large-robust"
SR = 16000
DURS = {"d010": 1600, "d019": 3000, "d050": 8000, "d100": 16000}  # muestras a 16 kHz
BATCH_MUESTRAS = 1_500_000  # ~94 s de audio por lote de GPU


@torch.no_grad()
def embeber_trozos(y, largo, fe, model):
    n = len(y) // largo
    trozos = y[: n * largo].reshape(n, largo)            # contiguos, sin solape; se descarta la cola
    bs = max(1, BATCH_MUESTRAS // largo)
    out = []
    for i in range(0, n, bs):
        x = fe(list(trozos[i:i + bs]), sampling_rate=SR, return_tensors="pt").input_values
        h = model(x.cuda().half()).last_hidden_state.float().mean(dim=1)
        out.append(h.cpu().numpy())
    return np.concatenate(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID).cuda().eval().half()
    wavs = sorted(AUDIO.glob("*_recortado_denoised.wav"))
    pendientes = [w for w in wavs if not (OUT / f"{int(w.name[:3]):03d}.npz").exists()]
    print(f"{len(wavs)} audios, {len(pendientes)} pendientes", flush=True)
    for k, w in enumerate(pendientes, 1):
        y, _ = librosa.load(str(w), sr=SR, mono=True)
        y = y / (np.max(np.abs(y)) + 1e-9)
        guardar = {}
        for nombre, largo in DURS.items():
            E = embeber_trozos(y, largo, fe, model)
            guardar[nombre] = E.mean(axis=0).astype(np.float32)
            guardar["s" + nombre[1:]] = E.std(axis=0).astype(np.float32)
            guardar["n_" + nombre] = E.shape[0]
        np.savez_compressed(OUT / f"{int(w.name[:3]):03d}.npz", **guardar)
        if k % 10 == 0 or k == len(pendientes):
            print(f"  {k}/{len(pendientes)}  (último {w.name[:3]}: {guardar['n_d019']} trozos de 0.19 s)",
                  flush=True)


if __name__ == "__main__":
    main()
