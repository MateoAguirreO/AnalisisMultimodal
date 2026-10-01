"""Robustez de la voz R (micro-ventanas de 0.19 s, una cada 2.5 s) frente al PUNTO DE
MUESTREO.

R toma la micro-ventana al inicio de cada ventana de 5 s: posiciones 0, 2.5, 5.0... s.
Las micro-ventanas DENSAS de 0.19 s (que contienen a las de R y 12 veces más) rinden
bastante menos (0.61 vs 0.68). Si la señal de R es real, desplazar el punto de muestreo
(0.5, 1.0, 1.5, 2.0 s) no debería cambiar mucho el AUC; si varía mucho, el 0.68 fue en
parte suerte del muestreo.

Misma receta que la réplica R; mismo protocolo (semilla 2026, 5x5, selección anidada,
AUC estándar). Las predicciones de V se reutilizan de predicciones_<dx>_r3.csv.

Uso:  python experimento_embeddings/robustez_offset_R.py --extraer
      python experimento_embeddings/robustez_offset_R.py --evaluar
"""
import argparse

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

import evaluar_ramas as ev
from evaluar_emb_psiquiatria import configs_psiquiatria
from replicar_emb_psiquiatria import AUDIO, HOP, MODEL_ID, SNIP, SR, WIN

OFFSETS_S = [0.0, 0.5, 1.0, 1.5, 2.0]
NPZ = ev.REPO / "features" / "emb_R_offsets.npz"


def ventanas_desde(y, off):
    """Igual que ventanas() de la réplica, pero empezando en `off` muestras."""
    segs, s = [], off
    while s < len(y):
        c = y[s:s + WIN]
        if len(c) >= SR:
            segs.append(np.pad(c, (0, WIN - len(c))) if len(c) < WIN else c)
        s += HOP
    return segs


@torch.no_grad()
def extraer():
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    m = Wav2Vec2Model.from_pretrained(MODEL_ID).cuda().eval().half()
    out = {}
    wavs = sorted(AUDIO.glob("*_recortado_denoised.wav"))
    for k, w in enumerate(wavs, 1):
        y, _ = librosa.load(str(w), sr=SR, mono=True)
        y = y / (np.max(np.abs(y)) + 1e-9)
        for o in OFFSETS_S:
            x = fe(ventanas_desde(y, int(o * SR)), sampling_rate=SR, return_tensors="pt",
                   padding="max_length", max_length=SNIP, truncation=True).input_values
            out[f"o{int(o * 10):02d}_{w.name[:3]}"] = (
                m(x.cuda().half()).last_hidden_state.float().mean(dim=1).mean(dim=0).cpu().numpy())
        if k % 20 == 0:
            print(f"  {k}/{len(wavs)}", flush=True)
    np.savez_compressed(NPZ, **out)
    print(f"OK -> {NPZ}", flush=True)


def evaluar(dx="depresion"):
    vid, _ = ev.cargar(dx)
    pids, y = vid.index.values, vid["label"].values
    z = np.load(NPZ)
    V = pd.read_csv(ev.OUT_DIR / f"predicciones_{dx}_r3.csv").set_index(["repeat", "pid"])["V"]
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=ev.SEED).split(pids, y))
    filas = []
    for o in OFFSETS_S:
        X = np.stack([z[f"o{int(o * 10):02d}_{p:03d}"] for p in pids])
        preds = []
        for i, (tr, te) in enumerate(splits):
            s = ev.elegir_tabular(X, y, tr, te, configs_psiquiatria())[1]
            v = V.loc[[(i // 5, p) for p in pids[te]]].values
            preds.append(pd.DataFrame({"repeat": i // 5, "y": y[te], "R": s, "R+V": (s + v) / 2}))
        P = pd.concat(preds)
        for m in ["R", "R+V"]:
            rep = [roc_auc_score(g.y, g[m]) for _, g in P.groupby("repeat")]
            filas.append({"offset_s": o, "modelo": m, "auc_media": np.mean(rep), "auc_de": np.std(rep)})
        print(f"  offset {o:.1f}s listo", flush=True)
    R = pd.DataFrame(filas).round(4)
    R.to_csv(ev.OUT_DIR / f"robustez_offset_R_{dx}.csv", index=False)
    print(R.pivot(index="offset_s", columns="modelo", values="auc_media").to_string())
    for m in ["R", "R+V"]:
        a = R[R.modelo == m].auc_media
        print(f"{m}: AUC según punto de muestreo -> media {a.mean():.3f}, rango {a.min():.3f}–{a.max():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraer", action="store_true")
    ap.add_argument("--evaluar", action="store_true")
    a = ap.parse_args()
    if a.extraer:
        extraer()
    if a.evaluar:
        evaluar()
