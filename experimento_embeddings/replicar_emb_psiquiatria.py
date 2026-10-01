"""EXPLORATORIO (ronda 3). ¿Por qué los embeddings originales de Psiquiatria tienen señal en
depresión y los nuestros no, con el mismo modelo y el mismo audio?

Hallazgo en su código (embedding_pipeline_5s.py:460-467): cada ventana de 5 s pasa por
processor(..., padding="max_length", max_length=3000, truncation=True) -> solo entran los
PRIMEROS 3000 muestras = 0.19 s de cada ventana (9 frames), última capa, attention pooling
con pesos aleatorios. Su representación = micro-muestras de 0.19 s cada 2.5 s.

Este script, para los 79 participantes (audio editado + Demucs, validado):
  R      réplica DETERMINISTA: mismo recorte de 0.19 s, última capa, media de los 9 frames
         (sin pesos aleatorios -> reproducible en la plataforma).
  PAUSA  hipótesis interpretable: si las micro-muestras caen en habla o en silencio, su
         media refleja cuánto calla la persona. Se miden 4 features directas de pausa.
Y evalúa con el protocolo de evaluar_ramas (semilla 2026, selección anidada, AUC intra-día)
las ramas P (sus embeddings), R, PAUSA y sus fusiones con V.

Uso:  python experimento_embeddings/replicar_emb_psiquiatria.py
"""
import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

import evaluar_ramas as ev
from evaluar_emb_psiquiatria import NPZ, cargar_psq, configs_psiquiatria

AUDIO = ev.REPO / "features" / "audio_editado_denoised"
OUT_NPZ = ev.REPO / "features" / "emb_replica_psiquiatria.npz"
MODEL_ID = "facebook/wav2vec2-large-robust"
SR, WIN, HOP, SNIP = 16000, 80000, 40000, 3000


def cargar_audio(pid):
    """Como load_and_preprocess de Psiquiatria: 16 kHz mono + normalización por pico."""
    y, _ = librosa.load(str(AUDIO / f"{pid:03d}_recortado_denoised.wav"), sr=SR, mono=True)
    return y / (np.max(np.abs(y)) + 1e-9)


def ventanas(y):
    """Como segment_audio de Psiquiatria: 5 s, salto 2.5 s, mínimo 1 s, padding con ceros."""
    segs, s = [], 0
    while s < len(y):
        c = y[s:s + WIN]
        if len(c) >= SR:
            segs.append(np.pad(c, (0, WIN - len(c))) if len(c) < WIN else c)
        s += HOP
    return segs


def features_pausa(y):
    """Pausas medidas directamente sobre la señal (frames de 25 ms, salto 10 ms).
    Silencio = energía < -35 dB respecto al percentil 95 de energía del participante."""
    rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
    db = 20 * np.log10(rms + 1e-10)
    sil = db < np.percentile(db, 95) - 35
    cambios = np.diff(np.r_[0, sil.astype(int), 0])
    ini, fin = np.where(cambios == 1)[0], np.where(cambios == -1)[0]
    dur = (fin - ini) * 0.01
    pausas = dur[dur >= 0.25]  # pausa = silencio de al menos 250 ms
    minutos = len(y) / SR / 60
    # fracción de micro-muestras de 0.19 s (las que ve su embedding) que caen en silencio
    snips = [s[:SNIP] for s in ventanas(y)]
    snip_db = np.array([20 * np.log10(np.sqrt(np.mean(s ** 2)) + 1e-10) for s in snips])
    return {"frac_silencio": sil.mean(), "pausas_por_min": len(pausas) / minutos,
            "dur_media_pausa": pausas.mean() if len(pausas) else 0.0,
            "frac_microm_silencio": np.mean(snip_db < np.percentile(db, 95) - 35)}


@torch.no_grad()
def replica(pids):
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    m = Wav2Vec2Model.from_pretrained(MODEL_ID).cuda().eval().half()
    R, F = {}, []
    for k, pid in enumerate(pids, 1):
        y = cargar_audio(pid)
        x = fe(ventanas(y), sampling_rate=SR, return_tensors="pt", padding="max_length",
               max_length=SNIP, truncation=True).input_values.cuda().half()
        R[f"{pid:03d}"] = m(x).last_hidden_state.float().mean(dim=1).cpu().numpy()  # (n_seg, 1024)
        F.append({"pid": pid, **features_pausa(y)})
        if k % 20 == 0:
            print(f"  réplica {k}/{len(pids)}", flush=True)
    np.savez_compressed(OUT_NPZ, **R)
    return R, pd.DataFrame(F).set_index("pid")


def configs_pausa():
    pipe = Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))])
    return {"pausa_logreg": (pipe, {"clf__C": [0.01, 0.1, 1.0]})}


def main():
    theirs = np.load(NPZ)
    dia_s = pd.read_csv(ev.SESIONES).set_index("pid").dia
    vid0, _ = ev.cargar("depresion")
    pids_all = vid0.index.values
    R, F = replica(pids_all)
    F.to_csv(ev.REPO / "features" / "pausas.csv")

    # ¿La réplica reproduce la variación ENTRE participantes de sus vectores?
    P_mean = np.stack([theirs[f"dep_{p:03d}"].mean(0) for p in pids_all])
    R_mean = np.stack([R[f"{p:03d}"].mean(0) for p in pids_all])
    r_dim = [np.corrcoef(P_mean[:, d], R_mean[:, d])[0, 1] for d in range(P_mean.shape[1])]
    print(f"\nCorrelación entre participantes, por dimensión (suyos vs réplica): "
          f"mediana {np.nanmedian(r_dim):.3f}, p10 {np.nanpercentile(r_dim, 10):.3f}")
    print("Pausas vs etiqueta (AUC univariado, depresión / ansiedad):")
    for c in F.columns:
        print(f"  {c:22s} dep {roc_auc_score(vid0.label, F.loc[pids_all, c]):.3f}  "
              f"ans {roc_auc_score(ev.cargar('ansiedad')[0].label, F.loc[pids_all, c]):.3f}")

    for dx in ev.cfg.DXS:
        vid, _ = ev.cargar(dx)
        pids, y = vid.index.values, vid["label"].values
        Xv = vid.drop(columns="label").values
        X = {"P": (cargar_psq(theirs, "dep", pids), configs_psiquiatria()),
             "R": (np.stack([R[f"{p:03d}"].mean(0) for p in pids]), configs_psiquiatria()),
             "PAUSA": (F.loc[pids].values, configs_pausa())}
        filas = []
        outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=ev.SEED)
        for i, (tr, te) in enumerate(outer.split(pids, y)):
            s = {"V": ev.elegir_tabular(Xv, y, tr, te, ev.configs_video())[1]}
            for r, (Xr, cf) in X.items():
                s[r] = ev.elegir_tabular(Xr, y, tr, te, cf)[1]
            for r in X:
                s[f"{r}+V"] = (s[r] + s["V"]) / 2
            filas.append(pd.DataFrame({"repeat": i // 5, "pid": pids[te], "y": y[te], **s}))
        P = pd.concat(filas, ignore_index=True)
        P.to_csv(ev.OUT_DIR / f"predicciones_{dx}_r3.csv", index=False)
        dia = P.pid.map(dia_s).values
        res = []
        for m in ["V", "P", "R", "PAUSA", "P+V", "R+V", "PAUSA+V"]:
            intra = [ev.auc_intra_dia(g.y.values, g[m].values, dia[g.index]) for _, g in P.groupby("repeat")]
            glob_ = [roc_auc_score(g.y, g[m]) for _, g in P.groupby("repeat")]
            ref = [ev.auc_intra_dia(g.y.values, g["V"].values, dia[g.index]) for _, g in P.groupby("repeat")]
            res.append({"metodo": m, "auc_intra_media": np.mean(intra), "auc_intra_de": np.std(intra),
                        "auc_rep_media": np.mean(glob_), "auc_rep_de": np.std(glob_),
                        "reps_gana_a_V_intra": int(np.sum(np.array(intra) > np.array(ref)))})
        Rt = pd.DataFrame(res).round(4)
        Rt.to_csv(ev.OUT_DIR / f"resumen_{dx}_r3.csv", index=False)
        print(f"\n=== {dx.upper()} — ronda 3 (exploratoria) ===")
        print(Rt.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
