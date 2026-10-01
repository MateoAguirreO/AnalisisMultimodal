"""¿Funciona la voz R (micro-ventanas de 0.19 s, última capa de wav2vec2-large-robust) sobre
audio CRUDO + Demucs? Es lo que recibe la plataforma: no tiene el audio editado a mano.

Variantes (mismo protocolo: semilla 2026, 5x5, selección anidada, AUC agrupado e intra-día):
  R_ed         entrena y prueba con audio editado (referencia, ronda 3)
  R_crudo      entrena y prueba con audio crudo (rama reentrenada de forma consistente)
  R_ed2crudo   entrena con editado y prueba con crudo (desplegar el modelo actual tal cual)
y sus fusiones con V. Las predicciones de V se reutilizan de predicciones_<dx>_r3.csv:
mismos folds y semilla, así que son idénticas a recalcularlas.

Uso:  python experimento_embeddings/evaluar_R_crudo.py --extraer      (GPU, ~3 min)
      python experimento_embeddings/evaluar_R_crudo.py --dx depresion
"""
import argparse

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

import evaluar_ramas as ev
from evaluar_emb_psiquiatria import configs_psiquiatria
from replicar_emb_psiquiatria import MODEL_ID, SNIP, SR, ventanas

CRUDO = ev.REPO / "features" / "audio_crudo_demucs"
NPZ_ED = ev.REPO / "features" / "emb_replica_psiquiatria.npz"
NPZ_CRUDO = ev.REPO / "features" / "emb_replica_crudo.npz"


@torch.no_grad()
def extraer():
    fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    m = Wav2Vec2Model.from_pretrained(MODEL_ID).cuda().eval().half()
    out = {}
    wavs = sorted(CRUDO.glob("*.wav"))
    for k, w in enumerate(wavs, 1):
        y, _ = librosa.load(str(w), sr=SR, mono=True)
        y = y / (np.max(np.abs(y)) + 1e-9)          # normalización por pico, como Psiquiatria
        x = fe(ventanas(y), sampling_rate=SR, return_tensors="pt", padding="max_length",
               max_length=SNIP, truncation=True).input_values.cuda().half()
        out[f"{int(w.stem):03d}"] = m(x).last_hidden_state.float().mean(dim=1).cpu().numpy()
        if k % 20 == 0:
            print(f"  {k}/{len(wavs)}", flush=True)
    np.savez_compressed(NPZ_CRUDO, **out)
    ed = np.load(NPZ_ED)
    comunes = sorted(set(out) & set(ed.files))
    A = np.stack([out[p].mean(0) for p in comunes]); B = np.stack([ed[p].mean(0) for p in comunes])
    r = [np.corrcoef(A[:, d], B[:, d])[0, 1] for d in range(A.shape[1])]
    print(f"OK -> {NPZ_CRUDO}. Correlación entre participantes crudo vs editado, por dimensión: "
          f"mediana {np.nanmedian(r):.3f}, p10 {np.nanpercentile(r, 10):.3f}", flush=True)


def elegir_cruzado(X_fit, X_pred, y, tr, te, configs):
    """Como ev.elegir_tabular, pero ajusta con X_fit[tr] y predice X_pred[te]."""
    mejor_nombre, mejor_score, mejor_est = None, -np.inf, None
    for nombre, (pipe, grid) in configs.items():
        if grid:
            gs = GridSearchCV(clone(pipe), grid, cv=ev.inner_cv(), scoring="roc_auc",
                              n_jobs=ev.N_JOBS).fit(X_fit[tr], y[tr])
            score, est = gs.best_score_, gs.best_estimator_
        else:
            score = cross_val_score(clone(pipe), X_fit[tr], y[tr], cv=ev.inner_cv(),
                                    scoring="roc_auc", n_jobs=ev.N_JOBS).mean()
            est = pipe
        if score > mejor_score:
            mejor_nombre, mejor_score, mejor_est = nombre, score, est
    if not configs[mejor_nombre][1]:
        mejor_est = clone(mejor_est).fit(X_fit[tr], y[tr])
    return mejor_nombre, mejor_est.predict_proba(X_pred[te])[:, 1]


def evaluar(dx):
    vid, _ = ev.cargar(dx)
    pids, y = vid.index.values, vid["label"].values
    ed, cr = np.load(NPZ_ED), np.load(NPZ_CRUDO)
    X_ed = np.stack([ed[f"{p:03d}"].mean(0) for p in pids])
    X_cr = np.stack([cr[f"{p:03d}"].mean(0) for p in pids])
    V = pd.read_csv(ev.OUT_DIR / f"predicciones_{dx}_r3.csv").set_index(["repeat", "pid"])["V"]
    cfg = configs_psiquiatria()
    filas = []
    for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                                                         random_state=ev.SEED).split(pids, y)):
        s = {"V": V.loc[[(i // 5, p) for p in pids[te]]].values,
             "R_ed": ev.elegir_tabular(X_ed, y, tr, te, cfg)[1],
             "R_crudo": ev.elegir_tabular(X_cr, y, tr, te, cfg)[1],
             "R_ed2crudo": elegir_cruzado(X_ed, X_cr, y, tr, te, cfg)[1]}
        for r in ("R_ed", "R_crudo", "R_ed2crudo"):
            s[f"{r}+V"] = (s[r] + s["V"]) / 2
        filas.append(pd.DataFrame({"repeat": i // 5, "pid": pids[te], "y": y[te], **s}))
    P = pd.concat(filas, ignore_index=True)
    P.to_csv(ev.OUT_DIR / f"predicciones_{dx}_R_crudo.csv", index=False)
    dia = P.pid.map(pd.read_csv(ev.SESIONES).set_index("pid").dia).values
    res = []
    ref = [ev.auc_intra_dia(g.y.values, g["V"].values, dia[g.index]) for _, g in P.groupby("repeat")]
    for m in ["V", "R_ed", "R_crudo", "R_ed2crudo", "R_ed+V", "R_crudo+V", "R_ed2crudo+V"]:
        intra = [ev.auc_intra_dia(g.y.values, g[m].values, dia[g.index]) for _, g in P.groupby("repeat")]
        glob_ = [roc_auc_score(g.y, g[m]) for _, g in P.groupby("repeat")]
        res.append({"metodo": m, "auc_intra_media": np.mean(intra), "auc_intra_de": np.std(intra),
                    "auc_rep_media": np.mean(glob_), "auc_rep_de": np.std(glob_),
                    "reps_gana_a_V_intra": int(np.sum(np.array(intra) > np.array(ref)))})
    R = pd.DataFrame(res).round(4)
    R.to_csv(ev.OUT_DIR / f"resumen_{dx}_R_crudo.csv", index=False)
    print(f"=== {dx.upper()} — voz R sobre audio crudo ===\n{R.to_string(index=False)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extraer", action="store_true")
    ap.add_argument("--dx", choices=ev.cfg.DXS)
    a = ap.parse_args()
    if a.extraer:
        extraer()
    if a.dx:
        evaluar(a.dx)
