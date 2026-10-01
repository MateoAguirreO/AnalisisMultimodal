"""Evaluación TOTALMENTE ANIDADA de las ramas V (rostro AU), A (voz eGeMAPS) y
E (voz wav2vec) y de sus fusiones por soft vote. Protocolo en INVENTARIO_Y_PROTOCOLO.md.

- Externo: RepeatedStratifiedKFold(5, 5, random_state=2026). Semilla nueva, nunca usada
  para elegir nada en este repo (el pipeline anterior usó 42 tanto para screening como
  para confirmación).
- En cada fold externo cada rama elige su configuración con CV interna SOLO sobre el
  train externo y luego predice el test externo. Soft vote = promedio simple, sin
  parámetros, así que no agrega selección.
- Rama A en 3 variantes (ver protocolo): A_ed, A_ed2crudo, A_crudo.
- Cada eje por separado (sin modelado conjunto ansiedad–depresión).

Salidas (experimento_embeddings/resultados/, gitignored por la regla `resultados/`):
  predicciones_<dx>.csv   una fila por (repeat, fold, pid) con el score de cada rama
  elecciones_<dx>.csv     config elegida por rama y fold
  resumen_<dx>.csv        AUC por fold (media ± DE) y AUC por repetición

Uso:  python experimento_embeddings/evaluar_ramas.py [--dx ansiedad|depresion|all]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "pipeline_multimodal_xai"))
import config_mm as cfg  # noqa: E402  (también agrega experimentos_sin_pca al path)
from audio_branch import AUDIO_CONFIGS, RamaAudio  # noqa: E402
from data_multimodal import cargar  # noqa: E402
from fusion import VIDEO_CONFIGS, construir_video_config  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 2026
EMB_DIR = REPO / "features" / "emb_w2v_large_robust"               # audio crudo + Demucs
EMB_DIR_ED = REPO / "features" / "emb_w2v_large_robust_editado"    # audio editado (ronda 2)
EGEMAPS_CRUDO = REPO / "features" / "egemaps_audio_crudo.csv"
SESIONES = REPO / "features" / "sesiones.csv"
OUT_DIR = REPO / "experimento_embeddings" / "resultados"
# núcleos para la CV interna; los scripts que ya paralelizan por fuera (permutaciones) lo
# ponen en 1 para no sobre-suscribir la CPU
N_JOBS = -1

FUSIONES = {
    "A_ed+V": ["A_ed", "V"],
    "E+V": ["E", "V"],
    "A_ed+E+V": ["A_ed", "E", "V"],
    "A_ed+E": ["A_ed", "E"],
    "A_crudo+V": ["A_crudo", "V"],
    "A_crudo+E+V": ["A_crudo", "E", "V"],
    "A_ed2crudo+V": ["A_ed2crudo", "V"],
    # ronda 2 (audio editado)
    "E_ed+V": ["E_ed", "V"],
    "A_ed+E_ed+V": ["A_ed", "E_ed", "V"],
    "A_ed+E_ed": ["A_ed", "E_ed"],
}


def inner_cv():
    return StratifiedKFold(5, shuffle=True, random_state=SEED)


def configs_embeddings():
    logreg = Pipeline([("sc", StandardScaler()), ("pca", "passthrough"),
                       ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))])
    grid = {"pca": ["passthrough", PCA(16, random_state=SEED), PCA(32, random_state=SEED)],
            "clf__C": [1e-3, 1e-2, 1e-1]}
    et = Pipeline([("clf", ExtraTreesClassifier(n_estimators=500, max_features="sqrt",
                                                min_samples_leaf=3, class_weight="balanced",
                                                random_state=SEED, n_jobs=1))])
    return {"e_logreg": (logreg, grid), "e_et": (et, {})}


def configs_video():
    return {c: construir_video_config(c) for c in VIDEO_CONFIGS}


def cargar_embeddings(pids, emb_dir=EMB_DIR) -> np.ndarray:
    """Representación principal (fijada a priori): media de capas transformer 1..24."""
    return np.stack([np.load(emb_dir / f"{p:03d}.npz")["mean_layers"][1:].mean(axis=0)
                     for p in pids])


def auc_intra_dia(y, score, dia) -> float:
    """AUC estratificado por día de grabación: fracción de pares (positivo, negativo) DEL
    MISMO DÍA ordenados correctamente (empates = 0.5). Elimina el confusor de sesión."""
    ok = tot = 0.0
    for d in np.unique(dia):
        m = dia == d
        pos, neg = score[m & (y == 1)], score[m & (y == 0)]
        if len(pos) and len(neg):
            dif = pos[:, None] - neg[None, :]
            ok += (dif > 0).sum() + 0.5 * (dif == 0).sum()
            tot += dif.size
    return ok / tot


def elegir_tabular(X, y, tr, te, configs):
    """Elige la config por AUC de CV interna sobre tr; devuelve (nombre, score_test)."""
    mejor_nombre, mejor_score, mejor_est = None, -np.inf, None
    for nombre, (pipe, grid) in configs.items():
        if grid:  # GridSearchCV reajusta la mejor combinación sobre todo X[tr]
            gs = GridSearchCV(clone(pipe), grid, cv=inner_cv(), scoring="roc_auc",
                              n_jobs=N_JOBS).fit(X[tr], y[tr])
            score, est = gs.best_score_, gs.best_estimator_
        else:
            score = cross_val_score(clone(pipe), X[tr], y[tr], cv=inner_cv(),
                                    scoring="roc_auc", n_jobs=N_JOBS).mean()
            est = pipe
        if score > mejor_score:
            mejor_nombre, mejor_score, mejor_est = nombre, score, est
    if not configs[mejor_nombre][1]:  # sin grid: falta ajustarla sobre todo el train externo
        mejor_est = clone(mejor_est).fit(X[tr], y[tr])
    return mejor_nombre, mejor_est.predict_proba(X[te])[:, 1]


def elegir_audio(aud_tr_df, aud_te_df, pids_tr, pids_te, y_tr):
    """Elige config de segmento por AUC OOF de participante (StratifiedGroupKFold interno)."""
    mejor, mejor_auc = None, -np.inf
    for c in AUDIO_CONFIGS:
        s = RamaAudio(config=c, random_state=SEED).scores_oof_participante(aud_tr_df)
        auc = roc_auc_score(y_tr, s.reindex(pids_tr).values)
        if auc > mejor_auc:
            mejor, mejor_auc = c, auc
    ra = RamaAudio(config=mejor, random_state=SEED).fit(aud_tr_df)
    return mejor, ra, ra.scores_test_participante(aud_te_df).reindex(pids_te).values


def evaluar(dx, tag=""):
    vid, aud_ed = cargar(dx)
    pids = vid.index.values
    y = vid["label"].values
    Xv = vid.drop(columns="label").values
    Xe = cargar_embeddings(pids)
    Xe_ed = cargar_embeddings(pids, EMB_DIR_ED)

    aud_cr = pd.read_csv(EGEMAPS_CRUDO)
    aud_cr = aud_cr[aud_cr.pid.isin(pids)].copy()
    aud_cr["label"] = aud_cr.pid.map(vid["label"])
    feats = [c for c in aud_ed.columns if c not in ("pid", cfg.AUDIO_SEG_COL, "label")]
    aud_cr = aud_cr[["pid", "seg_idx"] + feats + ["label"]]
    assert set(aud_cr.pid) == set(pids), "faltan participantes en egemaps_audio_crudo.csv"

    preds, elecciones = [], []
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    t0 = time.time()
    for i, (tr, te) in enumerate(outer.split(pids, y)):
        rep, fold = divmod(i, 5)
        s, e = {}, {}
        e["V"], s["V"] = elegir_tabular(Xv, y, tr, te, configs_video())
        e["E"], s["E"] = elegir_tabular(Xe, y, tr, te, configs_embeddings())
        e["E_ed"], s["E_ed"] = elegir_tabular(Xe_ed, y, tr, te, configs_embeddings())

        p_tr, p_te = pids[tr], pids[te]
        e["A_ed"], ra_ed, s["A_ed"] = elegir_audio(aud_ed[aud_ed.pid.isin(p_tr)],
                                                  aud_ed[aud_ed.pid.isin(p_te)], p_tr, p_te, y[tr])
        # mismo modelo entrenado con audio editado, alimentado con eGeMAPS de audio crudo
        s["A_ed2crudo"] = ra_ed.scores_test_participante(aud_cr[aud_cr.pid.isin(p_te)]).reindex(p_te).values
        e["A_ed2crudo"] = e["A_ed"]
        e["A_crudo"], _, s["A_crudo"] = elegir_audio(aud_cr[aud_cr.pid.isin(p_tr)],
                                                     aud_cr[aud_cr.pid.isin(p_te)], p_tr, p_te, y[tr])
        for nombre, ramas in FUSIONES.items():
            s[nombre] = np.mean([s[r] for r in ramas], axis=0)

        for j, pid in enumerate(p_te):
            preds.append({"repeat": rep, "fold": fold, "pid": pid, "y": y[te][j],
                          **{k: v[j] for k, v in s.items()}})
        elecciones.append({"repeat": rep, "fold": fold, **e})
        print(f"  [{dx}] fold {i + 1}/25  ({time.time() - t0:.0f}s)  "
              f"E={roc_auc_score(y[te], s['E']):.2f} E_ed={roc_auc_score(y[te], s['E_ed']):.2f} "
              f"V={roc_auc_score(y[te], s['V']):.2f} A_ed={roc_auc_score(y[te], s['A_ed']):.2f}",
              flush=True)

    P = pd.DataFrame(preds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    P.to_csv(OUT_DIR / f"predicciones_{dx}{tag}.csv", index=False)
    pd.DataFrame(elecciones).to_csv(OUT_DIR / f"elecciones_{dx}{tag}.csv", index=False)
    return resumir(dx, P, tag)


def resumir(dx, P, tag=""):
    """AUC agrupado e intra-día por repetición; cuenta repeticiones en que cada método
    supera a V (rostro solo, referencia de la ronda 2) en AUC intra-día."""
    dia = P.pid.map(pd.read_csv(SESIONES).set_index("pid").dia)
    metodos = [c for c in P.columns if c not in ("repeat", "fold", "pid", "y")]
    filas = []
    for m in metodos:
        por_rep = P.groupby("repeat").apply(lambda g: roc_auc_score(g.y, g[m]))
        intra = P.groupby("repeat").apply(
            lambda g: auc_intra_dia(g.y.values, g[m].values, dia.loc[g.index].values))
        filas.append({"metodo": m,
                      "auc_intra_media": intra.mean(), "auc_intra_de": intra.std(),
                      "auc_rep_media": por_rep.mean(), "auc_rep_de": por_rep.std(),
                      **{f"intra_rep{r}": v for r, v in intra.items()}})
    R = pd.DataFrame(filas).sort_values("auc_intra_media", ascending=False).round(4)
    cols = [f"intra_rep{r}" for r in range(5)]
    ref = R.set_index("metodo").loc["V", cols]
    R["reps_gana_a_V_intra"] = [int((R.set_index("metodo").loc[m, cols] > ref).sum())
                                for m in R.metodo]
    R.to_csv(OUT_DIR / f"resumen_{dx}{tag}.csv", index=False)
    print(f"\n=== {dx.upper()} (n={P.pid.nunique()}, pos={int(P.groupby('pid').y.first().sum())}) ===")
    print(R[["metodo", "auc_intra_media", "auc_intra_de", "auc_rep_media", "auc_rep_de",
             "reps_gana_a_V_intra"]].to_string(index=False))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", choices=cfg.DXS + ["all"], default="all")
    ap.add_argument("--tag", default="", help="sufijo de los archivos de salida (p. ej. _r2)")
    args = ap.parse_args()
    for dx in (cfg.DXS if args.dx == "all" else [args.dx]):
        evaluar(dx, args.tag)


if __name__ == "__main__":
    main()
