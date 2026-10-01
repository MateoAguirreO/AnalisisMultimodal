"""Búsqueda HONESTA del mejor modelo por eje: AUC estándar sobre todas las muestras juntas,
5x5 anidado, semilla 2026 (ver memoria feedback_metrica_auc: nada de AUC intra-día).

Ramas:
  V   rostro: AU / pose / emoción de py-feat (dataset_au_features.csv, 66 features)
  VT  rostro: 256 features temporales de MediaPipe (features/video_features_v2.csv, v2/v3)
  R   voz: micro-ventanas de 0.19 s, última capa de wav2vec2-large-robust (audio editado;
      réplica determinista de Psiquiatria, features/emb_replica_psiquiatria.npz)
Candidatos: cada rama sola y sus fusiones por soft vote (promedio de probabilidades).

AUTO: en cada fold externo elige el candidato con mayor AUC OOF DENTRO del train externo y
lo aplica al test externo. Su AUC externo es la estimación honesta de "quedarse con el
mejor modelo" (el máximo de la tabla de candidatos es optimista por construcción).

Muestras: 79 si hay alguna rama de voz (el participante 66 no tiene audio editado), 80 con
--solo-rostro.
Reanudable: guarda cada fold en resultados/cache_busqueda_<dx>_<n>/ y sale al pasar
--minutos, para caber en el límite de tiempo de un comando. Al completar los 25 folds,
imprime el resumen.

Uso:  python experimento_embeddings/buscar_mejor_modelo.py --dx depresion
      python experimento_embeddings/buscar_mejor_modelo.py --dx ansiedad --solo-rostro
"""
import argparse
import itertools
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import evaluar_ramas as ev
from evaluar_emb_psiquiatria import configs_psiquiatria

AU_CSV = ev.REPO / "dataset_au_features.csv"
VT_CSV = ev.REPO / "features" / "video_features_v2.csv"
R_NPZ = ev.REPO / "features" / "emb_replica_psiquiatria.npz"
RD_DIR = ev.REPO / "features" / "emb_microdensas"
RD_BLOQUES = ["d010", "d019", "d050", "d100"]   # duración del trozo: 0.1, 0.19, 0.5, 1.0 s
TXT_NPZ = ev.REPO / "features" / "texto_features.npz"
PUNTAJES = ev.REPO / "features" / "puntajes_phq_gad.csv"
OBJETIVOS = {"dx": None, "phq5": ("PHQ9", 5), "gad5": ("GAD7", 5)}


def etiquetas(dx, objetivo):
    """dx: etiqueta 'DX IA' del proyecto (origen desconocido). phq5 / gad5: síntomas al menos
    leves en el cuestionario validado (PHQ-9 >= 5 / GAD-7 >= 5; 69 participantes)."""
    if objetivo == "dx":
        au = pd.read_csv(AU_CSV).set_index("video_id")
        return au[f"target_{dx}"].astype(int)
    col, corte = OBJETIVOS[objetivo]
    s = pd.read_csv(PUNTAJES, index_col=0)[col].dropna()
    return (s >= corte).astype(int)


def configs_texto(n_emb=768, n_lex=7):
    """Rama de texto: la CV interna elige entre embeddings de frases (configs de voz) y las 7
    features léxicas interpretables (logística regularizada)."""
    emb = ColumnTransformer([("b", "passthrough", list(range(n_emb)))], remainder="drop")
    lex = ColumnTransformer([("b", "passthrough", list(range(n_emb, n_emb + n_lex)))], remainder="drop")
    out = {f"{c}_emb": (Pipeline([("bloque", emb)] + list(p.steps)), dict(g))
           for c, (p, g) in configs_psiquiatria().items()}
    out["logreg_lex"] = (Pipeline([("bloque", lex), ("sc", StandardScaler()),
                                   ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]),
                         {"clf__C": [0.01, 0.1, 1.0]})
    return out


def configs_densas(dim=1024):
    """Configs de voz (las mismas de R) x duración de micro-ventana. Cada config selecciona
    su bloque de columnas: así la duración se elige en la CV interna, como un
    hiperparámetro más, y no a ojo."""
    out = {}
    for b, nombre in enumerate(RD_BLOQUES):
        sel = ColumnTransformer([("b", "passthrough", list(range(b * dim, (b + 1) * dim)))],
                                remainder="drop")
        for c, (pipe, grid) in configs_psiquiatria().items():
            out[f"{c}_{nombre}"] = (Pipeline([("bloque", sel)] + list(pipe.steps)), dict(grid))
    return out


def cargar_ramas(dx, solo_rostro, pedidas=("R", "V", "VT"), objetivo="dx"):
    au = pd.read_csv(AU_CSV).rename(columns={"video_id": "pid"}).set_index("pid")
    y_all = etiquetas(dx, objetivo)
    au_feats = [c for c in au.columns if c not in ("error", "n_frames_detected", "target_ansiedad",
                                                   "target_depresion")]
    vt = pd.read_csv(VT_CSV).set_index("codigo")
    vt_feats = [c for c in vt.columns if c != "participant"]
    pids = sorted(set(au.index) & set(vt.index) & set(y_all.index))
    if "T" in pedidas:
        t = np.load(TXT_NPZ)
        fila_t = {int(p): i for i, p in enumerate(t["pids"])}
        pids = [p for p in pids if p in fila_t]
    voz = {}
    if not solo_rostro and "R" in pedidas:
        z = np.load(R_NPZ)
        pids = [p for p in pids if f"{p:03d}" in z.files]
        voz["R"] = lambda ps: (np.stack([z[f"{p:03d}"].mean(0) for p in ps]), configs_psiquiatria())
    if not solo_rostro and "RD" in pedidas:
        pids = [p for p in pids if (RD_DIR / f"{p:03d}.npz").exists()]
        voz["RD"] = lambda ps: (np.stack([np.concatenate([np.load(RD_DIR / f"{p:03d}.npz")[b]
                                                          for b in RD_BLOQUES]) for p in ps]),
                                configs_densas())
    pids = np.array(pids)
    ramas = {r: f(pids) for r, f in voz.items()}
    if "T" in pedidas:
        idx = [fila_t[p] for p in pids]
        ramas["T"] = (np.hstack([t["emb"][idx], t["lex"][idx]]).astype(float),
                      configs_texto(t["emb"].shape[1], t["lex"].shape[1]))
    if "V" in pedidas:
        ramas["V"] = (au.loc[pids, au_feats].values.astype(float), ev.configs_video())
    if "VT" in pedidas:
        ramas["VT"] = (vt.loc[pids, vt_feats].values.astype(float), ev.configs_video())
    return pids, y_all.loc[pids].values, ramas


def candidatos(ramas):
    nombres = list(ramas)
    out = {}
    for k in range(1, len(nombres) + 1):
        for combo in itertools.combinations(nombres, k):
            out["+".join(combo)] = list(combo)
    return out


def elegir_con_oof(X, y, tr, te, configs):
    """Elige la config por CV interna sobre tr (como ev.elegir_tabular) y devuelve además
    las predicciones OOF de esa config dentro de tr (para elegir entre candidatos)."""
    mejor = (None, -np.inf, None)
    for nombre, (pipe, grid) in configs.items():
        if grid:
            gs = GridSearchCV(clone(pipe), grid, cv=ev.inner_cv(), scoring="roc_auc",
                              n_jobs=ev.N_JOBS).fit(X[tr], y[tr])
            score, est = gs.best_score_, gs.best_estimator_
        else:
            score = cross_val_score(clone(pipe), X[tr], y[tr], cv=ev.inner_cv(),
                                    scoring="roc_auc", n_jobs=ev.N_JOBS).mean()
            est = clone(pipe).fit(X[tr], y[tr])
        if score > mejor[1]:
            mejor = (nombre, score, est)
    nombre, _, est = mejor
    oof = cross_val_predict(clone(est), X[tr], y[tr], cv=ev.inner_cv(),
                            method="predict_proba", n_jobs=ev.N_JOBS)[:, 1]
    return nombre, oof, est.predict_proba(X[te])[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", required=True, choices=ev.cfg.DXS)
    ap.add_argument("--solo-rostro", action="store_true")
    ap.add_argument("--minutos", type=float, default=8.5)
    ap.add_argument("--ramas", default="R,V,VT", help="ramas que compiten, p. ej. R,RD,V,T")
    ap.add_argument("--objetivo", default="dx", choices=list(OBJETIVOS),
                    help="dx = 'DX IA'; phq5 / gad5 = PHQ-9 >= 5 / GAD-7 >= 5")
    args = ap.parse_args()
    t0 = time.time()

    pedidas = args.ramas.split(",")
    pids, y, ramas = cargar_ramas(args.dx, args.solo_rostro, pedidas, args.objetivo)
    cands = candidatos(ramas)
    sufijo = "" if pedidas == ["R", "V", "VT"] else "_" + "-".join(pedidas)
    if args.objetivo != "dx":
        sufijo += f"_{args.objetivo}"
    cache = ev.OUT_DIR / f"cache_busqueda_{args.dx}_{len(pids)}{sufijo}"
    cache.mkdir(parents=True, exist_ok=True)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=ev.SEED).split(pids, y))
    print(f"[{args.dx} | objetivo={args.objetivo}] n={len(pids)} (pos={int(y.sum())}) ramas={list(ramas)} "
          f"candidatos={list(cands)}", flush=True)

    for i, (tr, te) in enumerate(splits):
        f = cache / f"fold_{i:02d}.pkl"
        if f.exists():
            continue
        if (time.time() - t0) / 60 > args.minutos:
            print(f"  tiempo agotado; {sum(1 for _ in cache.glob('fold_*.pkl'))}/25 folds en caché", flush=True)
            return
        oof, test, elegidas = {}, {}, {}
        for r, (X, cfgs) in ramas.items():
            elegidas[r], oof[r], test[r] = elegir_con_oof(X, y, tr, te, cfgs)
        fila = {"repeat": i // 5, "pid": pids[te], "y": y[te], "config_rama": elegidas}
        auc_tr = {}
        for c, rs in cands.items():
            fila[c] = np.mean([test[r] for r in rs], axis=0)
            auc_tr[c] = roc_auc_score(y[tr], np.mean([oof[r] for r in rs], axis=0))
        fila["AUTO_eleccion"] = max(auc_tr, key=auc_tr.get)
        fila["AUTO"] = fila[fila["AUTO_eleccion"]]
        fila["auc_train_oof"] = auc_tr
        with open(f, "wb") as fh:
            pickle.dump(fila, fh)
        print(f"  fold {i + 1}/25  AUTO eligió {fila['AUTO_eleccion']}  ({time.time() - t0:.0f}s)", flush=True)

    folds = [pickle.load(open(cache / f"fold_{i:02d}.pkl", "rb")) for i in range(25)]
    P = pd.concat([pd.DataFrame({k: v for k, v in d.items()
                                 if k not in ("config_rama", "AUTO_eleccion", "auc_train_oof")})
                   for d in folds], ignore_index=True)
    P.to_csv(ev.OUT_DIR / f"predicciones_busqueda_{args.dx}_{len(pids)}{sufijo}.csv", index=False)
    filas = []
    for c in list(cands) + ["AUTO"]:
        rep = [roc_auc_score(g.y, g[c]) for _, g in P.groupby("repeat")]
        filas.append({"modelo": c, "auc_media": np.mean(rep), "auc_de": np.std(rep),
                      **{f"rep{r}": v for r, v in enumerate(rep)}})
    R = pd.DataFrame(filas).sort_values("auc_media", ascending=False).round(4)
    R.to_csv(ev.OUT_DIR / f"resumen_busqueda_{args.dx}_{len(pids)}{sufijo}.csv", index=False)
    eleccion = pd.Series([d["AUTO_eleccion"] for d in folds]).value_counts().to_dict()
    if "RD" in ramas:
        durs = pd.Series([d["config_rama"]["RD"].rsplit("_", 1)[1] for d in folds]).value_counts().to_dict()
        print(f"RD: duración de micro-ventana elegida en la CV interna (25 folds): {durs}")
    print(f"\n=== {args.dx.upper()} | objetivo={args.objetivo} — n={len(pids)} — AUC estándar "
          f"(media de 5 repeticiones) ===")
    print(R[["modelo", "auc_media", "auc_de"]].to_string(index=False))
    print(f"AUTO eligió (25 folds): {eleccion}", flush=True)


if __name__ == "__main__":
    main()
