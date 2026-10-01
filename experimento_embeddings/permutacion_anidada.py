"""Test de permutación que REPITE TODA LA SELECCIÓN ANIDADA con etiquetas barajadas.

A diferencia de pipeline_multimodal_xai/permutation_multimodal.py (que congela la
arquitectura ganadora y solo reentrena), aquí cada permutación vuelve a elegir la
configuración de cada rama en la CV interna: el p-valor incluye el sesgo de selección.

Dos nulos:
  por defecto   barajado global; estadístico = AUC agrupado (79 participantes) de UNA
                repetición 5-fold externa (misma semilla para real y barajadas).
  --intra-dia   barajado DENTRO de cada día de grabación (conserva cuántos positivos hubo
                cada día, es decir, el efecto de sesión queda también en el nulo);
                estadístico principal = AUC intra-día. Es el nulo correcto si se quiere
                afirmar señal "más allá de la sesión".
p = (1 + #{estadístico_perm >= estadístico_real}) / (N_PERM + 1)   (Ojala & Garriga 2010)

Ramas: V (rostro), E / E_ed (nuestros embeddings, audio crudo / editado),
R (réplica determinista de los embeddings de Psiquiatria), P (sus embeddings originales),
A_ed / A_crudo (eGeMAPS por segmento). Fusión = soft vote, p. ej. --metodo R+V.

Uso:  python experimento_embeddings/permutacion_anidada.py --dx depresion --metodo R+V --intra-dia
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

import evaluar_ramas as ev

OUT_DIR = ev.OUT_DIR
NPZ_REPLICA = ev.REPO / "features" / "emb_replica_psiquiatria.npz"
NPZ_PSQ = ev.REPO.parent / "psiquiatria-audio-privado" / "emb_psiquiatria_w2v_large_robust.npz"


def cargar_datos(dx, ramas):
    vid, aud_ed = ev.cargar(dx)
    pids = vid.index.values
    tab = {}
    if "V" in ramas:
        tab["V"] = (vid.drop(columns="label").values, ev.configs_video())
    if "E" in ramas:
        tab["E"] = (ev.cargar_embeddings(pids), ev.configs_embeddings())
    if "E_ed" in ramas:
        tab["E_ed"] = (ev.cargar_embeddings(pids, ev.EMB_DIR_ED), ev.configs_embeddings())
    if "R" in ramas or "P" in ramas:
        from evaluar_emb_psiquiatria import configs_psiquiatria
        if "R" in ramas:
            z = np.load(NPZ_REPLICA)
            tab["R"] = (np.stack([z[f"{p:03d}"].mean(0) for p in pids]), configs_psiquiatria())
        if "P" in ramas:
            z = np.load(NPZ_PSQ)
            tab["P"] = (np.stack([z[f"dep_{p:03d}"].mean(0) for p in pids]), configs_psiquiatria())
    aud = {}
    if "A_ed" in ramas:
        aud["A_ed"] = aud_ed.drop(columns="label")
    if "A_crudo" in ramas:
        cr = pd.read_csv(ev.EGEMAPS_CRUDO)
        feats = [c for c in aud_ed.columns if c not in ("pid", "seg_idx", "label")]
        aud["A_crudo"] = cr[cr.pid.isin(pids)][["pid", "seg_idx"] + feats]
    dia = pd.read_csv(ev.SESIONES).set_index("pid").dia.loc[pids].values
    return vid["label"].values, pids, dia, tab, aud


def una_corrida(y, pids, dia, tab, aud, ramas, seed_split, n_rep=1):
    """Selección anidada completa con las etiquetas y dadas, sobre
    RepeatedStratifiedKFold(5, n_rep, seed) -> medias sobre repeticiones de
    (AUC agrupado, AUC intra-día). Con n_rep=5 y seed=2026 son exactamente los folds de
    evaluar_ramas.py, así que el valor real debe coincidir con sus tablas."""
    ev.N_JOBS = 1  # corre dentro de un proceso de joblib: la CV interna va en serie
    score = np.zeros((n_rep, len(y)))
    splits = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed_split)
    for i, (tr, te) in enumerate(splits.split(pids, y)):
        partes = []
        for r in ramas:
            if r in tab:
                X, cfgs = tab[r]
                partes.append(ev.elegir_tabular(X, y, tr, te, cfgs)[1])
            else:
                df = aud[r].assign(label=aud[r].pid.map(dict(zip(pids, y))))
                partes.append(ev.elegir_audio(df[df.pid.isin(pids[tr])], df[df.pid.isin(pids[te])],
                                              pids[tr], pids[te], y[tr])[2])
        score[i // 5, te] = np.mean(partes, axis=0)
    return (float(np.mean([roc_auc_score(y, s) for s in score])),
            float(np.mean([ev.auc_intra_dia(y, s, dia) for s in score])))


def barajar(y, dia, rng, intra_dia):
    if not intra_dia:
        return rng.permutation(y)
    yp = y.copy()
    for d in np.unique(dia):
        idx = np.where(dia == d)[0]
        yp[idx] = rng.permutation(y[idx])
    return yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", required=True)
    ap.add_argument("--metodo", required=True, help="rama o fusión, p. ej. V, R, R+V, A_ed+V")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=os.cpu_count())
    ap.add_argument("--intra-dia", action="store_true")
    ap.add_argument("--n-rep", type=int, default=1,
                    help="repeticiones 5-fold externas promediadas en el estadístico (5 = folds de evaluar_ramas)")
    ap.add_argument("--lote", type=int, default=None,
                    help="calcula solo este número de permutaciones pendientes y sale (reanudable: "
                         "cada llamada cabe en el límite de tiempo de un comando)")
    ap.add_argument("--finalizar", action="store_true",
                    help="calcula el p-valor con las permutaciones que haya en caché")
    args = ap.parse_args()

    ramas = args.metodo.split("+")
    etiqueta = f"[{args.dx} {args.metodo}{' intra-dia' if args.intra_dia else ''} rep={args.n_rep}]"
    y, pids, dia, tab, aud = cargar_datos(args.dx, ramas)
    # la secuencia de barajes depende solo de la semilla: el baraje k es el mismo en cualquier lote
    rng = np.random.default_rng(ev.SEED)
    perms = [barajar(y, dia, rng, args.intra_dia) for _ in range(args.n_perm)]
    nulo = "intra_dia" if args.intra_dia else "global"
    cache = OUT_DIR / f"perm_cache_{args.dx}_{args.metodo.replace('+', '-')}_{nulo}_rep{args.n_rep}.csv"
    C = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["idx", "glob", "intra"])
    t0 = time.time()

    if not args.finalizar:
        if -1 not in set(C.idx):  # idx = -1 es la corrida con las etiquetas reales
            g, i_ = una_corrida(y, pids, dia, tab, aud, ramas, ev.SEED, args.n_rep)
            C = pd.concat([C, pd.DataFrame([{"idx": -1, "glob": g, "intra": i_}])], ignore_index=True)
            C.to_csv(cache, index=False)
            print(f"{etiqueta} real: AUC agrupado={g:.4f}  intra-día={i_:.4f}  "
                  f"({time.time() - t0:.0f}s/corrida)", flush=True)
            if args.lote:
                return
        pendientes = [k for k in range(args.n_perm) if k not in set(C.idx)]
        if args.lote:
            pendientes = pendientes[:args.lote]
        if pendientes:
            res = Parallel(n_jobs=args.n_jobs)(
                delayed(una_corrida)(perms[k], pids, dia, tab, aud, ramas, ev.SEED, args.n_rep)
                for k in pendientes)
            nuevos = pd.DataFrame([{"idx": k, "glob": g, "intra": i_} for k, (g, i_) in zip(pendientes, res)])
            C = pd.concat([C, nuevos], ignore_index=True)
            C.to_csv(cache, index=False)
        hechos = int((C.idx >= 0).sum())
        print(f"{etiqueta} permutaciones en caché: {hechos}/{args.n_perm}  "
              f"({time.time() - t0:.0f}s en esta llamada)", flush=True)
        if hechos < args.n_perm:
            return

    real_row = C[C.idx == -1].iloc[0]
    N = C[C.idx >= 0]
    est = "intra" if args.intra_dia else "glob"   # estadístico principal según el nulo
    real, nulos = float(real_row[est]), N[est].values
    p = (1 + np.sum(nulos >= real)) / (len(nulos) + 1)
    fila = {"dx": args.dx, "metodo": args.metodo, "nulo": nulo,
            "estadistico": "auc_intra_dia" if args.intra_dia else "auc_agrupado",
            "real": round(real, 4), "nulo_media": round(float(nulos.mean()), 4),
            "nulo_p95": round(float(np.percentile(nulos, 95)), 4),
            "p_valor": round(float(p), 4), "n_perm": len(nulos), "n_rep": args.n_rep,
            "real_auc_agrupado": round(float(real_row["glob"]), 4),
            "real_auc_intra": round(float(real_row["intra"]), 4)}
    print(fila, flush=True)
    out = OUT_DIR / "permutacion_anidada.csv"
    prev = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=list(fila))
    col = lambda c, defecto: prev[c] if c in prev.columns else pd.Series(defecto, index=prev.index)
    # filas viejas (ronda 1) no traen 'nulo' ni 'n_rep': eran nulo global con 1 repetición
    clave = (prev.dx == fila["dx"]) & (prev.metodo == fila["metodo"]) \
        & (col("nulo", "global").fillna("global") == fila["nulo"]) \
        & (col("n_rep", 1).fillna(1) == fila["n_rep"])
    pd.concat([prev[~clave], pd.DataFrame([fila])], ignore_index=True).to_csv(out, index=False)


if __name__ == "__main__":
    main()
