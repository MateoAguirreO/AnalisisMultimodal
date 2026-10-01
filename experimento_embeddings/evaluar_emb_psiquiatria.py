"""Evalúa los embeddings ORIGINALES del repo Psiquiatria (wav2vec2-large-robust, attention
pooling con pesos aleatorios fijos durante su corrida) con el MISMO protocolo que
evaluar_ramas.py: RepeatedStratifiedKFold(5, 5, semilla 2026), selección anidada de la
configuración, AUC agrupado y AUC intra-día (pares del mismo día de grabación).

Entrada: ../psiquiatria-audio-privado/emb_psiquiatria_w2v_large_robust.npz
         claves 'dep_NNN' / 'ans_NNN' -> (n_segmentos, 1024), exportadas desde
         Clasificación Final/<Depresión|Ansiedad>/embeddings_v2 en la workstation.
Representación por participante: media de sus segmentos (lo que consumía su
ExtraTrees de embeddings). Candidatas: las mismas de E + su ExtraTrees tal cual
(ET_BEST_PARAMS de build_oof_matrix_v4).

Ramas: V (rostro), P_mismo (embeddings de la carpeta del mismo eje), P_otro (los de la
otra carpeta: misma voz, otra corrida con otro pooling aleatorio) y P_mismo+V.

Uso:  python experimento_embeddings/evaluar_emb_psiquiatria.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

import evaluar_ramas as ev

NPZ = ev.REPO.parent / "psiquiatria-audio-privado" / "emb_psiquiatria_w2v_large_robust.npz"
TAG = {"depresion": "dep", "ansiedad": "ans"}


def configs_psiquiatria():
    cfg = ev.configs_embeddings()
    et_psq = Pipeline([("clf", ExtraTreesClassifier(
        n_estimators=200, criterion="gini", max_features="log2", min_samples_leaf=1,
        min_samples_split=2, class_weight=None, random_state=ev.SEED, n_jobs=1))])
    cfg["et_psiquiatria"] = (et_psq, {})
    return cfg


def cargar_psq(npz, tag, pids):
    faltan = [p for p in pids if f"{tag}_{p:03d}" not in npz.files]
    assert not faltan, f"faltan participantes en {tag}: {faltan}"
    return np.stack([npz[f"{tag}_{p:03d}"].mean(axis=0) for p in pids])


def main():
    npz = np.load(NPZ)
    dia_s = pd.read_csv(ev.SESIONES).set_index("pid").dia
    for dx in ev.cfg.DXS:
        vid, _ = ev.cargar(dx)
        pids = vid.index.values
        y = vid["label"].values
        Xv = vid.drop(columns="label").values
        otro = TAG["ansiedad" if dx == "depresion" else "depresion"]
        X = {"P_mismo": cargar_psq(npz, TAG[dx], pids), "P_otro": cargar_psq(npz, otro, pids)}
        print(f"\n[{dx}] segmentos por participante (mediana): "
              f"{int(np.median([npz[f'{TAG[dx]}_{p:03d}'].shape[0] for p in pids]))}", flush=True)

        filas, elecciones = [], []
        outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=ev.SEED)
        for i, (tr, te) in enumerate(outer.split(pids, y)):
            s, e = {}, {}
            e["V"], s["V"] = ev.elegir_tabular(Xv, y, tr, te, ev.configs_video())
            for r, Xr in X.items():
                e[r], s[r] = ev.elegir_tabular(Xr, y, tr, te, configs_psiquiatria())
            s["P_mismo+V"] = (s["P_mismo"] + s["V"]) / 2
            filas.append(pd.DataFrame({"repeat": i // 5, "pid": pids[te], "y": y[te], **s}))
            elecciones.append({"repeat": i // 5, "fold": i % 5, **e})
            if (i + 1) % 5 == 0:
                print(f"  fold {i + 1}/25", flush=True)
        P = pd.concat(filas, ignore_index=True)
        P.to_csv(ev.OUT_DIR / f"predicciones_{dx}_psiquiatria.csv", index=False)
        E = pd.DataFrame(elecciones)
        E.to_csv(ev.OUT_DIR / f"elecciones_{dx}_psiquiatria.csv", index=False)

        dia = P.pid.map(dia_s).values
        res = []
        for m in ["V", "P_mismo", "P_otro", "P_mismo+V"]:
            intra = [ev.auc_intra_dia(g.y.values, g[m].values, dia[g.index]) for _, g in P.groupby("repeat")]
            glob_ = [roc_auc_score(g.y, g[m]) for _, g in P.groupby("repeat")]
            res.append({"metodo": m, "auc_intra_media": np.mean(intra), "auc_intra_de": np.std(intra),
                        "auc_rep_media": np.mean(glob_), "auc_rep_de": np.std(glob_)})
        R = pd.DataFrame(res).round(4)
        R.to_csv(ev.OUT_DIR / f"resumen_{dx}_psiquiatria.csv", index=False)
        print(f"=== {dx.upper()} — embeddings originales de Psiquiatria ===")
        print(R.to_string(index=False))
        print("configs elegidas:", {k: E[k].value_counts().to_dict() for k in ["P_mismo", "P_otro"]},
              flush=True)


if __name__ == "__main__":
    main()
