"""Cuantifica la fuga de segundo orden en la rama de destilación (KD) del repo Psiquiatria.

En Kd_bilstm_lazypredict.py y en build_oof_matrix_v4_balanceado.py el Student se
entrena con y_KD = p_teacher_OOF de los pacientes de train del fold k. Esas
probabilidades las generaron Teachers de OTROS folds, que se entrenaron CON las
etiquetas de los pacientes del fold k (el test del Student). La caché
teacher_oof_inner_cache/ (repeats 1-3) trae la versión limpia: p del Teacher
calculada con CV interna solo sobre los ~63 pacientes de train.

Mismo Student (ExtraTreesRegressor default, como en v4), mismos folds, tres
objetivos de entrenamiento distintos; AUC contra la etiqueta real en el fold de test.

Uso (desde la raíz de Multimodal/, con ../Psiquiatria clonado al lado):
    python auditoria_psiquiatria/verificar_fuga_kd.py
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
KD_DIR = REPO.parent / "Psiquiatria" / "KD_LazyPredict_Results"
EGEMAPS = REPO / "features_depresion_egemaps.csv"


def _key(s: pd.Series) -> pd.Series:
    return s.str.lower().str.replace(".wav", "", regex=False)


def main():
    teacher = pd.read_csv(KD_DIR / "teacher_oof_depresion.csv")
    teacher["key"] = _key(teacher.audio_id)
    eg = pd.read_csv(EGEMAPS)
    eg["key"] = _key(eg.audio_id)
    feats = [c for c in eg.columns if c not in ("audio_id", "seg_idx", "label", "key")]

    filas = []
    for path in sorted(glob.glob(str(KD_DIR / "teacher_oof_inner_cache" / "*.csv"))):
        r, k = map(int, re.findall(r"repeat_(\d+)_fold_(\d+)", path)[0])
        inner = pd.read_csv(path)
        inner["key"] = inner.audio_id_norm.str.lower()
        tr = teacher[teacher.repeat == r]
        test_keys = set(tr.loc[tr.fold == k, "key"])
        train_keys = set(tr.loc[tr.fold != k, "key"])
        # la caché interna debe cubrir exactamente el train externo de ese fold
        assert set(inner.key) == train_keys, (r, k)

        y = tr.set_index("key").y_real
        objetivos = {
            "fugado (OOF externo, como v4)": tr.set_index("key").p_teacher_OOF,
            "limpio (OOF interno)": inner.set_index("key").p_teacher_inner_oof,
            "etiqueta real": y.astype(float),
        }
        X_tr = eg[eg.key.isin(train_keys)]
        X_te = eg[eg.key.isin(test_keys)]
        for nombre, objetivo in objetivos.items():
            reg = ExtraTreesRegressor(random_state=42, n_jobs=-1)
            reg.fit(X_tr[feats].values, X_tr.key.map(objetivo).values)
            p = pd.Series(np.clip(reg.predict(X_te[feats].values), 0, 1)).groupby(X_te.key.values).mean()
            filas.append({"repeat": r, "fold": k, "objetivo": nombre,
                          "auc": roc_auc_score(y.loc[p.index], p.values)})

    df = pd.DataFrame(filas)
    print(df.groupby("objetivo").auc.agg(["mean", "std", "count"]).round(3))
    pv = df.pivot_table(index=["repeat", "fold"], columns="objetivo", values="auc")
    d = pv["fugado (OOF externo, como v4)"] - pv["limpio (OOF interno)"]
    print(f"\nfugado - limpio: {d.mean():+.3f} AUC en promedio; fugado > limpio en {(d > 0).sum()}/{len(d)} folds")


if __name__ == "__main__":
    main()
