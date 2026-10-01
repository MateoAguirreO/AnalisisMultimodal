"""Sesión (día) de grabación de cada participante, desde la fecha de modificación del mp4.

Los mp4/m4a no traen creation_time en metadatos, pero la fecha de modificación de los
archivos conserva la hora de grabación (coincide con el creation_time de los m4a).
Resultado: 7 días en dos campañas (19-22 nov y 9-11 dic 2025). El día se usa para el
AUC intra-día (control del confusor de sesión, ver RESULTADOS.md §3).

Salida: features/sesiones.csv (pid, t, dia)  (gitignored)
Uso:  python experimento_embeddings/sesiones_grabacion.py
"""
import datetime as dt
import glob
import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
MUESTRAS = REPO.parent / "analisis-espectrogramas" / ".MUESTRAS_SAMANÁ" / "Muestras"
OUT = REPO / "features" / "sesiones.csv"


def main():
    filas = []
    for carpeta in sorted(MUESTRAS.iterdir()):
        if not (carpeta.is_dir() and carpeta.name[:3].isdigit()):
            continue
        mp4 = [f for f in glob.glob(str(carpeta / "*.mp4")) if Path(f).name.startswith(carpeta.name[:3])]
        if mp4:
            t = dt.datetime.fromtimestamp(os.path.getmtime(mp4[0]))
            filas.append({"pid": int(carpeta.name[:3]), "t": t, "dia": t.date().isoformat()})
    df = pd.DataFrame(filas).sort_values("pid")
    df.to_csv(OUT, index=False)
    print(df.groupby("dia").pid.agg(["size", "min", "max"]))
    print(f"OK -> {OUT}")


if __name__ == "__main__":
    main()
